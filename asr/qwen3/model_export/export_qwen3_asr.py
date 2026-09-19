# SPDX-License-Identifier: Apache-2.0
"""Export Qwen3-ASR with explicit tensor interfaces; checkpoint BF16 is the default."""

import argparse
import json
import sys
from pathlib import Path

import onnx
import torch
import transformers
from torch import nn
from torch.nn import functional as F
from transformers import (
    AttentionInterface,
    AutoProcessor,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRForTokenClassification,
)
from transformers.models.qwen3_asr.processing_qwen3_asr import LANGUAGE_CODE_TO_NAME

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.model_export.log_mel import LogMel as WhisperMel  # noqa: E402


class LogMel(WhisperMel):
    def __init__(self, processor):
        fe = processor.feature_extractor
        if (fe.n_fft, fe.hop_length, fe.feature_size, fe.sampling_rate, fe.dither) != (400, 160, 128, 16000, 0):
            raise ValueError("Native frontend requires the standard Qwen3 16 kHz configuration")
        super().__init__(fe.mel_filters.T.copy(), torch.float32, frames=None)


def save_native_assets(processor, output, task):
    tokenizer = processor.tokenizer

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    data = {
        "audio_start": encode(processor.audio_bos_token),
        "audio_end": encode(processor.audio_eos_token),
    }
    if task == "asr":
        prefixes = {}
        suffixes = {}
        languages = {}
        for language in [None, *LANGUAGE_CODE_TO_NAME.values()]:
            messages = [{"role": "user", "content": [{"type": "audio"}]}]
            rendered = tokenizer.apply_chat_template(
                messages, chat_template=processor.chat_template, tokenize=False, add_generation_prompt=True
            )
            prefix, suffix = rendered.split(processor.audio_token)
            prefixes[language or "auto"] = encode(prefix)
            suffixes[language or "auto"] = encode(suffix + (f"language {language}<asr_text>" if language else ""))
            languages[language or "auto"] = language or ""
        for code, language in LANGUAGE_CODE_TO_NAME.items():
            prefixes[code] = prefixes[language]
            suffixes[code] = suffixes[language]
            languages[code] = language
        data["prefixes"] = prefixes
        data["suffixes"] = suffixes
        data["languages"] = languages
        data["suffix"] = suffixes["auto"]
    (output / "native.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def decode_attention(module, query, key, value, attention_mask, scaling=None, **kwargs):
    if query.dtype == torch.float32:
        return export_attention(module, query, key, value, attention_mask, scaling, **kwargs)
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        scale=scaling,
        enable_gqa=query.shape[1] != key.shape[1],
    )
    return output.transpose(1, 2), None


class DecodeCache:
    def __init__(self, past, positions):
        self.past, self.positions, self.present = past, positions.reshape(-1), []

    def update(self, key, value, layer_idx):
        updated = []
        for cache, data in zip(self.past[2 * layer_idx : 2 * layer_idx + 2], (key, value), strict=True):
            if torch.onnx.is_in_onnx_export():
                data = torch.onnx.ops.symbolic(
                    "TensorScatter",
                    (cache, data, self.positions[:1]),
                    {"axis": 2, "mode": "linear"},
                    dtype=data.dtype,
                    shape=cache.shape,
                    version=24,
                )
            else:
                data = cache.index_copy(2, self.positions, data)
            updated.append(data)
        self.present.extend(updated)
        return tuple(updated)


def export_decode(decoder, config, dtype, output, capacities):
    AttentionInterface.register("qwen3_onnx_decode", decode_attention)
    decoder.decoder.config._attn_implementation = "qwen3_onnx_decode"
    decoder.cache_type = DecodeCache
    past = tuple(
        torch.zeros(1, config.num_key_value_heads, 4, config.head_dim, dtype=dtype)
        for _ in range(2 * config.num_hidden_layers)
    )
    inputs = (
        torch.ones(1, 1, dtype=torch.int64),
        torch.zeros(1, 1, config.hidden_size, dtype=dtype),
        torch.zeros(1, 1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.int64),
        torch.zeros(1, 1, 1, 4, dtype=dtype),
        *past,
    )
    cap = torch.export.Dim("capacity", min=4)
    names = ["input_ids", "audio_embeddings", "audio_mask", "position_ids", "attention_bias"]
    # Export weights once; specialized graph headers share the same external data.
    template = output / "decode.onnx"
    torch.onnx.export(
        decoder.eval(),
        inputs,
        str(template),
        dynamo=True,
        opset_version=24,
        external_data=True,
        input_names=names + [f"past_{i}" for i in range(len(past))],
        output_names=["logits", "next_token"] + [f"present_{i}" for i in range(len(past))],
        dynamic_shapes={
            "input_ids": {},
            "audio_embeddings": {},
            "audio_mask": {},
            "position_ids": {},
            "attention_bias": {3: cap},
            "past": tuple({2: cap} for _ in past),
        },
    )
    for capacity in sorted(set(capacities)):
        graph = onnx.load(template, load_external_data=False)
        for value in list(graph.graph.input) + list(graph.graph.output) + list(graph.graph.value_info):
            for dim in value.type.tensor_type.shape.dim:
                if dim.dim_param == "capacity":
                    dim.dim_value = capacity
        path = output / f"decode_{capacity}.onnx"
        onnx.save(graph, path)
        onnx.checker.check_model(str(path))
        print(f"Checked {path}", flush=True)


def attention(q, k, v, bias, scale):
    scores = q @ k.transpose(-1, -2) * scale + bias
    return torch.softmax(scores.float(), dim=-1).to(v.dtype) @ v


class AudioEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.audio_tower
        self.projector = model.model.multi_modal_projector

    def forward(self, mel_chunks, valid_indices, attention_bias):
        enc = self.encoder
        x = F.gelu(enc.conv2d1(mel_chunks.unsqueeze(1)))
        x = F.gelu(enc.conv2d2(x))
        x = F.gelu(enc.conv2d3(x))
        x = x.permute(0, 3, 1, 2).flatten(2)
        x = enc.conv_out(x)
        x = x + enc.positional_embedding.positional_embedding[: x.shape[1]].to(x.dtype)
        x = x.flatten(0, 1).index_select(0, valid_indices).unsqueeze(0)
        for layer in enc.layers:
            y = layer.self_attn_layer_norm(x)
            attn = layer.self_attn
            q, k, v = [
                p(y).reshape(1, -1, attn.num_heads, attn.head_dim).transpose(1, 2)
                for p in (attn.q_proj, attn.k_proj, attn.v_proj)
            ]
            y = attention(q, k, v, attention_bias, attn.scaling)
            x = x + attn.out_proj(y.transpose(1, 2).flatten(2))
            y = layer.final_layer_norm(x)
            x = x + layer.fc2(layer.activation_fn(layer.fc1(y)))
        return self.projector(enc.ln_post(x)).squeeze(0)


def export_attention(module, query, key, value, attention_mask, scaling=None, **kwargs):
    # Fold query groups into the query axis. TensorRT RTX otherwise lowers GQA
    # to full-size KV-head replication. This is ordinary MHA with identical math
    # and the original smaller KV tensors; only the small query/mask is repeated.
    batch, heads, sequence, width = query.shape
    kv_heads = key.shape[1]
    groups = heads // kv_heads
    query = query.reshape(batch, kv_heads, groups * sequence, width)
    if attention_mask is not None:
        attention_mask = attention_mask.repeat(1, 1, groups, 1)
    # TensorRT RTX 1.6 has no fused FP32 Attention kernel for this graph.
    if query.dtype == torch.float32:
        output = attention(query, key, value, attention_mask, scaling)
    else:
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, scale=scaling)
    output = output.reshape(batch, heads, sequence, width)
    return output.transpose(1, 2), None


AttentionInterface.register("qwen3_onnx", export_attention)


class ExportCache:
    """Tensor-only HF cache adapter, using Whisper's fixed-capacity index updates."""

    def __init__(self, past, positions):
        self.past = past
        self.positions = positions.reshape(-1)
        self.present = []

    def update(self, key, value, layer_idx):
        key = self.past[2 * layer_idx].index_copy(2, self.positions, key)
        value = self.past[2 * layer_idx + 1].index_copy(2, self.positions, value)
        self.present.extend((key, value))
        return key, value


class TextBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.language_model
        self.decoder.config._attn_implementation = "qwen3_onnx"
        self.cache_type = ExportCache

    def hidden(self, input_ids, audio_embeddings, audio_mask, position_ids, attention_bias, *past):
        dec = self.decoder
        x = torch.where(audio_mask, audio_embeddings, dec.embed_tokens(input_ids))
        rotary = dec.rotary_emb(x, position_ids)
        cache = self.cache_type(past, position_ids) if past else None
        for layer in dec.layers:
            x = layer(x, attention_mask=attention_bias, position_embeddings=rotary, past_key_values=cache)
        return dec.norm(x), cache.present if cache else []


class TextDecoder(TextBackbone):
    """Prompt prefill and single-token decoding with fixed-capacity KV tensors."""

    def __init__(self, model):
        super().__init__(model)
        self.lm_head = model.lm_head

    def forward(self, input_ids, audio_embeddings, audio_mask, position_ids, attention_bias, *past):
        hidden, present = self.hidden(input_ids, audio_embeddings, audio_mask, position_ids, attention_bias, *past)
        logits = self.lm_head(hidden[:, -1])
        return logits, logits.argmax(-1), *present


class ForcedAligner(TextBackbone):
    """Predict all timestamp slots together, without an autoregressive cache."""

    def __init__(self, model):
        super().__init__(model)
        self.score = model.score

    def forward(self, input_ids, audio_embeddings, audio_mask, position_ids, attention_bias, timestamp_indices):
        hidden, _ = self.hidden(input_ids, audio_embeddings, audio_mask, position_ids, attention_bias)
        logits = self.score(hidden.index_select(1, timestamp_indices))
        return logits, logits.argmax(-1)


def export(module, args, path, names, outputs, shapes):
    print(f"Exporting {path.name}", flush=True)
    torch.onnx.export(
        module.eval(),
        args,
        str(path),
        input_names=names,
        output_names=outputs,
        dynamo=True,
        dynamic_shapes=shapes,
        opset_version=23,
        external_data=True,
    )
    onnx.checker.check_model(str(path))
    print(f"Checked {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--checkpoint", dest="model", help="HF model ID or local checkpoint directory")
    parser.add_argument("--size", choices=["0.6B", "1.7B"], default="0.6B", help="ASR model size")
    parser.add_argument("--revision", help="Optional HF revision or commit")
    parser.add_argument("--output", type=Path, help="Defaults to the ONNX artifact directory for --task")
    parser.add_argument("--task", choices=["asr", "aligner"], default="asr")
    parser.add_argument("--dtype", choices=["original", "fp16", "fp32"], default="original")
    parser.add_argument("--only", choices=["mel", "encoder", "decoder", "decode", "aligner"])
    parser.add_argument(
        "--decode-capacities",
        type=int,
        nargs="+",
        help="Export optional in-place one-token graphs for these KV capacities (TensorRT RTX)",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-capacity", type=int, default=32768)
    args = parser.parse_args()
    prefix = "aligner-" if args.task == "aligner" else ""
    args.model = args.model or (
        "Qwen/Qwen3-ForcedAligner-0.6B-hf" if args.task == "aligner" else f"Qwen/Qwen3-ASR-{args.size}-hf"
    )
    precision = "bf16" if args.dtype == "original" else args.dtype
    size_suffix = "-1.7b" if args.task == "asr" and "1.7b" in args.model.lower() else ""
    args.output = args.output or Path(f"artifacts/qwen3/{prefix}onnx-{precision}{size_suffix}")
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.cache_capacity < 4:
        parser.error("cache-capacity must be at least 4")
    if args.decode_capacities and (
        args.task != "asr" or any(c < 4 or c > args.cache_capacity for c in args.decode_capacities)
    ):
        parser.error("decode-capacities requires ASR and capacities between 4 and cache-capacity")
    if args.decode_capacities and max(args.decode_capacities) > 16384:
        parser.error("TensorRT RTX 1.6 GQA decode supports up to 16384 slots; larger buckets use the general decoder")
    if args.only == "decode" and (not args.decode_capacities or not (args.output / "metadata.json").exists()):
        parser.error("--only decode requires --decode-capacities and an existing ASR export")
    metadata_path = args.output / "metadata.json"
    if args.only not in (None, "mel") and metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        requested = {"original": "bfloat16", "fp16": "float16", "fp32": "float32"}[args.dtype]
        if existing["dtype"] != requested:
            parser.error("Partial export must keep the existing precision; use a separate output directory")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.only == "decoder" and args.task != "asr" or args.only == "aligner" and args.task != "aligner":
        parser.error("--only must match --task")
    if args.only == "mel":
        processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
        processor.save_pretrained(args.output / "processor")
        save_native_assets(processor, args.output, args.task)
        export_mel(processor, args.output)
        return
    model_class = Qwen3ASRForConditionalGeneration if args.task == "asr" else Qwen3ASRForTokenClassification
    model = model_class.from_pretrained(
        args.model,
        dtype={"original": "auto", "fp16": torch.float16, "fp32": torch.float32}[args.dtype],
        attn_implementation="eager",
        revision=args.revision,
    ).eval()
    revision = model.config._commit_hash or args.revision
    if args.only != "decode":
        processor = AutoProcessor.from_pretrained(args.model, revision=revision)
        processor.save_pretrained(args.output / "processor")
        save_native_assets(processor, args.output, args.task)
    if args.only is None:
        export_mel(processor, args.output)
    cfg = model.config
    dtype = model.dtype
    if args.dtype == "original" and dtype != torch.bfloat16:
        raise ValueError(f"Expected the original BF16 Qwen3 checkpoint, got {dtype}")
    if cfg.audio_config.n_window != 50 or cfg.audio_config.num_mel_bins != 128:
        raise ValueError("This export contract requires n_window=50 and 128 mel bins")
    if cfg.text_config.rope_parameters["rope_type"] != "default":
        raise ValueError("Only default RoPE is supported by this draft")
    if args.only == "decode":

        metadata = json.loads((args.output / "metadata.json").read_text(encoding="utf-8"))
        if (
            metadata["task"] != "asr"
            or metadata["dtype"] != str(dtype).removeprefix("torch.")
            or max(args.decode_capacities) > metadata["cache_capacity"]
        ):
            raise ValueError("Decode export must match the existing ASR precision and cache capacity")
        for key in ("hidden_size", "num_hidden_layers", "num_key_value_heads", "head_dim", "vocab_size"):
            if metadata["text_config"][key] != getattr(cfg.text_config, key):
                raise ValueError(f"Decode export differs from the existing ASR config: {key}")
        with torch.inference_mode():
            export_decode(TextDecoder(model), cfg.text_config, dtype, args.output, args.decode_capacities)
        metadata["decode_capacities"] = sorted(set(args.decode_capacities))
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return
    d = torch.export.Dim
    with torch.inference_mode():
        if args.only in (None, "encoder"):
            export(
                AudioEncoder(model),
                (torch.zeros(2, 128, 100, dtype=dtype), torch.arange(26), torch.zeros(1, 1, 26, 26, dtype=dtype)),
                args.output / "encoder.onnx",
                ["mel_chunks", "valid_indices", "attention_bias"],
                ["audio_embeddings"],
                {
                    "mel_chunks": {0: d("chunks", min=1)},
                    "valid_indices": {0: d("audio_tokens", min=1)},
                    "attention_bias": {2: d("audio_tokens", min=1), 3: d("audio_tokens", min=1)},
                },
            )
        if args.task == "asr" and args.only in (None, "decoder"):
            c = cfg.text_config
            past = tuple(
                torch.zeros(1, c.num_key_value_heads, 4, c.head_dim, dtype=dtype)
                for _ in range(2 * c.num_hidden_layers)
            )
            seq = d("sequence", min=1, max=args.cache_capacity)
            capacity = d("capacity", min=4, max=args.cache_capacity) if args.cache_capacity > 4 else None
            inputs = (
                torch.ones(1, 3, dtype=torch.int64),
                torch.zeros(1, 3, c.hidden_size, dtype=dtype),
                torch.zeros(1, 3, 1, dtype=torch.bool),
                torch.arange(3)[None],
                torch.zeros(1, 1, 3, 4, dtype=dtype),
                *past,
            )
            names = ["input_ids", "audio_embeddings", "audio_mask", "position_ids", "attention_bias"]
            export(
                TextDecoder(model),
                inputs,
                args.output / "decoder.onnx",
                names + [f"past_{i}" for i in range(len(past))],
                ["logits", "next_token"] + [f"present_{i}" for i in range(len(past))],
                {
                    "input_ids": {1: seq},
                    "audio_embeddings": {1: seq},
                    "audio_mask": {1: seq},
                    "position_ids": {1: seq},
                    "attention_bias": {2: seq, **({3: capacity} if capacity else {})},
                    "past": tuple({2: capacity} if capacity else {} for _ in past),
                },
            )
        if args.task == "aligner" and args.only in (None, "aligner"):
            seq = d("sequence", min=1)
            inputs = (
                torch.ones(1, 3, dtype=torch.int64),
                torch.zeros(1, 3, cfg.text_config.hidden_size, dtype=dtype),
                torch.zeros(1, 3, 1, dtype=torch.bool),
                torch.arange(3)[None],
                torch.zeros(1, 1, 3, 3, dtype=dtype),
                torch.tensor([1, 2]),
            )
            export(
                ForcedAligner(model),
                inputs,
                args.output / "aligner.onnx",
                ["input_ids", "audio_embeddings", "audio_mask", "position_ids", "attention_bias", "timestamp_indices"],
                ["timestamp_logits", "timestamp_bins"],
                {
                    "input_ids": {1: seq},
                    "audio_embeddings": {1: seq},
                    "audio_mask": {1: seq},
                    "position_ids": {1: seq},
                    "attention_bias": {2: seq, 3: seq},
                    "timestamp_indices": {0: d("timestamp_slots", min=1)},
                },
            )
    if args.decode_capacities:

        with torch.inference_mode():
            export_decode(TextDecoder(model), cfg.text_config, dtype, args.output, args.decode_capacities)
    eos = cfg.eos_token_id
    metadata = {
        "format_version": 2,
        "cache_capacity": args.cache_capacity if args.task == "asr" else None,
        "dynamic_cache_capacity": args.task == "asr" and args.cache_capacity > 4,
        "task": args.task,
        "dtype": str(dtype).removeprefix("torch."),
        "opset": 23,
        "batch_size": 1,
        "audio_config": cfg.audio_config.to_dict(),
        "text_config": cfg.text_config.to_dict(),
        "audio_token_id": cfg.audio_token_id,
        "eos_token_ids": list(eos) if isinstance(eos, (list, tuple)) else [eos],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    if args.task == "aligner":
        metadata.update(
            timestamp_bins=True,
            timestamp_token_id=cfg.timestamp_token_id,
            num_labels=cfg.num_labels,
            timestamp_segment_time=processor.timestamp_segment_time,
        )
    metadata["source"] = {"model": args.model, "revision": revision}
    if args.decode_capacities:
        metadata["decode_capacities"] = sorted(set(args.decode_capacities))
    if args.only == "encoder" and (args.output / "metadata.json").exists():
        previous = json.loads((args.output / "metadata.json").read_text(encoding="utf-8"))
        metadata["cache_capacity"] = previous["cache_capacity"]
        metadata["dynamic_cache_capacity"] = previous.get("dynamic_cache_capacity", False)
        if "decode_capacities" in previous and not args.decode_capacities:
            metadata["decode_capacities"] = previous["decode_capacities"]
        if args.task == "aligner":
            metadata["timestamp_bins"] = previous.get("timestamp_bins", False)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def export_mel(processor, output):
    export(
        LogMel(processor),
        (torch.zeros(1, 16000),),
        output / "mel.onnx",
        ["samples"],
        ["features"],
        {"samples": {1: torch.export.Dim("samples", min=8000, max=1205 * 16000)}},
    )


if __name__ == "__main__":
    main()
