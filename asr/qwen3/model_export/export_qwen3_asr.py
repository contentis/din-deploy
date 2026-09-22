# SPDX-License-Identifier: Apache-2.0
"""Export Qwen3-ASR with explicit tensor interfaces; checkpoint BF16 is the default."""

import argparse
import hashlib
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


class FP8Linear(nn.Module):
    def __init__(self, linear, activation_amax):
        super().__init__()
        scale = (linear.weight.detach().float().abs().amax() / 448).clamp_min(1e-8).to(torch.bfloat16)
        self.register_buffer("weight_scale", scale)
        self.register_buffer("input_scale", torch.tensor(max(activation_amax / 448, 1e-8), dtype=torch.bfloat16))
        self.register_buffer(
            "weight", (linear.weight.detach().float() / scale.float()).clamp(-448, 448).to(torch.float8_e4m3fn)
        )
        self.register_buffer("bias", linear.bias)

    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            x = torch.onnx.ops.symbolic(
                "QuantizeLinear",
                (x, self.input_scale),
                {"output_dtype": onnx.TensorProto.FLOAT8E4M3FN},
                dtype=torch.float8_e4m3fn,
                shape=x.shape,
                version=23,
            )
            x = torch.onnx.ops.symbolic(
                "DequantizeLinear",
                (x, self.input_scale),
                dtype=torch.bfloat16,
                shape=x.shape,
                version=23,
            )
            weight = torch.onnx.ops.symbolic(
                "DequantizeLinear",
                (self.weight, self.weight_scale),
                dtype=torch.bfloat16,
                shape=self.weight.shape,
                version=23,
            )
        else:
            x = (x.float() / self.input_scale.float()).clamp(-448, 448).to(torch.float8_e4m3fn)
            x = (x.float() * self.input_scale.float()).to(torch.bfloat16)
            weight = (self.weight.float() * self.weight_scale.float()).to(torch.bfloat16)
        return F.linear(x, weight, self.bias)


@torch.inference_mode()
def calibrate_fp8(model, processor, audio):
    maxima, hooks = {}, []

    def observe(name):
        def collect(module, inputs):
            maximum = inputs[0].float().abs().amax()
            maxima[name] = torch.maximum(maxima.get(name, maximum), maximum)

        return collect

    for name, module in model.model.language_model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_pre_hook(observe(name)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model.to(device)
        for path in audio:
            print(f"Calibrating FP8: {path}", flush=True)
            inputs = processor.apply_transcription_request(audio=str(path), return_tensors="pt")
            inputs = {
                k: v.to(device=device, dtype=model.dtype if v.is_floating_point() else v.dtype)
                for k, v in inputs.items()
            }
            model.generate(**inputs, do_sample=False, max_new_tokens=128)
        return {name: value.item() for name, value in maxima.items()}
    finally:
        for hook in hooks:
            hook.remove()
        model.cpu()


def apply_fp8(model, maxima):
    modules = {
        name: module for name, module in model.model.language_model.named_modules() if isinstance(module, nn.Linear)
    }
    if set(modules) != set(maxima) or any(not 0 <= value < float("inf") for value in maxima.values()):
        raise ValueError("FP8 calibration must contain finite maxima for every decoder projection")
    for name, module in modules.items():
        model.model.language_model.set_submodule(name, FP8Linear(module, maxima[name]))


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


class TextBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.language_model
        self.decoder.config._attn_implementation = "qwen3_onnx"

    def hidden(self, input_ids, audio_embeddings, audio_mask, position_ids, attention_bias, *past):
        dec = self.decoder
        x = torch.where(audio_mask, audio_embeddings, dec.embed_tokens(input_ids))
        rotary = dec.rotary_emb(x, position_ids)
        cache = DecodeCache(past, position_ids) if past else None
        for layer in dec.layers:
            x = layer(x, attention_mask=attention_bias, position_embeddings=rotary, past_key_values=cache)
        return dec.norm(x), cache.present if cache else []


class TextDecoder(TextBackbone):
    """Prompt prefill and single-token decoding with fixed-capacity KV tensors."""

    def __init__(self, model):
        super().__init__(model)
        self.lm_head = model.lm_head
        AttentionInterface.register("qwen3_onnx_decode", decode_attention)
        self.decoder.config._attn_implementation = "qwen3_onnx_decode"

    def forward(self, input_ids, audio_embeddings, audio_mask, position_ids, attention_bias, logits_index, *past):
        hidden, present = self.hidden(input_ids, audio_embeddings, audio_mask, position_ids, attention_bias, *past)
        logits = self.lm_head(hidden.index_select(1, logits_index).squeeze(1))
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
    path.unlink(missing_ok=True)
    path.with_suffix(".onnx.data").unlink(missing_ok=True)
    torch.onnx.export(
        module.eval(),
        args,
        str(path),
        input_names=names,
        output_names=outputs,
        dynamo=True,
        dynamic_shapes=shapes,
        opset_version=24,
        external_data=True,
    )
    onnx.checker.check_model(str(path))
    print(f"Checked {path}", flush=True)


def save_metadata(output, metadata):
    metadata["graphs"] = {}
    for name in ("mel", "encoder", "decoder" if metadata["task"] == "asr" else "aligner"):
        if not (output / (name + ".onnx")).is_file():
            continue
        digest = hashlib.sha256()
        for suffix in (".onnx", ".onnx.data"):
            path = output / (name + suffix)
            if path.exists():
                with path.open("rb") as file:
                    while data := file.read(8 * 1024 * 1024):
                        digest.update(data)
        metadata["graphs"][name] = digest.hexdigest()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--checkpoint", dest="model", help="HF model ID or local checkpoint directory")
    parser.add_argument("--size", choices=["0.6B", "1.7B"], default="0.6B", help="ASR model size")
    parser.add_argument("--revision", help="Optional HF revision or commit")
    parser.add_argument("--output", type=Path, help="Defaults to the ONNX artifact directory for --task")
    parser.add_argument("--task", choices=["asr", "aligner"], default="asr")
    parser.add_argument("--dtype", choices=["original", "fp16", "fp32"], default="original")
    parser.add_argument(
        "--quantization", choices=["fp8"], help="ASR decoder W8A8; keeps encoder, attention and KV in BF16"
    )
    parser.add_argument(
        "--calibration-audio", type=Path, nargs="+", help="Representative audio for FP8 activation calibration"
    )
    parser.add_argument("--only", choices=["mel", "encoder", "decoder", "aligner"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-capacity", type=int, default=8192)
    args = parser.parse_args()
    prefix = "aligner-" if args.task == "aligner" else ""
    args.model = args.model or (
        "Qwen/Qwen3-ForcedAligner-0.6B-hf" if args.task == "aligner" else f"Qwen/Qwen3-ASR-{args.size}-hf"
    )
    precision = args.quantization or ("bf16" if args.dtype == "original" else args.dtype)
    size_suffix = "-1.7b" if args.task == "asr" and "1.7b" in args.model.lower() else ""
    args.output = args.output or Path(f"artifacts/qwen3/{prefix}onnx-{precision}{size_suffix}")
    if args.quantization and (args.task != "asr" or args.dtype != "original" or args.only not in (None, "decoder")):
        parser.error("FP8 requires ASR with --dtype original and a full export or --only decoder")
    if args.quantization and args.only != "decoder" and not args.calibration_audio:
        parser.error("FP8 requires --calibration-audio")
    if args.calibration_audio and (not args.quantization or args.only == "decoder"):
        parser.error("--calibration-audio requires a full FP8 export; --only decoder reuses saved scales")
    if args.calibration_audio and any(not path.is_file() for path in args.calibration_audio):
        parser.error("Every calibration audio path must be an existing file")
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.task == "asr" and (not 512 <= args.cache_capacity <= 16384 or args.cache_capacity % 512):
        parser.error("cache-capacity must be a multiple of 512 between 512 and 16384")
    metadata_path = args.output / "metadata.json"
    if args.only not in (None, "mel") and metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        requested = {"original": "bfloat16", "fp16": "float16", "fp32": "float32"}[args.dtype]
        if existing["dtype"] != requested:
            parser.error("Partial export must keep the existing precision; use a separate output directory")
        if existing.get("quantization", {}).get("type") != args.quantization:
            parser.error("Partial export must keep the existing quantization")
    if args.quantization and args.only == "decoder" and not metadata_path.exists():
        parser.error("--only decoder with FP8 requires an existing export with saved calibration")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.only == "decoder" and args.task != "asr" or args.only == "aligner" and args.task != "aligner":
        parser.error("--only must match --task")
    if args.only == "mel":
        processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
        processor.save_pretrained(args.output / "processor")
        save_native_assets(processor, args.output, args.task)
        export_mel(processor, args.output)
        if metadata_path.exists():
            save_metadata(args.output, json.loads(metadata_path.read_text(encoding="utf-8")))
        return
    model_class = Qwen3ASRForConditionalGeneration if args.task == "asr" else Qwen3ASRForTokenClassification
    model = model_class.from_pretrained(
        args.model,
        dtype={"original": "auto", "fp16": torch.float16, "fp32": torch.float32}[args.dtype],
        attn_implementation="eager",
        revision=args.revision,
    ).eval()
    revision = model.config._commit_hash or args.revision
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
    quantization = None
    if args.quantization:
        if args.only == "decoder":
            quantization = existing["quantization"]
        else:
            quantization = {"type": "fp8", "activation_amax": calibrate_fp8(model, processor, args.calibration_audio)}
        apply_fp8(model, quantization["activation_amax"])
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
                torch.zeros(1, c.num_key_value_heads, args.cache_capacity, c.head_dim, dtype=dtype)
                for _ in range(2 * c.num_hidden_layers)
            )
            seq = d("sequence", min=1, max=512)
            inputs = (
                torch.ones(1, 3, dtype=torch.int64),
                torch.zeros(1, 3, c.hidden_size, dtype=dtype),
                torch.zeros(1, 3, 1, dtype=torch.bool),
                torch.arange(3)[None],
                torch.zeros(1, 1, 3, args.cache_capacity, dtype=dtype),
                torch.tensor([2]),
                *past,
            )
            names = ["input_ids", "audio_embeddings", "audio_mask", "position_ids", "attention_bias", "logits_index"]
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
                    "attention_bias": {2: seq},
                    "logits_index": {},
                    "past": tuple({} for _ in past),
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
    eos = cfg.eos_token_id
    metadata = {
        "format_version": 3 if args.task == "asr" else 2,
        "cache_capacity": args.cache_capacity if args.task == "asr" else None,
        "prefill_block": 512 if args.task == "asr" else None,
        "task": args.task,
        "dtype": str(dtype).removeprefix("torch."),
        "opset": 24,
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
    if quantization:
        metadata["quantization"] = quantization
    if args.only == "encoder" and (args.output / "metadata.json").exists():
        previous = json.loads((args.output / "metadata.json").read_text(encoding="utf-8"))
        metadata["cache_capacity"] = previous["cache_capacity"]
        metadata["format_version"] = previous["format_version"]
        metadata["prefill_block"] = previous.get("prefill_block")
        if args.task == "aligner":
            metadata["timestamp_bins"] = previous.get("timestamp_bins", False)
    save_metadata(args.output, metadata)
    if args.task == "asr" and args.only in (None, "decoder"):
        for path in [
            args.output / "decode.onnx",
            args.output / "decode.onnx.data",
            *args.output.glob("decode_[0-9]*.onnx"),
        ]:
            path.unlink(missing_ok=True)


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
