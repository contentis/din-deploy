# SPDX-License-Identifier: Apache-2.0
"""python -m vision.qwen3_vl.export --output artifacts/qwen3-vl-2b-bf16"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .modeling import TextDecoder, VisionEncoder, export_graph


def graph_digest(path):
    digest = hashlib.sha256()
    for file in (path, path.with_suffix(".onnx.data")):
        if file.exists():
            with file.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def export_vision(model, output, buckets):
    v, dtype = model.config.vision_config, model.dtype
    n = min(16, max(buckets))
    head = v.hidden_size // v.num_heads
    patches = 4 if max(buckets) == 4 else 4 * torch.export.Dim("visual_tokens", min=1, max=max(buckets) // 4)
    export_graph(
        VisionEncoder(model),
        (
            torch.zeros(n, 3 * v.temporal_patch_size * v.patch_size**2, dtype=dtype),
            torch.zeros(4, n, dtype=torch.int64),
            torch.zeros(4, n),
            torch.ones(n, head),
            torch.zeros(n, head),
            torch.zeros(1, 1, 1, n, dtype=dtype),
        ),
        output / "vision.onnx",
        ["pixels", "bilinear_indices", "bilinear_weights", "rotary_cos", "rotary_sin", "attention_bias"],
        ["visual_features"],
        {
            "pixels": {0: patches},
            "bilinear_indices": {1: patches},
            "bilinear_weights": {1: patches},
            "rotary_cos": {0: patches},
            "rotary_sin": {0: patches},
            "attention_bias": {3: patches},
        },
    )


@torch.inference_mode()
def export_model(
    model,
    output,
    capacity=2048,
    prefill=128,
    vision_buckets=(256, 1024, 4096),
    attention_buckets=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    c, v = model.config.text_config, model.config.vision_config
    dtype = model.dtype
    if prefill < 2 or capacity < prefill or capacity % prefill:
        raise ValueError("cache-capacity must be a multiple of prefill-block (at least 2)")
    if v.spatial_merge_size != 2 or v.temporal_patch_size != 2:
        raise ValueError("Only the Qwen3-VL spatial/temporal merge contract is supported")
    if c.rope_parameters["rope_type"] != "default":
        raise ValueError("Export currently requires default interleaved MRoPE")
    buckets = sorted(set(vision_buckets))
    if not buckets or any(b < 4 or b % 4 for b in buckets):
        raise ValueError("Vision buckets must be positive multiples of four raw patches")
    attention_buckets = sorted(
        set(
            attention_buckets
            if attention_buckets is not None
            else [*[b for b in (256, 512, 1024) if b < capacity], capacity]
        )
    )
    if not attention_buckets or attention_buckets[-1] != capacity or any(b < 2 for b in attention_buckets):
        raise ValueError("Attention buckets must be at least two slots and end at cache-capacity")
    export_vision(model, output, buckets)
    past = tuple(
        torch.zeros(1, c.num_key_value_heads, capacity, c.head_dim, dtype=dtype) for _ in range(c.num_hidden_layers * 2)
    )
    seq = torch.export.Dim("sequence", min=1, max=prefill)
    example_seq = min(3, prefill)
    attention_shape = {2: seq}
    example_window = capacity
    if len(attention_buckets) > 1:
        attention_shape[3] = torch.export.Dim("attention_capacity", min=2, max=capacity)
        example_window = attention_buckets[0]
    export_graph(
        TextDecoder(model),
        (
            torch.zeros(1, example_seq, dtype=torch.int64),
            torch.zeros(1 + len(v.deepstack_visual_indexes), 1, example_seq, c.hidden_size, dtype=dtype),
            torch.zeros(1, example_seq, 1, dtype=torch.bool),
            torch.ones(1, example_seq, c.head_dim, dtype=dtype),
            torch.zeros(1, example_seq, c.head_dim, dtype=dtype),
            torch.zeros(1, 1, example_seq, example_window, dtype=dtype),
            torch.zeros(1, dtype=torch.int64),
            torch.tensor([example_seq - 1]),
            *past,
        ),
        output / "decoder.onnx",
        [
            "input_ids",
            "visual_features",
            "visual_mask",
            "rotary_cos",
            "rotary_sin",
            "attention_bias",
            "cache_position",
            "logits_index",
        ]
        + [f"past_{i}" for i in range(len(past))],
        ["logits", "next_token", "hidden"] + [f"present_{i}" for i in range(len(past))],
        {
            "input_ids": {1: seq},
            "visual_features": {2: seq},
            "visual_mask": {1: seq},
            "rotary_cos": {1: seq},
            "rotary_sin": {1: seq},
            "attention_bias": attention_shape,
            "cache_position": {},
            "logits_index": {},
            "past": tuple({} for _ in past),
        },
    )
    eos = model.generation_config.eos_token_id
    metadata = {
        "format_version": 1,
        "dtype": str(dtype).removeprefix("torch."),
        "cache_capacity": capacity,
        "prefill_block": prefill,
        "vision_buckets": buckets,
        "attention_buckets": attention_buckets,
        "decoder_attention": "folded_fp32",
        "text_config": c.to_dict(),
        "vision_config": v.to_dict(),
        "image_token_id": model.config.image_token_id,
        "video_token_id": model.config.video_token_id,
        "eos_token_ids": eos if isinstance(eos, list) else [eos],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "graphs": {name: graph_digest(output / f"{name}.onnx") for name in ("vision", "decoder")},
    }
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["2B", "8B"], default="2B")
    parser.add_argument("--model", help="Local checkpoint or HF ID; defaults to Qwen/Qwen3-VL-{size}-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=["original", "fp16", "fp32"], default="original")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-capacity", type=int, default=2048)
    parser.add_argument("--prefill-block", type=int, default=128)
    parser.add_argument("--vision-buckets", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument(
        "--attention-buckets", type=int, nargs="+", help="Decode prefix buckets; last must equal cache capacity"
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if (args.output / "metadata.json").exists():
        parser.error("Use a fresh output directory to avoid mixing exports and compiled caches")
    torch.set_num_threads(args.threads)
    source = args.model or f"Qwen/Qwen3-VL-{args.size}-Instruct"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        source,
        revision=args.revision,
        dtype={"original": "auto", "fp16": torch.float16, "fp32": torch.float32}[args.dtype],
        attn_implementation="eager",
    ).eval()
    revision = model.config._commit_hash or args.revision
    processor = AutoProcessor.from_pretrained(source, revision=revision)
    metadata = export_model(
        model,
        args.output,
        args.cache_capacity,
        args.prefill_block,
        args.vision_buckets,
        args.attention_buckets,
    )
    metadata["source"] = {"model": source, "revision": revision}
    processor.save_pretrained(args.output / "processor")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Export complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
