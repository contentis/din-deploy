# SPDX-License-Identifier: Apache-2.0
"""Validate ONNX vision/DeepStack, multimodal positions, logits, all KV slots and generation against HF."""

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen3VLForConditionalGeneration

from .cli import add_input_arguments, prepare_from_args
from .inputs import position_ids
from .runtime import Qwen3VL


def compare(actual, expected, atol, rtol=0.02):
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    error = (a - b).abs()
    return {
        "passed": bool(
            torch.isfinite(a).all() and torch.isfinite(b).all() and torch.allclose(a, b, atol=atol, rtol=rtol)
        ),
        "max_abs": float(error.max()),
        "mean_abs": float(error.mean()),
        "relative_l2": float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12)),
    }


def compare_cache(runner, reference, atol):
    checks = []
    for i, layer in enumerate(reference.layers):
        for j, value in enumerate((layer.keys, layer.values)):
            checks.append(compare(runner.cache[2 * i + j][:, :, : runner.length], value, atol))
    return {"passed": all(c["passed"] for c in checks), "max_abs": max(c["max_abs"] for c in checks), "layers": checks}


@torch.inference_mode()
def validate(runner, reference, inputs, max_new_tokens=16, cached_steps=4, atol=0.25, cache_atol=None):
    cache_atol = atol if cache_atol is None else cache_atol
    if cached_steps < 1 or max_new_tokens < 1 or min(atol, cache_atol) < 0:
        raise ValueError("Validation steps must be positive and tolerances nonnegative")
    if inputs["input_ids"].shape[1] + max(cached_steps, max_new_tokens) > runner.capacity:
        raise ValueError("Validation exceeds exported cache capacity")
    device, dtype = next(reference.parameters()).device, reference.dtype
    ref_inputs = {
        k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
        for k, v in inputs.items()
        if isinstance(v, torch.Tensor)
    }
    report, checks = {}, {}
    encoded = runner.encode(inputs)
    runner.synchronize()
    for kind, pixels, grid in ((1, "pixel_values", "image_grid_thw"), (2, "pixel_values_videos", "video_grid_thw")):
        if pixels in inputs:
            expected = reference.model.visual(ref_inputs[pixels], grid_thw=ref_inputs[grid])
            checks[f"vision_{kind}"] = compare(encoded[kind][0], expected.pooler_output, atol)
            for i, features in enumerate(expected.deepstack_features):
                checks[f"deepstack_{kind}_{i}"] = compare(encoded[kind][i + 1], features, atol)
    positions, delta = position_ids(inputs)
    expected_positions, expected_delta = reference.model.get_rope_index(
        inputs["input_ids"],
        inputs["mm_token_type_ids"],
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
    )
    checks["positions"] = {"passed": torch.equal(positions, expected_positions) and delta == int(expected_delta.item())}
    expected = reference(**ref_inputs, use_cache=True, logits_to_keep=1)
    result = runner.prefill(inputs, encoded)
    runner.synchronize()
    checks["prefill_logits"] = compare(result["logits"], expected.logits[:, -1], atol)
    checks["prefill_cache"] = compare_cache(runner, expected.past_key_values, cache_atol)
    pointers = [t.data_ptr() for t in runner.cache]
    for step in range(cached_steps):
        old_length = runner.length
        before = [tensor.clone() for tensor in runner.cache]
        token = expected.logits[:, -1].argmax(-1, keepdim=True)
        expected = reference(
            input_ids=token, past_key_values=expected.past_key_values, use_cache=True, logits_to_keep=1
        )
        result = runner.decode(int(token.item()))
        runner.synchronize()
        checks[f"cache_write_bounds_{step}"] = {
            "passed": all(
                torch.equal(a[:, :, :old_length], b[:, :, :old_length])
                and torch.equal(a[:, :, old_length + 1 :], b[:, :, old_length + 1 :])
                for a, b in zip(before, runner.cache, strict=True)
            )
        }
        del before
        checks[f"cached_logits_{step}"] = compare(result["logits"], expected.logits[:, -1], atol)
        checks[f"cached_kv_{step}"] = compare_cache(runner, expected.past_key_values, cache_atol)
    checks["stable_kv_addresses"] = {"passed": pointers == [t.data_ptr() for t in runner.cache]}
    expected_ids = reference.generate(**ref_inputs, do_sample=False, max_new_tokens=max_new_tokens)
    expected_tokens = expected_ids[0, inputs["input_ids"].shape[1] :].tolist()
    actual = runner.generate(inputs, max_new_tokens=max_new_tokens)
    repeated = runner.generate(inputs, max_new_tokens=max_new_tokens)
    checks["greedy_tokens"] = {"passed": actual["tokens"] == expected_tokens}
    checks["reset_repeat"] = {"passed": repeated["tokens"] == actual["tokens"]}
    report.update(
        checks=checks,
        reference_tokens=expected_tokens,
        reference_text=runner.processor.tokenizer.decode(expected_tokens, skip_special_tokens=True),
        onnx=actual,
        warm_run=repeated,
        atol=atol,
        cache_atol=cache_atol,
        rtol=0.02,
        cache_integrity_passed=all(
            value["passed"]
            for key, value in checks.items()
            if key.startswith("cache_write_bounds_") or key in ("stable_kv_addresses", "reset_repeat", "positions")
        ),
        strict_numerical_passed=all(
            value["passed"]
            for key, value in checks.items()
            if "logits" in key or "kv_" in key or key == "prefill_cache" or key.startswith(("vision_", "deepstack_"))
        ),
        passed=all(check["passed"] for check in checks.values()),
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--provider", choices=["cpu", "trt-rtx"], default="trt-rtx")
    parser.add_argument("--reference-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--cached-steps", type=int, default=4)
    parser.add_argument("--atol", type=float, default=0.25)
    parser.add_argument("--cache-atol", type=float, help="Optional separate tolerance for unbounded KV activations")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--ep-library", type=Path)
    parser.add_argument("--reference-attention", choices=["eager", "sdpa"], default="sdpa")
    add_input_arguments(parser)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    runner = Qwen3VL(
        args.model_dir,
        args.provider,
        cache_dir=args.cache_dir,
        threads=args.threads,
        profile=args.profile,
        ep_library=args.ep_library,
    )
    inputs, media = prepare_from_args(runner.processor, args, max(runner.metadata["vision_buckets"]) // 4)
    source = runner.metadata["source"]
    reference = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            source["model"],
            revision=source["revision"],
            dtype=runner.dtype,
            attn_implementation=args.reference_attention,
        )
        .to(args.reference_device)
        .eval()
    )
    report = validate(runner, reference, inputs, args.max_new_tokens, args.cached_steps, args.atol, args.cache_atol)
    report["export"] = runner.metadata
    report["reference_attention"] = args.reference_attention
    report["reference_device"] = args.reference_device
    report["provider"] = args.provider
    report["media"] = media
    report["profiles"] = runner.finish_profiles()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": {k: {a: b for a, b in v.items() if a != "layers"} for k, v in report["checks"].items()},
                "reference": report["reference_text"],
                "onnx": report["onnx"]["text"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
