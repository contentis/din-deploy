# SPDX-License-Identifier: Apache-2.0
"""Diagnostic decoder benchmarks; fixed token counts deliberately continue after EOS.

Uses private runtime buffers to compare the existing host token loop with an
experimental GPU feedback loop. This is an analysis tool, not a generation API.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from .inputs import prepare_inputs
from .runtime import Qwen3VL, on_execution_stream
from .sampling import CudaSampler, sample_top_p


def select(logits, generator, full_sort=False):
    if full_sort:
        probabilities = torch.softmax(logits[0] / 0.7, dim=-1)
        probabilities, indices = probabilities.sort(descending=True)
        probabilities.masked_fill_(probabilities.cumsum(-1) - probabilities > 0.9, 0)
        return indices[torch.multinomial(probabilities, 1, generator=generator)]
    return sample_top_p(logits[0], 0.7, 0.9, generator)


@torch.inference_mode()
def device_decode(runner, previous):
    """Same fixed decoder, GPU token feedback; all positions still host controlled."""
    graph = runner._decoder(1)
    feed = graph.inputs
    if runner._decode_dirty:
        feed["visual_features"].zero_()
        feed["visual_mask"].zero_()
        runner._decode_dirty = False
    controls = runner._decode_controls_host[runner.length]
    controls[1] = runner.length
    runner._decode_controls.copy_(controls, non_blocking=True)
    feed["input_ids"].copy_(previous.reshape(1, 1))
    rotary = runner._rotary_table[runner.length + runner.rope_delta]
    runner._decode_rotary.copy_(rotary.reshape(2, 1, 1, -1))
    start = runner.capacity - runner.length - 1
    window = feed["attention_bias"].shape[-1]
    feed["attention_bias"].copy_(runner._causal_line[start : start + window].reshape(1, 1, 1, -1))
    graph.run()
    runner.length += 1
    return graph.outputs


@torch.inference_mode()
@on_execution_stream
def trial(runner, inputs, encoded, mode, steps, iteration):
    result = runner.prefill(inputs, encoded)
    generator = torch.Generator(device="cuda").manual_seed(123)
    if mode == "host_top_p_captured":
        if runner._sampler is None:
            runner._sampler = CudaSampler(runner.config["vocab_size"])
        runner._sampler.configure(0.7, 0.9)
        runner._sampler.select(result["logits"][0], generator)
        generator.manual_seed(123)
    previous = result["next_token"]
    host_token = int(previous.item())
    history = torch.empty(steps, dtype=torch.int64, device="cuda")
    runner.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    label = f"qwen3_vl/decoder_analysis/{mode}/{iteration}"
    torch.cuda.nvtx.range_push(label)
    begin.record()
    start = time.perf_counter()
    for i in range(steps):
        if mode == "graph_replay":
            # Hold position/IDs/mask fixed: isolates EP replay, not valid generation.
            result = runner._decoder(1).run()
        elif mode == "gpu_feedback":
            result = device_decode(runner, previous)
        else:
            torch.cuda.nvtx.range_push("host_decode")
            result = runner.decode(host_token)
            torch.cuda.nvtx.range_pop()
        if mode != "graph_replay":
            torch.cuda.nvtx.range_push("token_selection")
            previous = (
                runner._sampler.select(result["logits"][0], generator)
                if mode == "host_top_p_captured"
                else (
                    select(result["logits"], generator, full_sort=mode == "host_top_p_full")
                    if mode in ("host_top_p", "host_top_p_full")
                    else result["next_token"]
                )
            )
            history[i].copy_(previous.reshape(()))
            if mode != "gpu_feedback":
                host_token = int(previous.item())
            torch.cuda.nvtx.range_pop()
    end.record()
    end.synchronize()
    seconds = time.perf_counter() - start
    torch.cuda.nvtx.range_pop()
    return {
        "mode": mode,
        "iteration": iteration,
        "steps": steps,
        "wall_ms_per_token": seconds * 1000 / steps,
        "cuda_span_ms_per_token": begin.elapsed_time(end) / steps,
        "tokens_per_second": steps / seconds,
        "tokens": history.cpu().tolist() if mode != "graph_replay" else None,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["host_greedy", "gpu_feedback", "graph_replay", "host_top_p"],
        choices=["host_greedy", "gpu_feedback", "graph_replay", "host_top_p", "host_top_p_full", "host_top_p_captured"],
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ep-library", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.repeat < 1:
        parser.error("steps and repeat must be positive")
    torch.set_num_threads(4)
    runner = Qwen3VL(args.model_dir, cache_dir=args.cache_dir, ep_library=args.ep_library)
    inputs, media = prepare_inputs(
        runner.processor, "Describe what you see concisely.", images=args.image, max_visual_tokens=64
    )
    if inputs["input_ids"].numel() + args.steps > runner.capacity:
        parser.error("prompt plus steps exceeds capacity")
    encoded = runner.encode(inputs)
    # Warm prefill/decode bindings on the shared session before measured ranges.
    for _ in range(3):
        result = runner.prefill(inputs, encoded)
        for _ in range(3):
            result = runner.decode(int(result["next_token"].item()))
    runs = []
    # Interleave modes to reduce systematic thermal/clock bias.
    for iteration in range(args.repeat):
        for mode in args.modes:
            result = trial(runner, inputs, encoded, mode, args.steps, iteration)
            runs.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "tokens"}), flush=True)
    summary = {
        mode: statistics.median(r["wall_ms_per_token"] for r in runs if r["mode"] == mode) for mode in args.modes
    }
    greedy = [r["tokens"] for r in runs if r["mode"] == "host_greedy"]
    feedback = [r["tokens"] for r in runs if r["mode"] == "gpu_feedback"]
    report = {
        "model_dir": str(args.model_dir),
        "graph_hashes": runner.metadata["graphs"],
        "capacity": runner.capacity,
        "prompt_tokens": inputs["input_ids"].numel(),
        "media": media,
        "median_wall_ms_per_token": summary,
        "gpu_feedback_matches_greedy": all(t == greedy[0] for t in greedy + feedback) if greedy and feedback else None,
        "runs": runs,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("runs", "media")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
