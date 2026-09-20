# SPDX-License-Identifier: Apache-2.0
"""Image and video-segment understanding with ONNX Runtime."""

import argparse
import json
from pathlib import Path

import torch

from .inputs import VideoSegment, prepare_inputs
from .runtime import Qwen3VL


def add_input_arguments(parser):
    parser.add_argument("--images", nargs="*", default=[])
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-start", type=float, default=0)
    parser.add_argument("--video-end", type=float)
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-visual-tokens", type=int, default=256)
    parser.add_argument("--prompt", default="Describe what you see concisely.")


def prepare_from_args(processor, args, max_image_tokens=None):
    segments = (
        [VideoSegment(args.video, args.video_start, args.video_end, args.fps, args.max_frames)] if args.video else []
    )
    return prepare_inputs(processor, args.prompt, args.images, segments, args.max_visual_tokens, max_image_tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=["cpu", "trt-rtx"], default="trt-rtx")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--repeat", type=int, default=1, help="Repeat identical request; first run includes compilation"
    )
    parser.add_argument("--profile", action="store_true", help="Write ORT JSON profiles")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--ep-library", type=Path, help="Optional TensorRT RTX EP library override")
    parser.add_argument("--report", type=Path)
    add_input_arguments(parser)
    args = parser.parse_args()
    if args.repeat < 1 or args.threads < 1:
        parser.error("repeat and threads must be positive")
    torch.set_num_threads(args.threads)
    runner = Qwen3VL(
        args.model_dir,
        args.provider,
        args.cache_dir,
        args.threads,
        args.profile,
        not args.no_cuda_graph,
        ep_library=args.ep_library,
    )
    inputs, report = prepare_from_args(runner.processor, args, max(runner.metadata["vision_buckets"]) // 4)
    report["runs"] = []
    for i in range(args.repeat):
        if runner.device == "cuda":
            torch.cuda.nvtx.range_push(f"qwen3_vl/request_{i}")
        result = runner.generate(inputs, args.max_new_tokens, args.temperature, args.top_p, args.seed)
        if runner.device == "cuda":
            torch.cuda.nvtx.range_pop()
        report["runs"].append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    report["profiles"] = runner.finish_profiles()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
