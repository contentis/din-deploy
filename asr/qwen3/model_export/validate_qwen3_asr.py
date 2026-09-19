# SPDX-License-Identifier: Apache-2.0
"""Validate Qwen3 ASR or forced alignment against HF in checkpoint precision."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from _internal.runtime import OnnxAligner, OnnxASR, load_audio
from transformers import Qwen3ASRForConditionalGeneration, Qwen3ASRForTokenClassification


def validate_asr(args):
    metadata = json.loads((args.onnx_dir / "metadata.json").read_text(encoding="utf-8"))
    dtype = getattr(torch, metadata["dtype"])
    reference = (
        Qwen3ASRForConditionalGeneration.from_pretrained(
            args.model, revision=args.revision, dtype=dtype, attn_implementation="eager"
        )
        .to(args.reference_device)
        .eval()
    )
    runner = OnnxASR(args.onnx_dir, args.threads, args.provider)
    reports = []
    with torch.inference_mode():
        for path in args.audio:
            print(f"Validating {path}", flush=True)
            audio = load_audio(path)
            inputs = runner.processor.apply_transcription_request(
                audio=audio, language=args.language, return_tensors="pt"
            )
            ref_inputs = {
                k: v.to(device=args.reference_device, dtype=dtype if v.is_floating_point() else v.dtype)
                for k, v in inputs.items()
            }
            reference_audio = (
                reference.get_audio_features(ref_inputs["input_features"], ref_inputs["input_features_mask"])
                .pooler_output.float()
                .cpu()
                .numpy()
            )
            actual_audio = runner.encode(inputs)
            encoder_error = float(np.max(np.abs(reference_audio - actual_audio)))
            start = time.perf_counter()
            output = reference.generate(
                **ref_inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )
            reference_seconds = time.perf_counter() - start
            expected = output.sequences[0, inputs["input_ids"].shape[1] :].tolist()
            start = time.perf_counter()
            cache = runner.empty_cache(inputs["input_ids"].shape[1] + args.max_new_tokens)
            ids, generated, errors = inputs["input_ids"].numpy(), [], []
            embeddings = actual_audio
            cache_pointers = {v.data_ptr() for v in cache.values + cache.spare}
            for index in range(args.max_new_tokens):
                logits, cache = runner.step(ids, embeddings, cache)
                assert cache_pointers == {v.data_ptr() for v in cache.values + cache.spare}
                if runner.provider == "trt-rtx":
                    assert all(v.device_name() == "cuda" for v in cache.values)
                token = int(logits.argmax(-1)[0])
                generated.append(token)
                # Compare scores only while both generation histories are identical.
                if index < len(output.scores) and generated[:-1] == expected[:index]:
                    errors.append(float(np.max(np.abs(logits - output.scores[index].float().cpu().numpy()))))
                if token in runner.metadata["eos_token_ids"]:
                    break
                ids, embeddings = np.array([[token]], np.int64), None
            onnx_seconds = time.perf_counter() - start
            stopped = (
                generated[-1] in runner.metadata["eos_token_ids"] and expected[-1] in runner.metadata["eos_token_ids"]
            )
            token_match = generated == expected
            numeric_parity = bool(
                np.isfinite(encoder_error)
                and encoder_error <= args.atol
                and len(errors) == len(expected)
                and all(np.isfinite(e) and e <= args.atol for e in errors)
            )
            passed = bool(stopped and token_match and numeric_parity)
            report = {
                "audio": str(path.resolve()),
                "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "duration_seconds": len(audio) / 16000,
                "passed": passed,
                "numeric_parity": numeric_parity,
                "encoder_max_abs_error": encoder_error,
                "logits_max_abs_error": max(errors, default=None),
                "exact_token_match": token_match,
                "reached_eos": stopped,
                "reference_tokens": expected,
                "onnx_tokens": generated,
                "reference_text": runner.processor.decode(expected, return_format="parsed"),
                "onnx_text": runner.processor.decode(generated, return_format="parsed"),
                "reference_generate_seconds": reference_seconds,
                "onnx_decode_seconds": onnx_seconds,
            }
            reports.append(report)
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    payload = {
        "passed": all(r["passed"] for r in reports),
        "outputs_match": all(r["exact_token_match"] and r["reached_eos"] for r in reports),
        "atol": args.atol,
        "provider": runner.provider,
        "dtype": metadata["dtype"],
        "reference_device": args.reference_device,
        "source": runner.metadata.get("source"),
        "cases": reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(0 if payload["passed"] else 1)


def validate_aligner(args):
    audio = load_audio(args.audio[0])
    if len(audio) > 300 * 16000:
        raise ValueError("Standalone aligner inputs must be at most 300 seconds")
    text = args.transcript.read_text(encoding="utf-8").strip()
    metadata = json.loads((args.onnx_dir / "metadata.json").read_text(encoding="utf-8"))
    dtype = getattr(torch, metadata["dtype"])
    reference = (
        Qwen3ASRForTokenClassification.from_pretrained(
            args.model, revision=args.revision, dtype=dtype, attn_implementation="eager"
        )
        .to(args.reference_device)
        .eval()
    )
    runner = OnnxAligner(args.onnx_dir, args.threads, args.provider)
    inputs, words = runner.processor.prepare_forced_aligner_inputs(
        audio=audio, transcript=text, language=args.language, return_tensors="pt"
    )
    if not words[0]:
        raise ValueError("Transcript must contain alignable words")
    with torch.inference_mode():
        ref_inputs = {
            k: v.to(device=args.reference_device, dtype=dtype if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()
        }
        ref_audio = reference.model.get_audio_features(
            ref_inputs["input_features"], ref_inputs["input_features_mask"]
        ).pooler_output
        encoder_error = float(np.max(np.abs(ref_audio.float().cpu().numpy() - runner.encode(inputs))))
        expected_logits = reference(**ref_inputs, use_cache=False).logits.float().cpu()
        timestamp_id = reference.config.timestamp_token_id
        expected_slots = expected_logits[:, inputs["input_ids"][0] == timestamp_id].numpy()
        expected_items = runner.processor.decode_forced_alignment(
            expected_logits, inputs["input_ids"], words, timestamp_id
        )[0]
        actual_slots, actual_items = runner.align(inputs, words)
    error = float(np.max(np.abs(expected_slots - actual_slots)))
    classes_match = bool(np.array_equal(expected_slots.argmax(-1), actual_slots.argmax(-1)))
    numeric_parity = bool(
        np.isfinite(error) and error <= args.atol and np.isfinite(encoder_error) and encoder_error <= args.atol
    )
    passed = bool(numeric_parity and classes_match and expected_items == actual_items)
    report = {
        "passed": passed,
        "numeric_parity": numeric_parity,
        "provider": runner.provider,
        "dtype": metadata["dtype"],
        "reference_device": args.reference_device,
        "atol": args.atol,
        "source": runner.metadata.get("source"),
        "audio": str(args.audio[0].resolve()),
        "audio_sha256": hashlib.sha256(args.audio[0].read_bytes()).hexdigest(),
        "transcript": text,
        "encoder_max_abs_error": encoder_error,
        "timestamp_logits_max_abs_error": error,
        "exact_timestamp_classes": classes_match,
        "exact_spans": expected_items == actual_items,
        "reference_spans": expected_items,
        "onnx_spans": actual_items,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if passed else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["asr", "aligner"], default="asr")
    parser.add_argument("--onnx-dir", type=Path)
    parser.add_argument(
        "--model", "--checkpoint", dest="model", help="HF model ID or local checkpoint; defaults to export metadata"
    )
    parser.add_argument("--revision", help="Optional HF revision; defaults to export metadata")
    parser.add_argument("--audio", type=Path, nargs="+", default=[Path("assets/sample.wav")])
    parser.add_argument("--transcript", type=Path, help="UTF-8 transcript file, required for alignment")
    parser.add_argument("--language")
    parser.add_argument("--provider", choices=["cpu", "trt-rtx"])
    parser.add_argument("--reference-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--atol", type=float, default=0.005)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.max_new_tokens < 1 or args.atol <= 0:
        parser.error("threads, max-new-tokens and atol must be positive")
    if args.task == "aligner" and (args.transcript is None or len(args.audio) != 1):
        parser.error("Alignment requires one --audio and a --transcript")
    prefix = "aligner-" if args.task == "aligner" else ""
    args.onnx_dir = args.onnx_dir or Path(f"artifacts/qwen3/{prefix}onnx-bf16")
    args.report = args.report or Path(f"artifacts/qwen3/validation-{prefix}bf16.json")
    metadata = json.loads((args.onnx_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["task"] != args.task:
        parser.error("--task does not match the exported model")
    source = metadata["source"]
    if args.model is None:
        args.model = source["model"]
        args.revision = args.revision or source.get("revision")
    torch.set_num_threads(args.threads)
    if args.task == "aligner":
        args.language = args.language or "English"
        validate_aligner(args)
    else:
        validate_asr(args)


if __name__ == "__main__":
    main()
