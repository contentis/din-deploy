# Qwen3 ASR and forced alignment

Offline C++ inference with TensorRT RTX in BF16 (default), FP16 or FP32, with optional FP8 decoder quantization.

## Supported models

| Model | Hugging Face ID | Checkpoint | BF16 export | FP16 export | FP32 export | FP8 decoder | Recommended export |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ASR 0.6B | `Qwen/Qwen3-ASR-0.6B-hf` | BF16 | ✓ | ✓ | ✓ | ✓ | BF16 |
| ASR 1.7B | `Qwen/Qwen3-ASR-1.7B-hf` | BF16 | ✓ | ✓ | ✓ | ✓ | BF16 |
| Forced Aligner 0.6B | `Qwen/Qwen3-ForcedAligner-0.6B-hf` | BF16 | ✓ | ✓ | ✓ | — | BF16 |

## Supported capabilities

| Model / upstream toolkit capability | C++ sample |
|---|:---:|
| Offline, single stream | ✓ |
| Online / streaming | — |
| Batched inference | — |
| Long-form audio | ✓ |
| ASR with / without forced alignment | ✓ |
| Standalone alignment of supplied text | ✓ |
| Automatic language identification / language hint | ✓ |
| Multilingual ASR: 30 languages and 22 Chinese dialects | ✓ |
| English word timestamps | ✓ |
| Chinese / Cantonese character timestamps | ✓ |
| All 11 upstream alignment languages | — |

ASR uses the [upstream model's language support](https://github.com/QwenLM/Qwen3-ASR).
English and Chinese are validated; coverage of every language, dialect and singing
is still missing. Alignment supports English words and Chinese/Cantonese characters
(with Latin words preserved). English per-letter timestamps and the remaining
alignment languages are not implemented.

## Export

Run Python commands from `asr/qwen3/model_export`, with the dependencies in
`../requirements.txt` and a CUDA-enabled PyTorch installation.

```bash
python -X utf8 export_qwen3_asr.py --size 0.6B --output D:/models/qwen3-asr-0.6b-onnx-bf16
python -X utf8 export_qwen3_asr.py --size 1.7B --output D:/models/qwen3-asr-1.7b-onnx-bf16
python -X utf8 export_qwen3_asr.py --task aligner --output D:/models/qwen3-aligner-onnx-bf16
```

Use `--dtype fp16` or `--dtype fp32` for converted exports; `--dtype original`
keeps BF16. This applies to both ASR sizes and the aligner.

```bash
python -X utf8 export_qwen3_asr.py --dtype fp16 --output D:/models/qwen3-asr-0.6b-onnx-fp16
python -X utf8 export_qwen3_asr.py --dtype fp32 --output D:/models/qwen3-asr-0.6b-onnx-fp32
```

For FP8 W8A8 decoder projections, supply representative calibration audio:

```bash
python -X utf8 export_qwen3_asr.py --quantization fp8 --calibration-audio speech-en.wav speech-zh.wav --decode-capacities 256 512 1024 2048 4096 8192 --output D:/models/qwen3-asr-0.6b-onnx-fp8
```

Add `--size 1.7B` for the larger model. HF generation calibrates activation scales;
the encoder, output head, attention and KV cache stay BF16. FP8 needs a compatible
GPU and can change transcription. `--only decode --quantization fp8` reuses saved
calibration scales when adding cache buckets.

The C++ pipeline reads precision from each export; ASR and aligner can use different
precisions. FP32 uses decomposed attention for TensorRT RTX compatibility; BF16/FP16
use fused attention. Log-mel stays FP32. Use separate output directories per precision.

HF downloads checkpoints automatically. Use `--model` for a local checkpoint or
`--revision` to pin the source. Keep each export directory intact. Log-mel
processing reuses the shared Whisper frontend.

For faster cached decoding, add `--decode-capacities 1024 2048 4096 8192` to the
ASR export command. These graphs use standard ONNX operations, including TensorScatter;
there are no contrib ops. KV updates are in-place only on this specialized path.
The general decoder uses separate cache banks. Attention scans allocated capacity.

## Verify

```bash
python -X utf8 validate_qwen3_asr.py --onnx-dir D:/models/qwen3-asr-0.6b-onnx-bf16 --audio audio.mp3
python -X utf8 validate_qwen3_asr.py --onnx-dir D:/models/qwen3-asr-1.7b-onnx-bf16 --audio audio.mp3
python -X utf8 validate_qwen3_asr.py --task aligner --onnx-dir D:/models/qwen3-aligner-onnx-bf16 --audio audio.mp3 --transcript transcript.txt --language Chinese
```

The validator compares encoder, prefill and cached-token outputs against HF,
then uses HF `generate()` for a short end-to-end token/EOS check. Alignment uses
HF transcript preparation and span decoding. Strict BF16 numerical comparisons
can fail despite matching tokens/spans. Long-form specialized decoding can change
words; long-form alignment has small endpoint differences from HF.
FP8 is compared against the original BF16 HF model; quantization differences remain
visible in the report rather than being treated as a numerical pass.

## Build

Build from the repository root with the same TensorRT RTX setup as Whisper.
An NVIDIA GPU supporting the selected TensorRT RTX precision is required.

```powershell
cmake --build out\build\windows-x64 --target din_asr_qwen3_cli
```

## Run

```powershell
out\build\windows-x64\bin\din_asr_qwen3_cli.exe audio.mp3 --model-dir D:\models\qwen3-asr-1.7b-onnx-bf16
out\build\windows-x64\bin\din_asr_qwen3_cli.exe audio.mp3 --model-dir D:\models\qwen3-asr-0.6b-onnx-bf16 --aligner-dir D:\models\qwen3-aligner-onnx-bf16
out\build\windows-x64\bin\din_asr_qwen3_cli.exe audio.mp3 --aligner-dir D:\models\qwen3-aligner-onnx-bf16 --transcript transcript.txt --lang-id zh
```

Multi-configuration builds add the configuration (for example, `Release`) under `bin`.
`--transcript` loads only the aligner and accepts text from any ASR model or a
supplied transcript. C++ callers can use `Qwen3ForcedAligner::Align` or `AlignFile`.

Reuse pipeline/aligner instances; calls on one instance are synchronous.
Set `Qwen3Config::progress` for audio loading, model loading/compilation, ASR chunk
completion and standalone alignment start/completion. Callbacks run on the calling
thread; inference progress is stage/chunk-based, not a token-level percentage.
Standalone alignment accepts segments up to 180 seconds; longer recordings need
matching audio/text segments. ASR splits long audio using upstream's quiet-boundary
algorithm: 1200-second targets, or 180 with alignment, and a ±5-second search.
`--max-chunk-seconds` overrides the target (6–1200, or 6–180 with alignment);
zero uses the defaults. This is a target, not a strict duration cap.
Increase `--max-new-tokens` from 512 for longer speech.
