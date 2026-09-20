# Qwen3 ASR and forced alignment

Offline C++ inference with CPU or TensorRT RTX. Use FP32 exports for CPU; TensorRT RTX supports BF16 (default), FP16, FP32 and optional FP8 decoder quantization.

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
| Word timestamps: en, de, es, fr, it, pt, ru, ko | ✓ |
| Chinese / Cantonese character timestamps | ✓ |
| Japanese character timestamps | ✓ |
| All 11 upstream alignment languages | ✓ |

ASR uses the [upstream model's language support](https://github.com/QwenLM/Qwen3-ASR).
ASR accepts all 30 upstream language codes/names and `auto`; dialects use automatic
recognition or the corresponding language hint, not separate dialect switches.
Alignment supports Chinese, Cantonese, English, German, Spanish, French, Italian,
Portuguese, Russian, Korean and Japanese. Japanese uses character timestamps;
upstream's Nagisa word boundaries differ. Latin words in CJK text stay together.
Language names/codes are case-insensitive. ASR's other languages require alignment
to be disabled. Language coverage is not an accuracy guarantee for every dialect.

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
python -X utf8 export_qwen3_asr.py --quantization fp8 --calibration-audio speech-en.wav speech-zh.wav --output D:/models/qwen3-asr-0.6b-onnx-fp8
```

Add `--size 1.7B` for the larger model. Only decoder projections use FP8; other
components stay BF16. Use a compatible GPU and check transcription accuracy.
`--only decoder --quantization fp8` reuses saved calibration.

The C++ pipeline reads precision from each export; ASR and aligner can use different
precisions. FP32 uses decomposed attention for TensorRT RTX compatibility; BF16/FP16
use fused attention. Log-mel stays FP32. Use separate output directories per precision.

HF downloads checkpoints automatically. Use `--model` for a local checkpoint or
`--revision` to pin the source. Keep each export directory intact. Log-mel
processing reuses the shared Whisper frontend.

Exports contain one encoder and one decoder, each with one weight file, plus the
shared log-mel graph. Prefill and token generation update one KV bank in place.
The decoder uses two fixed TensorRT profiles (512-token prefill and one-token
steps), compiled once and cached; audio length does not create more encoder/decoder
profiles. The two engines may each retain weights in GPU memory. Changed exports
get new cache keys; clear compiled caches when changing the GPU or runtime.

`--cache-capacity` sets the token ceiling (default 8192; multiples of 512 up to
16384). Long audio uses upstream quiet-boundary splitting with enough room reserved
for `--max-new-tokens`. Reaching that generation limit returns `reached_eos=false`.
Attention still scans the allocated cache. BF16/FP16 KV uses 896 MiB at 8192 slots;
FP8 weights do not quantize KV. Re-export older ASR artifacts for format 3.

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
FP8 validation uses the original BF16 HF reference.

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
Standalone alignment uses `forced_aligner.h` and `ForcedAlignerConfig`; it loads no ASR decoder.
`--provider cpu|trt-rtx` selects execution (default: `trt-rtx`);
`--aligner-provider` optionally overrides it for alignment. C++ configs expose the same choices.

Reuse pipeline/aligner instances; calls on one instance are synchronous.
Set `Qwen3Config::progress` for audio loading, model loading/compilation, ASR chunk
completion and standalone alignment start/completion. Callbacks run on the calling
thread; inference progress is stage/chunk-based, not a token-level percentage.
Standalone alignment accepts segments up to 180 seconds; longer recordings need
matching audio/text segments. ASR splits long audio using upstream's quiet-boundary
algorithm: 1200-second targets, or 180 with alignment, and a ±5-second search.
`--max-chunk-seconds` overrides the target (6–1200, or 6–180 with alignment);
zero uses the defaults. The KV budget limits this target further. This is a target,
not a strict duration cap.
`--max-new-tokens` defaults to 1024 per chunk. If `reached_eos` is false, increase
the budget or reduce `--max-chunk-seconds`; the transcript is incomplete.
