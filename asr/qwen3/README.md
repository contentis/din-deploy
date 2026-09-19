# Qwen3 ASR and forced alignment

Offline C++ inference with TensorRT RTX and original BF16 weights.

| Model / upstream toolkit capability | C++ sample |
|---|:---:|
| 0.6B / 1.7B ASR | ✓ |
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
| Original BF16 precision | ✓ |

ASR uses the [upstream model's language support](https://github.com/QwenLM/Qwen3-ASR).
English and Chinese are validated; coverage of every language, dialect and singing
is still missing. Alignment supports English words and Chinese/Cantonese characters
(with Latin words preserved). English per-letter timestamps and the remaining
alignment languages are not implemented.

## Export

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r asr/qwen3/requirements.txt
./.venv/Scripts/python.exe -m pip install --upgrade torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
./.venv/Scripts/python.exe -X utf8 asr/qwen3/model_export/export_qwen3_asr.py
./.venv/Scripts/python.exe -X utf8 asr/qwen3/model_export/export_qwen3_asr.py --size 1.7B
./.venv/Scripts/python.exe -X utf8 asr/qwen3/model_export/export_qwen3_asr.py --task aligner
```

HF downloads checkpoints automatically. Outputs are `artifacts/qwen3/onnx-bf16`,
`artifacts/qwen3/onnx-bf16-1.7b` and `artifacts/qwen3/aligner-onnx-bf16`.
Use `--model`, `--revision` and `--output` to override. Keep each export directory
intact. Log-mel processing reuses the shared Whisper frontend.

For faster cached decoding, add `--decode-capacities 1024 2048 4096 8192` to the
ASR export command. These graphs use standard ONNX Attention and TensorScatter;
there are no contrib ops. KV updates are in-place only on this specialized path.
The general decoder uses separate cache banks. Attention scans allocated capacity.

## Run

Build with the root CMake project and the same TensorRT RTX setup as Whisper.
A BF16-capable NVIDIA GPU is required. Set `TRT_RTX_ROOT` to the SDK directory.

```powershell
cmake --build build --config Release --target din_asr_qwen3_cli
./build/bin/din_asr_qwen3_cli.exe recording.wav --model-dir artifacts/qwen3/onnx-bf16-1.7b
./build/bin/din_asr_qwen3_cli.exe recording.wav --aligner-dir artifacts/qwen3/aligner-onnx-bf16
./build/bin/din_asr_qwen3_cli.exe recording.wav --transcript transcript.txt --lang-id zh
```

Multi-configuration builds place the executable in `build/bin/Release`.
`--transcript` loads only the aligner and accepts text from any ASR model or a
supplied transcript. The reusable C++ API is:

```cpp
din::asr::qwen3::Qwen3ForcedAligner aligner;
auto spans = aligner.AlignFile("recording.wav", "Supplied transcript", "English");
```

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

## Validate

```powershell
./.venv/Scripts/python.exe -X utf8 asr/qwen3/model_export/validate_qwen3_asr.py --audio assets/sample.wav
./.venv/Scripts/python.exe -X utf8 asr/qwen3/model_export/validate_qwen3_asr.py --task aligner --audio recording.wav --transcript transcript.txt --language Chinese
```

The validator compares encoder, prefill and cached-token outputs against HF,
then uses HF `generate()` for a short end-to-end token/EOS check. Alignment uses
HF transcript preparation and span decoding. It checks correctness, not speed.
Use `--onnx-dir` for a different export. Short English fixtures match HF tokens
for both ASR sizes; English word and Chinese character spans match HF. Repeated
calls and simultaneous ASR/standalone alignment pass. Strict BF16 logit tolerances
can still fail despite matching tokens/spans. Long-form specialized decoding can
change words; long-form alignment has small endpoint differences from HF.
