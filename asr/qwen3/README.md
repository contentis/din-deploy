# Qwen3 ASR and forced alignment

Offline, single-stream C++ inference using TensorRT RTX and original BF16 weights.
There are two public Python entry points: `export_qwen3_asr.py` and
`validate_qwen3_asr.py`. Supporting modules and profiling tools are internal.

## Support matrix

Rows describe model/upstream capabilities. A checkmark means the C++ sample
supports the capability within the validation limits below.

| Model / upstream toolkit capability | C++ sample |
|---|:---:|
| Offline transcription | ✓ |
| Online / streaming (upstream vLLM) | — |
| Single-stream inference | ✓ |
| Batched offline inference | — |
| Long-form audio | ✓ |
| ASR without forced aligner | ✓ |
| ASR with forced aligner | ✓ |
| Standalone alignment of supplied text | — |
| Automatic language identification | ✓ |
| Forced language / language hint | ✓ |
| Multilingual ASR: 30 languages and 22 Chinese dialects | — |
| Word timestamps | ✓ |
| Character timestamps | — |
| Singing and speech with background music | — |
| 0.6B ASR | ✓ |
| 1.7B ASR | — |
| 0.6B forced aligner | ✓ |
| Original BF16 precision | ✓ |
| Local operation without network after export | ✓ |

English speech and silence are validated; other languages and singing remain
unvalidated. Native alignment supports ASCII English words with HF punctuation
cleanup. The Python validator also supports standalone alignment. Upstream live
streaming supports neither batching nor timestamps. See the
[official toolkit](https://github.com/QwenLM/Qwen3-ASR#readme).

## Setup and export

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r asr/qwen3/requirements.txt
./.venv/Scripts/python.exe -m pip install --upgrade torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
./.venv/Scripts/python.exe asr/qwen3/model_export/export_qwen3_asr.py
./.venv/Scripts/python.exe asr/qwen3/model_export/export_qwen3_asr.py --task aligner
```

HF `from_pretrained()` downloads and caches the model and processor. Defaults are
`Qwen/Qwen3-ASR-0.6B-hf` and `Qwen/Qwen3-ForcedAligner-0.6B-hf`. Use `--model` for
a local checkpoint or another HF ID and `--revision` to pin the source.

Exports go to `artifacts/qwen3/onnx-bf16` and
`artifacts/qwen3/aligner-onnx-bf16`. Keep ONNX graphs, external `.onnx.data`
weights, processor files, metadata and `native.json` together. Each export
contains its own encoder and decoder/timestamp head. The log-mel frontend is
shared with Whisper. HF generates native prompt tokens; C++ does not reimplement
the chat template. `--only` can refresh individual graphs.

Weights and KV buffers remain BF16. HF-defined FP32 accumulation and RoPE are
preserved. `--dtype fp32` is a diagnostic option; native inference requires BF16.

### Optional faster cached decode

```powershell
./.venv/Scripts/python.exe asr/qwen3/model_export/export_qwen3_asr.py --only decode --decode-capacities 1024 2048 4096 8192 16384
```

Use the original export's `--model`, `--revision` and `--output`. The resulting
`decode_<capacity>.onnx` headers share `decode.onnx.data`. These use standard
opset-24 `Attention` and `TensorScatter`, with no contrib ops or custom plugins.
The pipeline automatically uses listed capacity specializations and falls back
to the general opset-23 decoder otherwise. SDK 1.6.1 rejects native GQA at 32K,
so that capacity uses the general decoder.

This path is opt-in because long-form ASR-only tokens can change; see validation
limits below. The Python validator selects the same capacity buckets. Pass
`--max-new-tokens 512` with the short sample to exercise the 1K specialization.

## Native pipeline

Build through the root CMake project using the same TensorRT RTX EP deployment
as Whisper. A BF16-capable NVIDIA GPU is required. Validation used MSVC 19.44,
ORT 1.27, EP 0.4.0 and TensorRT RTX 1.6.1.120. Set `TRT_RTX_ROOT` to the SDK;
the ORT provider DLL must also be deployed beside the executable.

```powershell
cmake --build build --config Release --target din_asr_qwen3_cli
./build/bin/din_asr_qwen3_cli.exe assets/sample.wav --model-dir artifacts/qwen3/onnx-bf16
./build/bin/din_asr_qwen3_cli.exe recording.mp3 --model-dir artifacts/qwen3/onnx-bf16 --aligner-dir artifacts/qwen3/aligner-onnx-bf16 --max-new-tokens 2048
```

Multi-configuration generators place executables in `build/bin/Release`.
Reuse a `Qwen3Pipeline` instance across calls to `Transcribe` or `TranscribeFile`;
concurrent calls on one instance are unsupported. Shared helpers provide audio
loading/resampling, tokenization, ORT sessions, buffers, streams and NVTX ranges.
The CLI emits JSON, including tokens, word spans, chunk count and `reached_eos`.
Token-limited chunks still produce partial output and alignment, as upstream
does; the CLI returns nonzero when any chunk fails to reach EOS.

### Long-form audio

The splitter follows the [official algorithm](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/utils.py):
1200-second ASR targets or 180-second aligned targets, a ±5-second boundary
search, the quietest 100 ms absolute-amplitude window, then its quietest sample.
Chunks have no overlap or gaps. Each chunk gets a fresh prompt and globally
normalized log-mel features. Short tails are padded to 0.5 seconds. Text is
joined literally; timestamps are shifted and rounded to milliseconds.

The default generation budget is 512 tokens, matching HF. Increase
`--max-new-tokens` for long speech; this does not change chunk boundaries.
Prefill uses 512-position blocks with absolute positions. Cache capacity is a
fixed power-of-two bucket based on prompt length plus generation budget, capped
at the exported maximum of 32768. Two BF16 banks occupy 448 MiB at 2K,
1.75 GiB at 8K and 7 GiB at 32K. The waveform remains in host memory.

The splitter uses FP32 reductions. Near-tied boundaries can differ from HF's
convolution reduction order; native decoding/resampling also differs from
librosa. The 388.6-second fixture is validated; a full 20-minute speech chunk
has not been exercised.

## Validation

```powershell
./.venv/Scripts/python.exe asr/qwen3/model_export/validate_qwen3_asr.py --audio assets/sample.wav
./.venv/Scripts/python.exe asr/qwen3/model_export/validate_qwen3_asr.py --task aligner --audio assets/sample.wav --transcript asr/qwen3/tests/sample.reference.txt
./.venv/Scripts/python.exe -m pytest asr/qwen3/tests -q
```

The validator loads the HF reference recorded in export metadata; `--model` can
override its location. It checks encoder outputs, logits, greedy tokens/EOS,
cache reuse or timestamp classes/spans. Reports are saved under `artifacts/qwen3`;
failed comparisons exit nonzero. The fixture transcript is model-generated.

- Native short speech matches all 36 HF tokens and 23 word spans. Silence,
  repeated calls, input bounds and general-decoder fallback pass.
- Specialized Python decoding matches short-fixture tokens through EOS, but
  strict numeric parity at tolerance 0.005 fails: encoder maximum error 0.00317,
  logit maximum error 1.25. Exact tokens do not imply numerical parity.
- Long-form aligned output preserves 1258 tokens and 1031 timestamps against
  the previous native implementation. Against HF alignment of identical text,
  four of 2062 endpoints differed by 80–240 ms in the saved comparison.
- Specialized ASR-only decoding produces 1229 tokens versus 1237 with the general
  graph, differing by one normalized word. Both have three normalized word edits
  against the saved 1033-word HF output. These are model comparisons, not WER
  against human labels.
- Small-model tests cover FP32/BF16 packing, cache handoff, untouched slots,
  final-slot updates and the shared frontend. Whisper's graph interface is
  unchanged and its extracted frontend was checked bit-for-bit.

With `BUILD_TESTING=ON`, internal targets provide text contracts and GPU checks:

```powershell
cmake --build build --config Release --target din_asr_qwen3_test din_asr_qwen3_cache_test
ctest --test-dir build -C Release -R qwen3_text_contract --output-on-failure
./build/bin/din_asr_qwen3_test.exe artifacts/qwen3/onnx-bf16 assets/sample.wav artifacts/qwen3/validation-consolidated-asr.json artifacts/qwen3/aligner-onnx-bf16 artifacts/qwen3/validation-consolidated-aligner.json
python asr/qwen3/tests/cache_fixture.py artifacts/qwen3/cache-test.onnx
./build/bin/din_asr_qwen3_cache_test.exe artifacts/qwen3/cache-test.onnx
```

## Attention and efficiency

The export reuses HF text layers, RoPE, Q/K normalization and MLPs. The general
decoder folds query groups into the sequence axis to avoid physical KV-head
replication. Its ordinary Scatter cache uses separate banks: **do not alias
them**. The specialized single-token graph uses native GQA and the explicit
`TensorScatter` in-place contract. Neither graph grows KV with concatenation;
attention still scans the allocated capacity, including unused slots.

General decoding retains one context per bank to keep CUDA graph bindings
stable. Specialized decoding retains at most two capacity engines. Prefill
reuses full/tail staging per bank; unchanged audio masks are not re-uploaded.
Cache handoff requires at most one GPU copy per chunk. Autoregressive steps
download only argmax tokens, and alignment downloads timestamp bins. Encoder
windows reuse buffers and assemble their output on the GPU, allowing CPU
staging to overlap computation.

## Performance and profiling

Warm measurements on RTX PRO 4000 Blackwell SFF, TensorRT RTX 1.6.1.120 / EP
0.4.0, for the same 388.5975-second English recording:

| Mode | Token budget | General decoder | Specialized decoder | RTF |
|---|---:|---:|---:|---:|
| ASR only, one chunk | 2048 | 24.13 s | 9.55 s | 0.0246 |
| ASR + alignment, three chunks | 1024 | 14.69 s | 8.09 s | 0.0208 |

These exclude loading and first-use compilation. The last pipeline change
measured 8.290 to 8.089 s in a paired aligned run (2.4% less time), within earlier
roughly 8.1–8.3 s variation, preserving exact tokens and timestamps.

The retained Nsight capture has 78.8% overall and 88.1% decode GPU-active time.
Prefill fell from 771 to 628 ms, graph updates from 11 to 6, and decode mask
uploads from 1255 to 2. Decode medians are about 5.72 ms at 4K capacity and
5.03 ms at 2K. The traced total is 9.333 s; use untraced results for RTF.
Timeline occupancy is not hardware SOL: permissions prevented SM/DRAM counter
collection. Remaining opportunities are encoder/prefill scheduling, host launch
overhead and SDK support for valid-prefix attention. Separate encoder contexts
and asynchronous prefill experiments were reverted after repeatability failures.

Build with `DIN_ENABLE_NVTX=ON` and `BUILD_TESTING=ON`. The profiling driver warms
the workload and checks exact repeated tokens and timestamps. Arguments are
`AUDIO [SECONDS [TOKENS [ALIGNER_DIR [RESULT_JSON [MODEL_DIR]]]]]`;
use an empty aligner directory for ASR only.

```powershell
cmake --build build --config Release --target din_asr_qwen3_profile
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown -o artifacts/qwen3/profiles/run ./build/bin/din_asr_qwen3_profile.exe recording.mp3 90 256
nsys export --type sqlite --output artifacts/qwen3/profiles/run.sqlite artifacts/qwen3/profiles/run.nsys-rep
python asr/qwen3/tests/analyze_nsys.py artifacts/qwen3/profiles/run.sqlite --output artifacts/qwen3/profiles/run-summary.json
```

The validation machine retains `artifacts/qwen3/sol/retained-final-trace.nsys-rep`
and final summary, parity and numerical-validation reports. Scratch experiments,
ablation binaries and superseded captures are not required by the sample.
