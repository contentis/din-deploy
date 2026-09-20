# Local validation, 2026-09-20

Current single-weight runtime results and its 19-test suite are documented in
[SINGLE_WEIGHTS.md](SINGLE_WEIGHTS.md). Measurements below are earlier baselines.

The 2B implementation runs on TensorRT RTX and CPU, but **reduced-precision
numerical parity is not yet certified**. Functional smoke tests and cache-integrity
tests pass; strict BF16/FP16 comparisons still fail. These are separate results.
8B has a configuration/shape test, not a checkpoint export or hardware run.

The subsequent decoder optimization uses `*-prefix` exports and passes 17 tests,
including window transitions and sampling. See [OPTIMIZATION.md](OPTIMIZATION.md)
for current performance, trace evidence and numerical results. The measurements
below describe the earlier full-window decoder and remain historical baselines.

## Environment and artifacts

- Windows, NVIDIA RTX PRO 4000 Blackwell SFF Edition, 24 GB, driver 596.59.
- PyTorch 2.14.0+cu130, Transformers 5.13.0, ONNX Runtime 1.30.0,
  standalone `nv_tensorrt_rtx` EP, TensorRT RTX 1.6, Nsight Systems 2026.4.1.
- `Qwen/Qwen3-VL-2B-Instruct` revision
  `89644892e4d85e24eaac8bacfd4f463576704203`.
- Current local exports: `artifacts/qwen3-vl-2b-bf16-accurate`,
  `artifacts/qwen3-vl-2b-fp16-accurate`, `artifacts/qwen3-vl-2b-fp32`.
  The `accurate` suffix identifies the latest precision investigation, not a
  claim that parity passed. Earlier directories without this suffix are
  exploratory artifacts and should not be used.
- Batch 1, cache capacity 2048, prefill block 128, vision buckets
  256/1024/4096 raw patches. Models, reports and traces are local ignored artifacts.

## Correctness

`python -m pytest vision/qwen3_vl/tests -q` passed **14 tests** in 20.95 seconds.
The four warnings concern redundant exported dynamic-axis names.
Tests cover budgeting, variable timestamps and frame sampling, spatial padding,
DeepStack, multimodal rotary positions, real CPU ONNX execution with aliased KV
buffers, chunked prefill, reset, cache bounds, retrieval storage, and the actual
8B layer/head/hidden dimensions using a meta-device model.

The numerical validator compares the same export/reference dtype, using HF SDPA.
It checks vision and every DeepStack output, positions, prefill logits and all
layer caches, four teacher-forced decode steps, unchanged cache slots outside the
write, stable pointers, greedy tokens, and reset/repeat behavior. It exits with
failure when numerical comparisons fail, even if the words match.

| Export/provider/input | Numerical result | Greedy tokens and cache integrity |
| --- | --- | --- |
| FP32 / CPU / image | Pass, atol 0.005 | Pass |
| FP32 / CPU / two images | Pass, atol 0.005 | Pass |
| FP32 / CPU / video | Pass, atol 0.005; separate cache atol 0.02 | Pass, including EOS |
| FP16 / CPU / image | Fail on KV values; vision/logits pass atol 0.25 | Pass, including EOS |
| FP16 / TensorRT RTX / video | Fail at atol 0.25 | Pass, including EOS |
| BF16 / TensorRT RTX / video | Fail at atol 0.25 | Pass, including EOS |

All comparisons also use rtol 0.02. Image and multi-image FP32 greedy comparisons
use bounded token prefixes; they do not claim EOS completion. Video FP32 cache
max absolute error was 0.01399, with worst layer relative L2 about 0.0000294.
The original video run at cache atol 0.005 failed; the separate 0.02 cache
tolerance is explicit because KV activations are unbounded.

Reduced-precision differences remain material:

| TensorRT RTX video | Vision max abs / relative L2 | Prefill logits max abs | KV max abs |
| --- | --- | --- | --- |
| BF16 | 4.15625 / 0.27077 | 0.8125 | 102.875 |
| FP16 | 0.84766 / 0.05613 | 0.4375 | 41.52734 |

CPU FP16 image KV max absolute error was 0.93652. Matching generated text on
these examples does not establish broad model accuracy. The implementation keeps
the original patch convolution and requests FP32 normalization/attention/vision
linear accumulation, but this has not eliminated provider/reference differences.
The next accuracy work should isolate the first divergent intermediate on RTX
and validate a larger held-out image/video set before accepting reduced precision.

An additional first-block probe found exact patch-convolution agreement on the
shapes fixture, projection relative L2 about 0.000105, and a rise to 0.00385 at
GELU / 0.00353 at the first block output. Exposing intermediate outputs can change
fusion, so this does not prove the cause in the production graph. An attempted
FP32 GELU promotion made full-video agreement substantially worse and was reverted;
the `*-gelu` local exports are rejected experiments, not current deliverables.

Reports: [FP32 image](../../artifacts/qwen3-vl-fp32-cpu.json),
[FP32 multi-image](../../artifacts/qwen3-vl-fp32-multi.json),
[FP32 video](../../artifacts/qwen3-vl-fp32-video-final.json),
[FP16 CPU](../../artifacts/qwen3-vl-fp16-cpu-final.json),
[FP16 RTX](../../artifacts/qwen3-vl-fp16-video-accurate.json),
[BF16 RTX](../../artifacts/qwen3-vl-bf16-video-accurate.json).

Six BF16 RTX known-answer smoke cases passed with EOS: blue circle, red rectangle,
solid green, second image green, a 275-token prompt with 252 visual tokens spanning
prefill blocks, and a dog visible in the local video. Repeated sampling with
temperature 0.7, top-p 0.9 and seed 123 produced identical 16-token sequences.
See [acceptance report](../../artifacts/qwen3-vl-acceptance.json).

## Performance and system trace

The current BF16 image benchmark generated 15 tokens including EOS. With a
64-token visual budget (63 used), the last two untraced warm requests measured:

| Stage | Warm request 2 | Warm request 3 |
| --- | --- | --- |
| Vision | 10.55 ms | 11.38 ms |
| Prefill | 19.42 ms | 20.12 ms |
| Autoregressive decode | 77.54 tokens/s | 78.58 tokens/s |

Decode timing includes sampling/scalar readback. Preprocessing and session
construction are outside these warm stage times. The first request, even with
saved engines, still paid session loading and capture costs (about 1.99 s vision
and 2.43 s prefill); cold starts are not represented by warm throughput.
All four requests reported three sessions, zero newly compiled contexts, and a
224 MiB shared KV bank. See [benchmark](../../artifacts/qwen3-vl-benchmark-untraced.json).
These measurements are a baseline, not evidence that hardware limits are reached.

The [Nsight trace](../../artifacts/qwen3-vl-final.nsys-rep) captures fully warm
request 2. ORT profiling confirms **one RTX partition each** for vision, prefill
and decode, without CPU fallback. The trace shows:

- 16 CUDA graph launches: one vision, one prefill, 14 decode calls.
- No graph instantiation or JIT compilation APIs in the captured request.
- 1.097 MB total device-to-device copies, largest 1.032 MB; no 224 MiB KV copy.
- 0.958 MB host-to-device transfers; 16 small scalar device-to-host transfers.
- 16 asynchronous pool allocations remain inside EP execution.
- Roughly 45% of GPU kernel time in the largest generated matmul kernel,
  17% in matmul/pointwise work, 12% in MHA GEMV, and 7% in cast/repeated-KV work.

The observed transfers corroborate the cache alias/pointer/write-bound tests.
Remaining tuning opportunities include eliminating repeated-KV materialization,
reducing casts and EP temporary allocations, and sharing decoder weights between
prefill/decode engines. Re-profile after resolving numerical differences; older
faster exploratory graphs are not the reported current implementation.

## Retrieval demo

The separate SQLite + NumPy demo ingested three images. Querying with the shapes
image retrieved itself first (cosine 0.99999988), and reasoning returned:
"The image contains two shapes: a red rectangle and a blue circle."

Text query "red rectangle and blue circle" incorrectly ranked the outdoor video
frame first (0.7805) and the shapes image second (0.7469). This establishes that
the text-query path executes, **not useful semantic retrieval accuracy**. Use the
demo to iterate on storage/retrieval/reasoning; replacing Instruct hidden states
with a trained multimodal embedding model is the next retrieval-quality step.

## Remaining scope

- Resolve strict BF16/FP16 parity and evaluate accuracy beyond smoke examples.
- Export and run an actual 8B checkpoint; two engine weight copies may exceed
  this GPU's memory. Only its dimensions have been tested.
- Profile larger vision buckets, longer videos/contexts and more hardware.
- Native C++ application integration is not part of this initial Python module.
