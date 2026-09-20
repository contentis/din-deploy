# Decoder gap analysis, 2026-09-20

The prefix-window and grouped-query changes have subsequently been implemented.
See [OPTIMIZATION.md](OPTIMIZATION.md) for the 95.4 tokens/s result, GPU sampling,
trace evidence and remaining numerical limitations. The analysis below records
the preceding full-window implementation.

The main recoverable gap is attention and its temporary tensors. The decoder's
weight-streaming kernels already achieve roughly 334–406 GB/s of inferred useful
weight throughput. Keeping greedy tokens on the GPU buys much less than removing
KV conversion/replication and avoiding work over unused cache slots.

This analysis concerns the existing BF16 2B export on the RTX PRO 4000 Blackwell
SFF (432 GB/s advertised bandwidth), batch 1. Production inference code was not
changed. Experimental benchmarks continue for a fixed token count after EOS;
their outputs must not be treated as conversational generations.

## What the decoder trace says

The decoder is graph ID 2 in `artifacts/qwen3-vl-final.sqlite`: 14 decode calls,
424 GPU kernels per call. Vision and prefill are excluded from this breakdown.

| Work | Summed kernel time per token |
| --- | ---: |
| QKV projection | 1.308 ms |
| Attention output projection | 0.645 ms |
| MLP gate/up projection | 3.643 ms |
| MLP down projection | 2.111 ms |
| Vocabulary projection | 1.532 ms |
| Attention computation | 1.724 ms |
| KV cast/replication/transposition | 1.096 ms |
| Other norms, rotary, scatter, activation, selection | 0.694 ms |

The projection total is 9.239 ms/token. Kernel times overlap: their 12.754 ms sum
is not the critical-path duration. The graph's mean first-kernel-to-last-kernel
span is 12.382 ms, with 12.337 ms covered by at least one kernel. There is little
idle time inside the graph. The 1.421 ms between graphs in this old trace includes
profiling overhead and should not be assumed to be recoverable host overhead.

The inferred weight throughput below divides known BF16 matrix bytes by each
kernel's duration; these are **not measured DRAM counters**. Fused arithmetic and
activation/cache traffic also consume time. Projection order is checked against
four matmuls per layer plus the vocabulary projection.

| Projection | Mean kernel duration | Useful weight rate | Fraction of 432 GB/s |
| --- | ---: | ---: | ---: |
| QKV | 46.71 us | 359 GB/s | 83% |
| Attention output | 23.05 us | 364 GB/s | 84% |
| Gate/up | 130.09 us | 387 GB/s | 90% |
| Down | 75.41 us | 334 GB/s | 77% |
| Vocabulary | 1532.10 us | 406 GB/s | 94% |

The combined projection rate is approximately 372 GB/s, 86% of the advertised
bandwidth. With 3.441 GB of decoder weights, the ideal weight-only floor is
7.966 ms/token. Only about 1.27 ms separates that floor from the observed matrix
kernels. Most of the remaining end-to-end gap is elsewhere.

## KV storage versus attention temporaries

Persistent K/V tensors are **BF16**, not FP32: 224 MiB for 28 layers, 8 KV heads,
2048 slots and head dimension 128. CPU BF16 is unsupported; this analysis uses RTX.

`modeling.attention` converts those tensors to FP32 and repeats 8 KV heads to
16 query heads. Across the layers, this produces 896 MiB of expanded temporary
K/V outputs per generated token. Including input reads and subsequent attention
reads gives roughly 2 GiB of logical traffic, not necessarily off-chip traffic:
one layer's 32 MiB expanded K/V fits within this GPU's reported 48 MiB L2 cache.
Only hardware counters can establish how much reaches DRAM.

Attention always consumes the exported 2048-slot shape, even with only 81 prompt
tokens and a short generated continuation. Masking preserves semantics but does
not remove those casts, replication or all attention work.

BF16 K/V storage is compatible with FP32 accumulation and softmax inside a fused
kernel. Full FP32 materialization is an implementation workaround, not a model
requirement. Query-group folding can also avoid duplicating K/V while retaining
explicit FP32 arithmetic. The ASR exporter already uses this idea for prefill.

## Controlled token-loop benchmark

`decoder_analysis.py` uses the same engines, 81 prompt tokens, 64 generated steps
and three interleaved repetitions. Engines and graph captures are warmed first.
The GPU feedback loop retains the host position controls and existing engine;
it only feeds the selected token back on-device. It produces the same 64 tokens
as the normal host loop on every trial.

| Mode | Median wall time/token | Rate |
| --- | ---: | ---: |
| Existing host greedy loop | 12.916 ms | 77.42 tokens/s |
| GPU token feedback | 12.717 ms | 78.63 tokens/s |
| Bare graph replay, fixed inputs/position | 12.463 ms | 80.24 calls/s |
| Host loop with temperature 0.7, top-p 0.9 | 14.041 ms | 71.22 tokens/s |

GPU feedback saves about 0.20 ms (1.5%). Removing all per-token preparation gives
an intentionally unrealistic bound of about 0.45 ms (3.5%); fixed-input replay is
not valid autoregressive generation. Top-p's full-vocabulary sort adds about
1.1 ms/token once warmed. The first sampled repetition was slower due to lazy
initialization. Greedy argmax itself is about 6 us/token in the existing trace.

The large `cudaStreamSynchronize` total in a CPU API summary mostly represents
waiting for GPU work. It must not be added on top of GPU kernel time or labeled
as independently removable overhead.

## Capacity experiment

Re-exporting with the same BF16 checkpoint and arithmetic, but 256 cache slots,
improved the same 81-token prompt / 64-step benchmark:

| Mode | 2048 slots | 256 slots |
| --- | ---: | ---: |
| Host greedy | 12.916 ms / 77.42 tokens/s | 10.683 ms / 93.61 tokens/s |
| GPU feedback | 12.717 ms / 78.63 tokens/s | 10.406 ms / 96.10 tokens/s |
| Bare graph replay | 12.463 ms | 10.202 ms |

Host greedy saves **2.233 ms/token**, or **20.9% higher throughput**. All 64 greedy
tokens match the 2048-slot run, and GPU feedback matches within each export. This
is an ablation on one short request, not broad accuracy certification or a reason
to reduce the default context capacity for long videos.

A few reusable capacity buckets selected once per request could recover much of
this cost. Prefer a fused kernel that receives the valid KV length while retaining
fixed allocation/engine shapes; provider support and actual work-skipping must be
tested. Simply changing tensor shape every step would risk compilation/capture
churn and defeat the current cache design.

See [256-slot benchmark](../../artifacts/qwen3-vl-decoder-c256.json).

## Attention compiler probes

Six small exports isolate one 16-query-head / 8-KV-head attention call with 2048
cache slots, head size 128 and 96 valid positions. Each was traced for 300 calls
after warming the GPU. NVML reported 9001 MHz memory clocks before/after every
range; SM clocks varied from 1087 to 1230 MHz. An earlier run with idle clock-state
variation was discarded. Times below come from the GPU timeline, not Python call
timings, which are dominated by launch overhead for these tiny graphs.

| Variant | GPU span/call | Observed lowering |
| --- | ---: | --- |
| Existing FP32 expression, standalone | 18.86 us | One fused MHA kernel |
| Same, dynamic query length with fixed decode profile | 18.87 us | Same fused kernel |
| Same, with aliased TensorScatter cache update/output | 19.85 us | Scatter plus fused MHA |
| Direct BF16 ONNX GQA attention | 15.81 us | One fused MHA kernel |
| Query-group folding, explicit FP32 arithmetic | 33.95 us | Seven kernels |
| Query-group folding, BF16 attention | 51.09 us | Cast plus GEMM attention |

The standalone current expression fuses without full K/V conversion buffers.
Dynamic query dimensions and aliased scatter do not individually prevent this.
The full decoder instead materializes K/V and has about 61.6 us of attention plus
39.1 us of K/V preparation per layer. This points to a **full-graph fusion/layout
opportunity**, rather than FP32 accumulation being intrinsically this expensive.
The exact interaction that prevents that fusion in the full decoder remains to
be isolated. Standalone probes have hot K/V and different surrounding work, so
their times cannot simply be multiplied by 28 to promise end-to-end speed.

Direct BF16 attention also needs care: at a synthetic Q/K input scale of four,
it failed an atol 0.01 / rtol 0.02 check against PyTorch SDPA (max absolute error
0.1328, relative L2 0.02054). The current expression and query-folded variants
passed the tested scales 1, 4 and 16. These three synthetic cases are probes,
not model-accuracy certification. Do not remove explicit precision constraints
solely because the fastest standalone kernel is BF16.

Evidence: [kernel breakdown](../../artifacts/qwen3-vl-attention-kernels.json),
[numeric checks and clocks](../../artifacts/qwen3-vl-attention-probe/report.json),
[probe trace](../../artifacts/qwen3-vl-attention-probe-trace.nsys-rep).

## Recommended order of work

1. Preserve BF16 persistent K/V and FP32 accumulation. Isolate why the full
   decoder misses the fusion already demonstrated by the standalone expression;
   this could remove the large temporary tensors without weakening numerics.
2. Avoid processing the full capacity for short requests: valid-length-aware
   fused attention or a small reusable set of request-level capacity buckets.
   The 256-slot experiment proves a 2.23 ms/token opportunity on this workload.
3. Tune the down projection and other matrix kernels after attention. Matrix
   kernels collectively have about 1.27 ms of remaining weight-only theoretical
   headroom; the vocabulary projection is already near its bandwidth floor.
4. Optimize GPU top-p sampling if stochastic decoding matters. Greedy host token
   feedback is a much smaller opportunity (about 0.20 ms/token).

These savings overlap and should not be added together. Validate numerical
agreement, shared-cache correctness and continuous graph execution after each
change, and then remeasure the complete decoder.

## Hardware sampling status

Nsight Systems 2026.4.1 is installed. Its GPU metric query identifies GB203 but
returns `ERR_NVGPUCTRPERM` / insufficient privilege, including outside the sandbox.
The one-time administrator capture was canceled at the Windows UAC prompt.
No driver permission settings were changed. There are no measured DRAM/L2/SM
utilization counters or instruction-stall samples in these results.

The prepared `artifacts/qwen3-vl-profile-admin.ps1` captures the warmed decoder
with the `gb20x` metric set at 10 kHz. To obtain counters, run that fixed capture
from an administrator session or enable profiling access through the NVIDIA
control panel. See [NVIDIA's counter permission documentation](https://developer.nvidia.com/ERR_NVGPUCTRPERM).

## Reproduction and evidence

```powershell
python -m vision.qwen3_vl.decoder_analysis --model-dir artifacts/qwen3-vl-2b-bf16-accurate --image artifacts/qwen3-vl-fixtures/shapes.png --steps 64 --repeat 3 --report artifacts/qwen3-vl-decoder-analysis.json
python -m vision.qwen3_vl.nsys_decoder_summary artifacts/qwen3-vl-final.sqlite --metadata artifacts/qwen3-vl-2b-bf16-accurate/metadata.json --report artifacts/qwen3-vl-decoder-kernels.json
```

- [Token-loop benchmark](../../artifacts/qwen3-vl-decoder-analysis.json)
- [Decoder-only kernel summary](../../artifacts/qwen3-vl-decoder-kernels.json)
- [Original system trace](../../artifacts/qwen3-vl-final.nsys-rep)

Numerical parity remains subject to the limitations in [VALIDATION.md](VALIDATION.md).
Performance experiments are not a substitute for full decoder/vision validation.
