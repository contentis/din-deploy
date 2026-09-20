# Shared weights and decoder execution, 2026-09-20

The runtime now uses one vision session and one decoder session. Prefill, all
decode attention windows, and all vision buckets reuse their component's engine.
The previous fixed-session implementation was faster on long decode benchmarks,
but duplicated weights and is no longer the runtime path.

## Retained changes

- One dynamic engine per component, with stable bindings for each supported shape.
- BF16/FP16 persistent KV storage and in-place TensorScatter writes. Folded GQA
  avoids duplicated heads; FP32 attention temporaries cover the active prefix.
- An explicit CUDA stream with event ordering at the Python API boundary removes
  the provider's extra default-stream synchronization.
- Captured GPU categorical/nucleus sampling; logits stay on device and only the
  chosen token is read back. Seeds and mutable sampling parameters are tested.
- Optional [isolated provider build](patches/README.md): persistent decoder
  workspace and cached execution contexts sharing one engine. This eliminates
  repeated prefill/decode capture in the measured warm range. Vision retains
  transient workspace to avoid reserving its large maximum-shape workspace.
- Source ONNX exports reject nonstandard node domains. No contrib attention or
  custom sampling operator was added. CPU uses the same source ONNX graphs;
  TensorRT's compiled EPContext file is provider-specific cache packaging.

## Measurements

RTX PRO 4000 Blackwell SFF Edition, 24 GB, theoretical 432 GB/s. Qwen3-VL-2B-Instruct
BF16, batch 1, 81-token image prompt, capacity 2048, 64 decode steps, 256-slot
attention prefix. Diagnostic generation continues beyond EOS. These timings
exclude vision, prefill, engine loading and cold JIT compilation.

| Configuration | ms/token | tokens/s |
|---|---:|---:|
| Original full-window decoder | 12.916 | 77.4 |
| Historical prefix decoder with duplicated engines | 10.485 | 95.4 |
| Shared engine, installed provider, explicit stream | 10.889 | 91.8 |
| Shared engine, persistent workspace | 10.916 | 91.6 |
| Shared engine, persistent workspace and cached contexts | 11.106 | 90.0 |
| Cached contexts, diagnostic GPU token feedback | 10.792 | 92.7 |
| Cached contexts, fixed-input replay only | 10.556 | 94.7 |

The cached-context run uses three interleaved repetitions. The other configurations
were measured separately, so small differences may include clock/system variation.
Keeping captures avoids transition overhead; it does not improve the matrix
kernels in this shared dynamic engine. The single-weight constraint costs about
4–6% against the earlier duplicated-engine benchmark.

With 3.441 GB decoder weights, the user's weights/bandwidth lower bound is 7.966 ms,
or 125.5 tokens/s. Current host-driven decode achieves approximately **72–73%** of
that ideal; fixed replay achieves about 75%. This is an idealized comparison, not
a measured DRAM efficiency counter.

A five-repetition sampling comparison before context caching measured 11.514 ms
for uncaptured CDF sampling and 11.055 ms for captured CDF sampling, versus
10.954 ms greedy: about **0.46 ms/token saved**, leaving about 0.10 ms sampling
overhead. The final cached-context run measured 11.107 ms for captured sampling.

## Trace and memory evidence

`artifacts/qwen3-vl-contexts-trace.nsys-rep` and
`artifacts/qwen3-vl-contexts-kernels.json` contain the final warm decoder trace:

- 64 graph replays, 538 kernels per token; **no graph creation/recapture** in the
  captured range and no CUDA workspace allocation/free calls there.
- **9.62% idle** across the captured GPU kernel span, counting kernels and copies
  as activity: 71.06 ms over 738.96 ms. Restricting to the first/last decoder graph
  kernels gives 9.45%. These are traced-process timeline gaps, not SM occupancy
  or device-wide idle counters. Profiling increases host overhead.
- One token-readback synchronization per decode step; the previous redundant
  provider wait is gone. CPU time spent in this wait mostly overlaps GPU work.
- Controls transfer 24 bytes H2D and the selected token 8 bytes D2H per step.
  Device copies are at most 512 bytes; no full-KV transfer or copy appears.
- Mean graph span 10.479 ms, kernel union 10.431 ms: only 0.048 ms idle inside
  each graph. Most remaining idle time lies between graph launches.
- ORT profiling across the acceptance cases reports one continuous RTX partition
  per component (10 vision and 46 decoder calls), with no CPU fallback.

The earlier 11.7% and 9.3% figures used the first/last decoder graph span. The
single-context 9.3% result excluded its initial recapture overhead; its complete
captured GPU span was about 16.9% idle. The new cached-context trace has no such
recapture. Compare like scopes, and use unprofiled throughput for speed claims.

`artifacts/qwen3-vl-weight-audit.json` exercises windows 256, 512, 1024, 2048 and
back to 256. It finds exactly one active serialized engine for each component:
decoder 3,447,025,908 bytes and vision 1,628,581,828 bytes. There are two ORT
sessions throughout, one decoder session identity, and unchanged KV pointers.
Total observed device use rises from 6,757,027,840 to 6,767,513,600 bytes across
the four cached contexts: **10 MiB extra**, not another weight bank. These device
totals include other processes. KV itself is 224 MiB. Synthetic positions in
this audit check allocation behavior, not attention accuracy.

## Correctness and limits

- 19 tests pass, including tiny-model HF comparisons, standard ONNX domains,
  CPU chunked prefill/reset, one-session reuse, in-place KV bounds, sampling
  equivalence/reproducibility, and trace interval accounting.
- The real 2B decoder matches **260/260** baseline greedy choices, including the
  256→512 attention-window transition; write bounds, KV addresses and reset pass.
- Real-model known-answer acceptance covers single images, multiple images,
  a larger image requiring chunked prefill, and a video segment: **6/6 pass**.
  Repeated seeded generation also passes. Individual results are in
  `artifacts/qwen3-vl-acceptance-contexts.json`.
- The FP16 CPU run produces the expected HF description and passes the logit,
  cache-address, write-bound and reset checks. FP32 CPU was validated earlier.
  ORT CPU does not support this BF16 graph; choose FP16 or FP32 for CPU.
- **Strict BF16/FP16 numerical parity is still not certified.** The final 260-step
  comparison against the earlier export has prefill relative L2 0.022 and maximum
  absolute logit error 0.53125; cache comparisons also exceed strict tolerances.
  Functional/token agreement is not a substitute for a broad accuracy evaluation.
  No tolerance was relaxed to turn these failures into passes.
- 8B has configuration/shape coverage only; no 8B checkpoint hardware validation.

## Remaining performance gaps

Gate/up projections dominate at 4.263 ms/token and approximately 331 GB/s inferred
weight bandwidth, versus 408 GB/s for the LM head. Together with other matrix
kernels this is the main gap to the weights/bandwidth bound. Native BF16/FP16 KV
still needs FP32 conversion for the retained attention calculation. The folding
and prefix changes greatly reduce it but do not eliminate it.

The host still submits input updates and reads each token for EOS. Diagnostic GPU
feedback saves about 0.31 ms/token, but does not implement production EOS handling.
First use of a shape still specializes/captures. The context cache removes repeat
capture, not cold compilation. Hardware memory counters were unavailable without
administrator profiling privileges; all bandwidth figures above are inferred.

Standard ONNX Attention and outer CUDA capture were also tested. Attention either
failed the dynamic prefill backend or ran slower; outer capture was unsupported
with the installed allocator, and the working isolated-provider variant was
slower. Neither experiment is retained in the runtime.

## Reproduce

```powershell
python -X utf8 -m pytest vision/qwen3_vl/tests -q
python -X utf8 -m vision.qwen3_vl.decoder_analysis --model-dir artifacts/qwen3-vl-2b-bf16-prefix --ep-library artifacts/trt-ep-source/build/Release/onnxruntime_providers_nv_tensorrt_rtx.dll --image test.jpg --steps 64 --repeat 5 --modes host_greedy gpu_feedback graph_replay host_top_p host_top_p_captured --report artifacts/decoder.json
python -X utf8 -m vision.qwen3_vl.nsys_decoder_summary artifacts/qwen3-vl-contexts-trace.sqlite --metadata artifacts/qwen3-vl-2b-bf16-prefix/metadata.json --report artifacts/trace-summary.json
```

Omit `--ep-library` to use the installed provider. Both paths share one engine per
component; persistent workspace/context caching requires the optional build.
