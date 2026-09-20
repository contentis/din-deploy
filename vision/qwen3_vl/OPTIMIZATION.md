# Decoder optimization results, 2026-09-20

Historical duplicated-engine measurements below are superseded by the
[single-weight implementation](SINGLE_WEIGHTS.md), which shares one engine across
prefill/decode and all attention windows.

The 2B BF16 decoder improved from 77.4 to 95.4 tokens/s on the RTX PRO 4000
Blackwell SFF Edition. Both runs retain a 2048-slot KV bank and use the same
81-token image prompt, 64 decode steps, batch 1, warm engines and CUDA graphs.
These diagnostic runs deliberately continue after EOS. They measure decode,
excluding vision, prefill, compilation and engine loading.

## Changes

- Decode attention reads a fixed prefix window: 256, 512, 1024 or 2048 slots.
  It selects the smallest window containing the valid cache prefix. TensorScatter
  still writes into the original full-capacity KV bank, with aliased past/present
  bindings. Switching windows does not move or resize the cache.
- Grouped-query attention folds query groups into the query sequence dimension,
  eliminating K/V head duplication. Attention accumulation remains FP32; KV storage
  remains BF16 or FP16. The native low-precision attention experiment failed the
  earlier numerical stress test and was not adopted.
- All decode profiles share stable control and rotary input buffers. Prefill
  always uses its own profile, including a one-token final chunk, so decode
  profiles cannot retain stale visual features across requests.
- Sampling uses GPU softmax, sorting/CDF and one uniform draw. It avoids generating
  one random value per vocabulary entry and avoids intermediate host decisions.
  `top_p=1` skips sorting. The scalar selected token is still read for EOS handling.
  Same-seed repeatability is tested; token sequences need not match the previous
  multinomial implementation for the same seed.

For this short prompt, the summed logical FP32 K/V temporary footprint across 28
layers falls from 896 MiB to 56 MiB per token: 8x from the prefix and 2x from
removing repeated heads. This is an allocation/work estimate, not measured DRAM
traffic. The persistent KV bank remains 224 MiB. At full context the prefix
advantage disappears; avoiding repeated heads still halves this temporary size.

## Measurements

| Warm decoder mode | Previous | Optimized |
| --- | ---: | ---: |
| Host greedy loop, ms/token | 12.916 | 10.485 |
| Host greedy loop, tokens/s | 77.42 | 95.37 |
| GPU token feedback diagnostic, ms/token | 12.717 | 10.269 |
| Fixed-input graph replay diagnostic, ms/token | 12.463 | 10.028 |

The greedy throughput gain is 23.2%. With 3.441 GB of decoder weights and 432 GB/s
peak bandwidth, the weight-only floor is 7.966 ms/token or 125.54 tokens/s.
Measured greedy throughput is now **76.0% of that ceiling**, up from 61.7%.
This definition excludes vision weights, KV traffic and other obligatory work.

An independent five-repeat, interleaved sampling comparison on the optimized
decoder measured 11.951 ms/token for the CDF sampler versus 12.209 ms/token for
full-sort multinomial sampling. Both use temperature 0.7, top-p 0.9 and seed 123.
The final timing is recorded in `artifacts/qwen3-vl-sampling-cdf-final.json`:
about 0.26 ms/token saved. An earlier run measured a 0.45 ms gain. Greedy timing
in this final run was 10.685 ms/token (93.59 tokens/s), illustrating run variation.
Different random algorithms generate different token paths; the comparison is a
practical workload measurement, not an identical-logit microbenchmark.
An attempted top-256 shortcut plus mass check was slower (13.026 ms/token versus
11.951 ms for full-sort multinomial), and was discarded.

## Trace evidence

`artifacts/qwen3-vl-prefix-trace.nsys-rep` captures 64 warm greedy steps, with a
SQLite export and `qwen3-vl-prefix-kernels.json` summary. The CUDA timeline shows:

| Kernel work | Previous trace, ms/token | Optimized trace, ms/token |
| --- | ---: | ---: |
| Graph span | 12.382 | 9.804 |
| Weight projections, summed | 9.239 | 8.890 |
| Attention matmul/softmax, summed | 1.724 | 0.264 |
| K/V conversion/reformatting, summed | 1.096 | 0.371 |

Kernel sums can overlap and should not be added to graph span. The traces were
captured separately, so projection differences include clock/measurement effects.
The optimized graph has 479 kernels versus 424 previously: it does less data work,
but attention is still decomposed into several kernels per layer.

There are 64 graph launches, no graph instantiation/capture APIs in the warmed
range, and no full-cache memcpy. Per token transfers are 24 bytes host-to-device,
8 bytes device-to-host and three small device copies (maximum 512 bytes).
ORT profiles independently show one RTX partition each for vision, prefill and
256-slot decode, with no CPU fallback (`qwen3-vl-prefix-partitions.json`).
The host trace still contains one asynchronous pool allocation/free pair and two
stream synchronizations per token. These are remaining runtime/host-loop costs;
the trace does not establish which internal allocation owns the pool request.
Profiling adds overhead, so unprofiled wall timings determine reported throughput.

Hardware bandwidth counters remain unavailable under the existing driver
permissions. Rates inferred from weight sizes and kernel durations are not actual
DRAM-counter measurements. Administrator profiling was not retried.

### GPU idle-time follow-up

The detailed 64-step capture spans 708.103 ms from the first graph's first kernel
to the last graph's last kernel. The union of all recorded kernel and memcpy
activity is 625.197 ms, leaving **82.906 ms idle (11.71%, 1.295 ms/token)**.
Within individual decoder graphs, the difference between span and kernel union
is only **0.0397 ms/token (0.405% of graph span)**. Most recorded idle time lies
between graph launches, rather than between kernels inside a graph.

A second capture using graph-level rather than per-node tracing did not reduce
the observed gaps: 9.934 ms mean graph span, 1.431 ms mean inter-graph gap, and
12.39% idle outside graphs after accounting for memcpy. Its measured iteration
took 11.378 ms/token. See `qwen3-vl-idle-coarse.nsys-rep` and
`qwen3-vl-idle-summary.json`. Graph-level tracing cannot resolve idle inside graphs.

These are **profiled** idle fractions, not a measurement of normal unprofiled
idle. The unprofiled benchmark is faster at 10.485 ms/token. Its fixed-input
replay takes 10.028 ms/token, demonstrating about **0.458 ms/token (4.37%)** of
removable request-loop work in that diagnostic, while GPU feedback alone saves
0.216 ms. Those differences are host/control-path overhead, not direct idle
measurements, and fixed-input replay is not valid autoregressive generation.

The detailed trace attributes one stream synchronization to `host_decode`
(9.789 ms average, predominantly overlapping graph execution) and another to
token selection (0.0545 ms average). Do not count CPU synchronization duration as
GPU idle. Removing all traced gaps would imply an approximately 13% throughput
gain within that capture, not recovering the entire 24% weight-only SOL gap.

## Validation and remaining work

- 17 tests pass, including independent HF versus CPU ONNX logits/KV checks,
  cache-window transitions, stable pointers, reset and write bounds, plus sampler
  distribution/support and seed checks.
- Six full-model BF16 smoke cases pass: single-image colors, multiple images,
  chunked visual prefill and video. Repeated seeded sampling also passes
  (`qwen3-vl-acceptance-prefix.json`).
- A 260-step teacher-forced comparison with the previous BF16 decoder has identical
  greedy choices at every step. The 256-to-512 transition and reset preserve KV
  pointers and write bounds. Prefill logits are identical. Decode numerical checks
  still fail at atol 0.25 / rtol 0.02: sampled logits have maximum absolute error
  up to 0.40625, and active cache differences reach 5.0, worst layer relative L2
  0.02914 (`qwen3-vl-prefix-parity-corrected.json`).
- Full-model BF16 versus HF also still fails strict numerical checks, despite
  matching the 16-token image description and passing cache integrity. Vision
  max error is 0.59375 (relative L2 0.05206), prefill logits max error 0.46875,
  and cache max error 70.0625 (`qwen3-vl-validate-prefix.json`). Existing vision
  and reduced-precision discrepancies are not fixed by these optimizations.
- The new FP16 export runs on CPU and TensorRT RTX. Both produce the expected
  image description, and the CPU comparison passes logits, greedy tokens, reset
  and cache integrity. Strict CPU KV parity still fails, with max error 5.28125
  (`qwen3-vl-validate-fp16-prefix-cpu.json`). The RTX second request measured
  95.69 tokens/s over this short description (`qwen3-vl-fp16-prefix-rtx.json`);
  this is a functional check, not a repeated performance characterization.

The main remaining performance opportunities are fused attention that preserves
the FP32 accumulation contract, the remaining 0.371 ms of K/V conversion work,
and host/runtime overhead. Weight projections already approach the bandwidth
ceiling; further large gains likely require changing weight precision or batching.
The diagnostic GPU feedback loop saves roughly 0.2 ms, but does not implement EOS
handling and is not substituted for the generation API.

Profiles are compiled/loaded lazily once per window and cached persistently. A
first transition can therefore still incur compilation/capture. Additional
profiles can retain additional weight copies; this needs particular care on 8B.
Use fewer `--attention-buckets` to trade prefix efficiency for engine memory and
cold-start cost. 8B remains shape-tested, not measured on hardware.

Current optimized exports are `artifacts/qwen3-vl-2b-bf16-prefix` and
`artifacts/qwen3-vl-2b-fp16-prefix`. Re-export to obtain the new graph contract;
older exports remain loadable but retain their original attention work.
