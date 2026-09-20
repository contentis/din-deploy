# Qwen3-VL

Initial Python ONNX Runtime implementation for `Qwen/Qwen3-VL-2B-Instruct`
and `Qwen/Qwen3-VL-8B-Instruct`. Work and hardware validation start with 2B;
8B uses the same configuration-driven exporter/runtime but has not been run locally.
This module is independent of the existing C++ audio/video applications.

## Setup and export

Run from the repository root using its virtual environment, a CUDA PyTorch build,
and `pip install -r vision/qwen3_vl/requirements.txt`. The implementation uses
Transformers 5.13's Qwen3-VL interfaces and ONNX opset 24 `TensorScatter`.

```powershell
python -X utf8 -m vision.qwen3_vl.export --output artifacts/qwen3-vl-2b-bf16
python -X utf8 -m vision.qwen3_vl.export --dtype fp16 --output artifacts/qwen3-vl-2b-fp16
python -X utf8 -m vision.qwen3_vl.export --dtype fp32 --output artifacts/qwen3-vl-2b-fp32
python -X utf8 -m vision.qwen3_vl.export --size 8B --output artifacts/qwen3-vl-8b-bf16
```

`--dtype original` (default) retains checkpoint BF16. FP16 and FP32 convert the
checkpoint, without quantization. Normalization, rotary calculations and logits
use FP32 where needed. Attention uses explicit FP32 matmul/softmax accumulation;
vision linear layers also request FP32 accumulation while retaining checkpoint
weight storage. CPU supports FP32 and FP16 exports; ORT CPU lacks the
necessary BF16 kernels and rejects that combination explicitly. Use a fresh
directory for each export. `--model` accepts a local checkpoint or HF ID, and
`--revision` pins a source revision. The resolved revision is saved in metadata.

`--cache-capacity` defaults to 2048 tokens and `--prefill-block` to 128. Capacity
must be a multiple of the block. The default decode attention windows are
256/512/1024/2048 slots, capped at the exported capacity. Decode selects the smallest
window containing the valid prefix while retaining the full shared KV allocation.
`--attention-buckets 2048` selects one full-capacity decode profile instead.
2B's BF16/FP16 KV bank at 2048 slots occupies 224 MiB.
Prefill and all decode windows share one decoder session/engine and one set of
weights. Vision buckets likewise share one vision session. Bindings retain separate
activation buffers but do not load another model. No automatic offload or
quantization is implemented; 8B hardware memory/performance remains unmeasured.

## Images and video segments

```powershell
python -X utf8 -m vision.qwen3_vl.cli --model-dir artifacts/qwen3-vl-2b-bf16 --images first.jpg second.jpg --max-visual-tokens 512 --prompt "Compare these images."
python -X utf8 -m vision.qwen3_vl.cli --model-dir artifacts/qwen3-vl-2b-bf16 --video clip.mp4 --video-start 5 --video-end 15 --fps 2 --max-frames 16 --max-visual-tokens 512 --prompt "Describe the action."
python -X utf8 -m vision.qwen3_vl.cli --model-dir artifacts/qwen3-vl-2b-fp16 --provider cpu --images first.jpg
```

The visual budget counts **merged language-model visual tokens**, shared across
all images and segments in one request. Four raw vision patches become one visual
token. Each image or segment receives an equal share; a video share is divided
among temporal groups. Frames are sampled uniformly within the segment and retain
their actual presentation timestamps, including for variable frame rates. Spatial
dimensions are resized to multiples of 32 and checked after processing. The last
odd frame is repeated by the upstream processor for temporal pairing. Text,
timestamps and chat delimiters consume additional context tokens.

The default vision buckets hold 256, 1024 and 4096 **raw patches** per image or
temporal group. Runtime preprocessing additionally caps each frame/image to the
largest exported bucket. `--vision-buckets` changes the export contract. A batch
of images means multiple images in one conversational request, not independent
batched prompts. Python callers can pass several `VideoSegment` instances to
`prepare_inputs`; the CLI exposes one segment plus any number of images.

```python
from vision.qwen3_vl.inputs import VideoSegment, prepare_inputs
from vision.qwen3_vl.runtime import Qwen3VL

model = Qwen3VL("artifacts/qwen3-vl-2b-bf16")
inputs, budget = prepare_inputs(
    model.processor,
    "What changed?",
    images=["before.jpg"],
    videos=[VideoSegment("after.mp4", start=3, end=6)],
    max_visual_tokens=512,
    max_image_tokens=max(model.metadata["vision_buckets"]) // 4,
)
result = model.generate(inputs, max_new_tokens=128)
print(result["text"], result["reached_eos"])
```

Greedy decoding is default. `--temperature`, `--top-p` and `--seed` enable
autoregressive sampling. A generation limit is reported with `reached_eos=false`;
it is never presented as EOS. Requests exceeding the fixed cache fail before
generation. Instances process requests serially and reuse engines and buffers.
Sampling uses a device-side cumulative distribution and one uniform draw, with a
full vocabulary sort for nucleus sampling. CUDA execution reuses two small sampling
graphs (nucleus and unrestricted sampling) and stable parameter buffers. Random
draws remain outside capture to preserve per-request generator/seed semantics.
There is no intermediate host decision;
only the selected token is read back for EOS handling. Seeds reproduce results
within this implementation, not the previous multinomial sampler's token sequence.

## Cache and execution contract

The vision graph retains the patch convolution, learned bilinear position lookup,
spatial rotary attention and every DeepStack merger. Video temporal groups are
encoded separately because upstream vision attention is spatial within each group.
The decoder receives all DeepStack levels and interleaved multimodal rotary
positions; cache offsets are distinct from rotary positions.

Prefill and decode use one ONNX decoder session with a dynamic optimization range
covering the fixed prefill block (128 by default) and the bounded set of decode
windows, one shared KV bank and persistent input/output bindings. Window transitions
can specialize kernels/capture on first use; they do not load another engine or
resize/copy the KV bank. Decode
profiles also share stable token, position and rotary input addresses. Each layer's
`past_N` and `present_N` point at the same allocation on RTX and CPU. `TensorScatter`
writes only the current block. Padding in the last prefill block is masked from
future tokens and overwritten starting at the true prompt length. Request reset
does not clear or copy the entire bank. Device features and logits remain on the
GPU; greedy generation reads back only the selected scalar token.
Grouped-query attention folds query groups into the query sequence dimension,
avoiding duplicated K/V heads. KV storage retains the export dtype; FP32 attention
temporaries cover only the selected window. FP32 attention accumulation is retained.

TensorRT CPU fallback is disabled, so unsupported partitioning fails explicitly.
CUDA graphs are enabled by default. The runtime uses an explicit CUDA stream and
orders caller inputs/returned tensors with events, avoiding the provider's default
stream wait. Engine contexts and runtime kernels are cached
by graph digest, shape range, GPU, provider-library digest and runtime versions. New processes load the
saved contexts; a graph/profile/runtime change gets a new cache directory.
`--cache-dir` overrides the default directory inside the export. Delete that
directory to force a clean compilation. First use includes compilation/capture;
measure warm requests separately.

## Validation and profiling

```powershell
python -X utf8 -m pytest vision/qwen3_vl/tests -q
python -X utf8 -m vision.qwen3_vl.validate --model-dir artifacts/qwen3-vl-2b-bf16 --images test.jpg --report artifacts/validation.json
python -X utf8 -m vision.qwen3_vl.validate --model-dir artifacts/qwen3-vl-2b-fp32 --provider cpu --reference-device cpu --images test.jpg --atol 0.005 --report artifacts/validation-cpu.json
python -X utf8 -m vision.qwen3_vl.cli --model-dir artifacts/qwen3-vl-2b-bf16 --images test.jpg --repeat 3 --profile --report artifacts/benchmark.json
```

Validation compares against HF vision/DeepStack features, multimodal positions,
prefill logits and KV values, several teacher-forced cached steps, generated tokens
and repeat/reset behavior. It also checks stable cache addresses and verifies that
decode does not change any slot outside the current write. Reports retain maximum
absolute, mean absolute and relative L2 errors; numerical failures remain failures
even when generated tokens match. BF16 and FP16 comparisons are sensitive to kernel
rounding; consult [VALIDATION.md](VALIDATION.md) for measured results and limitations.

ORT `--profile` JSONs can be checked with
`python -m vision.qwen3_vl.profile --require-continuous profile1.json profile2.json`.
For system tracing with Nsight Systems, capture a fully warmed request:

```powershell
$env:NSYS_NVTX_PROFILER_REGISTER_ONLY='0'
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=qwen3_vl/request_2 --capture-range-end=stop -o artifacts/qwen3-vl python -X utf8 -m vision.qwen3_vl.cli --model-dir artifacts/qwen3-vl-2b-bf16 --images test.jpg --repeat 3
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_size_sum artifacts/qwen3-vl.nsys-rep
```

The NVTX environment variable is necessary for PyTorch's dynamically named ranges.
Check graph replay, host/device transfers, CPU gaps and the absence of full-cache
copies; throughput alone does not establish those properties.

See [decoder gap analysis](DECODER_ANALYSIS.md) for decoder-only kernel timings,
cache-capacity experiments, GPU token feedback and hardware-counter availability.
See [optimization results](OPTIMIZATION.md) for the subsequent prefix-window and
sampling changes, measured gains, traces and remaining accuracy limitations.
See [single-weight results](SINGLE_WEIGHTS.md) for the current shared-session
implementation, captured sampling, memory audit and optional persistent workspace.
The optional [provider build](patches/README.md), selected with `--ep-library`,
also retains per-shape CUDA captures while sharing one decoder engine and workspace.

Reference: [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL),
[2B config](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/config.json),
[8B config](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json),
[TensorRT RTX EP](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html).

An optional [SQLite retrieval demo](demo/README.md) is kept separate from inference.
