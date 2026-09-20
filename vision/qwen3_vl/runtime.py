# SPDX-License-Identifier: Apache-2.0
"""Persistent ONNX sessions and I/O bindings for Qwen3-VL.

TensorRT uses shared past/present storage, stable addresses, fixed shape profiles,
one CUDA stream and device-resident visual features/KV/logits. Instances are serial.
"""

import hashlib
import importlib.metadata
import json
import math
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import onnxruntime as ort
import torch
from filelock import FileLock
from transformers import AutoProcessor
from transformers.vision_utils import get_vision_bilinear_indices_and_weights, get_vision_position_ids

from .inputs import position_ids, text_rotary
from .sampling import CudaSampler, sample_top_p

_REGISTERED = None


def on_execution_stream(function):
    @wraps(function)
    def invoke(self, *args, **kwargs):
        with self.execution_context():
            return function(self, *args, **kwargs)

    return invoke


def ort_value(tensor):
    if tensor.dtype == torch.bool:
        capsule = torch.utils.dlpack.to_dlpack(tensor.view(torch.uint8))
        return ort.OrtValue(ort.capi._pybind_state.OrtValue.from_dlpack(capsule, True))
    return ort.OrtValue.from_dlpack(tensor)


class BoundGraph:
    def __init__(
        self,
        path,
        inputs,
        outputs,
        provider,
        cache_dir,
        identity,
        threads=4,
        profile=False,
        cuda_graph=True,
        session=None,
        shape_ranges=None,
        ep_library=None,
        persistent_workspace=False,
    ):
        global _REGISTERED
        self.inputs, self.outputs = inputs, outputs
        self.engine_cache = None
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.enable_profiling = profile
        options.profile_file_prefix = str(cache_dir / identity)
        providers = ["CPUExecutionProvider"]
        if provider == "trt-rtx" and session is None:
            import onnxruntime_ep_nv_tensorrt_rtx as ep

            library = str(Path(ep_library or ep.get_library_path()).resolve())
            if _REGISTERED is None:
                _REGISTERED = (ep.get_ep_name(), library)
                ort.register_execution_provider_library(_REGISTERED[0], library)
            elif _REGISTERED[1] != library:
                raise RuntimeError("A different RTX provider library is already registered in this process")
            devices = [d for d in ort.get_ep_devices() if d.ep_name == _REGISTERED[0]]
            if not devices:
                raise RuntimeError("TensorRT RTX registered but no compatible GPU was found")
            shape_ranges = shape_ranges or {
                bound: {name: tuple(value.shape) for name, value in inputs.items()} for bound in ("min", "opt", "max")
            }
            profiles = {
                f"nv_profile_{bound}_shapes": ",".join(
                    f"{name}:{'x'.join(str(x) for x in shape)}" for name, shape in shapes.items()
                )
                for bound, shapes in shape_ranges.items()
            }
            compatibility = json.dumps(
                {
                    "ort": ort.__version__,
                    "ep": importlib.metadata.version("onnxruntime-ep-nv-tensorrt-rtx"),
                    "gpu": torch.cuda.get_device_name(),
                    "capability": torch.cuda.get_device_capability(),
                    "profiles": profiles,
                    "library_sha256": hashlib.sha256(Path(library).read_bytes()).hexdigest(),
                    "persistent_workspace": bool(ep_library and persistent_workspace),
                    "ep_cuda_graph": bool(cuda_graph),
                }
            )
            engine_cache = cache_dir / (identity + "_" + hashlib.sha256(compatibility.encode()).hexdigest()[:12])
            engine_cache.mkdir(parents=True, exist_ok=True)
            self.engine_cache = engine_cache
            ep_options = {
                "enable_cuda_graph": str(int(cuda_graph)),
                "nv_runtime_cache_path": str(engine_cache),
                "user_compute_stream": str(torch.cuda.current_stream().cuda_stream),
                "has_user_compute_stream": "1",
                **profiles,
            }
            if ep_library:
                ep_options["nv_persistent_context_memory"] = str(int(persistent_workspace))
            options.add_provider_for_devices(devices, ep_options)
            # Fail on graph partitions falling back to CPU, rather than silently
            # measuring a path with transfers and discontinuous GPU execution.
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            providers = None
            context = engine_cache / "model.onnx"
            ready = engine_cache / "ready.json"
            with FileLock(str(engine_cache / "compile.lock"), timeout=600):
                self.compiled = not (context.is_file() and ready.is_file())
                if self.compiled:
                    compiler = ort.ModelCompiler(
                        options,
                        str(path.resolve()),
                        embed_compiled_data_into_model=False,
                        external_initializers_file_path="weights.data",
                    )
                    compiler.compile_to_file(str(context))
                    del compiler
                    ready.write_text(compatibility, encoding="utf-8")
            path = context
        self.session = session if session is not None else ort.InferenceSession(str(path), options, providers=providers)
        self.run_options = ort.RunOptions()
        if provider == "trt-rtx":
            self.run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")
        self.binding = self.session.io_binding()
        self.values = []
        for name, tensor in inputs.items():
            value = ort_value(tensor)
            self.values.append(value)
            self.binding.bind_ortvalue_input(name, value)
        for name, tensor in outputs.items():
            value = ort_value(tensor)
            self.values.append(value)
            self.binding.bind_ortvalue_output(name, value)

    def run(self):
        self.session.run_with_iobinding(self.binding, self.run_options)
        return self.outputs


class Qwen3VL:
    def __init__(
        self,
        directory,
        provider="trt-rtx",
        cache_dir=None,
        threads=4,
        profile=False,
        cuda_graph=True,
        ep_library=None,
    ):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata["format_version"] != 1:
            raise ValueError("Incompatible export; re-export with the current exporter")
        if provider not in ("cpu", "trt-rtx"):
            raise ValueError("provider must be cpu or trt-rtx")
        self.provider, self.device = provider, "cuda" if provider == "trt-rtx" else "cpu"
        # A nonzero stream handle is required for the RTX EP to recognize an
        # external stream and avoid its unconditional post-enqueue CPU wait.
        self.stream = None
        if self.device == "cuda":
            current = torch.cuda.current_stream()
            self.stream = current if current.cuda_stream else torch.cuda.Stream()
        self.dtype = getattr(torch, self.metadata["dtype"])
        if provider == "cpu" and self.dtype == torch.bfloat16:
            raise ValueError("ORT CPU lacks BF16 kernels; use --dtype fp32 (or fp16) export for CPU inference")
        self.config = self.metadata["text_config"]
        self.vision_config = self.metadata["vision_config"]
        self.capacity = self.metadata["cache_capacity"]
        self.attention_buckets = self.metadata.get("attention_buckets", [self.capacity])
        self.block = self.metadata["prefill_block"]
        self.profile = profile
        self.cuda_graph = cuda_graph
        self.ep_library = ep_library
        self.threads = threads
        self.cache_dir = Path(cache_dir or self.directory / "runtime_cache").resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processor = AutoProcessor.from_pretrained(self.directory / "processor", local_files_only=True)
        c = self.config
        self.cache = [
            self.zeros(1, c["num_key_value_heads"], self.capacity, c["head_dim"])
            for _ in range(c["num_hidden_layers"] * 2)
        ]
        self.vision_graphs = {}
        self.decoder_graphs = {}
        self._sessions = {}
        self._sampler = None
        self.length = 0
        self.rope_delta = 0
        self._cache_positions = torch.arange(self.capacity, device=self.device)
        # Rotary lookup and a sliding slice of a causal mask avoid rebuilding
        # dozens of tiny GPU tensors/kernels at every generated token.
        text_positions = torch.arange(self.capacity).reshape(1, 1, -1).expand(3, 1, -1)
        cos, sin = text_rotary(text_positions, self.config, self.dtype)
        self._rotary_table = torch.stack([cos[0], sin[0]], dim=1).to(self.device)
        self._causal_line = self.zeros(2 * self.capacity)
        self._causal_line[self.capacity :] = -1e4
        self._decode_dirty = True
        # All decode profiles share these stable input addresses. Recreating them
        # for a new prefix bucket would leave older bindings pointing at stale data.
        self._decode_controls = self.zeros(3, dtype=torch.int64)
        self._decode_controls_host = torch.zeros(self.capacity, 3, dtype=torch.int64, pin_memory=self.device == "cuda")
        self._decode_controls_host[:, 1] = torch.arange(self.capacity)
        self._decode_controls_numpy = self._decode_controls_host.numpy()
        self._decode_rotary = self.zeros(2, 1, 1, c["head_dim"])

    def zeros(self, *shape, dtype=None):
        return torch.zeros(shape, dtype=dtype or self.dtype, device=self.device)

    @contextmanager
    def execution_context(self):
        """Order caller inputs and returned outputs without a CPU synchronization."""
        if self.stream is None or torch.cuda.current_stream() == self.stream:
            yield
            return
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)
        try:
            with torch.cuda.stream(self.stream):
                yield
        finally:
            caller.wait_stream(self.stream)

    def _graph(self, name, variant, inputs, outputs):
        identity = f"{name}_{self.metadata['graphs'][name][:20]}_{self.provider}_shared_v1"
        ranges = {}
        for bound in ("min", "opt", "max"):
            shapes = {key: list(value.shape) for key, value in inputs.items()}
            if name == "decoder":
                sequence = self.block if bound == "max" else 1
                window = self.capacity if bound == "max" else min(self.attention_buckets)
                for key, axis in (
                    ("input_ids", 1),
                    ("visual_features", 2),
                    ("visual_mask", 1),
                    ("rotary_cos", 1),
                    ("rotary_sin", 1),
                    ("attention_bias", 2),
                ):
                    shapes[key][axis] = sequence
                shapes["attention_bias"][3] = window
            else:
                patches = (
                    max(self.metadata["vision_buckets"]) if bound == "max" else min(self.metadata["vision_buckets"])
                )
                for key, axis in (
                    ("pixels", 0),
                    ("bilinear_indices", 1),
                    ("bilinear_weights", 1),
                    ("rotary_cos", 0),
                    ("rotary_sin", 0),
                    ("attention_bias", 3),
                ):
                    shapes[key][axis] = patches
            ranges[bound] = shapes
        graph = BoundGraph(
            self.directory / f"{name}.onnx",
            inputs,
            outputs,
            self.provider,
            self.cache_dir,
            identity,
            self.threads,
            self.profile,
            self.cuda_graph,
            session=self._sessions.get(name),
            shape_ranges=ranges,
            ep_library=self.ep_library,
            persistent_workspace=name == "decoder",
        )
        self._sessions[name] = graph.session
        return graph

    def _decoder(self, size):
        window = next(b for b in self.attention_buckets if b >= self.length + 1) if size == 1 else self.capacity
        key = (size, window)
        if key not in self.decoder_graphs:
            c = self.config
            inputs = {
                "input_ids": self.zeros(1, size, dtype=torch.int64),
                "visual_features": self.zeros(
                    1 + len(self.vision_config["deepstack_visual_indexes"]), 1, size, c["hidden_size"]
                ),
                "visual_mask": self.zeros(1, size, 1, dtype=torch.bool),
                "rotary_cos": self.zeros(1, size, c["head_dim"]),
                "rotary_sin": self.zeros(1, size, c["head_dim"]),
                "attention_bias": self.zeros(1, 1, size, window),
                "cache_position": self.zeros(1, dtype=torch.int64),
                "logits_index": self.zeros(1, dtype=torch.int64),
                **{f"past_{i}": tensor for i, tensor in enumerate(self.cache)},
            }
            if size == 1:
                inputs.update(
                    input_ids=self._decode_controls[:1].reshape(1, 1),
                    cache_position=self._decode_controls[1:2],
                    logits_index=self._decode_controls[2:3],
                    rotary_cos=self._decode_rotary[0],
                    rotary_sin=self._decode_rotary[1],
                )
            outputs = {
                "logits": self.zeros(1, c["vocab_size"], dtype=torch.float32),
                "next_token": self.zeros(1, dtype=torch.int64),
                "hidden": self.zeros(1, c["hidden_size"]),
            }
            # TensorScatter supports aliasing on both RTX and ORT CPU. The CPU
            # implementation is covered by the shared-buffer integration test.
            outputs.update({f"present_{i}": tensor for i, tensor in enumerate(self.cache)})
            variant = f"s{size}" if "attention_buckets" not in self.metadata else f"s{size}_a{window}"
            self.decoder_graphs[key] = self._graph("decoder", variant, inputs, outputs)
        return self.decoder_graphs[key]

    def _vision(self, patches):
        bucket = next((b for b in self.metadata["vision_buckets"] if b >= patches), None)
        if bucket is None:
            raise ValueError(f"Image/frame needs {patches} patches, exceeding largest exported vision bucket")
        if bucket not in self.vision_graphs:
            v = self.vision_config
            head = v["hidden_size"] // v["num_heads"]
            inputs = {
                "pixels": self.zeros(bucket, 3 * v["temporal_patch_size"] * v["patch_size"] ** 2),
                "bilinear_indices": self.zeros(4, bucket, dtype=torch.int64),
                "bilinear_weights": self.zeros(4, bucket, dtype=torch.float32),
                "rotary_cos": self.zeros(bucket, head, dtype=torch.float32),
                "rotary_sin": self.zeros(bucket, head, dtype=torch.float32),
                "attention_bias": self.zeros(1, 1, 1, bucket),
            }
            outputs = {
                "visual_features": self.zeros(1 + len(v["deepstack_visual_indexes"]), bucket // 4, v["out_hidden_size"])
            }
            self.vision_graphs[bucket] = self._graph("vision", f"p{bucket}", inputs, outputs)
        return self.vision_graphs[bucket]

    @torch.inference_mode()
    @on_execution_stream
    def encode(self, inputs):
        encoded = {}
        for pixel_key, grid_key, kind in (
            ("pixel_values", "image_grid_thw", 1),
            ("pixel_values_videos", "video_grid_thw", 2),
        ):
            if pixel_key not in inputs:
                continue
            offset, features = 0, []
            for t, h, w in inputs[grid_key].tolist():
                n = h * w
                grid = torch.tensor([[1, h, w]])
                indices, weights = get_vision_bilinear_indices_and_weights(
                    grid, int(math.isqrt(self.vision_config["num_position_embeddings"])), 2
                )
                positions = get_vision_position_ids(grid, 2)
                head = self.vision_config["hidden_size"] // self.vision_config["num_heads"]
                inv = 1.0 / (10000.0 ** (torch.arange(0, head // 2, 2).float() / (head // 2)))
                frequencies = (positions[..., None] * inv).flatten(1)
                frequencies = torch.cat([frequencies, frequencies], dim=-1)
                graph = self._vision(n)
                feed = graph.inputs
                feed["bilinear_indices"].zero_()
                feed["bilinear_indices"][:, :n].copy_(indices)
                feed["bilinear_weights"].zero_()
                feed["bilinear_weights"][:, :n].copy_(weights)
                feed["rotary_cos"].fill_(1)
                feed["rotary_sin"].zero_()
                feed["rotary_cos"][:n].copy_(frequencies.cos())
                feed["rotary_sin"][:n].copy_(frequencies.sin())
                feed["attention_bias"].zero_()
                feed["attention_bias"][..., n:] = -1e4
                for _ in range(t):
                    feed["pixels"].zero_()
                    feed["pixels"][:n].copy_(inputs[pixel_key][offset : offset + n])
                    offset += n
                    graph.run()
                    features.append(graph.outputs["visual_features"][:, : n // 4].clone())
            if offset != len(inputs[pixel_key]):
                raise ValueError("Grid and pixel patch lengths differ")
            encoded[kind] = torch.cat(features, dim=1)
        return encoded

    def reset(self):
        if self.length:
            self.synchronize()  # Retire any pending DMA before reusing host control rows.
        self.length = 0
        self.rope_delta = 0
        self._decode_dirty = True
        # No zeroing/copying of the entire cache on GPU: all unread slots are
        # masked, and subsequent valid slots are overwritten by TensorScatter.

    @torch.inference_mode()
    def _step(self, ids, positions, visual=None, mask=None):
        # Keep prefill on its own profile even when the final chunk has one token.
        # Decode profiles then always retain zero visual inputs across requests.
        size = self.block
        valid = ids.numel()
        if valid > size or self.length + size > self.capacity:
            raise ValueError("Decoder block exceeds the fixed KV capacity")
        graph = self._decoder(size)
        feed = graph.inputs
        feed["input_ids"].zero_()
        feed["input_ids"][:, :valid].copy_(ids.reshape(1, -1))
        feed["visual_features"].zero_()
        feed["visual_mask"].zero_()
        if visual is not None:
            feed["visual_features"][:, :, :valid].copy_(visual)
            feed["visual_mask"][:, :valid].copy_(mask)
        cos, sin = text_rotary(positions, self.config, self.dtype)
        feed["rotary_cos"].fill_(1)
        feed["rotary_sin"].zero_()
        feed["rotary_cos"][:, :valid].copy_(cos)
        feed["rotary_sin"][:, :valid].copy_(sin)
        query_positions = torch.arange(size, device=self.device) + self.length
        feed["attention_bias"].zero_().masked_fill_(self._cache_positions[None, :] > query_positions[:, None], -1e4)
        feed["cache_position"].fill_(self.length)
        feed["logits_index"].fill_(valid - 1)
        graph.run()
        self.length += valid
        return graph.outputs

    @torch.inference_mode()
    @on_execution_stream
    def prefill(self, inputs, encoded=None):
        self.reset()
        ids = inputs["input_ids"]
        length = ids.shape[1]
        if math.ceil(length / self.block) * self.block > self.capacity:
            raise ValueError("Prompt including prefill padding exceeds the exported KV capacity")
        positions, self.rope_delta = position_ids(inputs)
        encoded = self.encode(inputs) if encoded is None else encoded
        features = self.zeros(
            1 + len(self.vision_config["deepstack_visual_indexes"]), 1, length, self.config["hidden_size"]
        )
        mask = torch.zeros(1, length, 1, dtype=torch.bool)
        for kind, token_id in ((1, self.metadata["image_token_id"]), (2, self.metadata["video_token_id"])):
            selection = ids[0] == token_id
            count = int(selection.sum())
            if count:
                if kind not in encoded or encoded[kind].shape[1] != count:
                    raise ValueError("Visual features and placeholders differ")
                features[:, 0, selection.to(self.device)] = encoded[kind]
                mask[0, selection, 0] = True
        for offset in range(0, length, self.block):
            result = self._step(
                ids[:, offset : offset + self.block],
                positions[..., offset : offset + self.block],
                features[:, :, offset : offset + self.block],
                mask[:, offset : offset + self.block],
            )
        return result

    @torch.inference_mode()
    @on_execution_stream
    def decode(self, token):
        if self.length < 1 or self.length >= self.capacity:
            raise ValueError("Decode requires a prefilled prompt and available cache capacity")
        graph = self._decoder(1)
        feed = graph.inputs
        if self._decode_dirty:
            feed["visual_features"].zero_()
            feed["visual_mask"].zero_()
            feed["logits_index"].zero_()
            self._decode_dirty = False
        controls = self._decode_controls_host[self.length]
        self._decode_controls_numpy[self.length, 0] = int(token)
        self._decode_controls.copy_(controls, non_blocking=self.device == "cuda")
        rotary = self._rotary_table[self.length + self.rope_delta]
        self._decode_rotary.copy_(rotary.reshape(2, 1, 1, -1))
        start = self.capacity - self.length - 1
        window = feed["attention_bias"].shape[-1]
        feed["attention_bias"].copy_(self._causal_line[start : start + window].reshape(1, 1, 1, -1))
        graph.run()
        self.length += 1
        return graph.outputs

    def synchronize(self):
        if self.device == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    @on_execution_stream
    def generate(self, inputs, max_new_tokens=128, temperature=0.0, top_p=1.0, seed=0):
        if max_new_tokens < 1 or not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("Invalid generation limit, temperature or top_p")
        if inputs["input_ids"].shape[1] + max_new_tokens > self.capacity:
            raise ValueError("Prompt plus generation budget exceeds cache-capacity")
        self.synchronize()
        start = time.perf_counter()
        encoded = self.encode(inputs)
        self.synchronize()
        vision_end = time.perf_counter()
        result = self.prefill(inputs, encoded)
        self.synchronize()
        prefill_end = time.perf_counter()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        if temperature and self.device == "cuda" and self.cuda_graph:
            if self._sampler is None:
                self._sampler = CudaSampler(self.config["vocab_size"])
            self._sampler.configure(temperature, top_p)
        tokens, latencies = [], []
        tick = None
        for _ in range(max_new_tokens):
            if temperature == 0:
                token = int(result["next_token"].item())
            else:
                selected = (
                    self._sampler.select(result["logits"][0], generator)
                    if self.device == "cuda" and self.cuda_graph
                    else sample_top_p(result["logits"][0], temperature, top_p, generator)
                )
                token = int(selected.item())
            if tick is not None:
                latencies.append(time.perf_counter() - tick)
            tokens.append(token)
            if token in self.metadata["eos_token_ids"] or len(tokens) == max_new_tokens:
                break
            tick = time.perf_counter()
            result = self.decode(token)
        return {
            "text": self.processor.tokenizer.decode(tokens, skip_special_tokens=True),
            "tokens": tokens,
            "reached_eos": tokens[-1] in self.metadata["eos_token_ids"],
            "vision_seconds": vision_end - start,
            "prefill_seconds": prefill_end - vision_end,
            "decode_seconds": latencies,
            "decode_tokens_per_second": len(latencies) / sum(latencies) if latencies else None,
            "session_count": len(self._sessions),
            "compiled_sessions": sum(
                getattr(g, "compiled", False) for g in [*self.vision_graphs.values(), *self.decoder_graphs.values()]
            ),
            "kv_bytes": sum(t.numel() * t.element_size() for t in self.cache),
        }

    def finish_profiles(self):
        return [session.end_profiling() for session in self._sessions.values() if self.profile]
