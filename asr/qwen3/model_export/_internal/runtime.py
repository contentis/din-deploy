# SPDX-License-Identifier: Apache-2.0
"""Internal ONNX runner used by the Qwen3 validation sample."""

import json
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoProcessor

_REGISTERED_EP = None


def session_options(provider, threads, extra_options=None):
    global _REGISTERED_EP
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    if provider == "cpu":
        return options, ["CPUExecutionProvider"]
    if provider != "trt-rtx":
        raise ValueError(f"Unknown provider: {provider}")
    import onnxruntime_ep_nv_tensorrt_rtx as ep

    if _REGISTERED_EP is None:
        _REGISTERED_EP = ep.get_ep_name()
        ort.register_execution_provider_library(_REGISTERED_EP, ep.get_library_path())
    devices = [device for device in ort.get_ep_devices() if device.ep_name == _REGISTERED_EP]
    if not devices:
        raise RuntimeError("TensorRT RTX registered but no GPU was discovered")
    options.add_provider_for_devices(devices, {"enable_cuda_graph": "0", **(extra_options or {})})
    return options, None


def ort_numpy(value):
    """Promote only diagnostic/host output, since NumPy has no native BF16 type."""
    return torch.from_dlpack(value).float().cpu().numpy()


def load_audio(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != 16000:
        factor = gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // factor, rate // factor).astype(np.float32)
    if not audio.size:
        raise ValueError("Audio is empty")
    return audio


def pack_audio(features, mask, config):
    """Reproduce the native encoder's packing without tracing nonzero/list splits."""
    chunk_size = config["n_window"] * 2
    if features.shape[0] != 1 or features.shape[-1] % chunk_size:
        raise ValueError("Expected batch-one features padded to the encoder chunk size")
    chunks = features.reshape(1, features.shape[1], -1, chunk_size)[0].transpose(1, 0, 2).copy()
    lengths = mask.reshape(-1, chunk_size).sum(axis=1).astype(np.int64)
    for _ in range(3):
        lengths = (lengths + 1) // 2
    max_length = (chunk_size + 7) // 8
    indices = np.flatnonzero((np.arange(max_length)[None] < lengths[:, None]).reshape(-1)).astype(np.int64)
    if not len(indices):
        raise ValueError("Audio has no valid feature frames")
    window = int(lengths.max()) * (config["n_window_infer"] // chunk_size)
    groups = np.arange(len(indices)) // window
    bias = np.where(groups[:, None] == groups[None], 0.0, np.finfo(np.float32).min)
    return chunks, indices, bias[None, None].astype(np.float32)


def decoder_inputs(ids, embeddings, audio_token_id, hidden_size, past_length, capacity=None):
    ids = np.asarray(ids, dtype=np.int64).reshape(1, -1)
    seq = ids.shape[1]
    if not seq or past_length < 0 or (capacity is not None and past_length + seq > capacity):
        raise ValueError("Input exceeds KV cache capacity or has invalid length")
    mask = (ids == audio_token_id)[..., None] if embeddings is not None else np.zeros((1, seq, 1), dtype=bool)
    padded = np.zeros((1, seq, hidden_size), dtype=np.float32)
    if embeddings is not None:
        if int(mask.sum()) != len(embeddings):
            raise ValueError("Audio placeholder count differs from encoder output length")
        padded[mask[..., 0]] = embeddings
    positions = np.arange(past_length, past_length + seq, dtype=np.int64)[None]
    allowed = np.arange(capacity or past_length + seq)[None, :] <= positions.T
    # Same finite mask as Whisper, avoiding NaNs in fully masked flash tiles.
    bias = np.where(allowed, 0.0, -1e4).astype(np.float32)[None, None]
    return {
        "input_ids": ids,
        "audio_embeddings": padded,
        "audio_mask": mask,
        "position_ids": positions,
        "attention_bias": bias,
    }


class OnnxAudioModel:
    def __init__(self, directory, threads=4, provider=None):
        directory = Path(directory)
        self.directory = directory
        self.threads = threads
        self.decode_session = None
        self.decode_capacity = 0
        self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        self.processor = AutoProcessor.from_pretrained(directory / "processor", local_files_only=True)
        self.dtype = getattr(torch, self.metadata["dtype"])
        self.provider = provider or ("trt-rtx" if self.dtype == torch.bfloat16 else "cpu")
        options, providers = session_options(self.provider, threads)
        self.encoder = ort.InferenceSession(str(directory / "encoder.onnx"), options, providers=providers)
        if self.text_graph == "decoder.onnx":
            # Exclude zero-length queries and include the exported 32768-slot
            # capacity, which TRT's implicit dynamic range does not cover.
            c = self.metadata["text_config"]
            capacity = self.metadata["cache_capacity"]

            def shapes(sequence, keys):
                shape = (
                    f"input_ids:1x{sequence},audio_embeddings:1x{sequence}x{c['hidden_size']},"
                    f"audio_mask:1x{sequence}x1,position_ids:1x{sequence},"
                    f"attention_bias:1x1x{sequence}x{keys}"
                )
                if self.metadata.get("dynamic_cache_capacity", False):
                    for i in range(2 * c["num_hidden_layers"]):
                        shape += f",past_{i}:1x{c['num_key_value_heads']}x{keys}x{c['head_dim']}"
                return shape

            options, providers = session_options(
                self.provider,
                threads,
                {
                    "nv_profile_min_shapes": shapes(1, 4 if self.metadata.get("dynamic_cache_capacity") else capacity),
                    "nv_profile_opt_shapes": shapes(
                        min(128, capacity),
                        min(2048, capacity) if self.metadata.get("dynamic_cache_capacity") else capacity,
                    ),
                    "nv_profile_max_shapes": shapes(capacity, capacity),
                },
            )
        self.decoder = ort.InferenceSession(str(directory / self.text_graph), options, providers=providers)

    def run(self, session, feed):
        values = {}
        for name, value in feed.items():
            if isinstance(value, ort.OrtValue):
                values[name] = value
                continue
            if not np.issubdtype(value.dtype, np.floating):
                values[name] = ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(value))
                continue
            tensor = torch.from_numpy(np.ascontiguousarray(value))
            if tensor.is_floating_point():
                if name == "attention_bias":
                    tensor = tensor.clamp(min=torch.finfo(self.dtype).min)
                tensor = tensor.to(self.dtype)
            values[name] = ort.OrtValue.from_dlpack(tensor)
        # OrtValues preserve BF16 outputs/caches without a NumPy FP32 round trip.
        return session.run_with_ort_values(None, values)

    def encode(self, inputs):
        packed = pack_audio(
            inputs["input_features"].numpy(), inputs["input_features_mask"].numpy(), self.metadata["audio_config"]
        )
        return ort_numpy(
            self.run(self.encoder, dict(zip(("mel_chunks", "valid_indices", "attention_bias"), packed, strict=True)))[0]
        )


@dataclass
class DecodeCache:
    values: list
    spare: list
    capacity: int
    length: int = 0


class OnnxASR(OnnxAudioModel):
    text_graph = "decoder.onnx"

    def empty_cache(self, required=None):
        c = self.metadata["text_config"]
        device = "cuda" if self.provider == "trt-rtx" else "cpu"
        if self.metadata.get("format_version") != 2:
            raise ValueError("Re-export the model for the fixed-capacity cache contract (format 2)")
        capacity = self.metadata["cache_capacity"]
        if required is not None and self.metadata.get("dynamic_cache_capacity"):
            capacity = min(capacity, max(4, 1 << (required - 1).bit_length()))
        values = [
            ort.OrtValue.from_dlpack(
                torch.zeros(
                    (1, c["num_key_value_heads"], capacity, c["head_dim"]),
                    dtype=self.dtype,
                    device=device,
                )
            )
            for _ in range(4 * c["num_hidden_layers"])
        ]
        if device == "cuda":
            torch.cuda.synchronize()
        count = 2 * c["num_hidden_layers"]
        return DecodeCache(values[:count], values[count:], capacity)

    def step(self, ids, embeddings, cache):
        feed = decoder_inputs(
            ids,
            embeddings,
            self.metadata["audio_token_id"],
            self.metadata["text_config"]["hidden_size"],
            cache.length,
            cache.capacity,
        )
        device = "cuda" if self.provider == "trt-rtx" else "cpu"
        inplace = (
            device == "cuda"
            and cache.length > 0
            and feed["input_ids"].shape[1] == 1
            and cache.capacity in self.metadata.get("decode_capacities", [])
        )
        session = self.decoder
        if inplace:
            if self.decode_capacity != cache.capacity:
                options, providers = session_options(self.provider, self.threads)
                path = self.directory / f"decode_{cache.capacity}.onnx"
                self.decode_session = ort.InferenceSession(str(path), options, providers=providers)
                self.decode_capacity = cache.capacity
            session = self.decode_session
        binding = session.io_binding()
        # Keep tensor owners alive through execution; bind pointers also supports bool.
        tensors = []
        for name, value in feed.items():
            tensor = torch.from_numpy(np.ascontiguousarray(value)).to(device)
            if tensor.is_floating_point():
                tensor = tensor.to(self.dtype)
            tensors.append(tensor)
            element_type = 16 if tensor.dtype == torch.bfloat16 else value.dtype
            binding.bind_input(name, device, 0, element_type, tuple(tensor.shape), tensor.data_ptr())
        for i, value in enumerate(cache.values):
            binding.bind_ortvalue_input(f"past_{i}", value)
            binding.bind_ortvalue_output(f"present_{i}", value if inplace else cache.spare[i])
        binding.bind_output("logits", device)
        binding.bind_output("next_token", device)
        if device == "cuda":
            torch.cuda.synchronize()
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        # Outputs follow binding order, not graph order.
        result = binding.get_outputs()
        if not inplace:
            cache.values, cache.spare = cache.spare, cache.values
        logits = result[-2]
        cache.length += np.asarray(ids).size
        return ort_numpy(logits), cache


class OnnxAligner(OnnxAudioModel):
    text_graph = "aligner.onnx"

    def align(self, inputs, word_lists):
        audio = self.encode(inputs)
        feed = decoder_inputs(
            inputs["input_ids"].numpy(),
            audio,
            self.metadata["audio_token_id"],
            self.metadata["text_config"]["hidden_size"],
            0,
        )
        timestamp_id = self.metadata["timestamp_token_id"]
        indices = np.flatnonzero(feed["input_ids"][0] == timestamp_id).astype(np.int64)
        if not len(indices):
            raise ValueError("Transcript has no alignable words")
        feed["timestamp_indices"] = indices
        logits = ort_numpy(self.run(self.decoder, feed)[0])
        items = self.processor.decode_forced_alignment(
            torch.from_numpy(logits),
            torch.full((1, len(indices)), timestamp_id),
            word_lists,
            timestamp_id,
            self.metadata["timestamp_segment_time"],
        )[0]
        return logits, items
