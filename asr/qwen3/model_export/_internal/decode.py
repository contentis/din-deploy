# SPDX-License-Identifier: Apache-2.0
"""Standard ONNX decode specialization; TensorRT requires aliased past/present I/O."""

import onnx
import torch
from torch.nn import functional as F
from transformers import AttentionInterface


def decode_attention(module, query, key, value, attention_mask, scaling=None, **kwargs):
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        scale=scaling,
        enable_gqa=query.shape[1] != key.shape[1],
    )
    return output.transpose(1, 2), None


class DecodeCache:
    def __init__(self, past, positions):
        self.past, self.positions, self.present = past, positions.reshape(-1), []

    def update(self, key, value, layer_idx):
        updated = []
        for cache, data in zip(self.past[2 * layer_idx : 2 * layer_idx + 2], (key, value), strict=True):
            if torch.onnx.is_in_onnx_export():
                data = torch.onnx.ops.symbolic(
                    "TensorScatter",
                    (cache, data, self.positions[:1]),
                    {"axis": 2, "mode": "linear"},
                    dtype=data.dtype,
                    shape=cache.shape,
                    version=24,
                )
            else:
                data = cache.index_copy(2, self.positions, data)
            updated.append(data)
        self.present.extend(updated)
        return tuple(updated)


def export_decode(decoder, config, dtype, output, capacities):
    AttentionInterface.register("qwen3_onnx_decode", decode_attention)
    decoder.decoder.config._attn_implementation = "qwen3_onnx_decode"
    decoder.cache_type = DecodeCache
    past = tuple(
        torch.zeros(1, config.num_key_value_heads, 4, config.head_dim, dtype=dtype)
        for _ in range(2 * config.num_hidden_layers)
    )
    inputs = (
        torch.ones(1, 1, dtype=torch.int64),
        torch.zeros(1, 1, config.hidden_size, dtype=dtype),
        torch.zeros(1, 1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.int64),
        torch.zeros(1, 1, 1, 4, dtype=dtype),
        *past,
    )
    cap = torch.export.Dim("capacity", min=4)
    names = ["input_ids", "audio_embeddings", "audio_mask", "position_ids", "attention_bias"]
    # Export weights once; specialized graph headers share the same external data.
    template = output / "decode.onnx"
    torch.onnx.export(
        decoder.eval(),
        inputs,
        str(template),
        dynamo=True,
        opset_version=24,
        external_data=True,
        input_names=names + [f"past_{i}" for i in range(len(past))],
        output_names=["logits", "next_token"] + [f"present_{i}" for i in range(len(past))],
        dynamic_shapes={
            "input_ids": {},
            "audio_embeddings": {},
            "audio_mask": {},
            "position_ids": {},
            "attention_bias": {3: cap},
            "past": tuple({2: cap} for _ in past),
        },
    )
    for capacity in sorted(set(capacities)):
        graph = onnx.load(template, load_external_data=False)
        for value in list(graph.graph.input) + list(graph.graph.output) + list(graph.graph.value_info):
            for dim in value.type.tensor_type.shape.dim:
                if dim.dim_param == "capacity":
                    dim.dim_value = capacity
        path = output / f"decode_{capacity}.onnx"
        onnx.save(graph, path)
        onnx.checker.check_model(str(path))
        print(f"Checked {path}", flush=True)
