# SPDX-License-Identifier: Apache-2.0
"""Generate the standard-ONNX fixture consumed by din_asr_qwen3_cache_test."""

import sys
from pathlib import Path

import onnx
from onnx import TensorProto as T
from onnx import helper as h

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
model = h.make_model(
    h.make_graph(
        [h.make_node("TensorScatter", ["cache", "update", "position"], ["out"], axis=2, mode="linear")],
        "cache_update",
        [
            h.make_tensor_value_info("cache", T.BFLOAT16, [1, 8, 8192, 128]),
            h.make_tensor_value_info("update", T.BFLOAT16, [1, 8, 1, 128]),
            h.make_tensor_value_info("position", T.INT64, [1]),
        ],
        [h.make_tensor_value_info("out", T.BFLOAT16, [1, 8, 8192, 128])],
    ),
    opset_imports=[h.make_opsetid("", 24)],
    ir_version=10,
)
onnx.checker.check_model(model)
onnx.save(model, path)
