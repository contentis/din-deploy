# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from vision.qwen3_vl.demo.retrieval import VectorStore


def test_store_roundtrip_and_identity(tmp_path):
    path = tmp_path / "images.sqlite"
    store = VectorStore(path, "model-a")
    store.upsert(tmp_path / "a.png", [1, 0, 0])
    store.upsert(tmp_path / "b.png", [0, 1, 0])
    store.upsert(tmp_path / "a.png", [0.8, 0.2, 0])
    result = store.search([1, 0, 0], k=8)
    assert len(result) == 2 and result[0]["path"].endswith("a.png")
    assert np.isclose(result[0]["cosine"], 0.8 / np.hypot(0.8, 0.2))
    with pytest.raises(ValueError):
        store.upsert("bad.png", [float("nan"), 0, 0])
    store.close()
    with pytest.raises(ValueError, match="differs"):
        VectorStore(path, "model-b")
    store = VectorStore(path, "model-a")
    assert len(store.search([0, 1, 0])) == 2
    store.close()
