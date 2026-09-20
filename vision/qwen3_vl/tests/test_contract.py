# SPDX-License-Identifier: Apache-2.0
"""Small, independent reference checks without downloading model weights."""

import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.vision_utils import get_vision_bilinear_indices_and_weights, get_vision_position_ids
from vision.qwen3_vl.inputs import position_ids, resize_shape, text_rotary
from vision.qwen3_vl.modeling import TextDecoder, VisionEncoder


@torch.inference_mode()
def test_onnx_cpu_cache_alias(tmp_path):
    """Exercise ORT TensorScatter with actual shared input/output storage."""
    import onnxruntime as ort
    from vision.qwen3_vl.modeling import FixedCache, export_graph
    from vision.qwen3_vl.runtime import ort_value

    class Scatter(torch.nn.Module):
        def forward(self, past, updates, position):
            cache = FixedCache([past, past], position)
            return cache.update(updates, updates, 0)[0]

    bank = torch.arange(64, dtype=torch.float32).reshape(1, 2, 8, 4)
    updates = torch.full((1, 2, 1, 4), -3.0)
    position = torch.tensor([3])
    path = tmp_path / "scatter.onnx"
    export_graph(Scatter(), (bank, updates, position), path, ["past", "updates", "position"], ["present"], None)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    binding = session.io_binding()
    value = ort_value(bank)
    binding.bind_ortvalue_input("past", value)
    binding.bind_ortvalue_input("updates", ort_value(updates))
    binding.bind_ortvalue_input("position", ort_value(position))
    binding.bind_ortvalue_output("present", value)
    original = bank.clone()
    for index in (3, 4, 7, 0):
        position.fill_(index)
        session.run_with_iobinding(binding)
        original[:, :, index : index + 1] = updates
        torch.testing.assert_close(bank, original, atol=0, rtol=0)


@pytest.fixture
def model():
    torch.manual_seed(42)
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 5000000,
                "mrope_section": [3, 3, 2],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 3,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 16,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "deepstack_visual_indexes": [0, 1],
        },
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
        eos_token_id=127,
    )
    return Qwen3VLForConditionalGeneration(config).eval()


@torch.inference_mode()
def test_8b_tensor_contract_without_allocating_weights():
    # Official 8B dimensions: verify the configuration-driven contract without
    # pretending this is checkpoint accuracy or hardware validation.
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 5000000,
                "mrope_section": [24, 20, 20],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_heads": 16,
            "out_hidden_size": 4096,
            "num_position_embeddings": 2304,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "deepstack_visual_indexes": [8, 16, 24],
        },
        tie_word_embeddings=False,
    )
    with torch.device("meta"):
        large = Qwen3VLForConditionalGeneration(config).eval()
        encoded = VisionEncoder(large)(
            torch.zeros(16, 1536),
            torch.zeros(4, 16, dtype=torch.long),
            torch.zeros(4, 16),
            torch.ones(16, 72),
            torch.zeros(16, 72),
            torch.zeros(1, 1, 1, 16),
        )
        assert encoded.shape == (4, 4, 4096)
        cache = [torch.zeros(1, 8, 16, 128) for _ in range(72)]
        result = TextDecoder(large)(
            torch.zeros(1, 3, dtype=torch.long),
            torch.zeros(4, 1, 3, 4096),
            torch.zeros(1, 3, 1, dtype=torch.bool),
            torch.ones(1, 3, 128),
            torch.zeros(1, 3, 128),
            torch.zeros(1, 1, 3, 16),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            *cache,
        )
        assert result[0].shape == (1, 151936)
        assert len(result[3:]) == 72
        assert all(tensor.shape == (1, 8, 16, 128) for tensor in result[3:])


@pytest.mark.parametrize(
    "width,height,budget", [(1, 10000, 1), (10000, 1, 3), (640, 480, 256), (71, 89, 7), (4096, 2048, 17)]
)
def test_resize_strict_budget(width, height, budget):
    w, h = resize_shape(width, height, budget)
    assert w % 32 == h % 32 == 0
    assert w * h // 1024 <= budget
    assert w >= 32 and h >= 32


def test_video_segment_pts_and_bounds(tmp_path):
    import av
    import numpy as np
    from vision.qwen3_vl.inputs import VideoSegment, read_segment

    path = tmp_path / "timestamps.mkv"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("ffv1", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "bgr0"
        for i in range(10):
            frame = av.VideoFrame.from_ndarray(np.full((32, 64, 3), i * 20, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    frames, times = read_segment(VideoSegment(path, start=0.25, end=0.85, fps=10), 3)
    assert times == pytest.approx([0.3, 0.5, 0.7])
    assert len(frames) == 3
    with pytest.raises(ValueError):
        read_segment(VideoSegment(path, start=0.8, end=0.2), 3)


@torch.inference_mode()
def test_vision_padding_deepstack(model):
    grid = torch.tensor([[1, 4, 6]])
    pixels = torch.randn(24, 1536)
    expected = model.model.visual(pixels, grid_thw=grid)
    indices, weights = get_vision_bilinear_indices_and_weights(grid, 4, 2)
    positions = get_vision_position_ids(grid, 2)
    frequencies = model.model.visual.rotary_pos_emb(positions)
    emb = torch.cat([frequencies, frequencies], dim=-1)
    for pad in (0, 8):
        n = len(pixels) + pad
        pp = torch.zeros(n, 1536)
        pp[:24] = pixels
        ii, ww = torch.zeros(4, n, dtype=torch.long), torch.zeros(4, n)
        ii[:, :24], ww[:, :24] = indices, weights
        cos, sin = torch.ones(n, 16), torch.zeros(n, 16)
        cos[:24], sin[:24] = emb.cos(), emb.sin()
        bias = torch.zeros(1, 1, 1, n)
        bias[..., 24:] = -1e4
        actual = VisionEncoder(model)(pp, ii, ww, cos, sin, bias)[:, :6]
        torch.testing.assert_close(
            actual, torch.stack([expected.pooler_output, *expected.deepstack_features]), atol=1e-5, rtol=1e-4
        )


@torch.inference_mode()
def test_multimodal_positions(model):
    inputs = {
        "input_ids": torch.tensor([[1, 120, 120, 120, 120, 2, 121, 121, 121, 121, 3, 121, 121, 121, 121, 4]]),
        "mm_token_type_ids": torch.tensor([[0, 1, 1, 1, 1, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0]]),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
        "video_grid_thw": torch.tensor([[2, 4, 4]]),
    }
    positions, delta = position_ids(inputs)
    ref_pos, ref_delta = model.model.get_rope_index(**inputs)
    assert torch.equal(positions, ref_pos)
    assert delta == ref_delta.item()
    cos, sin = text_rotary(positions, model.config.text_config.to_dict(), torch.float32)
    expected = model.model.language_model.rotary_emb(torch.zeros(1, 16, 64), positions)
    torch.testing.assert_close(cos, expected[0])
    torch.testing.assert_close(sin, expected[1])


@torch.inference_mode()
def test_cached_decoder_partial_prefill(model):
    # Independent HF path, then a padded prefill whose dead cache slots must be
    # overwritten/masked when decoding resumes at the true prompt length.
    ids = torch.tensor([[1, 120, 120, 120, 120, 2]])
    grid = torch.tensor([[1, 4, 4]])
    pixels = torch.randn(16, 1536)
    types = torch.tensor([[0, 1, 1, 1, 1, 0]])
    inputs = {"input_ids": ids, "pixel_values": pixels, "image_grid_thw": grid, "mm_token_type_ids": types}
    reference = model(**inputs, use_cache=True)
    expected_vision = model.model.visual(pixels, grid_thw=grid)
    positions, delta = position_ids(inputs)
    features = torch.zeros(3, 1, 8, 64)
    features[:, 0, 1:5] = torch.stack([expected_vision.pooler_output, *expected_vision.deepstack_features])
    mask = torch.zeros(1, 8, 1, dtype=torch.bool)
    mask[:, 1:5] = True
    padded_ids = torch.zeros(1, 8, dtype=torch.long)
    padded_ids[:, :6] = ids
    padded_pos = torch.zeros(3, 1, 8, dtype=torch.long)
    padded_pos[..., :6] = positions
    cos, sin = text_rotary(padded_pos, model.config.text_config.to_dict(), torch.float32)
    cache = [torch.zeros(1, 2, 32, 16) for _ in range(6)]
    bias = torch.zeros(1, 1, 8, 32).masked_fill(torch.arange(32)[None, :] > torch.arange(8)[:, None], -1e4)
    decoder = TextDecoder(model)
    logits, _, _, *cache = decoder(
        padded_ids, features, mask, cos, sin, bias, torch.tensor([0]), torch.tensor([5]), *cache
    )
    torch.testing.assert_close(logits, reference.logits[:, -1], atol=1e-5, rtol=1e-4)
    # Restore the independent reference attention backend after wrapper setup.
    model.model.language_model.config._attn_implementation = "eager"
    for step in range(3):
        token = reference.logits[:, -1].argmax(-1, keepdim=True)
        reference = model(input_ids=token, past_key_values=reference.past_key_values, use_cache=True)
        model.model.language_model.config._attn_implementation = "din_qwen3_vl"
        length = 6 + step
        cos, sin = text_rotary(torch.full((3, 1, 1), length + delta), model.config.text_config.to_dict(), torch.float32)
        bias = torch.zeros(1, 1, 1, 32).masked_fill(torch.arange(32) > length, -1e4)
        logits, _, _, *cache = decoder(
            token,
            torch.zeros(3, 1, 1, 64),
            torch.zeros(1, 1, 1, dtype=torch.bool),
            cos,
            sin,
            bias,
            torch.tensor([length]),
            torch.tensor([0]),
            *cache,
        )
        torch.testing.assert_close(logits, reference.logits[:, -1], atol=1e-5, rtol=1e-4)
        for i, layer in enumerate(reference.past_key_values.layers):
            torch.testing.assert_close(cache[2 * i][:, :, : length + 1], layer.keys, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(cache[2 * i + 1][:, :, : length + 1], layer.values, atol=1e-5, rtol=1e-4)
        model.model.language_model.config._attn_implementation = "eager"


@torch.inference_mode()
@pytest.mark.parametrize("prefill", [2, 8])
def test_exported_runtime_chunks_reset_and_capacity(model, tmp_path, monkeypatch, prefill):
    import json

    from vision.qwen3_vl.export import export_model
    from vision.qwen3_vl.runtime import Qwen3VL

    metadata = export_model(
        model, tmp_path, capacity=16, prefill=prefill, vision_buckets=(4, 32), attention_buckets=(4, 8, 16)
    )
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    import onnx

    for name in ("vision", "decoder"):
        exported = onnx.load(tmp_path / f"{name}.onnx", load_external_data=False)
        assert all(node.domain in ("", "ai.onnx") for node in exported.graph.node)
    monkeypatch.setattr("vision.qwen3_vl.runtime.AutoProcessor.from_pretrained", lambda *a, **kw: None)
    runner = Qwen3VL(tmp_path, provider="cpu")
    model.model.language_model.config._attn_implementation = "eager"
    pixels = torch.randn(16, 1536)
    small_grid = torch.tensor([[1, 2, 2]])
    small_reference = model.model.visual(pixels[:4], grid_thw=small_grid)
    small_encoded = runner.encode({"pixel_values": pixels[:4], "image_grid_thw": small_grid})[1]
    torch.testing.assert_close(
        small_encoded,
        torch.stack([small_reference.pooler_output, *small_reference.deepstack_features]),
        atol=1e-5,
        rtol=1e-4,
    )
    for length in (9, 6):
        ids = torch.tensor([[1, 120, 120, 120, 120] + [2] * (length - 5)])
        types = torch.tensor([[0, 1, 1, 1, 1] + [0] * (length - 5)])
        inputs = {
            "input_ids": ids,
            "mm_token_type_ids": types,
            "pixel_values": pixels,
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }
        expected = model(**inputs, use_cache=True)
        actual = runner.prefill(inputs)
        torch.testing.assert_close(actual["logits"], expected.logits[:, -1], atol=1e-5, rtol=1e-4)
        assert runner.length == length
        pointers = [tensor.data_ptr() for tensor in runner.cache]
        for _ in range(16 - length):
            token = expected.logits[:, -1].argmax(-1, keepdim=True)
            before = [tensor.clone() for tensor in runner.cache]
            old_length = runner.length
            expected = model(input_ids=token, past_key_values=expected.past_key_values, use_cache=True)
            actual = runner.decode(int(token.item()))
            torch.testing.assert_close(actual["logits"], expected.logits[:, -1], atol=1e-5, rtol=1e-4)
            for i, layer in enumerate(expected.past_key_values.layers):
                torch.testing.assert_close(runner.cache[2 * i][:, :, : runner.length], layer.keys, atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(
                    runner.cache[2 * i + 1][:, :, : runner.length], layer.values, atol=1e-5, rtol=1e-4
                )
            for old, current in zip(before, runner.cache, strict=True):
                assert torch.equal(old[:, :, :old_length], current[:, :, :old_length])
                assert torch.equal(old[:, :, old_length + 1 :], current[:, :, old_length + 1 :])
            assert pointers == [tensor.data_ptr() for tensor in runner.cache]
        with pytest.raises(ValueError, match="capacity"):
            runner.decode(1)
    assert (1, 8) in runner.decoder_graphs and (1, 16) in runner.decoder_graphs
    for graph in runner.decoder_graphs.values():
        assert graph.inputs["past_0"].data_ptr() == runner.cache[0].data_ptr()
    assert len({id(graph.session) for graph in runner.decoder_graphs.values()}) == 1
    assert len({id(graph.session) for graph in runner.vision_graphs.values()}) == 1
