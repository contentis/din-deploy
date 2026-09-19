# SPDX-License-Identifier: Apache-2.0
"""Small random-model checks against independent Transformers forward methods."""

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, Qwen3ASRForTokenClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_export"))
from _internal.frontend import LogMel
from _internal.runtime import decoder_inputs, pack_audio
from export_qwen3_asr import AudioEncoder, ForcedAligner, TextDecoder


@pytest.fixture(params=[torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def model(request):
    torch.manual_seed(7)
    torch.set_num_threads(2)
    config = Qwen3ASRConfig(
        audio_config={
            "d_model": 32,
            "encoder_attention_heads": 4,
            "encoder_layers": 2,
            "encoder_ffn_dim": 64,
            "downsample_hidden_size": 8,
            "output_dim": 32,
        },
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 128,
        },
        audio_token_id=100,
    )
    config._attn_implementation = "eager"
    return Qwen3ASRForConditionalGeneration(config).to(request.param).eval()


def typed_inputs(values, dtype):
    tensors = [torch.from_numpy(x) for x in values]
    return [x.clamp(min=torch.finfo(dtype).min).to(dtype) if x.is_floating_point() else x for x in tensors]


@pytest.mark.parametrize("frames", [50, 100, 101, 800, 845])
def test_audio_packing_and_window_attention(model, frames):
    padded = ((frames + 99) // 100) * 100
    features = torch.randn(1, 128, padded)
    features[..., frames:] = 0
    mask = (torch.arange(padded)[None] < frames).long()
    packed = pack_audio(features.numpy(), mask.numpy(), model.config.audio_config.to_dict())
    with torch.inference_mode():
        expected = model.get_audio_features(features.to(model.dtype), mask).pooler_output
        actual = AudioEncoder(model)(*typed_inputs(packed, model.dtype))
    torch.testing.assert_close(actual, expected)
    # Native long-form execution groups by exactly HF's independent attention windows.
    window = model.config.audio_config.n_window_infer
    parts = []
    with torch.inference_mode():
        for start in range(0, frames, window):
            end = min(start + window, padded)
            packed = pack_audio(
                features[..., start:end].numpy(), mask[..., start:end].numpy(), model.config.audio_config.to_dict()
            )
            parts.append(AudioEncoder(model)(*typed_inputs(packed, model.dtype)))
    torch.testing.assert_close(torch.cat(parts), expected)


@pytest.mark.parametrize("chunk_lengths", [(4, 1, 1, 1), (2, 2, 3), (7,)])
def test_prefill_and_cached_decode(model, chunk_lengths, capacity=7):
    wrapper = TextDecoder(deepcopy(model)).eval()
    all_ids = np.array([[3, 100, 100, 5, 10, 11, 12]])
    all_audio = np.random.default_rng(7).normal(size=(2, 32)).astype(np.float32)
    # Nonzero unused slots expose missing padding masks; fill the final slot too.
    cache = tuple(torch.randn(1, 2, capacity, 8, dtype=model.dtype) for _ in range(4))
    length = 0
    ref_cache = None
    # BF16 fused SDPA rounds differently from HF eager attention near zero.
    tolerance = {"atol": 0.004, "rtol": 0.016} if model.dtype == torch.bfloat16 else {}
    with torch.inference_mode():
        for chunk in chunk_lengths:
            ids = all_ids[:, length : length + chunk]
            audio_start = int((all_ids[:, :length] == 100).sum())
            audio_count = int((ids == 100).sum())
            audio = all_audio[audio_start : audio_start + audio_count] if audio_count else None
            feed = decoder_inputs(ids, audio, 100, 32, length, capacity)
            tensors = typed_inputs(feed.values(), model.dtype)
            embeds = torch.where(tensors[2], tensors[1], model.get_input_embeddings()(tensors[0]))
            reference = model.model.language_model(inputs_embeds=embeds, past_key_values=ref_cache, use_cache=True)
            expected = model.lm_head(reference.last_hidden_state[:, -1])
            old_cache = cache
            actual, token, *cache = wrapper(*tensors, *cache)
            length += ids.size
            torch.testing.assert_close(actual, expected, **tolerance)
            torch.testing.assert_close(token, actual.argmax(-1))
            ref_cache = reference.past_key_values
            for index, layer in enumerate(ref_cache.layers):
                torch.testing.assert_close(cache[2 * index][:, :, :length], layer.keys, **tolerance)
                torch.testing.assert_close(cache[2 * index + 1][:, :, :length], layer.values, **tolerance)
            for before, after in zip(old_cache, cache, strict=True):
                assert after.shape[2] == capacity
                torch.testing.assert_close(after[:, :, length:], before[:, :, length:])
    with pytest.raises(ValueError, match="capacity"):
        decoder_inputs(ids, None, 100, 32, capacity, capacity)


@pytest.mark.parametrize("model", [torch.float32], indirect=True, ids=["fp32"])
def test_larger_cache_bucket(model):
    # FP32 checks the bucket/mask math independently of BF16 backend rounding.
    test_prefill_and_cached_decode(model, (2, 2, 3), capacity=32)


def test_placeholder_mismatch_is_rejected():
    with pytest.raises(ValueError, match="placeholder"):
        decoder_inputs([[100]], np.zeros((2, 32), np.float32), 100, 32, 0)


def test_decode_specialization_preserves_prefill_cache(model):
    from _internal.decode import DecodeCache, decode_attention
    from transformers import AttentionInterface

    prefill = TextDecoder(deepcopy(model))
    decode = TextDecoder(deepcopy(model))
    AttentionInterface.register("qwen3_onnx_decode", decode_attention)
    decode.decoder.config._attn_implementation = "qwen3_onnx_decode"
    decode.cache_type = DecodeCache
    capacity = 8
    cache = tuple(torch.randn(1, 2, capacity, 8, dtype=model.dtype) for _ in range(4))
    ref_cache = None
    tolerance = {"atol": 0.004, "rtol": 0.016} if model.dtype == torch.bfloat16 else {}
    with torch.inference_mode():
        for position, sequence in [(0, 5), (5, 1), (6, 1), (7, 1)]:
            ids = torch.arange(3 + position, 3 + position + sequence)[None]
            positions = torch.arange(position, position + sequence)[None]
            visible = torch.arange(capacity)[None, None, None, :] <= positions[:, None, :, None]
            inputs = (
                ids,
                torch.zeros(1, sequence, 32, dtype=model.dtype),
                torch.zeros(1, sequence, 1, dtype=torch.bool),
                positions,
                torch.where(visible, 0.0, -1e4).to(model.dtype),
            )
            reference = model.model.language_model(input_ids=ids, past_key_values=ref_cache, use_cache=True)
            expected = model.lm_head(reference.last_hidden_state[:, -1])
            old = cache
            actual, _, *cache = (prefill if position == 0 else decode)(*inputs, *cache)
            torch.testing.assert_close(actual, expected, **tolerance)
            ref_cache = reference.past_key_values
            for i, layer in enumerate(ref_cache.layers):
                for value, ref in [(cache[2 * i], layer.keys), (cache[2 * i + 1], layer.values)]:
                    # Native GQA and eager attention differ by BF16 rounding in
                    # later-layer values; FP32 remains a strict math comparison.
                    cache_tolerance = {"atol": 0.008, "rtol": 0.016} if model.dtype == torch.bfloat16 else {}
                    torch.testing.assert_close(value[:, :, : position + sequence], ref, **cache_tolerance)
            for before, after in zip(old, cache, strict=True):
                assert torch.equal(before[:, :, :position], after[:, :, :position])
                assert torch.equal(before[:, :, position + sequence :], after[:, :, position + sequence :])


def test_timestamp_head_matches_reference(model):
    aligner = Qwen3ASRForTokenClassification(model.config).to(model.dtype).eval()
    ids = np.array([[3, 100, 100, 5, 6, 7]])
    embeddings = np.random.default_rng(7).normal(size=(2, 32)).astype(np.float32)
    feed = decoder_inputs(ids, embeddings, 100, 32, 0)
    values = typed_inputs(feed.values(), model.dtype)
    indices = torch.tensor([4, 5])
    with torch.inference_mode():
        combined = torch.where(values[2], values[1], aligner.get_input_embeddings()(values[0]))
        expected = aligner(inputs_embeds=combined, use_cache=False).logits[:, indices]
        actual, bins = ForcedAligner(deepcopy(aligner))(*values, indices)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(bins, expected.argmax(-1))


@pytest.mark.parametrize("samples", [8000, 16001, 118960, 480000])
def test_shared_log_mel_matches_hf(samples):
    from types import SimpleNamespace

    from transformers import Qwen3ASRFeatureExtractor

    fe = Qwen3ASRFeatureExtractor()
    audio = np.random.default_rng(3).normal(0, 0.1, samples).astype(np.float32)
    expected = fe(audio, sampling_rate=16000, padding=True, return_tensors="pt")["input_features"]
    with torch.inference_mode():
        actual = LogMel(SimpleNamespace(feature_extractor=fe))(torch.from_numpy(audio)[None])
    assert actual.shape == (1, 128, samples // 160)
    torch.testing.assert_close(actual, expected[..., : samples // 160], atol=1e-3, rtol=1e-3)
