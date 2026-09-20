# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from vision.qwen3_vl.sampling import CudaSampler, sample_top_p


def test_sampling_peaked_distribution_and_seed():
    # Two supported tokens hold 0.99 mass; all other vocabulary entries together
    # hold 0.01. A 0.9 nucleus must include both, including the boundary token.
    p = torch.full((1000,), 0.01 / 998)
    p[5], p[733] = 0.6, 0.39
    logits = p.log()

    def draw(seed):
        generator = torch.Generator().manual_seed(seed)
        return [int(sample_top_p(logits, 1, 0.9, generator)) for _ in range(400)]

    actual = draw(123)
    assert actual == draw(123)
    assert set(actual) == {5, 733}
    # A broad distribution check catches accidental top-1 truncation/reweighting.
    assert 0.48 < actual.count(5) / len(actual) < 0.72


def test_sampling_diffuse_distribution_keeps_full_nucleus():
    # More than 256 candidates are needed. Descending unique probabilities avoid
    # ambiguous cutoff ties and make the expected nucleus independent of sorting.
    p = torch.arange(1000, 0, -1, dtype=torch.float32)
    p /= p.sum()
    cutoff = int(torch.searchsorted(p.cumsum(0), torch.tensor(0.9)))
    generator = torch.Generator().manual_seed(41)
    samples = [int(sample_top_p(p.log(), 1, 0.9, generator)) for _ in range(400)]
    assert all(0 <= token <= cutoff for token in samples)
    assert any(token > 256 for token in samples)


def test_sampling_top_p_one_keeps_categorical_support():
    generator = torch.Generator().manual_seed(11)
    logits = torch.full((1000,), -torch.inf)
    logits[23], logits[999] = 0, 0
    samples = [int(sample_top_p(logits, 1, 1, generator)) for _ in range(40)]
    assert set(samples) == {23, 999}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA sampler requires GPU")
@torch.inference_mode()
def test_cuda_sampler_replay_parameters_and_seed():
    with torch.cuda.stream(torch.cuda.Stream()):
        sampler = CudaSampler(1000)
        logits = torch.linspace(-4, 4, 1000, device="cuda")
        for temperature, top_p in ((0.7, 0.9), (1.3, 1.0), (0.4, 0.7)):
            sampler.configure(temperature, top_p)
            captured_rng = torch.Generator(device="cuda").manual_seed(81)
            reference_rng = torch.Generator(device="cuda").manual_seed(81)
            for _ in range(30):
                actual = sampler.select(logits, captured_rng)
                expected = sample_top_p(logits, temperature, top_p, reference_rng)
                assert actual.item() == expected.item()
            logits.neg_()
        assert len(sampler.graphs) == 2
