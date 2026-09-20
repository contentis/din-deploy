# SPDX-License-Identifier: Apache-2.0
"""Device-side nucleus sampling using one uniform draw and a cumulative CDF."""

import torch


def sample_top_p(logits, temperature, top_p, generator):
    probabilities = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1:
        probabilities, indices = probabilities.sort(descending=True)
    cumulative = probabilities.cumsum(-1)
    if top_p < 1:
        # Include the first token crossing the threshold, including the next
        # token when the preceding cumulative mass equals top_p exactly.
        cutoff = torch.searchsorted(cumulative, torch.full_like(cumulative[:1], top_p), right=True)
        mass = cumulative[cutoff.clamp_max(cumulative.numel() - 1)]
    else:
        mass = cumulative[-1:]
    draw = torch.rand(1, device=logits.device, generator=generator) * mass
    selected = torch.searchsorted(cumulative, draw, right=True).clamp_max(cumulative.numel() - 1)
    return indices[torch.minimum(selected, cutoff)] if top_p < 1 else selected


class CudaSampler:
    """Two reusable CUDA graphs for categorical/nucleus sampling; no model weights.

    RNG stays outside capture, so caller-owned generators and request seeds retain
    their normal semantics. Parameters and logits use stable device buffers.
    """

    def __init__(self, vocabulary_size):
        self.logits = torch.empty(vocabulary_size, dtype=torch.float32, device="cuda")
        self.temperature = torch.ones(1, device="cuda")
        self.top_p = torch.ones(1, device="cuda")
        self.uniform = torch.zeros(1, device="cuda")
        self.graphs = {}
        self.outputs = {}

    def configure(self, temperature, top_p):
        self.temperature.fill_(temperature)
        self.top_p.fill_(top_p)
        self.nucleus = top_p < 1

    def _select(self):
        probabilities = torch.softmax(self.logits / self.temperature, dim=-1)
        if self.nucleus:
            probabilities, indices = probabilities.sort(descending=True)
        cumulative = probabilities.cumsum(-1)
        if self.nucleus:
            cutoff = torch.searchsorted(cumulative, self.top_p, right=True).clamp_max(cumulative.numel() - 1)
            mass = cumulative[cutoff]
        else:
            mass = cumulative[-1:]
        selected = torch.searchsorted(cumulative, self.uniform * mass, right=True).clamp_max(cumulative.numel() - 1)
        return indices[torch.minimum(selected, cutoff)] if self.nucleus else selected

    @torch.inference_mode()
    def select(self, logits, generator):
        self.logits.copy_(logits)
        torch.rand(1, out=self.uniform, generator=generator)
        if self.nucleus not in self.graphs:
            self._select()  # Initialize kernels/allocator before capture.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
                self.outputs[self.nucleus] = self._select()
            self.graphs[self.nucleus] = graph
        self.graphs[self.nucleus].replay()
        return self.outputs[self.nucleus]
