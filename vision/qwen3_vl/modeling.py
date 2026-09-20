# SPDX-License-Identifier: Apache-2.0
"""Tensor-only export contracts, preserving Qwen3-VL DeepStack and interleaved MRoPE."""

import onnx
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AttentionInterface
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision


def attention(q, k, v, bias, scale):
    dtype = q.dtype
    # RTX 1.6 rejects ONNX Attention.softmax_precision. Express accumulation
    # explicitly so the graph has the FP32 SDPA contract on both providers.
    q, k, v = q.float(), k.float(), v.float()
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    return (torch.softmax(q @ k.transpose(-1, -2) * scale + bias.float(), dim=-1) @ v).to(dtype)


def decoder_attention(module, query, key, value, attention_mask, scaling=None, **kwargs):
    # Attend to a fixed prefix bucket while keeping the full shared KV bank.
    # Folding query groups into the query axis avoids materializing repeated KV.
    batch, heads, sequence, width = query.shape
    kv_heads = key.shape[1]
    groups = heads // kv_heads
    length = attention_mask.shape[-1]
    key, value = key[:, :, :length], value[:, :, :length]
    query = query.reshape(batch, kv_heads, groups * sequence, width)
    attention_mask = attention_mask.repeat(1, 1, groups, 1)
    output = attention(query, key, value, attention_mask, scaling)
    return output.reshape(batch, heads, sequence, width).transpose(1, 2), None


def layer_norm(module, x):
    # Match PyTorch's FP32 accumulation and affine transform even when the RTX
    # optimizer would otherwise execute LayerNormalization in activation dtype.
    return F.layer_norm(x.float(), module.normalized_shape, module.weight.float(), module.bias.float(), module.eps).to(
        x.dtype
    )


def merge_patches(module, x):
    x = x.reshape(-1, module.hidden_size) if module.use_postshuffle_norm else x
    x = layer_norm(module.norm, x).reshape(-1, module.hidden_size)
    return module.linear_fc2(module.act_fn(module.linear_fc1(x)))


class AccumulationLinear(nn.Module):
    """Keep checkpoint weights/activations, explicitly request FP32 accumulation."""

    def __init__(self, module):
        super().__init__()
        self.weight = module.weight
        self.bias = module.bias

    def forward(self, x):
        return F.linear(x.float(), self.weight.float(), self.bias.float() if self.bias is not None else None).to(
            x.dtype
        )


class FixedCache:
    def __init__(self, past, cache_position):
        self.past = past
        self.position = cache_position
        self.present = []

    def update(self, key, value, layer_idx):
        updated = []
        for cache, data in zip(self.past[2 * layer_idx : 2 * layer_idx + 2], (key, value), strict=True):
            if torch.onnx.is_in_onnx_export():
                result = torch.onnx.ops.symbolic(
                    "TensorScatter",
                    (cache, data, self.position),
                    {"axis": 2, "mode": "linear"},
                    dtype=data.dtype,
                    shape=cache.shape,
                    version=24,
                )
            else:
                indices = torch.arange(data.shape[2], device=data.device) + self.position[0]
                result = cache.index_copy(2, indices, data)
            updated.append(result)
        self.present.extend(updated)
        return tuple(updated)


class VisionEncoder(nn.Module):
    """One image or temporal patch per call; pad to a small fixed set of patch buckets.

    All attention is within a single spatial grid, exactly as in the upstream ViT.
    Bilinear position lookup and rotary inputs are computed from the original grid,
    not the padded bucket. Padded keys are masked in every block.
    """

    def __init__(self, model):
        super().__init__()
        self.visual = model.model.visual
        for name, module in list(self.visual.named_modules()):
            if isinstance(module, nn.Linear):
                self.visual.set_submodule(name, AccumulationLinear(module))

    def forward(self, pixels, bilinear_indices, bilinear_weights, rotary_cos, rotary_sin, attention_bias):
        visual = self.visual
        # Preserve the checkpoint's convolution: a flattened linear projection is
        # algebraically equivalent, but changes BF16 accumulation/rounding.
        x = visual.patch_embed(pixels)
        pos = (visual.pos_embed(bilinear_indices).float() * bilinear_weights[..., None]).sum(0)
        x = x + pos.to(x.dtype)
        # ONNX Attention permits broadcasting, but ORT's CPU kernel currently
        # requires the explicit query dimension. RTX folds this broadcast.
        attention_bias = attention_bias.expand(1, 1, pixels.shape[0], pixels.shape[0])
        deep = []
        for i, block in enumerate(visual.blocks):
            normalized = layer_norm(block.norm1, x)
            a = block.attn
            q, k, v = a.qkv(normalized).reshape(-1, 3, a.num_heads, a.head_dim).permute(1, 0, 2, 3).unbind(0)
            q, k = apply_rotary_pos_emb_vision(q, k, rotary_cos, rotary_sin)
            y = attention(
                q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], attention_bias, a.scaling
            )
            x = x + a.proj(y.transpose(1, 2).reshape(-1, a.dim))
            x = x + block.mlp(layer_norm(block.norm2, x))
            if i in visual.deepstack_visual_indexes:
                deep.append(merge_patches(visual.deepstack_merger_list[visual.deepstack_visual_indexes.index(i)], x))
        return torch.stack([merge_patches(visual.merger, x), *deep])


class TextDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.language_model
        self.lm_head = model.lm_head
        AttentionInterface.register("din_qwen3_vl", decoder_attention)
        self.decoder.config._attn_implementation = "din_qwen3_vl"
        self.deep_count = len(model.config.vision_config.deepstack_visual_indexes)

    def forward(
        self,
        input_ids,
        visual_features,
        visual_mask,
        rotary_cos,
        rotary_sin,
        attention_bias,
        cache_position,
        logits_index,
        *past,
    ):
        x = torch.where(visual_mask, visual_features[0], self.decoder.embed_tokens(input_ids))
        cache = FixedCache(past, cache_position)
        for i, layer in enumerate(self.decoder.layers):
            x = layer(
                x,
                position_embeddings=(rotary_cos, rotary_sin),
                attention_mask=attention_bias,
                past_key_values=cache,
                use_cache=True,
            )
            if i < self.deep_count:
                x = x + visual_features[i + 1]
        hidden = self.decoder.norm(x.index_select(1, logits_index)).squeeze(1)
        logits = self.lm_head(hidden).float()
        return logits, logits.argmax(-1), hidden, *cache.present


def export_graph(module, inputs, path, names, outputs, shapes):
    print(f"Exporting {path}", flush=True)
    torch.onnx.export(
        module.eval(),
        inputs,
        str(path),
        input_names=names,
        output_names=outputs,
        dynamo=True,
        dynamic_shapes=shapes,
        opset_version=24,
        external_data=True,
    )
    onnx.checker.check_model(str(path))
    model = onnx.load(str(path), load_external_data=False)
    unsupported = sorted({node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")})
    if unsupported:
        raise ValueError(f"Export must use standard ONNX operators only; found domains: {unsupported}")
