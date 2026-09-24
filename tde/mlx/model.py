"""MLX port of the joint readout (JointDecisionModel, topology 'seq') over ModernBERT, for training on Apple GPUs.

Parameters stay fp32 (master weights) while every matmul runs in the activation dtype (bf16 in training): MPLinear,
MPEmbedding and MPLayerNorm cast their weights at use, so gradients and AdamW updates are fp32. Parameter names mirror
the PyTorch state_dict (backbone.layers.3.attn.Wqkv.weight, scorer.q.weight, level.emb.weight, ...), so release
safetensors load directly and trained weights convert back to a best.pt that the PyTorch evaluation reads.
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class MPLinear(nn.Module):
    def __init__(self, n_in: int, n_out: int, bias: bool):
        super().__init__()
        self.weight = mx.zeros((n_out, n_in))
        if bias:
            self.bias = mx.zeros((n_out,))

    def __call__(self, x):
        y = x @ self.weight.astype(x.dtype).T
        return y + self.bias.astype(x.dtype) if "bias" in self else y


class MPEmbedding(nn.Module):
    def __init__(self, n: int, d: int):
        super().__init__()
        self.weight = mx.zeros((n, d))

    def __call__(self, ids):
        return self.weight[ids]


class MPLayerNorm(nn.Module):
    def __init__(self, d: int, bias: bool, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((d,))
        if bias:
            self.bias = mx.zeros((d,))

    def __call__(self, x):
        b = self.bias.astype(x.dtype) if "bias" in self else None
        return mx.fast.layer_norm(x, self.weight.astype(x.dtype), b, self.eps)


class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int, rope_base: float):
        super().__init__()
        self.heads, self.head_dim, self.rope_base = heads, hidden // heads, rope_base
        self.Wqkv = MPLinear(hidden, 3 * hidden, bias=False)
        self.Wo = MPLinear(hidden, hidden, bias=False)

    def __call__(self, x, mask):
        b, t, _ = x.shape
        qkv = self.Wqkv(x).reshape(b, t, 3, self.heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q = mx.fast.rope(qkv[0], self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=0)
        k = mx.fast.rope(qkv[1], self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=0)
        o = mx.fast.scaled_dot_product_attention(q, k, qkv[2], scale=self.head_dim ** -0.5, mask=mask)
        return self.Wo(o.transpose(0, 2, 1, 3).reshape(b, t, -1))


class MLP(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.Wi = MPLinear(hidden, 2 * inter, bias=False)
        self.Wo = MPLinear(inter, hidden, bias=False)

    def __call__(self, x):
        a, g = mx.split(self.Wi(x), 2, axis=-1)
        return self.Wo(nn.gelu(a) * g)


class Layer(nn.Module):
    def __init__(self, i: int, hidden: int, heads: int, inter: int, every: int):
        super().__init__()
        self.is_global = i % every == 0
        self.attn_norm = MPLayerNorm(hidden, bias=False) if i else nn.Identity()  # HF ModernBERT: no norm before layer 0
        self.attn = Attention(hidden, heads, 160000.0 if self.is_global else 10000.0)
        self.mlp_norm = MPLayerNorm(hidden, bias=False)
        self.mlp = MLP(hidden, inter)

    def __call__(self, x, masks):
        x = x + self.attn(self.attn_norm(x), masks[0] if self.is_global else masks[1])
        return x + self.mlp(self.mlp_norm(x))


class Embeddings(nn.Module):
    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.tok_embeddings = MPEmbedding(vocab, hidden)
        self.norm = MPLayerNorm(hidden, bias=False)

    def __call__(self, ids, dtype):
        return self.norm(self.tok_embeddings(ids).astype(dtype))


class ModernBert(nn.Module):
    def __init__(self, vocab: int = 50370, hidden: int = 768, layers: int = 22, heads: int = 12, inter: int = 1152,
                 window: int = 128, every: int = 3):
        super().__init__()
        self.window = window
        self.embeddings = Embeddings(vocab, hidden)
        self.layers = [Layer(i, hidden, heads, inter, every) for i in range(layers)]
        self.final_norm = MPLayerNorm(hidden, bias=False)

    def __call__(self, ids, valid, dtype):
        """valid: [B,T] bool, or None when no row is padded. Global layers mask padded keys; local layers also keep
        |i-j| <= window/2. Padded queries may see every valid key (their outputs are never read) so no softmax row is
        empty. Without padding, global layers run unmasked and local layers share one [T,T] band."""
        pos = mx.arange(ids.shape[1])
        band = mx.abs(pos[:, None] - pos[None, :]) <= self.window // 2
        if valid is None:
            full, local = None, band
        else:
            full = valid[:, None, None, :]
            local = (band[None, None] | ~valid[:, None, :, None]) & full
        x = self.embeddings(ids, dtype)
        for layer in self.layers:
            x = layer(x, (full, local))
        return self.final_norm(x)


class Level(nn.Module):
    def __init__(self, hidden: int, max_levels: int = 10):
        super().__init__()
        self.emb = MPEmbedding(max_levels + 1, hidden)  # index 0 = "not ordinal"


class PointerScorer(nn.Module):
    """s_i = (W_q LN(h_decide)) . (W_k LN(h_opt_i)) / sqrt(d) + b, as tde.model.heads.PointerScorer."""

    def __init__(self, hidden: int):
        super().__init__()
        self.q = MPLinear(hidden, hidden, bias=True)
        self.k = MPLinear(hidden, hidden, bias=True)
        self.norm_q = MPLayerNorm(hidden, bias=True)
        self.norm_k = MPLayerNorm(hidden, bias=True)
        self.bias = mx.zeros((1,))
        self._scale = 1.0 / math.sqrt(hidden)

    def __call__(self, h_dec, h_opts, cand_mask):
        q = self.q(self.norm_q(h_dec))[:, None, :]
        k = self.k(self.norm_k(h_opts))
        s = (q * k).sum(-1).astype(mx.float32) * self._scale + self.bias
        return mx.where(cand_mask, s, -1e4)


class JointDecisionModel(nn.Module):
    """[CLS] state [SEP] question [SEP] [MASK] c1 ... [MASK] [SEP] -> one logit per candidate (pool marker+span)."""

    def __init__(self, vocab: int = 50370, hidden: int = 768):
        super().__init__()
        self.backbone = ModernBert(vocab, hidden)
        self.level = Level(hidden)
        self.scorer = PointerScorer(hidden)

    def __call__(self, b: dict, dtype=mx.bfloat16):
        h = self.backbone(b["input_ids"], b.get("valid"), dtype)
        B, T, D = h.shape
        K = b["opt_positions"].shape[1]
        marker = mx.take_along_axis(h, mx.broadcast_to(b["opt_positions"][..., None], (B, K, D)), axis=1)
        pos = mx.arange(T)[None, None, :]
        m = ((pos >= b["span_start"][..., None]) & (pos < b["span_end"][..., None])).astype(h.dtype)
        span = (m @ h) / mx.maximum(m.sum(-1, keepdims=True), 1.0)
        level = self.level.emb(mx.clip(b["level_index"] + 1, 0, self.level.emb.weight.shape[0] - 1)).astype(h.dtype)
        h_dec = mx.take_along_axis(h, mx.broadcast_to(b["decide_positions"][:, None, None], (B, 1, D)), axis=1)[:, 0]
        return self.scorer(h_dec, marker + span + level, b["cand_mask"])


def load_joint(path: str) -> JointDecisionModel:
    """Build the model with the embedding size found in `path` (a safetensors state_dict) and load it strictly."""
    weights = mx.load(path)
    model = JointDecisionModel(vocab=weights["backbone.embeddings.tok_embeddings.weight"].shape[0])
    model.load_weights(list(weights.items()), strict=True)
    return model
