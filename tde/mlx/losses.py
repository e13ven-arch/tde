"""MLX ports of tde.losses for the MLX trainers."""
from __future__ import annotations

import mlx.core as mx


def log_probs(logits, cand_mask):
    x = mx.where(cand_mask, logits, -1e4)
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def _sample(p, u):
    """Categorical draws by inverse CDF: p [B,K], u [B,M] uniform(0,1) -> indices [B,M]."""
    cdf = mx.cumsum(p, axis=-1)
    return mx.minimum((u[..., None] > cdf[:, None, :]).sum(-1), p.shape[-1] - 1).astype(mx.int32)


def paired_brier_pg(logits, target, cand_mask, u_y, u_a):
    """tde.losses.paired_brier_pg (RLCD-style proper reward, score-function surrogate) with the randomness passed in.

    u_y [B,1] draws the outcome label Y ~ target and u_a [B,M] the M >= 2 predictions A_i ~ p, so a compiled step stays
    a pure function of its inputs. Expected gradient: grad ||p - q||^2 (Brier)."""
    M = u_a.shape[1]
    lp = log_probs(logits, cand_mask)
    p = mx.stop_gradient(mx.exp(lp))
    K = p.shape[-1]
    Y = _sample(target, u_y)                                         # [B,1]
    A = _sample(p, u_a)                                              # [B,M]
    counts = (A[..., None] == mx.arange(K)).astype(p.dtype).sum(1)  # [B,K]
    c_A = mx.take_along_axis(counts, A, axis=1)
    hit = (A == Y).astype(p.dtype)
    r = (2.0 / M) * hit - 2.0 * (c_A - 1.0) / (M * (M - 1.0))
    p_A = mx.take_along_axis(p, A, axis=1)
    p_Y = mx.take_along_axis(p, Y, axis=1)
    b = (2.0 / M) * p_Y - 2.0 * (p_A.sum(1, keepdims=True) - p_A) / (M * (M - 1.0))
    return -(mx.stop_gradient(r - b) * mx.take_along_axis(lp, A, axis=1)).sum(1).mean()
