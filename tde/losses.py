"""Training objectives. All take masked logits [B,K] and target distributions [B,K]."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def log_probs(logits: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.masked_fill(~cand_mask, -1e4), dim=-1)


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
    """-sum_k y_k log p_k; equals CE for one-hot y and KL(y||p)+const for soft y."""
    lp = log_probs(logits, cand_mask)
    return -(target * lp).masked_fill(~cand_mask, 0.0).sum(-1).mean()


def brier(logits: torch.Tensor, target: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
    p = log_probs(logits, cand_mask).exp()
    return ((p - target) ** 2).masked_fill(~cand_mask, 0.0).sum(-1).mean()


def ranked_probability_score(logits: torch.Tensor, target: torch.Tensor, cand_mask: torch.Tensor, is_score: torch.Tensor) -> torch.Tensor:
    """RPS for ordinal questions: sum over levels of (CDF_p - CDF_y)^2 / (K-1). Zero for non-ordinal rows."""
    if not bool(is_score.any()):
        return logits.new_zeros(())
    p = log_probs(logits, cand_mask).exp()
    cp = torch.cumsum(p.masked_fill(~cand_mask, 0.0), dim=-1)
    cy = torch.cumsum(target.masked_fill(~cand_mask, 0.0), dim=-1)
    k = cand_mask.sum(-1).clamp_min(2).float()
    per = ((cp - cy) ** 2).masked_fill(~cand_mask, 0.0).sum(-1) / (k - 1)
    return (per * is_score.float()).sum() / is_score.float().sum()


def permutation_kl(logits_a: torch.Tensor, logits_b: torch.Tensor, perm: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
    """Symmetric KL between two forward passes whose candidates were permuted by `perm` (b = a[perm]).

    perm: [B,K] such that candidate j of pass b is candidate perm[b,j] of pass a (padded with arange).
    """
    lpa = log_probs(logits_a, cand_mask)
    lpb = log_probs(logits_b, cand_mask)
    lpb_aligned = torch.zeros_like(lpb)
    lpb_aligned.scatter_(1, perm, lpb)  # position perm[j] (in a's order) receives b's log-prob for that candidate
    pa, pb = lpa.exp(), lpb_aligned.exp()
    kl_ab = (pa * (lpa - lpb_aligned)).masked_fill(~cand_mask, 0.0).sum(-1)
    kl_ba = (pb * (lpb_aligned - lpa)).masked_fill(~cand_mask, 0.0).sum(-1)
    return 0.5 * (kl_ab + kl_ba).mean()


def confidence_bce(conf_logit: torch.Tensor, logits: torch.Tensor, target: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
    """BCE for the confidence head: label = 1 if argmax p equals argmax target."""
    pred = logits.masked_fill(~cand_mask, -1e4).argmax(-1)
    gold = target.masked_fill(~cand_mask, -1.0).argmax(-1)
    y = (pred == gold).float()
    return F.binary_cross_entropy_with_logits(conf_logit, y)


def total_loss(out: dict, batch: dict, *, w_ce: float = 1.0, w_brier: float = 0.0, w_rps: float = 0.0,
               w_perm: float = 0.0, out_perm: dict | None = None, perm: torch.Tensor | None = None, w_conf: float = 0.0,
               w_pg: float = 0.0, w_correct_pg: float = 0.0, pg_samples: int = 32) -> tuple[torch.Tensor, dict]:
    logits, target, mask = out["logits"], batch["target"], batch["cand_mask"]
    parts = {}
    loss = logits.new_zeros(())
    if w_pg:
        parts["paired_pg"] = paired_brier_pg(logits, target, mask, pg_samples); loss = loss + w_pg * parts["paired_pg"]
    if w_correct_pg:
        parts["correct_pg"] = correctness_pg(logits, target, mask, pg_samples); loss = loss + w_correct_pg * parts["correct_pg"]
    if w_ce:
        parts["ce"] = soft_cross_entropy(logits, target, mask); loss = loss + w_ce * parts["ce"]
    if w_brier:
        parts["brier"] = brier(logits, target, mask); loss = loss + w_brier * parts["brier"]
    if w_rps:
        parts["rps"] = ranked_probability_score(logits, target, mask, batch["is_score"]); loss = loss + w_rps * parts["rps"]
    if w_perm and out_perm is not None and perm is not None:
        parts["perm_kl"] = permutation_kl(logits, out_perm["logits"], perm, mask); loss = loss + w_perm * parts["perm_kl"]
    if w_conf and "conf_logit" in out:
        parts["conf"] = confidence_bce(out["conf_logit"], logits.detach(), target, mask); loss = loss + w_conf * parts["conf"]
    return loss, {k: float(v.detach()) for k, v in parts.items()}


# --------------------------------------------------------------------------
# RLCD-style control arms: policy-gradient on a sampled reward.
# `paired_brier_pg` uses a strictly proper reward whose expectation is
# ||q||^2 - ||p - q||^2 (the negative Brier score up to a constant), so its
# expected gradient equals the gradient of the direct Brier loss (see
# tests/test_losses_pg.py). `correctness_pg` rewards 1[A = Y] only, which is
# linear in p and therefore NOT proper: it drives p towards a one-hot on the
# majority label instead of recovering the target distribution.
# --------------------------------------------------------------------------

def _sample_labels(target: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
    """Draw one observed outcome Y per row from the (soft or hard) target distribution."""
    t = target.masked_fill(~cand_mask, 0.0).clamp_min(0.0)
    t = t / t.sum(-1, keepdim=True).clamp_min(1e-9)
    return torch.multinomial(t, 1).squeeze(-1)


def paired_brier_pg(logits: torch.Tensor, target: torch.Tensor, cand_mask: torch.Tensor, n_samples: int = 32) -> torch.Tensor:
    """Score-function surrogate whose expected gradient is grad ||p - q||^2 (RLCD-style, proper reward).

    M >= 2 predictions A_i ~ p (with replacement); reward
        R = (2/M) sum_i 1[A_i = Y] - sum_k c_k (c_k - 1) / (M (M - 1)),
    per-sample reward r_i = (2/M) 1[A_i = Y] - 2 (c_{A_i} - 1) / (M (M - 1)),
    detached conditional baseline b_i = (2/M) p_Y - 2 sum_{j != i} p_{A_j} / (M (M - 1)).
    """
    M = max(2, int(n_samples))
    lp = log_probs(logits, cand_mask)
    p = lp.exp().detach()
    B, K = p.shape
    Y = _sample_labels(target, cand_mask)                                   # [B]
    A = torch.multinomial(p, M, replacement=True)                            # [B, M]
    counts = torch.zeros_like(p).scatter_add_(1, A, torch.ones_like(A, dtype=p.dtype))  # [B, K]
    c_A = torch.gather(counts, 1, A)                                         # [B, M]
    hit = (A == Y.unsqueeze(1)).to(p.dtype)
    r = (2.0 / M) * hit - 2.0 * (c_A - 1.0) / (M * (M - 1.0))
    p_A = torch.gather(p, 1, A)                                              # [B, M]
    p_Y = torch.gather(p, 1, Y.unsqueeze(1))                                 # [B, 1]
    b = (2.0 / M) * p_Y - 2.0 * (p_A.sum(1, keepdim=True) - p_A) / (M * (M - 1.0))
    lp_A = torch.gather(lp, 1, A)
    return -((r - b).detach() * lp_A).sum(1).mean()


def correctness_pg(logits: torch.Tensor, target: torch.Tensor, cand_mask: torch.Tensor, n_samples: int = 32) -> torch.Tensor:
    """REINFORCE with reward 1[A = Y] and a detached p_Y baseline. Improper: not a calibration objective."""
    M = max(1, int(n_samples))
    lp = log_probs(logits, cand_mask)
    p = lp.exp().detach()
    Y = _sample_labels(target, cand_mask)
    A = torch.multinomial(p, M, replacement=True)
    hit = (A == Y.unsqueeze(1)).to(p.dtype)
    b = torch.gather(p, 1, Y.unsqueeze(1))
    lp_A = torch.gather(lp, 1, A)
    return -((hit - b).detach() * lp_A).mean(1).mean()
