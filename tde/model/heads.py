"""Readout heads shared by the three models."""
from __future__ import annotations

import math

import torch
from torch import nn


class LevelIndexEmbedding(nn.Module):
    """Adds a learned ordinal-position embedding to candidate hidden states of `score` questions."""

    def __init__(self, hidden: int, max_levels: int = 10):
        super().__init__()
        self.emb = nn.Embedding(max_levels + 1, hidden)  # index 0 = "not ordinal"
        nn.init.normal_(self.emb.weight, std=0.02)
        with torch.no_grad():
            self.emb.weight[0].zero_()

    def forward(self, h: torch.Tensor, level_index: torch.Tensor) -> torch.Tensor:
        idx = (level_index + 1).clamp(min=0, max=self.emb.num_embeddings - 1)
        return h + self.emb(idx)


class PointerScorer(nn.Module):
    """s_i = (W_q h_decide) . (W_k h_opt_i) / sqrt(d) + b.  Candidate count is free."""

    def __init__(self, hidden: int, proj: int | None = None):
        super().__init__()
        proj = proj or hidden
        self.q = nn.Linear(hidden, proj)
        self.k = nn.Linear(hidden, proj)
        self.norm_q = nn.LayerNorm(hidden)
        self.norm_k = nn.LayerNorm(hidden)
        self.scale = 1.0 / math.sqrt(proj)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, h_decide: torch.Tensor, h_opts: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        q = self.q(self.norm_q(h_decide)).unsqueeze(1)          # [B,1,P]
        k = self.k(self.norm_k(h_opts))                         # [B,K,P]
        s = (q * k).sum(-1) * self.scale + self.bias            # [B,K]
        return s.masked_fill(~cand_mask, -1e4)


class ConfidenceHead(nn.Module):
    """Predicts P(top-1 is correct) from the decision hidden state and the max-prob."""

    def __init__(self, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden + 2, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, h_decide: torch.Tensor, probs: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        p = probs.masked_fill(~cand_mask, 0.0)
        maxp = p.max(-1).values.unsqueeze(-1)
        ent = -(p.clamp_min(1e-9).log() * p).sum(-1, keepdim=True)
        return self.mlp(torch.cat([h_decide, maxp, ent], dim=-1)).squeeze(-1)


def gather_positions(h: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """h: [B,L,D]; positions: [B,K] -> [B,K,D]."""
    idx = positions.unsqueeze(-1).expand(-1, -1, h.size(-1))
    return torch.gather(h, 1, idx)


def candidate_states(h: torch.Tensor, batch: dict, pool: str = "marker+span") -> torch.Tensor:
    """Per-candidate representation [B,K,D]: the marker hidden state, the mean over the candidate's text span, or their sum."""
    marker = gather_positions(h, batch["opt_positions"])
    if pool == "marker" or "span_start" not in batch:
        return marker
    B, L, D = h.shape
    pos = torch.arange(L, device=h.device).view(1, 1, L)
    m = ((pos >= batch["span_start"].unsqueeze(-1)) & (pos < batch["span_end"].unsqueeze(-1))).to(h.dtype)  # [B,K,L]
    span = torch.bmm(m, h) / m.sum(-1, keepdim=True).clamp_min(1.0)
    return span if pool == "span" else marker + span
