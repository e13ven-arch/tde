"""Bi-encoder ablation: z = pool(state+question), e_i = pool(candidate_i), s_i = z^T W e_i.

Candidates never see the state or the other candidates; kept only to measure
what listwise interaction is worth (Exp 001).
"""
from __future__ import annotations

import torch
from torch import nn

from tde.model.heads import LevelIndexEmbedding


def _pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp_min(1.0)


class BiEncoderDecisionModel(nn.Module):
    mode = "biencoder"

    def __init__(self, backbone: nn.Module, hidden: int, use_confidence_head: bool = False):
        super().__init__()
        self.backbone = backbone
        self.level = LevelIndexEmbedding(hidden)
        self.w = nn.Linear(hidden, hidden, bias=False)
        self.norm_z = nn.LayerNorm(hidden)
        self.norm_e = nn.LayerNorm(hidden)
        self.scale = nn.Parameter(torch.tensor(1.0 / hidden**0.5))
        self.conf = None

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hz = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        z = self.norm_z(_pool(hz, batch["attention_mask"]))                     # [B,D]
        he = self.backbone(input_ids=batch["cand_input_ids"], attention_mask=batch["cand_attention_mask"]).last_hidden_state
        e = self.norm_e(_pool(he, batch["cand_attention_mask"]))                # [N,D]
        B, K = batch["cand_mask"].shape
        E = z.new_zeros((B, K, z.size(-1)))
        E[batch["cand_owner"], batch["cand_slot"]] = e
        E = self.level(E, batch["level_index"])
        s = (self.w(z).unsqueeze(1) * E).sum(-1) * self.scale
        return {"logits": s.masked_fill(~batch["cand_mask"], -1e4)}
