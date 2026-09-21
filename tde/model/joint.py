"""Joint-sequence readout: one encoder pass over [state; question; candidates], pointer scores."""
from __future__ import annotations

import torch
from torch import nn

from tde.model.heads import ConfidenceHead, LevelIndexEmbedding, PointerScorer, candidate_states, gather_positions


class JointDecisionModel(nn.Module):
    mode = "joint"

    def __init__(self, backbone: nn.Module, hidden: int, use_confidence_head: bool = False, pool: str = "marker+span"):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.level = LevelIndexEmbedding(hidden)
        self.scorer = PointerScorer(hidden)
        self.conf = ConfidenceHead(hidden) if use_confidence_head else None

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        h = out.last_hidden_state
        h_opts = self.level(candidate_states(h, batch, self.pool), batch["level_index"])
        h_dec = h[torch.arange(h.size(0), device=h.device), batch["decide_positions"]]
        logits = self.scorer(h_dec, h_opts, batch["cand_mask"])
        res = {"logits": logits}
        if self.conf is not None:
            res["conf_logit"] = self.conf(h_dec, logits.softmax(-1), batch["cand_mask"])
        return res
