"""Causal-decoder readout (matched-data comparison with the encoder: Exp 002).

Same joint layout as JointDecisionModel, but the backbone is a causal LM
(e.g. Qwen3-0.6B) so a marker placed *before* a candidate cannot see its text.
Candidates are therefore read at the last token of their span, and the decision
query at the final [DECIDE] token, which sees everything. Pointer scoring,
ordinal level embedding and confidence head are shared with the encoder models
so the only difference under test is bidirectional-MLM vs causal-LM
representations at matched data (and, with LoRA, roughly matched trainable
parameters).
"""
from __future__ import annotations

import torch
from torch import nn

from tde.model.heads import ConfidenceHead, LevelIndexEmbedding, PointerScorer, gather_positions


class DecoderDecisionModel(nn.Module):
    mode = "joint"  # same encoding/collate as the joint encoder model

    def __init__(self, backbone: nn.Module, hidden: int, use_confidence_head: bool = False, pool: str = "span_end"):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.level = LevelIndexEmbedding(hidden)
        self.scorer = PointerScorer(hidden)
        self.conf = ConfidenceHead(hidden) if use_confidence_head else None

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        # last token of each candidate span (span_end is exclusive); padded slots point at position 0 and are masked
        end = (batch["span_end"] - 1).clamp_min(0)
        h_opts = self.level(gather_positions(h, end), batch["level_index"])
        h_dec = h[torch.arange(h.size(0), device=h.device), batch["decide_positions"]]
        logits = self.scorer(h_dec, h_opts, batch["cand_mask"])
        res = {"logits": logits}
        if self.conf is not None:
            res["conf_logit"] = self.conf(h_dec, logits.softmax(-1), batch["cand_mask"])
        return res
