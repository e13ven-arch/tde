"""Cached-state + per-question branch readout.

The state is encoded once by the backbone (memory H). Each question is a short
branch sequence [question; [OPT] c_1; ...; [DECIDE]] processed by a small
transformer decoder (self-attention inside the branch + cross-attention to H),
then read out with the same pointer scorer as the joint model. Several
questions over one state share H (collate de-duplicates identical states).
"""
from __future__ import annotations

import torch
from torch import nn

from tde.model.heads import ConfidenceHead, LevelIndexEmbedding, PointerScorer, gather_positions


class BranchDecisionModel(nn.Module):
    mode = "branch"

    def __init__(self, backbone: nn.Module, hidden: int, n_layers: int = 3, n_heads: int = 8, max_branch_len: int = 1024,
                 use_confidence_head: bool = False, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.embed = backbone.get_input_embeddings()  # shared token embeddings
        self.pos = nn.Embedding(max_branch_len, hidden)
        nn.init.normal_(self.pos.weight, std=0.02)
        layer = nn.TransformerDecoderLayer(d_model=hidden, nhead=n_heads, dim_feedforward=4 * hidden, dropout=dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.branch = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.in_norm = nn.LayerNorm(hidden)
        self.level = LevelIndexEmbedding(hidden)
        self.scorer = PointerScorer(hidden)
        self.conf = ConfidenceHead(hidden) if use_confidence_head else None

    def encode_state(self, state_ids: torch.Tensor, state_mask: torch.Tensor) -> torch.Tensor:
        return self.backbone(input_ids=state_ids, attention_mask=state_mask).last_hidden_state

    def forward(self, batch: dict[str, torch.Tensor], memory: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if memory is None:
            memory = self.encode_state(batch["state_ids"], batch["state_mask"])
        mem = memory[batch["state_index"]]                       # [B, Ls, D]
        mem_pad = batch["state_mask"][batch["state_index"]] == 0  # True = pad
        ids = batch["input_ids"]
        x = self.embed(ids) + self.pos(torch.arange(ids.size(1), device=ids.device)).unsqueeze(0)
        x = self.in_norm(x)
        tgt_pad = batch["attention_mask"] == 0
        h = self.branch(tgt=x, memory=mem, tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=mem_pad)
        h_opts = self.level(gather_positions(h, batch["opt_positions"]), batch["level_index"])
        h_dec = h[torch.arange(h.size(0), device=h.device), batch["decide_positions"]]
        logits = self.scorer(h_dec, h_opts, batch["cand_mask"])
        res = {"logits": logits}
        if self.conf is not None:
            res["conf_logit"] = self.conf(h_dec, logits.softmax(-1), batch["cand_mask"])
        return res
