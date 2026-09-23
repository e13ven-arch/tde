"""Joint-sequence readout: one encoder pass over [state; question; candidates], pointer scores.

topology selects how candidates are laid out inside the encoder:
    seq        candidates in sequence with ordinary position ids and full attention (the default)
    pointwise  every candidate block starts at the same position id; a candidate sees the prefix and itself,
               the prefix and [DECIDE] see only the prefix. Logits are per-candidate functions, so the output is
               permutation-equivariant and independent of which other candidates are present.
    set        pointwise plus [OPT] markers that see every marker, and a [DECIDE] that sees every marker:
               candidates interact, output still permutation-equivariant.
The two tied topologies need a backbone that accepts position ids and per-layer-type masks (ModernBERT).
"""
from __future__ import annotations

import torch
from torch import nn

from tde.model.heads import ConfidenceHead, LevelIndexEmbedding, PointerScorer, candidate_states, gather_positions

TOPOLOGIES = ("seq", "pointwise", "set")


def set_masks(batch: dict[str, torch.Tensor], topology: str, config) -> dict[str, torch.Tensor]:
    """{"full_attention", "sliding_attention"} masks [B,1,L,L] for the tied topologies.

    The sliding window is measured in position ids, not sequence index, so every candidate block sees the same
    stretch of prefix whatever its place in the sequence."""
    impl = getattr(config, "_attn_implementation", "sdpa")
    if impl not in ("sdpa", "eager"):
        raise ValueError(f"topology={topology!r} needs sdpa or eager attention, got {impl!r}")
    block, pos, valid = batch["block_ids"], batch["position_ids"], batch["attention_mask"].bool()
    q, k = block[:, :, None], block[:, None, :]
    allowed = (k == -1) | ((q == k) & (q != -3))  # all tokens see the prefix; candidate / reader tokens see their own group
    if topology == "set":
        cand = block >= 0
        marker = cand & torch.cat([torch.ones_like(cand[:, :1]), block[:, 1:] != block[:, :-1]], dim=1)  # first token of a block
        allowed |= (marker[:, :, None] | (q == -2)) & marker[:, None, :]
    elif topology != "pointwise":
        raise ValueError(topology)
    allowed &= valid[:, None, :]
    local = allowed & ((pos[:, :, None] - pos[:, None, :]).abs() <= config.sliding_window)
    masks = {"full_attention": allowed[:, None], "sliding_attention": local[:, None]}
    if impl == "eager":  # eager attention adds the mask to the scores
        masks = {t: torch.zeros(m.shape, device=m.device).masked_fill(~m, torch.finfo(torch.float32).min) for t, m in masks.items()}
    return masks


class JointDecisionModel(nn.Module):
    mode = "joint"

    def __init__(self, backbone: nn.Module, hidden: int, use_confidence_head: bool = False, pool: str = "marker+span",
                 topology: str = "seq"):
        super().__init__()
        if topology not in TOPOLOGIES:
            raise ValueError(topology)
        self.backbone = backbone
        self.pool = pool
        self.topology = topology
        self.level = LevelIndexEmbedding(hidden)
        self.scorer = PointerScorer(hidden)
        self.conf = ConfidenceHead(hidden) if use_confidence_head else None

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.topology == "seq":
            out = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        else:
            out = self.backbone(input_ids=batch["input_ids"], attention_mask=set_masks(batch, self.topology, self.backbone.config),
                                position_ids=batch["position_ids"])
        h = out.last_hidden_state
        h_opts = self.level(candidate_states(h, batch, self.pool), batch["level_index"])
        h_dec = h[torch.arange(h.size(0), device=h.device), batch["decide_positions"]]
        logits = self.scorer(h_dec, h_opts, batch["cand_mask"])
        res = {"logits": logits}
        if self.conf is not None:
            res["conf_logit"] = self.conf(h_dec, logits.softmax(-1), batch["cand_mask"])
        return res
