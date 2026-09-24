"""Figure 1: (a) joint sequence layout; (b) attention masks + position ids for seq / pointwise / set on a toy example."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
from types import SimpleNamespace
from tde.model.joint import set_masks
from tde.model.encoding import _tied_layout

# toy: prefix 6 tokens ([CLS] s s s [SEP] q [SEP] -> we use 7), 3 candidates of 2-3 tokens, [DECIDE] + [SEP]
prefix = ["[CLS]", "state₁", "state₂", "[SEP]", "q₁", "q₂", "[SEP]"]
cands = [["⟨opt⟩", "c₁"], ["⟨opt⟩", "c₂", "c₂'"], ["⟨opt⟩", "c₃"]]
tail = ["⟨decide⟩", "[SEP]"]
toks = prefix + sum(cands, []) + tail
P = len(prefix)
opt_pos = []; i = 0
for c in cands:
    opt_pos.append(i); i += len(c)
decide_pos = i
max_block = 3
block, tied = _tied_layout(P, opt_pos, decide_pos, len(tail), max_block)
L = len(toks)
seq_pos = list(range(L))
cfg = SimpleNamespace(_attn_implementation="sdpa", sliding_window=64)
batch = {"block_ids": torch.tensor([block]), "position_ids": torch.tensor([tied]), "attention_mask": torch.ones(1, L, dtype=torch.long)}
masks = {"seq": np.ones((L, L), dtype=bool)}
for t in ("pointwise", "set"):
    masks[t] = set_masks(batch, t, cfg)["full_attention"][0, 0].numpy()
positions = {"seq": seq_pos, "pointwise": tied, "set": tied}

fig = plt.figure(figsize=(12, 5.2))
gs = fig.add_gridspec(2, 3, height_ratios=[0.55, 3.2], hspace=0.45, wspace=0.25)
# (a) layout strip
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, L); ax.set_ylim(0, 1); ax.axis("off")
colors = {-1: "#dfe7f1", -2: "#f6d6a8"}
for j, (tk, b) in enumerate(zip(toks, block)):
    c = colors.get(b, ["#cfe8cf", "#bfe0bf", "#a9d8a9"][b % 3])
    ax.add_patch(plt.Rectangle((j, 0.15), 0.96, 0.7, color=c, ec="white"))
    ax.text(j + 0.48, 0.5, tk, ha="center", va="center", fontsize=8.5)
ax.text(0, 0.98, "(a) joint sequence: prefix (state, question) · candidate blocks, each opened by a ⟨opt⟩ = [MASK] marker · ⟨decide⟩ = [MASK]", fontsize=9, va="bottom")
ax.annotate("pointer score  sᵢ = q(h⟨decide⟩)·k(h⟨opt⟩ᵢ)/√d", xy=(decide_pos + P + 0.5, 0.15), xytext=(L * 0.55, -0.35),
            fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=0.8), annotation_clip=False)
# (b) masks
for k, t in enumerate(("seq", "pointwise", "set")):
    ax = fig.add_subplot(gs[1, k])
    m = masks[t].astype(float)
    ax.imshow(m, cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
    ax.set_xticks(range(L)); ax.set_yticks(range(L))
    ax.set_xticklabels([str(p) for p in positions[t]], fontsize=6.5)
    ax.set_yticklabels(toks, fontsize=6.5)
    ax.set_xlabel("key (label = position id)", fontsize=8)
    if k == 0: ax.set_ylabel("query token", fontsize=8)
    ax.set_title({"seq": "seq: sequential positions, full attention",
                  "pointwise": "pointwise: tied positions, block-diagonal",
                  "set": "set: pointwise + ⟨opt⟩↔⟨opt⟩, ⟨decide⟩→⟨opt⟩"}[t], fontsize=8.5)
    for b_edge in [P - 0.5, P + decide_pos - 0.5]:
        ax.axhline(b_edge, color="#c0392b", lw=0.6); ax.axvline(b_edge, color="#c0392b", lw=0.6)
fig.text(0.5, 0.02, "(b) attention masks (dark = attends); position ids on the x axis. Under the tied layouts every candidate block starts at position P and ⟨decide⟩ sits at P + B, whatever K.", ha="center", fontsize=8.5)
fig.savefig("docs/paper/figures/fig1_layout_topology.png", dpi=200, bbox_inches="tight")
fig.savefig("docs/paper/figures/fig1_layout_topology.pdf", bbox_inches="tight")
print("ok", L, block, tied)
