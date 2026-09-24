"""Figure 3: risk-coverage curves. (a) readout structures (Exp 001); (b) release general model per dataset."""
import json, glob, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def rc(rows):
    conf = np.array([r["conf"] for r in rows]); cor = np.array([r["correct"] for r in rows], dtype=float)
    o = np.argsort(-conf); cor = cor[o]
    cov = np.arange(1, len(cor) + 1) / len(cor); risk = 1 - np.cumsum(cor) / np.arange(1, len(cor) + 1)
    return cov, risk

fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
ax = axes[0]
for name, f, c in [("joint", "runs/host/rc_exp001_joint_full.json", "#2c3e50"), ("bi-encoder", "runs/host/rc_exp001_biencoder_full.json", "#27ae60"), ("branch", "runs/host/rc_exp001_branch_full.json", "#c0392b")]:
    d = json.load(open(f)); cov, risk = rc(d["rows"]); ax.plot(cov, risk, color=c, label=f"{name} (acc {np.mean([r['correct'] for r in d['rows']]):.3f})")
ax.axhline(0.05, color="grey", lw=0.6, ls=":"); ax.text(0.01, 0.055, "5% risk", fontsize=7, color="grey")
ax.set_xlabel("coverage"); ax.set_ylabel("risk (error rate among answered)"); ax.set_title("(a) readout structure, v0.1 test (6,000)", fontsize=9.5)
ax.set_xlim(0, 1); ax.set_ylim(0, 0.45); ax.legend(fontsize=8, frameon=False); ax.grid(lw=0.3, alpha=0.5)
ax = axes[1]
d = json.load(open("runs/host/rc_release_general.json")); rows = d["rows"]
for ds in ["banking77", "clinc150", "ag_news", "snli", "go_emotions", "boolq", "sst5"]:
    sub = [r for r in rows if r["dataset"] == ds]
    if len(sub) < 100: continue
    cov, risk = rc(sub); ax.plot(cov, risk, label=f"{ds} (n={len(sub)})")
ax.axhline(0.05, color="grey", lw=0.6, ls=":")
ax.set_xlabel("coverage"); ax.set_title("(b) released general model, per dataset", fontsize=9.5)
ax.set_xlim(0, 1); ax.set_ylim(0, 0.6); ax.legend(fontsize=7.5, frameon=False, ncol=2); ax.grid(lw=0.3, alpha=0.5)
fig.tight_layout(); fig.savefig("docs/paper/figures/fig3_risk_coverage.png", dpi=200, bbox_inches="tight"); fig.savefig("docs/paper/figures/fig3_risk_coverage.pdf", bbox_inches="tight"); print("ok")
