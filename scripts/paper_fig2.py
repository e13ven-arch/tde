"""Figure 2: what data buys — endpoints across data versions (Table 7 of the paper)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
labels = ["Stage 1\nv0.1", "Exp 005\nv0.2", "Exp 006\nv0.3", "Exp 008\nv0.3+init", "Exp 009\nv0.4", "Exp 010\nv0.5", "Exp 015\nv0.6"]
hard = [28.8, 33.3, 30.6, 35.1, 33.3, 34.2, 33.3]
typed = [36.1, 62.1, 61.7, 65.7, 67.2, 66.6, 65.8]
b77 = [63.0, 57.1, 72.7, 73.0, 73.2, 75.8, np.nan]
c150 = [64.6, 62.4, 79.6, 77.4, 78.9, 86.5, 86.6]
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.fill_between(x, 25, 44, color="#f2f2f2", label="JevBench hard, 111 items: ±9 pt 95% band")
ax.plot(x, hard, "o-", color="#c0392b", label="JevBench hard (public, self-run)")
ax.plot(x, typed, "s-", color="#2c3e50", label="typed-decisions test, mixture mode")
ax.plot(x, b77, "^-", color="#27ae60", label="banking77, all 77 labels, one pass")
ax.plot(x, c150, "v-", color="#16a085", label="clinc150, all 150 labels, one pass")
for ref, name, c in [(34.1, "Laya 421M", "#c0392b"), (38.2, "openJev-verdict", "#c0392b"), (40.0, "kev 0.6B", "#c0392b")]:
    ax.axhline(ref, color=c, lw=0.6, ls=":", alpha=0.7); ax.text(0.02, ref + 0.4, name, fontsize=7, color=c, ha="left")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("accuracy (%)"); ax.set_ylim(25, 100)
ax.legend(fontsize=7.5, loc="upper left", frameon=False, ncol=1); ax.grid(axis="y", lw=0.3, alpha=0.5)
ax.set_title("Data versions: in-distribution and full-label endpoints move; the hard tier does not", fontsize=9.5)
fig.savefig("docs/paper/figures/fig2_data_versions.png", dpi=200, bbox_inches="tight")
fig.savefig("docs/paper/figures/fig2_data_versions.pdf", bbox_inches="tight"); print("ok")
