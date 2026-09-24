"""Dump per-item (confidence, correct, dataset, primitive) for risk-coverage curves.

    python scripts/dump_confidences.py --run runs/exp001_joint_full --data_dir data/v0.1 --limit 6000 --out runs/rc_joint.json
"""
from __future__ import annotations
import argparse, json
import numpy as np
from tde.schema import load_jsonl
from tde.train import load_checkpoint, predict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test"); ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--batch_size", type=int, default=32); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dtok, model, cfg, device = load_checkpoint(a.run)
    ex = load_jsonl(f"{a.data_dir}/{a.split}.jsonl", a.limit)
    logits = predict(model, dtok, ex, device, a.batch_size)
    rows = []
    for e, z in zip(ex, logits):
        z = np.asarray(z, dtype=np.float64); p = np.exp(z - z.max()); p /= p.sum()
        rows.append({"conf": float(p.max()), "correct": int(int(p.argmax()) == int(np.argmax(e.target))),
                     "dataset": e.dataset,
                     "primitive": e.primitive})
    json.dump({"run": a.run, "n": len(rows), "rows": rows}, open(a.out, "w"))
    acc = np.mean([r["correct"] for r in rows]); print(f"{a.run}: n={len(rows)} acc={acc:.4f}")

if __name__ == "__main__":
    main()
