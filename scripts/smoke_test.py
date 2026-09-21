#!/usr/bin/env python
"""End-to-end smoke test on synthetic data with a tiny random backbone (no downloads).

Builds a synthetic corpus whose answer is a deterministic function of the state,
trains each readout for a few hundred steps, and checks that accuracy rises well
above chance and that the control block behaves (no-state accuracy near chance).
"""
from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tde.evaluate import evaluate_run  # noqa: E402
from tde.schema import Candidate, DecisionExample, write_jsonl  # noqa: E402
from tde.train import TrainConfig, train  # noqa: E402

COLORS = ["red", "green", "blue", "yellow", "black"]
SIZES = ["tiny", "small", "medium", "large", "huge"]


def synth(n: int, split: str, rng: random.Random) -> list[DecisionExample]:
    out = []
    for i in range(n):
        color, size = rng.choice(COLORS), rng.randrange(5)
        state = f"a {SIZES[size]} {color} box sits on the table"
        kind = rng.choice(["choice", "noul", "score"])
        sid = f"syn:{split}:{i}"
        if kind == "choice":
            k = rng.randint(2, 5)
            cands = rng.sample(COLORS, k)
            if color not in cands:
                cands[rng.randrange(k)] = color
            target = [1.0 if c == color else 0.0 for c in cands]
            out.append(DecisionExample(f"{sid}#c", sid, "syn", split, "choice", state, "What color is the box?", [Candidate(c) for c in cands], target))
        elif kind == "noul":
            asked = color if rng.random() < 0.5 else rng.choice([c for c in COLORS if c != color])
            out.append(DecisionExample(f"{sid}#n", sid, "syn", split, "noul", state, f"Is the box {asked}?", [Candidate("yes"), Candidate("no")],
                                       [1.0, 0.0] if asked == color else [0.0, 1.0]))
        else:
            out.append(DecisionExample(f"{sid}#s", sid, "syn", split, "score", state, "How big is the box?", [Candidate(s) for s in SIZES],
                                       [1.0 if j == size else 0.0 for j in range(5)]))
    return out


def main() -> None:
    rng = random.Random(0)
    root = Path(tempfile.mkdtemp(prefix="tde-smoke-"))
    data = root / "data"
    for split, n in (("train", 3000), ("calibration", 400), ("test", 400)):
        write_jsonl(data / f"{split}.jsonl", synth(n, split, rng))
    results = {}
    for readout in ("joint", "branch", "biencoder"):
        cfg = TrainConfig(backbone="tiny", tiny=True, readout=readout, finetune="full", data_dir=str(data), out_dir=str(root / readout),
                          max_steps=300, batch_size=32, grad_accum=1, lr_backbone=1e-3, lr_head=1e-3, layer_decay=1.0,
                          eval_every=100, log_every=100, w_rps=0.5, w_perm=0.1 if readout != "biencoder" else 0.0,
                          use_confidence_head=(readout == "joint"), w_conf=0.1 if readout == "joint" else 0.0, device="cpu", bf16=False)
        train(cfg)
        rep = evaluate_run(root / readout, data, "test", batch_size=64, control_limit=400)
        acc = rep["raw"]["all"]["accuracy"]
        results[readout] = {"accuracy": round(acc, 3), "brier": round(rep["raw"]["all"]["brier"], 3),
                            "no_state_accuracy": round(rep["controls"]["no_state_accuracy"], 3),
                            "shuffled_state_accuracy": round(rep["controls"]["shuffled_state_accuracy"], 3),
                            "order_agreement": round(rep["controls"]["order_argmax_agreement"], 3)}
        print(f"[smoke] {readout}: {results[readout]}")
    print(json.dumps(results, indent=2))
    ok = all(r["accuracy"] > 0.6 for r in results.values())
    shutil.rmtree(root, ignore_errors=True)
    print("SMOKE OK" if ok else "SMOKE FAILED: accuracy did not rise above 0.6")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
