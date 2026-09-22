#!/usr/bin/env python
"""Average the probability outputs of several runs on one jsonl and report metrics (plus each member alone).
    python scripts/ensemble_eval.py --runs runs/host/exp011_typed_specialist_v2 runs/host/exp012_typed_specialist --split data/typed_only/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tde.calibration.metrics import Predictions, summarize  # noqa: E402
from tde.schema import load_jsonl  # noqa: E402
from tde.train import load_checkpoint, predict  # noqa: E402


def softmax(z):
    z = z - z.max(); p = np.exp(z); return p / p.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    examples = load_jsonl(args.split, args.limit)
    targets = [np.array(e.target) for e in examples]
    ordinal = [e.primitive == "score" for e in examples]
    members = {}
    for r in args.runs:
        import torch
        dtok, model, cfg, device = load_checkpoint(r, torch.device(args.device) if args.device else None)
        probs = [softmax(z) for z in predict(model, dtok, examples, device, args.batch_size)]
        members[r] = probs
        s = summarize(Predictions(probs, targets, ordinal=ordinal))
        print(f"{Path(r).name:36s} acc {s['accuracy']:.3f} brier {s['brier']:.3f} nll {s['nll']:.3f} ece {s['ece']:.3f}")
        del model
    avg = [np.mean([members[r][i] for r in members], axis=0) for i in range(len(examples))]
    s = summarize(Predictions(avg, targets, ordinal=ordinal))
    print(f"{'ENSEMBLE(' + str(len(members)) + ')':36s} acc {s['accuracy']:.3f} brier {s['brier']:.3f} nll {s['nll']:.3f} ece {s['ece']:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"members": list(members), "ensemble": s}, indent=2))


if __name__ == "__main__":
    main()
