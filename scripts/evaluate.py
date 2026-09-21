#!/usr/bin/env python
"""Evaluate a run directory on a split with slices, temperature scaling and the control block."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tde.evaluate import evaluate_run, save_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data_dir", default="data/v0.1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--no-temperature", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = evaluate_run(args.run, args.data_dir, args.split, args.limit, args.batch_size,
                       controls=not args.no_controls, fit_temperature=not args.no_temperature)
    out = args.out or str(Path(args.run) / f"eval_{Path(args.split).stem if args.split.endswith(chr(46)+chr(106)+chr(115)+chr(111)+chr(110)+chr(108)) else args.split}.json")
    save_report(rep, out)
    brief = {k: {m: round(v, 4) for m, v in s.items() if m in ("n", "accuracy", "brier", "nll", "ece", "ece_noise_floor", "aurc")} for k, s in rep["raw"].items()}
    print(json.dumps({"raw": brief, "controls": rep.get("controls"), "saved": out}, indent=2))


if __name__ == "__main__":
    main()
