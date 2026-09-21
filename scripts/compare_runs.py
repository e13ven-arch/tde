#!/usr/bin/env python
"""Side-by-side slice table for several run directories (local copies under runs/host/<run>/).
    python scripts/compare_runs.py --runs exp001s1_joint_full exp005_breadth_joint --report eval_test
    python scripts/compare_runs.py --runs ... --report eval_typed_decisions.test --metrics accuracy nll cov@risk5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--root", default="runs/host")
    ap.add_argument("--report", default="eval_test")
    ap.add_argument("--metrics", nargs="+", default=["accuracy", "nll", "cov@risk5"])
    ap.add_argument("--prefix", default=None, help="only slices starting with this (e.g. dataset=)")
    ap.add_argument("--ts", action="store_true", help="use temperature-scaled block")
    args = ap.parse_args()
    reps = {}
    for r in args.runs:
        p = Path(args.root) / r / f"{args.report}.json"
        if not p.exists():
            print(f"missing: {p}")
            continue
        d = json.load(open(p))
        reps[r] = d["temperature_scaled"] if args.ts and "temperature_scaled" in d else d["raw"]
    if not reps:
        return
    slices = sorted(set().union(*[set(v) for v in reps.values()]), key=lambda s: (s != "all", s))
    if args.prefix:
        slices = [s for s in slices if s == "all" or s.startswith(args.prefix)]
    head = "| slice | " + " | ".join(reps) + " |"
    print(head); print("|---|" + "---|" * len(reps))
    for s in slices:
        cells = []
        for r, v in reps.items():
            m = v.get(s)
            cells.append(" / ".join(f"{m[k]:.3f}" if isinstance(m.get(k), float) else str(m.get(k)) for k in args.metrics) if m else "-")
        print(f"| {s} | " + " | ".join(cells) + " |")
    print(f"\nmetrics: {' / '.join(args.metrics)}")


if __name__ == "__main__":
    main()
