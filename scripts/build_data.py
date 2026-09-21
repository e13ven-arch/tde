#!/usr/bin/env python
"""Build the decision corpus. Example: python scripts/build_data.py --stage 0 --out data/v0.1"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tde.data.build import build  # noqa: E402
from tde.data.registry import REGISTRY  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None, help=f"subset of {list(REGISTRY)}")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--out", default="data/v0.1")
    ap.add_argument("--max-rows", type=int, default=None, help="cap source rows per dataset (overrides registry caps)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    report = build(args.datasets, args.out, stage=args.stage, max_rows=args.max_rows, seed=args.seed)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
