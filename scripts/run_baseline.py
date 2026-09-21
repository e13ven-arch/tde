#!/usr/bin/env python
"""Run a zero-training Track-B baseline on a split. Example:
    python scripts/run_baseline.py --baseline nli --limit 2000
    python scripts/run_baseline.py --baseline lm --model Qwen/Qwen3-0.6B --limit 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tde.baselines.common import report  # noqa: E402
from tde.schema import load_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["nli", "lm"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--data_dir", default="data/v0.1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    examples = load_jsonl(Path(args.data_dir) / f"{args.split}.jsonl", args.limit)
    if args.baseline == "nli":
        from tde.baselines.nli_zeroshot import NLIZeroShot
        m = NLIZeroShot(args.model or "MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c")
    else:
        from tde.baselines.lm_prefill import LMPrefill
        m = LMPrefill(args.model or "Qwen/Qwen3-0.6B")
    probs = m.predict(examples)
    name = f"{args.baseline}:{args.model or 'default'}"
    rep = report(examples, probs, name, args.out or f"runs/baselines/{args.baseline}_{args.split}.json")
    print(json.dumps({k: round(v, 4) for k, v in rep["all"].items() if isinstance(v, float)}, indent=2))


if __name__ == "__main__":
    main()
