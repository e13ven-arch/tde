#!/usr/bin/env python
"""N-gram overlap between training states and benchmark items (JevBench public, typed-decisions test).
Reports, per training dataset, the share of benchmark items that share any 13-gram with a training state."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tde.schema import read_jsonl  # noqa: E402


def grams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/v0.2/train.jsonl")
    ap.add_argument("--jevbench", default=None, help="path to jevbench clone (datasets/public/*.jsonl)")
    ap.add_argument("--typed", default="data/v0.2/eval_only/typed_decisions.test.jsonl")
    ap.add_argument("--n", type=int, default=13)
    args = ap.parse_args()
    bench: dict[str, list[tuple[str, str]]] = {}
    if args.jevbench:
        items = []
        for p in sorted(Path(args.jevbench, "datasets/public").glob("*.jsonl")):
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                st = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])
                items.append((r["id"], st + " " + r["question"]["instructions"]))
        bench["jevbench_public"] = items
    if Path(args.typed).exists():
        bench["typed_decisions_test"] = [(e.id, e.state + " " + e.question) for e in read_jsonl(args.typed)]
    bench_grams = {b: [(i, grams(t, args.n)) for i, t in items] for b, items in bench.items()}
    hits: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    seen_states: dict[str, set] = defaultdict(set)
    for e in read_jsonl(args.train):
        if e.state in seen_states[e.dataset]:
            continue
        seen_states[e.dataset].add(e.state)
        g = grams(e.state, args.n)
        if not g:
            continue
        for b, items in bench_grams.items():
            for iid, ig in items:
                if g & ig:
                    hits[b][e.dataset].add(iid)
    for b, items in bench.items():
        print(f"== {b}: {len(items)} items")
        for ds, ids in sorted(hits[b].items(), key=lambda kv: -len(kv[1])):
            print(f"   {ds:20s} items sharing a {args.n}-gram: {len(ids)} ({100*len(ids)/len(items):.1f}%)")
        if not hits[b]:
            print("   no overlap")


if __name__ == "__main__":
    main()
