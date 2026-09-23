#!/usr/bin/env python
"""Paired comparison of run directories on one split, plus two candidate-set probes.

    python scripts/paired_eval.py --runs runs/exp016_seq runs/exp016_pointwise runs/exp016_set \
        --split data/v0.1/eval_only/full_k.test.jsonl --limit 6000 --max_total 4096 --out runs/exp016_fullk_paired.json
    python scripts/paired_eval.py --runs ... --split data/v0.1/test.jsonl --limit 6000 --probes --out runs/exp016_test_paired.json

The first run is the reference. For each other run: accuracy and NLL differences with a 95% bootstrap CI that
resamples source items (rewrites of one source move together), and an exact McNemar p-value on accuracy.
--probes adds, per run, on non-score items: argmax agreement after a random candidate reordering, and the
drop-one effect (total variation between p over S \\ {c} and p over S renormalised to S \\ {c}, c a random
non-gold candidate; exactly 0 for a readout that satisfies IIA).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tde.schema import load_jsonl  # noqa: E402
from tde.train import load_checkpoint, predict, predict_chunked  # noqa: E402


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    p = np.exp(z - z.max())
    return p / p.sum()


def _mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    n01, n10 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _bootstrap(diff: np.ndarray, groups: list[str], reps: int = 10000, seed: int = 0) -> tuple[float, float]:
    ids = {g: i for i, g in enumerate(dict.fromkeys(groups))}
    gi = np.array([ids[g] for g in groups])
    sums, counts = np.bincount(gi, diff), np.bincount(gi)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(reps, len(sums)))
    stats = sums[draws].sum(1) / counts[draws].sum(1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _probes(model, dtok, examples, device, batch_size: int, base: list[np.ndarray], seed: int = 0, bf16: bool = True) -> dict:
    rng = random.Random(seed)
    items = [i for i, e in enumerate(examples) if e.primitive != "score" and e.k >= 3]
    orders = {i: rng.sample(range(examples[i].k), examples[i].k) for i in items}
    zp = predict(model, dtok, [examples[i].with_candidate_order(orders[i]) for i in items], device, batch_size, bf16)
    agree = [int(np.argmax(base[i]) == orders[i][int(np.argmax(z))]) for i, z in zip(items, zp)]
    keep = {}
    for i in items:
        e = examples[i]
        drop = rng.choice([j for j in range(e.k) if j != e.label])
        keep[i] = [j for j in range(e.k) if j != drop]
    zd = predict(model, dtok, [examples[i].with_candidate_order(keep[i]) for i in items], device, batch_size, bf16)
    tv = []
    for i, z in zip(items, zd):
        p_full = _softmax(base[i])[keep[i]]
        tv.append(0.5 * float(np.abs(p_full / p_full.sum() - _softmax(z)).sum()))
    return {"n": len(items), "order_argmax_agreement": float(np.mean(agree)), "drop_one_mean_tv": float(np.mean(tv)),
            "drop_one_p90_tv": float(np.percentile(tv, 90))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_total", type=int, default=None)
    ap.add_argument("--chunk_k", type=int, default=None)
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="no bf16 autocast (exactness checks)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    examples = load_jsonl(args.split, args.limit)
    gold = np.array([e.label for e in examples])
    groups = [e.source_id for e in examples]
    per_run, report = {}, {"split": args.split, "n": len(examples), "max_total": args.max_total, "chunk_k": args.chunk_k, "fp32": args.fp32, "runs": {}}
    for run in args.runs:
        dtok, model, cfg, device = load_checkpoint(run, max_total=args.max_total)
        bf16 = not args.fp32
        logits = (predict_chunked(model, dtok, examples, device, args.chunk_k, args.batch_size, bf16) if args.chunk_k
                  else predict(model, dtok, examples, device, args.batch_size, bf16))
        probs = [_softmax(z) for z in logits]
        correct = (np.array([int(p.argmax()) for p in probs]) == gold).astype(float)
        nll = np.array([-(np.asarray(e.target) * np.log(np.clip(p, 1e-6, 1))).sum() for e, p in zip(examples, probs)])
        per_run[run] = (correct, nll)
        rec = {"topology": cfg.get("topology", "seq"), "accuracy": float(correct.mean()), "nll": float(nll.mean())}
        if args.probes:
            rec["probes"] = _probes(model, dtok, examples, device, args.batch_size, logits, bf16=bf16)
        report["runs"][run] = rec
        print(run, json.dumps(rec), flush=True)
        del model
    ref = args.runs[0]
    report["paired_vs_" + ref] = {}
    for run in args.runs[1:]:
        (ca, na), (cb, nb) = per_run[ref], per_run[run]
        d_acc, d_nll = cb - ca, nb - na
        report["paired_vs_" + ref][run] = {"d_accuracy": float(d_acc.mean()), "d_accuracy_ci95": _bootstrap(d_acc, groups),
                                            "d_nll": float(d_nll.mean()), "d_nll_ci95": _bootstrap(d_nll, groups),
                                            "mcnemar_p": _mcnemar(ca, cb)}
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["paired_vs_" + ref], indent=2))


if __name__ == "__main__":
    main()
