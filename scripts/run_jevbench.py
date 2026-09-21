#!/usr/bin/env python
"""Run JevBench public items against a TDE run without modifying the benchmark repo.
    python scripts/run_jevbench.py --jevbench /path/to/jevbench --run runs/exp001_joint_full --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jevbench", required=True, help="path to a clone of fstandhartinger/jevbench")
    ap.add_argument("--run", required=True)
    ap.add_argument("--tasks", default=None, help="comma-separated jsonl; default: all datasets/public/*.jsonl")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-temperature", action="store_true")
    args = ap.parse_args()
    jb = Path(args.jevbench).resolve()
    sys.path.insert(0, str(jb))
    sys.path.insert(0, str(ROOT / "integrations" / "jevbench"))
    from jevbench.budget import Ledger
    from jevbench.cli import _load_tasks
    from jevbench.runner import Runner
    from jevbench.summarize import summarize
    from tde_local import TdeLocalAdapter

    tasks_spec = args.tasks or ",".join(str(p) for p in sorted((jb / "datasets" / "public").glob("*.jsonl")))
    tasks = _load_tasks(tasks_spec)
    out = Path(args.out or ROOT / "runs" / "jevbench" / Path(args.run).name)
    out.mkdir(parents=True, exist_ok=True)
    adapter = TdeLocalAdapter(endpoint=args.run, model=f"tde:{Path(args.run).name}", device=args.device,
                              use_temperature=not args.no_temperature)
    t0 = time.perf_counter(); adapter.load(); print(f"[jevbench] warm load {time.perf_counter() - t0:.1f}s")
    ledger = Ledger(str(out / "ledger.jsonl"), cap_usd=0.0)
    results_path = out / f"results_{int(time.time())}.jsonl"
    runner = Runner(adapter, ledger, raw_dir=str(out / "raw"), default_reserve_usd=0.0)
    records = runner.run_all(tasks, results_path=str(results_path))
    failed = sum(1 for r in records if r["status"] == "failed")
    print(f"[jevbench] {len(records)}/{len(tasks)} attempted, {failed} failed")
    summary = summarize(tasks, records, 0.0, headline_only=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    keep = {k: v for k, v in summary.items() if not isinstance(v, (dict, list))}
    print(json.dumps(keep, indent=2, sort_keys=True, default=str))
    for k, v in summary.items():
        if isinstance(v, dict) and k in ("by_family", "families", "by_split", "overall"):
            print(k, json.dumps(v, indent=1, default=str)[:1500])
    print(f"[jevbench] saved {out}/summary.json")


if __name__ == "__main__":
    main()
