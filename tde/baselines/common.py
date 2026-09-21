"""Shared helpers for zero-training baselines: they all return per-example probability vectors."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tde.calibration.metrics import Predictions, summarize
from tde.schema import DecisionExample


def report(examples: list[DecisionExample], probs: list[np.ndarray], name: str, out: str | Path | None = None) -> dict:
    P = Predictions(probs, [np.array(e.target) for e in examples], groups=[e.source_id for e in examples],
                    ordinal=[e.primitive == "score" for e in examples])
    rep = {"baseline": name, "n": len(examples), "all": summarize(P)}
    by_ds: dict[str, list[int]] = {}
    for i, e in enumerate(examples):
        by_ds.setdefault(e.dataset, []).append(i)
    rep["by_dataset"] = {d: summarize(Predictions([probs[i] for i in idx], [P.targets[i] for i in idx])) for d, idx in by_ds.items()}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rep, indent=2))
    return rep
