"""Evaluation harness: per-slice metrics, post-hoc temperature, and the control block.

Control block (reported for every evaluation):
    no_state          question + candidates only (state blanked)        -> partial-input accuracy
    shuffled_state    each item gets another item's state (same dataset) -> mismatched-input accuracy
    order_consistency candidates randomly permuted                        -> argmax agreement, mean TV distance
    held_out_templates items rendered with templates never seen in training
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from tde.calibration.metrics import Predictions, summarize
from tde.calibration.temperature import BucketTemperature
from tde.schema import DecisionExample, k_bucket, load_jsonl
from tde.train import load_checkpoint, predict


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def _preds(examples: list[DecisionExample], logits: list[np.ndarray], temp: BucketTemperature | None) -> Predictions:
    probs = [temp.apply(z, e.primitive) if temp else _softmax(z) for z, e in zip(logits, examples)]
    return Predictions(probs, [np.array(e.target) for e in examples], groups=[e.source_id for e in examples],
                       ordinal=[e.primitive == "score" for e in examples])


def _slices(examples: list[DecisionExample]) -> dict[str, list[int]]:
    s: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(examples):
        s["all"].append(i)
        s[f"dataset={e.dataset}"].append(i)
        s[f"primitive={e.primitive}"].append(i)
        s[f"kbucket={k_bucket(e.k)}"].append(i)
        tg = e.meta.get("template_group")
        if tg:
            s[f"templates={tg}"].append(i)
        ls = e.meta.get("label_source")
        if ls:
            s[f"labels={ls}"].append(i)
    return s


def _sub(P: Predictions, idx: list[int]) -> Predictions:
    return Predictions([P.probs[i] for i in idx], [P.targets[i] for i in idx],
                       groups=[P.groups[i] for i in idx] if P.groups else None,
                       ordinal=[P.ordinal[i] for i in idx] if P.ordinal else None)


def control_block(model, dtok, examples: list[DecisionExample], device, batch_size: int, base_logits: list[np.ndarray], seed: int = 0) -> dict:
    rng = random.Random(seed)
    base_pred = np.array([int(z.argmax()) for z in base_logits])
    gold = np.array([e.label for e in examples])
    out: dict = {"n": len(examples), "base_accuracy": float((base_pred == gold).mean())}
    # no state
    blank = [DecisionExample(**{**e.to_dict(), "state": "", "candidates": e.candidates, "meta": dict(e.meta, allow_empty_state=True)}) for e in examples]
    lz = predict(model, dtok, blank, device, batch_size)
    out["no_state_accuracy"] = float((np.array([int(z.argmax()) for z in lz]) == gold).mean())
    # shuffled state within dataset
    by_ds: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(examples):
        by_ds[e.dataset].append(i)
    donor = list(range(len(examples)))
    for idx in by_ds.values():
        if len(idx) > 1:
            shuffled = idx[:]
            rng.shuffle(shuffled)
            for a, b in zip(idx, shuffled):
                donor[a] = b
    mixed = [DecisionExample(**{**e.to_dict(), "state": examples[donor[i]].state, "candidates": e.candidates}) for i, e in enumerate(examples)]
    lz = predict(model, dtok, mixed, device, batch_size)
    out["shuffled_state_accuracy"] = float((np.array([int(z.argmax()) for z in lz]) == gold).mean())
    # order consistency
    perms = []
    permuted = []
    for e in examples:
        order = list(range(e.k)); rng.shuffle(order)
        perms.append(order); permuted.append(e.with_candidate_order(order))
    lz = predict(model, dtok, permuted, device, batch_size)
    agree, tv = [], []
    for z0, z1, order in zip(base_logits, lz, perms):
        p0, p1 = _softmax(z0), _softmax(z1)
        p1_aligned = np.zeros_like(p0)
        for j, o in enumerate(order):
            p1_aligned[o] = p1[j]
        agree.append(int(p0.argmax() == p1_aligned.argmax()))
        tv.append(0.5 * float(np.abs(p0 - p1_aligned).sum()))
    out["order_argmax_agreement"] = float(np.mean(agree))
    out["order_mean_tv"] = float(np.mean(tv))
    return out


def evaluate_run(run_dir: str | Path, data_dir: str | Path, split: str = "test", limit: int | None = None,
                 batch_size: int = 32, controls: bool = True, fit_temperature: bool = True, control_limit: int = 2000) -> dict:
    dtok, model, cfg, device = load_checkpoint(run_dir)
    data_dir = Path(data_dir)
    split_path = Path(split) if split.endswith(".jsonl") else data_dir / f"{split}.jsonl"  # split name or a jsonl path
    examples = load_jsonl(split_path, limit)
    logits = predict(model, dtok, examples, device, batch_size)
    report: dict = {"run": str(run_dir), "split": split, "n": len(examples), "readout": cfg["readout"], "backbone": cfg["backbone"]}
    temp = None
    if fit_temperature and (data_dir / "calibration.jsonl").exists() and split != "calibration":
        cal = load_jsonl(data_dir / "calibration.jsonl", limit)
        cal_logits = predict(model, dtok, cal, device, batch_size)
        temp = BucketTemperature().fit(cal_logits, [np.array(e.target) for e in cal], [e.primitive for e in cal])
        report["temperature"] = temp.table
    P_raw = _preds(examples, logits, None)
    slices = _slices(examples)
    report["raw"] = {name: summarize(_sub(P_raw, idx)) for name, idx in slices.items()}
    if temp:
        P_ts = _preds(examples, logits, temp)
        report["temperature_scaled"] = {name: summarize(_sub(P_ts, idx)) for name, idx in slices.items()}
    if controls:
        sub = examples[:control_limit]
        report["controls"] = control_block(model, dtok, sub, device, batch_size, logits[:control_limit])
    return report


def save_report(report: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2))
