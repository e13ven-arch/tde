"""Minimal inference API over a trained run: state + typed questions -> probability distributions.

    from tde.inference import Decider
    d = Decider.from_run("runs/exp001_joint_full")
    d.decide(state, {"type": "choice", "instructions": "...", "criteria": {"a": "desc", "b": "desc"}})
    -> {"probabilities": {"a": 0.8, "b": 0.2}, "label": "a", "confidence": 0.6}

Question format mirrors the typed-decision convention used by JevBench and typed-decisions:
  noul   criteria = {"true": ..., "false": ...} (optional)     -> probabilities over {"yes", "no"}
  choice criteria = {label: description}                        -> probabilities over labels
  score  criteria = [level descriptions]                         -> probabilities over "0".."n-1" + expected score
Several questions over one state are answered in one batched call; with the `branch` readout the state is
encoded once and shared.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tde.calibration.temperature import BucketTemperature
from tde.model.encoding import collate
from tde.schema import Candidate, DecisionExample
from tde.train import load_checkpoint


def _question_to_example(state: str, q: dict, qid: str = "q") -> DecisionExample:
    qtype = q["type"]
    crit = q.get("criteria")
    if qtype == "noul":
        crit = crit or {}
        cands = [Candidate("yes", str(crit.get("true", "") or "")), Candidate("no", str(crit.get("false", "") or ""))]
        prim = "noul"
    elif qtype == "score":
        cands = [Candidate(str(i), str(d)) for i, d in enumerate(crit or [])]
        prim = "score"
    else:
        if isinstance(crit, dict):
            cands = [Candidate(str(k), str(v) if v else "") for k, v in crit.items()]
        else:
            cands = [Candidate(str(k)) for k in (crit or q.get("options") or [])]
        prim = "choice"
    k = len(cands)
    return DecisionExample(id=qid, source_id="inference", dataset="inference", split="test", primitive=prim,
                           state=state if isinstance(state, str) else json.dumps(state, ensure_ascii=False),
                           question=str(q["instructions"]), candidates=cands, target=[1.0 / k] * k,
                           meta={"allow_empty_state": True})


def confidence_from_probs(p: np.ndarray, ordinal: bool) -> float:
    """TypeSafe-style concentration statistic (from the MIT system-one-adapter): choice = (max-1/K)/(1-1/K);
    score = 1 - mean absolute level distance from the argmax / that of the uniform distribution."""
    k = len(p)
    if k <= 1:
        return 1.0
    if not ordinal:
        return float((p.max() - 1.0 / k) / (1.0 - 1.0 / k))
    m = int(p.argmax())
    idx = np.arange(k)
    d = float((p * np.abs(idx - m)).sum())
    d_uniform = float(np.abs(idx - (k - 1) / 2).mean())
    return float(max(0.0, 1.0 - d / d_uniform)) if d_uniform > 0 else 1.0


class Decider:
    def __init__(self, dtok, model, cfg: dict, device: torch.device, temperature: BucketTemperature | None = None, batch_size: int = 16):
        self.dtok, self.model, self.cfg, self.device, self.temperature, self.batch_size = dtok, model, cfg, device, temperature, batch_size

    @classmethod
    def from_run(cls, run_dir: str | Path, device: str | None = None, use_temperature: bool = True, batch_size: int = 16,
                 revision: str | None = None) -> "Decider":
        """`run_dir` is a local run / release directory or a Hugging Face model id (e.g. "tdelab/tde-general-v0.1")."""
        if not Path(run_dir).exists() and "/" in str(run_dir) and not str(run_dir).startswith((".", "/")):
            from huggingface_hub import snapshot_download
            run_dir = snapshot_download(str(run_dir), revision=revision, allow_patterns=["config.json", "best.pt", "tokenizer/*", "temperature.json", "MANIFEST.json"])
        run_dir = Path(run_dir)
        dtok, model, cfg, dev = load_checkpoint(run_dir, torch.device(device) if device else None)
        temp = None
        tpath = run_dir / "temperature.json"
        if use_temperature and tpath.exists():
            temp = BucketTemperature.load(tpath)
        return cls(dtok, model, cfg, dev, temp, batch_size)

    @torch.no_grad()
    def decide_batch(self, state, questions: dict[str, dict]) -> dict[str, dict]:
        exs = [_question_to_example(state, q, qid) for qid, q in questions.items()]
        mode = self.model.mode
        out: dict[str, dict] = {}
        for i in range(0, len(exs), self.batch_size):
            chunk = exs[i : i + self.batch_size]
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                     for k, v in collate([self.dtok.encode(e, mode) for e in chunk], self.dtok.pad_id, mode).items()}
            use_amp = self.device.type == "cuda"
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = self.model(batch)["logits"].float().cpu().numpy()
            for j, e in enumerate(chunk):
                z = logits[j, : e.k]
                if self.temperature is not None:
                    p = self.temperature.apply(z, e.primitive)
                else:
                    z = z - z.max(); p = np.exp(z); p = p / p.sum()
                names = [c.name for c in e.candidates]
                rec = {"type": e.primitive, "probabilities": {n: float(x) for n, x in zip(names, p)},
                       "label": names[int(p.argmax())], "confidence": confidence_from_probs(p, e.primitive == "score")}
                if e.primitive == "noul":
                    rec["noul"] = float(p[0])
                if e.primitive == "score":
                    rec["score"] = float((p * np.arange(e.k)).sum())
                out[e.id] = rec
        return out

    def decide(self, state, question: dict) -> dict:
        return self.decide_batch(state, {"q": question})["q"]
