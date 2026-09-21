"""Unified decision example schema and JSONL IO.

One primitive covers all three question types:

    P(candidate_i | state, question, {candidate_j})

`noul` has exactly two candidates, `choice` 2..255, `score` 2..10 ordered levels.
`target` is a probability vector over candidates (one-hot for hard labels,
a vote distribution for human soft labels, or a teacher distribution).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

PRIMITIVES = ("noul", "choice", "score")
SPLITS = ("train", "calibration", "test")
MAX_CANDIDATES = {"noul": 2, "choice": 255, "score": 10}
MIN_CANDIDATES = {"noul": 2, "choice": 2, "score": 2}


@dataclass
class Candidate:
    name: str
    description: str = ""

    def text(self) -> str:
        return f"{self.name}: {self.description}" if self.description else self.name


@dataclass
class DecisionExample:
    id: str
    source_id: str  # dataset + row id; the split unit
    dataset: str
    split: str
    primitive: str
    state: str
    question: str
    candidates: list[Candidate]
    target: list[float]
    template_id: str = ""
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    @property
    def label(self) -> int:
        return max(range(len(self.target)), key=lambda i: self.target[i])

    @property
    def k(self) -> int:
        return len(self.candidates)

    def validate(self) -> None:
        if self.primitive not in PRIMITIVES:
            raise ValueError(f"{self.id}: unknown primitive {self.primitive!r}")
        if self.split not in SPLITS:
            raise ValueError(f"{self.id}: unknown split {self.split!r}")
        k = len(self.candidates)
        if not MIN_CANDIDATES[self.primitive] <= k <= MAX_CANDIDATES[self.primitive]:
            raise ValueError(f"{self.id}: {self.primitive} needs {MIN_CANDIDATES[self.primitive]}..{MAX_CANDIDATES[self.primitive]} candidates, got {k}")
        if len(self.target) != k:
            raise ValueError(f"{self.id}: target length {len(self.target)} != {k} candidates")
        if any(t < 0 for t in self.target):
            raise ValueError(f"{self.id}: negative target mass")
        s = sum(self.target)
        if not math.isclose(s, 1.0, abs_tol=1e-4):
            raise ValueError(f"{self.id}: target sums to {s:.5f}, not 1")
        if not self.state.strip() and not self.meta.get("allow_empty_state"):
            raise ValueError(f"{self.id}: empty state")
        if not self.question.strip():
            raise ValueError(f"{self.id}: empty question")
        names = [c.name for c in self.candidates]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.id}: duplicate candidate names {names}")

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionExample":
        d = dict(d)
        d["candidates"] = [Candidate(**c) if isinstance(c, dict) else Candidate(str(c)) for c in d["candidates"]]
        return cls(**d)

    def with_candidate_order(self, order: list[int]) -> "DecisionExample":
        """Return a copy with candidates (and target) permuted by `order`."""
        return DecisionExample(
            id=self.id, source_id=self.source_id, dataset=self.dataset, split=self.split,
            primitive=self.primitive, state=self.state, question=self.question,
            candidates=[self.candidates[i] for i in order], target=[self.target[i] for i in order],
            template_id=self.template_id, meta=dict(self.meta, permuted_from=order),
        )


def write_jsonl(path: str | Path, examples: Iterable[DecisionExample]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path, limit: int | None = None) -> Iterator[DecisionExample]:
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                yield DecisionExample.from_dict(json.loads(line))


def load_jsonl(path: str | Path, limit: int | None = None) -> list[DecisionExample]:
    return list(read_jsonl(path, limit))


def k_bucket(k: int) -> str:
    """Bucket candidate counts for per-bucket temperature / reporting."""
    if k <= 2:
        return "k2"
    if k <= 5:
        return "k3-5"
    if k <= 10:
        return "k6-10"
    if k <= 50:
        return "k11-50"
    return "k51+"
