"""Post-hoc temperature scaling fitted per (primitive, K-bucket) on the calibration split only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tde.schema import k_bucket


def _nll(logits: list[np.ndarray], targets: list[np.ndarray], T: float) -> float:
    tot = 0.0
    for z, y in zip(logits, targets):
        zz = z / T
        zz = zz - zz.max()
        lp = zz - np.log(np.exp(zz).sum())
        tot += -(y * lp).sum()
    return tot / max(len(logits), 1)


def fit_temperature(logits: list[np.ndarray], targets: list[np.ndarray], grid: np.ndarray | None = None) -> float:
    if not logits:
        return 1.0
    grid = grid if grid is not None else np.exp(np.linspace(np.log(0.2), np.log(10.0), 120))
    best = min(grid, key=lambda T: _nll(logits, targets, T))
    # refine locally
    lo, hi = best / 1.15, best * 1.15
    fine = np.exp(np.linspace(np.log(lo), np.log(hi), 60))
    return float(min(fine, key=lambda T: _nll(logits, targets, T)))


class BucketTemperature:
    def __init__(self, table: dict[str, float] | None = None):
        self.table = table or {}

    @staticmethod
    def key(primitive: str, k: int) -> str:
        return f"{primitive}/{k_bucket(k)}"

    def fit(self, logits: list[np.ndarray], targets: list[np.ndarray], primitives: list[str]) -> "BucketTemperature":
        groups: dict[str, tuple[list, list]] = {}
        for z, y, p in zip(logits, targets, primitives):
            groups.setdefault(self.key(p, len(z)), ([], []))
            groups[self.key(p, len(z))][0].append(z)
            groups[self.key(p, len(z))][1].append(y)
        self.table = {k: fit_temperature(*v) for k, v in groups.items()}
        self.table["__global__"] = fit_temperature(logits, targets)
        return self

    def temperature(self, primitive: str, k: int) -> float:
        return self.table.get(self.key(primitive, k), self.table.get("__global__", 1.0))

    def apply(self, logits: np.ndarray, primitive: str) -> np.ndarray:
        z = logits / self.temperature(primitive, len(logits))
        z = z - z.max()
        p = np.exp(z)
        return p / p.sum()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.table, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "BucketTemperature":
        return cls(json.loads(Path(path).read_text()))
