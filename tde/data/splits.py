"""Deterministic, source-level split assignment.

The split unit is the *source item* (dataset + row id). Every example rendered
from that row inherits its split, so paraphrases and multi-interface rewrites
never straddle train / calibration / test. Template ids are hashed the same way
so that ~20% of templates are held out for evaluation only.
"""
from __future__ import annotations

import hashlib

DEFAULT_RATIOS = {"train": 0.80, "calibration": 0.10, "test": 0.10}


def _unit(salt: str, key: str) -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def assign_split(source_id: str, salt: str = "tde-v0.1", ratios: dict[str, float] | None = None) -> str:
    ratios = ratios or DEFAULT_RATIOS
    u = _unit(salt, source_id)
    acc = 0.0
    for name, r in ratios.items():
        acc += r
        if u < acc:
            return name
    return list(ratios)[-1]


def template_is_held_out(template_id: str, salt: str = "tde-v0.1-templates", fraction: float = 0.20) -> bool:
    return _unit(salt, template_id) < fraction


def make_source_id(dataset: str, row_key: str | int) -> str:
    return f"{dataset}:{row_key}"
