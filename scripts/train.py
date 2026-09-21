#!/usr/bin/env python
"""Train from a YAML config with CLI overrides, e.g.
    python scripts/train.py --config configs/exp001_readout.yaml --readout branch --out_dir runs/exp001_branch
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tde.train import TrainConfig, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    for f in fields(TrainConfig):
        if f.name == "extra":
            continue
        ap.add_argument(f"--{f.name}", default=None)
    args = ap.parse_args()
    cfg_dict = {}
    if args.config:
        cfg_dict.update(yaml.safe_load(Path(args.config).read_text()) or {})
    types = {f.name: f.type for f in fields(TrainConfig)}
    for f in fields(TrainConfig):
        v = getattr(args, f.name, None)
        if v is None:
            continue
        t = types[f.name]
        if t == "bool" or t is bool:
            v = str(v).lower() in ("1", "true", "yes")
        elif "int" in str(t):
            v = int(v)
        elif "float" in str(t):
            v = float(v)
        cfg_dict[f.name] = v
    cfg = TrainConfig(**cfg_dict)
    result = train(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
