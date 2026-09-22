#!/usr/bin/env python
"""Export a run directory as a self-contained release folder (loadable with tde.inference.Decider.from_run).
    python scripts/export_release.py --run runs/host/exp010_fullk_v05 --out release/tde-general-v0.1 --name tde-general-v0.1
Writes: config.json, tokenizer/, best.pt (state_dict), model.safetensors (same weights), MANIFEST.json (hashes), README.md (model card stub if absent).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "config.json", out / "config.json")
    shutil.copytree(run / "tokenizer", out / "tokenizer", dirs_exist_ok=True)
    state = torch.load(run / "best.pt", map_location="cpu")
    torch.save(state, out / "best.pt")
    save_file({k: v.contiguous() for k, v in state.items()}, str(out / "model.safetensors"))
    cfg = json.loads((out / "config.json").read_text())
    # a release folder is its own backbone source for the tokenizer; the backbone weights are inside best.pt
    manifest = {"name": args.name, "source_run": str(run), "readout": cfg["readout"], "backbone_init": cfg["backbone"],
                "params": cfg.get("total_params"), "files": {f.name: sha(f) for f in out.iterdir() if f.is_file()}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: (v if k != "files" else {kk: vv[:16] for kk, vv in v.items()}) for k, v in manifest.items()}, indent=2))


if __name__ == "__main__":
    main()
