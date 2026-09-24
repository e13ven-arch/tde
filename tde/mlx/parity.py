#!/usr/bin/env python
"""Check the MLX joint model against the PyTorch one on real examples: same encodings, logits compared.

    python -m tde.mlx.parity --run release/tde-general-v0.1 --data data/v0.1/test.jsonl --n 64
"""
from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np
import torch

from tde.mlx.model import load_joint
from tde.mlx.train import make_batch
from tde.schema import load_jsonl
from tde.train import load_checkpoint, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="release/tde-general-v0.1")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    exs = load_jsonl(args.data, args.n)
    dtok, torch_model = load_checkpoint(args.run, device=torch.device("cpu"))[:2]
    ref = predict(torch_model, dtok, exs, torch.device("cpu"), batch_size=16, bf16=False)
    model = load_joint(f"{args.run}/model.safetensors")
    encs = [dtok.encode(e, "joint") for e in exs]
    kmax = max(len(e.target) for e in encs)
    for dtype in (mx.float32, mx.bfloat16):
        got = []
        for i in range(0, len(encs), 16):
            chunk = encs[i : i + 16]
            lg = model(make_batch(chunk, dtok.pad_id, kmax), dtype)
            mx.eval(lg)
            got += [np.array(lg[j, : len(e.target)]) for j, e in enumerate(chunk)]
        diff = max(float(np.abs(g - r).max()) for g, r in zip(got, ref))
        same = sum(int(g.argmax() == r.argmax()) for g, r in zip(got, ref))
        print(f"{dtype}: max |logit diff| {diff:.4f}, same argmax {same}/{len(exs)}")


if __name__ == "__main__":
    main()
