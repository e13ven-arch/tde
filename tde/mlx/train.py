#!/usr/bin/env python
"""Train the joint readout with MLX on Apple GPUs (about 2x the PyTorch MPS throughput for ModernBERT-base).

    python -m tde.mlx.train --data_dir data/snake/bfs_v1 --init release/tde-general-v0.1 --out_dir runs/tde-general-v0.2 \
        --max_state_tokens 480 --epochs 2

Mirrors tde/train.py for the joint readout (topology 'seq', marker 'mask', pool 'marker+span', CE loss): the same
DecisionTokenizer encodings, AdamW (betas 0.9/0.98, eps 1e-6) with layer-wise lr decay and no decay on norms and
biases, linear warmup then linear decay, global-norm clipping, and best-by-calibration-NLL selection. Master weights
are fp32 and matmuls bf16. Each batch is padded to a multiple of 64 tokens so the compiled step sees few shapes.
The run directory gets model.safetensors plus a PyTorch best.pt, config.json and tokenizer, so
scripts/evaluate.py reads it like any other run.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from tde.mlx.model import JointDecisionModel, load_joint
from tde.model.encoding import DecisionTokenizer
from tde.schema import load_jsonl

KEYS = ("input_ids", "valid", "opt_positions", "span_start", "span_end", "level_index", "cand_mask", "target",
        "decide_positions")


def make_batch(encs, pad_id: int, kmax: int, multiple: int = 64) -> dict:
    """EncodedExample list -> padded MLX arrays. Padded candidates are masked; their span covers [CLS] only."""
    B = len(encs)
    T = -(-max(len(e.input_ids) for e in encs) // multiple) * multiple
    ids = np.full((B, T), pad_id, np.int32)
    valid = np.zeros((B, T), bool)
    opt, ss, se = np.zeros((B, kmax), np.int32), np.zeros((B, kmax), np.int32), np.ones((B, kmax), np.int32)
    lvl, cm, tgt = np.full((B, kmax), -1, np.int32), np.zeros((B, kmax), bool), np.zeros((B, kmax), np.float32)
    dec = np.zeros((B,), np.int32)
    for i, e in enumerate(encs):
        n, k = len(e.input_ids), len(e.target)
        ids[i, :n], valid[i, :n] = e.input_ids, True
        opt[i, :k], lvl[i, :k], cm[i, :k], tgt[i, :k] = e.opt_positions, e.level_index, True, e.target
        ss[i, :k], se[i, :k] = [a for a, _ in e.spans], [b for _, b in e.spans]
        dec[i] = e.decide_position
    return {k: mx.array(v) for k, v in zip(KEYS, (ids, valid, opt, ss, se, lvl, cm, tgt, dec))}


def ce_loss(logits, target, cand_mask):
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -mx.where(cand_mask, target * lp, 0.0).sum(-1).mean()


def param_schedule(names: list[str], lr_backbone: float, lr_head: float, layer_decay: float, wd: float):
    """Per-parameter lr and weight decay, as tde.train._param_groups: backbone lr decays with depth (order of the
    PyTorch named_parameters), heads get lr_head; norms, biases and 1-D tensors get no decay."""
    order = ["embeddings.tok_embeddings.weight", "embeddings.norm.weight"]
    n_layers = 1 + max(int(n.split(".")[2]) for n in names if n.startswith("backbone.layers."))
    for i in range(n_layers):
        order += ([f"layers.{i}.attn_norm.weight"] if i else []) + [f"layers.{i}.attn.Wqkv.weight", f"layers.{i}.attn.Wo.weight",
                  f"layers.{i}.mlp_norm.weight", f"layers.{i}.mlp.Wi.weight", f"layers.{i}.mlp.Wo.weight"]
    order.append("final_norm.weight")
    depth = {"backbone." + n: i / (len(order) - 1) for i, n in enumerate(order)}
    lr = {n: lr_backbone * layer_decay ** ((1 - depth[n]) * 12) if n in depth else lr_head for n in names}
    decay = {n: 0.0 if n.startswith("backbone.") and (n.endswith("bias") or "norm" in n.lower()) else wd for n in names}
    missing = [n for n in names if n.startswith("backbone.") and n not in depth]
    assert not missing, f"parameters outside the layer-decay order: {missing}"
    return lr, decay


def build_step(model, names: list[str], lr_of: dict, wd_of: dict, loss_fn, max_grad_norm: float = 1.0):
    """Compiled AdamW step for loss_fn(model, batch) -> scalar, over a flat parameter dict: betas 0.9/0.98, eps 1e-6,
    decoupled weight decay as torch.optim.AdamW, global-norm clipping, per-parameter lr = lr_scale * lr_of[name].

    step(params, m, v, t, lr_scale, batch) -> (params, m, v, loss, grad_norm)."""
    def loss_of(p, batch):
        model.update(tree_unflatten_flat(p))
        return loss_fn(model, batch)

    def step(p, m, v, t, lr_scale, batch):
        loss, grads = mx.value_and_grad(loss_of)(p, batch)
        gnorm = mx.sqrt(sum((g.astype(mx.float32) ** 2).sum() for g in grads.values()))
        clip = mx.minimum(1.0, max_grad_norm / (gnorm + 1e-6))
        b1, b2, eps = 0.9, 0.98, 1e-6
        new_p, new_m, new_v = {}, {}, {}
        for n in names:
            g = grads[n] * clip
            new_m[n] = b1 * m[n] + (1 - b1) * g
            new_v[n] = b2 * v[n] + (1 - b2) * g * g
            lr = lr_scale * lr_of[n]
            upd = (new_m[n] / (1 - b1 ** t)) / (mx.sqrt(new_v[n] / (1 - b2 ** t)) + eps)
            new_p[n] = p[n] * (1 - lr * wd_of[n]) - lr * upd
        return new_p, new_m, new_v, loss, gnorm

    return mx.compile(step)


def evaluate(model, encs, pad_id, kmax, batch_size=64) -> dict:
    """Top-1 agreement with the target argmax, NLL and 15-bin equal-mass ECE of the top-1 probability."""
    probs, labels = [], []
    for i in range(0, len(encs), batch_size):
        chunk = encs[i : i + batch_size]
        b = make_batch(chunk, pad_id, kmax)
        p = mx.softmax(model(b), axis=-1)
        mx.eval(p)
        for j, e in enumerate(chunk):
            probs.append(np.array(p[j, : len(e.target)]))
            labels.append(int(np.argmax(e.target)))
    conf = np.array([p.max() for p in probs])
    correct = np.array([int(p.argmax() == y) for p, y in zip(probs, labels)], float)
    nll = float(np.mean([-math.log(max(p[y], 1e-12)) for p, y in zip(probs, labels)]))
    edges = np.quantile(conf, np.linspace(0, 1, 16))
    bins = np.clip(np.searchsorted(edges[1:-1], conf, side="right"), 0, 14)
    ece = sum(abs(conf[bins == k].mean() - correct[bins == k].mean()) * (bins == k).mean() for k in range(15) if (bins == k).any())
    return {"n": len(encs), "acc": float(correct.mean()), "nll": nll, "ece": float(ece)}


def export(model, out: Path, init: Path, cfg: dict):
    """model.safetensors (fp32) + best.pt, config.json and tokenizer in the PyTorch run layout."""
    import torch
    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), flat)
    torch.save({k: torch.from_numpy(np.array(v, dtype=np.float32)) for k, v in flat.items()}, out / "best.pt")
    base = json.loads((init / "config.json").read_text())
    (out / "config.json").write_text(json.dumps({**base, **cfg, "readout": "joint", "topology": "seq", "marker": "mask",
                                                 "pool": "marker+span", "device": "auto", "trainer": "mlx"}, indent=2))
    if not (out / "tokenizer").exists():
        shutil.copytree(init / "tokenizer", out / "tokenizer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--init", default="release/tde-general-v0.1", help="run dir with model.safetensors, config.json, tokenizer/")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_limit", type=int, default=None)
    ap.add_argument("--eval_limit", type=int, default=2000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_backbone", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--layer_decay", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--max_state_tokens", type=int, default=448)
    ap.add_argument("--max_total", type=int, default=1024)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = vars(args).copy()

    from transformers import AutoTokenizer
    init, out = Path(args.init), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_args.json").write_text(json.dumps(cfg, indent=2))
    dtok = DecisionTokenizer(AutoTokenizer.from_pretrained(init / "tokenizer"), max_state_tokens=args.max_state_tokens,
                             marker="mask", max_total=args.max_total)
    train_ex = load_jsonl(Path(args.data_dir) / "train.jsonl", args.train_limit)
    cal_ex = load_jsonl(Path(args.data_dir) / "calibration.jsonl", args.eval_limit)
    t0 = time.time()
    train_enc = [dtok.encode(e, "joint") for e in train_ex]
    cal_enc = [dtok.encode(e, "joint") for e in cal_ex]
    kmax = max(len(e.target) for e in train_enc + cal_enc)
    print(f"[mlx] encoded {len(train_enc)} train / {len(cal_enc)} cal in {time.time() - t0:.0f}s, kmax={kmax}", flush=True)

    model = load_joint(str(init / "model.safetensors"))
    params = dict(tree_flatten(model.trainable_parameters()))
    names = sorted(params)
    lr_of, wd_of = param_schedule(names, args.lr_backbone, args.lr_head, args.layer_decay, args.weight_decay)
    m_state = {n: mx.zeros_like(p) for n, p in params.items()}
    v_state = {n: mx.zeros_like(p) for n, p in params.items()}

    step = build_step(model, names, lr_of, wd_of, lambda m, b: ce_loss(m(b), b["target"], b["cand_mask"]),
                      args.max_grad_norm)
    rng = random.Random(args.seed)
    order = list(range(len(train_enc)))
    steps_per_epoch = len(order) // args.batch_size
    total = max(1, int(steps_per_epoch * args.epochs))
    warmup = max(1, int(total * args.warmup_ratio))
    log_f = (out / "log.jsonl").open("a")
    best_nll, n_step, running, t0 = float("inf"), 0, [], time.time()
    while n_step < total:
        rng.shuffle(order)
        window = args.batch_size * 64
        batches = []
        for i in range(0, len(order) - args.batch_size + 1, window):  # sort windows by length, drop the ragged tail
            w = sorted(order[i : i + window], key=lambda j: len(train_enc[j].input_ids))
            batches += [w[k : k + args.batch_size] for k in range(0, len(w) - args.batch_size + 1, args.batch_size)]
        rng.shuffle(batches)
        # Group equal padded lengths inside chunks of 256 batches: the compiled step caches only a few input shapes,
        # and cycling through ~9 in random order recompiled almost every step (1.5 s vs 0.8 s per step).
        t_of = lambda b: -(-max(len(train_enc[j].input_ids) for j in b) // 64)
        batches = [b for i in range(0, len(batches), 256) for b in sorted(batches[i : i + 256], key=t_of)]
        for idx in batches:
            if n_step >= total:
                break
            n_step += 1
            lr_scale = min(1.0, n_step / warmup) * max(0.0, (total - n_step + 1) / max(1, total - warmup))
            batch = make_batch([train_enc[j] for j in idx], dtok.pad_id, kmax)
            params, m_state, v_state, loss, gnorm = step(params, m_state, v_state, mx.array(float(n_step)),
                                                         mx.array(lr_scale), batch)
            mx.eval(params, m_state, v_state, loss)
            running.append(float(loss))
            if n_step % args.log_every == 0:
                rec = {"step": n_step, "elapsed_s": round(time.time() - t0, 1), "lr_scale": round(lr_scale, 4),
                       "ce": sum(running) / len(running), "gnorm": float(gnorm)}
                running = []
                print("[mlx]", json.dumps(rec), flush=True)
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            if n_step % args.eval_every == 0 or n_step >= total:
                model.update(tree_unflatten_flat(params))
                ev = evaluate(model, cal_enc, dtok.pad_id, kmax)
                print("[eval]", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in ev.items()}), flush=True)
                log_f.write(json.dumps({"step": n_step, "eval": ev}) + "\n"); log_f.flush()
                if ev["nll"] < best_nll:
                    best_nll = ev["nll"]
                    export(model, out, init, {**cfg, "best_step": n_step, "best_cal": ev})
    print(f"[mlx] done: {n_step} steps in {(time.time() - t0) / 60:.1f} min, best cal nll {best_nll:.4f}", flush=True)


def tree_unflatten_flat(flat: dict):
    from mlx.utils import tree_unflatten
    return tree_unflatten(list(flat.items()))


if __name__ == "__main__":
    main()
