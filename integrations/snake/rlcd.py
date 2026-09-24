#!/usr/bin/env python
"""RLCD-style training on Snake: the joint model learns which move is best from game outcomes alone, no teacher labels.

"RLCD-style" means the community proper-reward policy gradient in tde.losses (paired_brier_pg), which equals direct
Brier supervision in expectation; Jev's RLCD itself is unpublished and nothing here claims to match it.

Each iteration advances N actor games one move. For every actor state (a root), each move is scored by environment
playouts: the return is the discounted food eaten minus a discounted death penalty, and the outcome target q is
uniform over the moves with the best return. The model trains with the paired Brier policy gradient of
tde.losses.paired_brier_pg: M predictions A_i ~ p are scored against one outcome Y ~ q with the pairwise Brier reward
and a conditional baseline, so the expected gradient is grad ||p - q||^2. Roots and targets go to a replay buffer
that each iteration samples for a few steps. Actors play the greedy move of the noised logits or, with probability
follow_q, a move drawn from q, never a move that loses at once when another exists, so the buffer holds roughly the
boards the greedy model itself reaches. A greedy evaluation on fixed seeds picks the checkpoint exported per board.

Playouts, chosen per stage:
  mc      flat Monte Carlo: 256 uniformly random safe playouts per move, on CPU workers while the GPU trains.
  policy  one playout per move with moves sampled from the current policy (policy iteration), on the GPU.

Policy playouts from the untrained model stalled on 8x8. Greedy ones traced fixed loops that rarely reach food, so
every safe move tied and the policy learned to circle (0 deaths, 0.4 food in 500 moves after 6.4k roots). Sampled
ones made adjacent food a clear target (0.81 on the eating move) but left farther food near chance, and after 12.8k
roots the model still ignored food. Flat Monte Carlo ranks the moves toward food at every distance on small boards,
but not on 24x16, where random walks rarely reach it and a model that already seeks food is the better playout policy.
Its argmax is noisy: two independent evaluations of a mid-game state pick the same move 60% of the time with 16
playouts per move, 82% (8x8) and 70% (12x12) with 128, 88% and 83% with 256; with 16 the model's agreement with its
targets stalled at that ceiling. With 256 and actors that sampled the softmax and followed q half the time, actor games
lasted ~490 moves, the buffer held mostly long-snake boards, and the greedy model died early on open boards: 15 of 16
evaluation deaths were a losing top move (e.g. reversing into its neck toward food) with a safe move available.

    python -m integrations.snake.rlcd --out_dir runs/snake_rlcd_v4
    python -m integrations.snake.rlcd --out_dir runs/snake_rlcd_v4_ce --loss ce   # ablation: cross-entropy on q
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from integrations.snake.encode import FastEncoder
from integrations.snake.game import DIRS, SnakeGame
from integrations.snake.montecarlo import mc_returns
from integrations.snake.play import evaluate, model_policy, on_gpu
from tde.mlx.losses import paired_brier_pg
from tde.mlx.model import load_joint
from tde.mlx.train import build_step, ce_loss, export, param_schedule, tree_unflatten_flat
from tde.model.encoding import DecisionTokenizer

EVAL_MOVES = {(8, 8): 500, (12, 12): 800}


def parse_stages(s: str) -> list[tuple[int, int, int, int, str]]:
    """'8x8x3:60000:mc,24x16x6:10000:policy' -> [(width, height, initial_length, roots, playouts), ...]"""
    out = []
    for part in s.split(","):
        board, roots, *source = part.split(":")
        w, h, length = map(int, board.split("x"))
        out.append((w, h, length, int(roots), source[0] if source else "mc"))
    return out


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def draw(p: np.ndarray, u: float) -> int:
    return min(int(np.searchsorted(np.cumsum(p), u * p.sum(), side="right")), len(p) - 1)


def rollout_returns(model, enc, roots, horizon: int, gamma: float, death: float, rng) -> tuple[np.ndarray, int]:
    """R[i, k]: return of move k from root i followed by moves sampled from the current policy, `horizon` in all.
    The four moves of a root share food spawns and the uniforms that draw each later move (common random numbers)."""
    R = np.zeros((len(roots), len(DIRS)))
    U = np.array([[rng.random() for _ in range(horizon)] for _ in roots])
    sims = []
    for i, g in enumerate(roots):
        seed = rng.getrandbits(64)
        for k, d in enumerate(DIRS):
            if g.dies(d):
                R[i, k] = -death
                continue
            c = g.clone()
            c.rng.seed(seed)  # the four moves share food spawns the actor will not see
            c.step(d)
            R[i, k] = c.score - g.score
            sims.append((i, k, c))
    n_fwd = 0
    for t in range(1, horizon):
        live = [s for s in sims if s[2].alive and s[2].food is not None]
        if not live:
            break
        b = enc.batch([c for _, _, c in live])
        p = on_gpu(lambda: np.array(mx.softmax(model(b).astype(mx.float32), axis=-1), np.float64))
        n_fwd += len(live)
        for (i, k, c), row in zip(live, p):
            before = c.score
            c.step(DIRS[draw(row, U[i, t])])
            R[i, k] += gamma ** t * (c.score - before) if c.alive else -gamma ** t * death
    return R, n_fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="release/tde-general-v0.1")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stages", default="8x8x3:60000:mc,12x12x3:30000:mc,24x16x6:10000:policy",
                    help="WxHxLength:roots:playouts,... with playouts mc or policy")
    ap.add_argument("--actors", type=int, default=64, help="roots per iteration")
    ap.add_argument("--updates", type=int, default=4, help="gradient steps per iteration")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--buffer", type=int, default=50000)
    ap.add_argument("--loss", default="brier_pg", choices=["brier_pg", "ce"],
                    help="brier_pg: paired Brier policy gradient; ce: cross-entropy on q (ablation)")
    ap.add_argument("--samples", type=int, default=32, help="M predictions per root in the paired Brier reward")
    ap.add_argument("--playouts", type=int, default=256, help="random playouts per move in mc stages")
    ap.add_argument("--workers", type=int, default=12, help="CPU processes for mc playouts")
    ap.add_argument("--horizon", type=int, default=0, help="0: width + height, at most 32")
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--death", type=float, default=5.0)
    ap.add_argument("--sigma0", type=float, default=0.5, help="actor logit noise (std) at the start of each stage")
    ap.add_argument("--sigma1", type=float, default=0.2, help="... and at its end")
    ap.add_argument("--follow_q", type=float, default=0.2, help="probability that an actor plays a move drawn from q")
    ap.add_argument("--lr_backbone", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--layer_decay", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--max_state_tokens", type=int, default=480)
    ap.add_argument("--max_total", type=int, default=1024)
    ap.add_argument("--eval_every", type=int, default=100, help="iterations")
    ap.add_argument("--eval_games", type=int, default=16)
    ap.add_argument("--log_every", type=int, default=10, help="iterations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = vars(args).copy()
    # Rollout batches change size every call, so MLX's buffer cache fills with one-off sizes (17-25 GB after a few
    # dozen iterations) until macOS swaps and forwards run at a third of their speed. A small cache avoids it.
    mx.set_cache_limit(4 << 30)
    pool = multiprocessing.get_context("spawn").Pool(args.workers)

    from transformers import AutoTokenizer
    init, out = Path(args.init), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_args.json").write_text(json.dumps(cfg, indent=2))
    dtok = DecisionTokenizer(AutoTokenizer.from_pretrained(init / "tokenizer"), max_state_tokens=args.max_state_tokens,
                             marker="mask", max_total=args.max_total)
    enc = FastEncoder(dtok)
    model = load_joint(str(init / "model.safetensors"))
    params = dict(tree_flatten(model.trainable_parameters()))
    names = sorted(params)
    lr_of, wd_of = param_schedule(names, args.lr_backbone, args.lr_head, args.layer_decay, args.weight_decay)
    m_state = {n: mx.zeros_like(p) for n, p in params.items()}
    v_state = {n: mx.zeros_like(p) for n, p in params.items()}
    loss_fn = {"brier_pg": lambda m, b: paired_brier_pg(m(b), b["target"], b["cand_mask"], b["u_y"], b["u_a"]),
               "ce": lambda m, b: ce_loss(m(b), b["target"], b["cand_mask"])}[args.loss]
    step = build_step(model, names, lr_of, wd_of, loss_fn, args.max_grad_norm)
    rng, nrng = random.Random(args.seed), np.random.default_rng(args.seed)
    log_f = (out / "log.jsonl").open("a")

    def log(rec: dict):
        print(json.dumps(rec), flush=True)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    n_update, roots_seen, t_start = 0, 0, time.time()
    for w, h, length, n_roots, source in parse_stages(args.stages):
        board = f"{w}x{h}"
        horizon = args.horizon or min(w + h, 32)
        moves = EVAL_MOVES.get((w, h), 600)
        stage_dir = out / board
        stage_dir.mkdir(exist_ok=True)
        seeds = list(range(1000, 1000 + args.eval_games))
        best = None

        def run_eval(it: int):
            nonlocal best
            ev = evaluate(model_policy(model, enc), w, h, length, seeds, moves)
            sh = evaluate(model_policy(model, enc), w, h, length, seeds, moves, shield=True)
            log({"eval": board, "iter": it, "roots": roots_seen, "elapsed_s": round(time.time() - t_start),
                 "score": ev["score_mean"], "deaths": ev["deaths"], "steps": ev["steps"], "scores": ev["scores"],
                 "shield_score": sh["score_mean"], "shield_deaths": sh["deaths"], "interventions": sh["interventions"]})
            key = (ev["score_mean"], -ev["deaths"])
            if it and (best is None or key > best):
                best = key
                export(model, stage_dir, init, {**cfg, "board": board, "playouts_source": source, "best_iter": it,
                                                "best_eval": ev, "best_eval_shield": sh})

        actors = [SnakeGame(w, h, length, rng.getrandbits(32)) for _ in range(args.actors)]
        T = len(enc.encode(actors[0])[0])
        buf_ids, buf_q, n_buf = np.zeros((args.buffer, T), np.int32), np.zeros((args.buffer, len(DIRS)), np.float32), 0
        const = {k: v for k, v in enc.batch([actors[0]] * args.batch_size).items() if k != "input_ids"}
        iters = -(-n_roots // args.actors)
        run_eval(0)
        stats, episodes = [], []
        for it in range(1, iters + 1):
            sigma = args.sigma0 + (args.sigma1 - args.sigma0) * (it - 1) / max(1, iters - 1)
            t0 = time.time()
            b = enc.batch(actors)
            logits = on_gpu(lambda: np.array(model(b).astype(mx.float32), dtype=np.float64))
            if source == "mc":  # playouts run on the CPU workers during this iteration's gradient steps
                jobs = [(g, args.playouts, horizon, args.gamma, args.death, rng.getrandbits(32)) for g in actors]
                pending, n_fwd = pool.map_async(mc_returns, jobs), 0
            else:
                R, n_fwd = rollout_returns(model, enc, actors, horizon, args.gamma, args.death, rng)
            t1 = time.time()
            losses = []
            for _ in range(args.updates if n_buf else 0):
                idx = nrng.integers(0, min(n_buf, args.buffer), args.batch_size)
                batch = {**const, "input_ids": mx.array(buf_ids[idx]), "target": mx.array(buf_q[idx]),
                         "u_y": mx.array(nrng.random((args.batch_size, 1), np.float32)),
                         "u_a": mx.array(nrng.random((args.batch_size, args.samples), np.float32))}
                n_update += 1

                def train_step():
                    out = step(params, m_state, v_state, mx.array(float(n_update)),
                               mx.array(min(1.0, n_update / args.warmup)), batch)
                    mx.eval(out)
                    return out

                params, m_state, v_state, loss, gnorm = on_gpu(train_step)
                losses.append((float(loss), float(gnorm)))
            model.update(tree_unflatten_flat(params))
            t2 = time.time()
            if source == "mc":
                R = np.array(pending.get())
            best_set = R >= R.max(1, keepdims=True) - 1e-9
            q = best_set / best_set.sum(1, keepdims=True)
            ids = np.array(b["input_ids"])
            for i in range(len(actors)):
                buf_ids[n_buf % args.buffer], buf_q[n_buf % args.buffer] = ids[i], q[i]
                n_buf += 1
            roots_seen += len(actors)
            p = softmax(logits)
            safe = np.array([[not g.dies(d) for d in DIRS] for g in actors])
            noisy = logits + sigma * nrng.standard_normal(logits.shape)
            noisy[~safe & safe.any(1, keepdims=True)] = -np.inf  # all-fatal rows keep every move
            for i, g in enumerate(actors):
                g.step(DIRS[draw(q[i], nrng.random()) if nrng.random() < args.follow_q else int(noisy[i].argmax())])
                if not g.alive or g.food is None or g.steps >= moves:
                    episodes.append((g.score, g.steps, g.alive))
                    actors[i] = SnakeGame(w, h, length, rng.getrandbits(32))
            stats.append({"agree": float(best_set[np.arange(len(p)), p.argmax(1)].mean()),
                          "p_best": float((p * best_set).sum(1).mean()), "brier": float(((p - q) ** 2).sum(1).mean()),
                          "ties": float(best_set.sum(1).mean()), "fwd": n_fwd,
                          "loss": float(np.mean([l for l, _ in losses])) if losses else float("nan"),
                          "gnorm": float(np.mean([g for _, g in losses])) if losses else float("nan"),
                          "t_targets": (t1 - t0) + (time.time() - t2), "t_upd": t2 - t1})
            if it % args.log_every == 0 or it == iters:
                rec = {"board": board, "playouts": source, "iter": it, "roots": roots_seen, "updates": n_update,
                       "elapsed_s": round(time.time() - t_start), "sigma": round(sigma, 3)}
                rec.update({k: round(float(np.nanmean([s[k] for s in stats])), 4) for k in stats[0]})
                if episodes:
                    rec.update({"episodes": len(episodes),
                                "ep_score": round(float(np.mean([e[0] for e in episodes])), 2),
                                "ep_steps": round(float(np.mean([e[1] for e in episodes])), 1),
                                "ep_deaths": sum(not e[2] for e in episodes)})
                log(rec)
                stats, episodes = [], []
            if it % args.eval_every == 0 or it == iters:
                run_eval(it)
        mx.save_safetensors(str(stage_dir / "last.safetensors"), dict(tree_flatten(model.parameters())))
    log({"done": True, "roots": roots_seen, "updates": n_update, "minutes": round((time.time() - t_start) / 60, 1)})
    pool.close()


if __name__ == "__main__":
    main()
