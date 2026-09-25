"""Greedy Snake play and the evaluation protocols, for MLX joint models and the BFS teacher.

    python -m integrations.snake.play release/tde-general-v0.2 runs/snake_rlcd_v4/12x12 bfs mc --standard

A policy maps a list of games to move probabilities [n, 4]; play is greedy (top-1). Besides run directories, 'bfs' is
the BFS teacher and 'mc' the flat Monte Carlo player that makes the RLCD targets (16 random playouts per move). With
the shield, a move that loses at once gives way to the most probable move that does not; the trap shield also vetoes
moves that cut the head off from its tail. Each veto is counted. The standard protocol is: 24x16 board,
initial length 6, seeds 101-104, 600 moves.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import time

import mlx.core as mx
import numpy as np

from integrations.snake.game import DELTA, DIRS, SnakeGame

STANDARD = (24, 16, 6, [101, 102, 103, 104], 600)


def on_gpu(fn, tries: int = 5):
    """fn() again after a GPU reset: while another app's GPU work faults (seen with a game open), macOS discards the
    in-flight MLX command buffers ("victim of GPU error/recovery"). Inputs stay valid, so recomputing is safe."""
    for k in range(tries):
        try:
            return fn()
        except RuntimeError as e:
            if "Command buffer execution failed" not in str(e) or k == tries - 1:
                raise
            print(f"[gpu] command buffer discarded, retry {k + 1}: {e}", flush=True)
            time.sleep(10)


def probs(model, enc, games: list[SnakeGame], dtype=mx.bfloat16) -> np.ndarray:
    b = enc.batch(games)
    return on_gpu(lambda: np.array(mx.softmax(model(b, dtype).astype(mx.float32), axis=-1)))


def model_policy(model, enc):
    return lambda games: probs(model, enc, games)


def room(g: SnakeGame) -> tuple[int, bool]:
    """Free cells the head can reach (the tail cell counts as free, it moves on) and whether the tail is one of them."""
    blocked, seen, stack = g.cells - {g.body[-1]}, {g.body[0]}, [g.body[0]]
    while stack:
        x, y = stack.pop()
        for dx, dy in DELTA.values():
            n = (x + dx, y + dy)
            if 0 <= n[0] < g.w and 0 <= n[1] < g.h and n not in seen and n not in blocked:
                seen.add(n)
                stack.append(n)
    return len(seen) - 1, g.body[-1] in seen


def after(g: SnakeGame, d: str) -> SnakeGame:
    s = g.clone()
    s.step(d)
    return s


def choose(p_row: np.ndarray, g: SnakeGame, shield: bool | str) -> tuple[str, bool]:
    """Top-1 move. shield=True vetoes a move that dies at once; shield="trap" also vetoes a move after which the head
    cannot reach its tail, and with no such move left takes the surviving move with the most room. Vetoes go to the
    most probable remaining move."""
    order = np.argsort(-p_row, kind="stable")
    a = DIRS[order[0]]
    if shield == "trap":
        alive = [(d, after(g, d)) for d in (DIRS[i] for i in order) if not g.dies(d)]
        ok = [d for d, s in alive if room(s)[1]]
        pick = ok[0] if ok else max(alive, key=lambda x: room(x[1])[0])[0] if alive else a
        return pick, pick != a
    if shield and g.dies(a):
        alt = [DIRS[i] for i in order[1:] if not g.dies(DIRS[i])]
        if alt:
            return alt[0], True
    return a, False


def evaluate(policy, w: int, h: int, length: int, seeds, moves: int, shield: bool | str = False) -> dict:
    games = [SnakeGame(w, h, length, s) for s in seeds]
    hits, t_policy, calls = [0] * len(games), 0.0, 0
    while live := [i for i, g in enumerate(games) if g.alive and g.food is not None and g.steps < moves]:
        t0 = time.perf_counter()
        p = policy([games[i] for i in live])
        t_policy, calls = t_policy + time.perf_counter() - t0, calls + 1
        for i, row in zip(live, p):
            a, hit = choose(row, games[i], shield)
            hits[i] += hit
            games[i].step(a)
    scores = [g.score for g in games]
    return {"board": f"{w}x{h}", "length0": length, "moves": moves, "shield": shield, "seeds": list(seeds),
            "scores": scores, "score_mean": float(np.mean(scores)), "deaths": sum(not g.alive for g in games),
            "steps": [g.steps for g in games], "interventions": hits,
            "ms_per_batched_call": round(1000 * t_policy / max(1, calls), 2)}


def seed_range(s: str) -> list[int]:
    a, b = map(int, s.split("-"))
    return list(range(a, b + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policies", nargs="+", help="run dirs with model.safetensors, 'bfs' or 'mc'")
    ap.add_argument("--boards", default="8x8x3:500,12x12x3:800,24x16x6:600", help="WxHxLength:moves,...")
    ap.add_argument("--seeds", default="2000-2031")
    ap.add_argument("--standard", action="store_true", help="also the standard 24x16 protocol (600 moves, seeds 101-104)")
    ap.add_argument("--init", default="release/tde-general-v0.1", help="tokenizer source")
    ap.add_argument("--out", default=None, help="append results as JSON lines")
    args = ap.parse_args()
    mx.set_cache_limit(4 << 30)
    from transformers import AutoTokenizer

    from integrations.snake.encode import FastEncoder
    from integrations.snake.montecarlo import mc_returns
    from integrations.snake.teacher import teacher_target
    from tde.mlx.model import load_joint
    from tde.model.encoding import DecisionTokenizer
    enc = FastEncoder(DecisionTokenizer(AutoTokenizer.from_pretrained(f"{args.init}/tokenizer"), max_state_tokens=480,
                                        marker="mask", max_total=1024))
    protocols = []
    for part in args.boards.split(","):
        board, moves = part.split(":")
        w, h, length = map(int, board.split("x"))
        protocols.append(("heldout", w, h, length, seed_range(args.seeds), int(moves)))
    if args.standard:
        protocols.append(("standard", *STANDARD))
    out = open(args.out, "a") if args.out else None
    rng = random.Random(0)
    for name in args.policies:
        if name == "bfs":
            policy = lambda games: np.array([teacher_target(g) for g in games])
        elif name == "mc":
            pool = multiprocessing.get_context("spawn").Pool(12)
            jobs = lambda gs: [(g, 16, min(g.w + g.h, 32), 0.97, 5.0, rng.getrandbits(32)) for g in gs]
            policy = lambda games: np.eye(4)[np.argmax(pool.map(mc_returns, jobs(games)), axis=1)]
        else:
            policy = model_policy(load_joint(f"{name}/model.safetensors"), enc)
        for tag, w, h, length, seeds, moves in protocols:
            for shield in (False, True, "trap"):
                r = {"policy": name, "protocol": tag, **evaluate(policy, w, h, length, seeds, moves, shield)}
                print(json.dumps({k: r[k] for k in ("policy", "protocol", "board", "shield", "score_mean", "deaths",
                                                    "ms_per_batched_call")} | {"scores": r["scores"][:8]}), flush=True)
                if out:
                    out.write(json.dumps(r) + "\n")
                    out.flush()


if __name__ == "__main__":
    main()
