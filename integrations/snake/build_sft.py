#!/usr/bin/env python
"""BFS-teacher data for the supervised control arm, with the boards and state counts of the RLCD curriculum.

The teacher plays each board (a random safe move with probability eps, else one of its best moves) and every visited
state is labelled with teacher_target. Episodes end at death, a full board or the evaluation move cap. Splits are
disjoint by episode and shuffled, since tde.mlx.train reads the first rows of calibration.jsonl.

    python -m integrations.snake.build_sft --out data/snake/bfs_v1
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from integrations.snake.encode import to_example
from integrations.snake.game import DIRS, SnakeGame
from integrations.snake.rlcd import EVAL_MOVES, parse_stages
from integrations.snake.teacher import teacher_target
from tde.schema import write_jsonl


def teacher_states(w: int, h: int, length: int, n: int, eps: float, split: str, rng: random.Random) -> list:
    out, ep = [], 0
    cap = EVAL_MOVES.get((w, h), 600)
    while len(out) < n:
        g = SnakeGame(w, h, length, rng.getrandbits(32))
        while g.alive and g.food is not None and g.steps < cap and len(out) < n:
            target = teacher_target(g)
            out.append(to_example(g, target, split, f"{w}x{h}{split[:3]}{ep}-{g.steps}",
                                  {"board": f"{w}x{h}", "length": len(g.body), "score": g.score}))
            safe = [d for d in DIRS if not g.dies(d)]
            best = [d for d, t in zip(DIRS, target) if t == max(target)]
            g.step(rng.choice(safe) if safe and rng.random() < eps else rng.choice(best))
        ep += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/snake/bfs_v1")
    ap.add_argument("--stages", default="8x8x3:60000,12x12x3:30000,24x16x6:10000")
    ap.add_argument("--calibration", type=int, default=600, help="states per board")
    ap.add_argument("--test", type=int, default=1000, help="states per board")
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    splits = {"train": [], "calibration": [], "test": []}
    for w, h, length, n, _ in parse_stages(args.stages):
        for split, k in (("train", n), ("calibration", args.calibration), ("test", args.test)):
            splits[split] += teacher_states(w, h, length, k, args.eps, split, rng)
        print(f"{w}x{h}: {n} train states", flush=True)
    for split, exs in splits.items():
        rng.shuffle(exs)
        write_jsonl(Path(args.out) / f"{split}.jsonl", exs)
        print(split, len(exs))


if __name__ == "__main__":
    main()
