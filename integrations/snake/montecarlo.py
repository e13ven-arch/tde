"""Flat Monte Carlo move values for the RLCD targets: uniformly random safe playouts, no model and no teacher.

With 16 playouts of 16-24 moves this default policy plays 8x8 and 12x12 well (20-37 food in 500-800 moves, no deaths)
but finds little food on 24x16, where random walks rarely reach it.
"""
from __future__ import annotations

import random

from integrations.snake.game import DIRS


def mc_returns(job) -> list[float]:
    """job = (game, playouts, horizon, gamma, death, seed) -> mean discounted return of each first move, followed by
    uniformly random safe moves (death once none is left). Playout j of every first move shares its food spawns and
    move draws (common random numbers). A first move that loses at once scores -death."""
    g, playouts, horizon, gamma, death, seed = job
    rng = random.Random(seed)
    seeds = [rng.getrandbits(32) for _ in range(playouts)]
    out = []
    for d in DIRS:
        if g.dies(d):
            out.append(-death)
            continue
        total = 0.0
        for s in seeds:
            c, u = g.clone(), random.Random(s + 1)
            c.rng.seed(s)
            c.step(d)
            ret = c.score - g.score
            for t in range(1, horizon):
                if c.food is None:
                    break
                safe = [x for x in DIRS if not c.dies(x)]
                if not safe:
                    ret -= gamma ** t * death
                    break
                before = c.score
                c.step(safe[int(u.random() * len(safe))])
                ret += gamma ** t * (c.score - before)
            total += ret
        out.append(total / playouts)
    return out
