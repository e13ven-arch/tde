"""BFS teacher for the supervised control arm: shortest safe route to food, else stay alive.

A move is taken toward food only if a virtual snake that follows the shortest path and eats can still reach its own
tail afterwards; ties split the target. Without such a route the teacher keeps its tail reachable if it can, and
otherwise picks the largest open region. Used only to label the control arm's training data.
"""
from __future__ import annotations

from collections import deque

from integrations.snake.game import DELTA, DIRS, SnakeGame


def _neighbours(w, h, c):
    for d in DIRS:
        dx, dy = DELTA[d]
        n = (c[0] + dx, c[1] + dy)
        if 0 <= n[0] < w and 0 <= n[1] < h:
            yield d, n


def _path(g: SnakeGame, goal) -> list[str] | None:
    """Directions of a shortest path from the head to `goal`; body cells block except the tail."""
    start, blocked = g.body[0], g.cells - {g.body[-1]}
    prev = {start: None}
    q = deque([start])
    while q:
        c = q.popleft()
        if c == goal:
            out = []
            while prev[c] is not None:
                c, d = prev[c]
                out.append(d)
            return out[::-1]
        for d, n in _neighbours(g.w, g.h, c):
            if n not in prev and (n not in blocked or n == goal):
                prev[n] = (c, d)
                q.append(n)
    return None


def _tail_reachable(g: SnakeGame) -> bool:
    return len(g.body) < 4 or _path(g, g.body[-1]) is not None


def _region(g: SnakeGame) -> int:
    seen, q = {g.body[0]}, deque([g.body[0]])
    while q:
        c = q.popleft()
        for _, n in _neighbours(g.w, g.h, c):
            if n not in seen and n not in g.cells:
                seen.add(n)
                q.append(n)
    return len(seen)


def _after(g: SnakeGame, d: str) -> SnakeGame:
    s = g.clone()
    s.step(d)
    return s


def teacher_target(g: SnakeGame) -> list[float]:
    safe = [d for d in DIRS if not g.dies(d)]
    if not safe:
        return [0.25] * 4
    good = []
    for d in safe:
        s = _after(g, d)
        route = [] if s.score > g.score else _path(s, s.food) if s.food else None
        if route is None:
            continue
        v = s.clone()
        for step in route:
            v.step(step)
            if not v.alive:
                break
        if v.alive and _tail_reachable(v):
            good.append((len(route), d))
    if good:
        best_len = min(t for t, _ in good)
        best = [d for t, d in good if t == best_len]
    else:
        pool = [d for d in safe if _tail_reachable(_after(g, d))] or safe
        area = {d: _region(_after(g, d)) for d in pool}
        best = [d for d in pool if area[d] == max(area.values())]
    return [1.0 / len(best) if d in best else 0.0 for d in DIRS]


def teacher_move(g: SnakeGame) -> str:
    t = teacher_target(g)
    return DIRS[max(range(4), key=t.__getitem__)]
