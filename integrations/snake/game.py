"""Snake on a W x H grid, rendered as a text board for the joint decision readout.

Board text: a header line, then one line per row in which every cell is exactly one token of the TDE tokenizer:
' .' empty, ' F' food, ' H' head, ' 1'..' 9' body (steps until that cell is vacated; 9 means nine or more).
"""
from __future__ import annotations

import random
from collections import deque

DIRS = ("up", "down", "left", "right")
DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


class SnakeGame:
    def __init__(self, width: int = 8, height: int = 8, initial_length: int = 3, seed: int = 0):
        self.w, self.h = width, height
        self.rng = random.Random(seed)
        y, x0 = height // 2, width // 2
        self.body = deque((x0 - i, y) for i in range(initial_length))  # body[0] is the head, heading right
        self.cells = set(self.body)
        self.heading = "right"
        self.alive, self.score, self.steps = True, 0, 0
        self.food = None
        self._spawn()

    def _spawn(self):
        free = [(x, y) for y in range(self.h) for x in range(self.w) if (x, y) not in self.cells]
        self.food = self.rng.choice(free) if free else None

    def next_head(self, d: str) -> tuple[int, int]:
        (x, y), (dx, dy) = self.body[0], DELTA[d]
        return x + dx, y + dy

    def dies(self, d: str) -> bool:
        nx, ny = self.next_head(d)
        if not (0 <= nx < self.w and 0 <= ny < self.h):
            return True
        eats = (nx, ny) == self.food
        return (nx, ny) in self.cells and (eats or (nx, ny) != self.body[-1])  # the tail moves away unless we eat

    def step(self, d: str):
        assert self.alive
        self.steps += 1
        if self.dies(d):
            self.alive = False
            return
        nh = self.next_head(d)
        eats = nh == self.food
        if not eats:
            self.cells.discard(self.body.pop())
        self.body.appendleft(nh)
        self.cells.add(nh)
        self.heading = d
        if eats:
            self.score += 1
            self._spawn()

    def clone(self) -> "SnakeGame":
        g = SnakeGame.__new__(SnakeGame)
        g.w, g.h, g.heading, g.alive, g.score, g.steps, g.food = self.w, self.h, self.heading, self.alive, self.score, self.steps, self.food
        g.body, g.cells = deque(self.body), set(self.cells)
        g.rng = random.Random()
        g.rng.setstate(self.rng.getstate())
        return g

    def rows(self) -> list[str]:
        n = len(self.body)
        age = {c: n - i for i, c in enumerate(self.body)}  # the tail (i = n-1) is vacated next step
        out = []
        for y in range(self.h):
            row = []
            for x in range(self.w):
                c = (x, y)
                row.append(" H" if c == self.body[0] else f" {min(9, age[c])}" if c in age else " F" if c == self.food else " .")
            out.append("".join(row))
        return out

    def header(self) -> str:
        return f"Length {len(self.body)}. Heading {self.heading}."

    def text(self) -> str:
        return self.header() + "\n" + "\n".join(self.rows())
