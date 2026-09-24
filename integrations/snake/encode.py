"""Snake states as DecisionExamples and a fast joint-layout encoder for batched rollouts.

The fast encoder builds the same input_ids as tde.model.encoding.DecisionTokenizer.encode(ex, "joint") by table lookup
(each board cell is one token), so rollouts avoid a tokenizer call per state. `check` asserts the two agree.
"""
from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np

from integrations.snake.game import DIRS, SnakeGame
from tde.schema import Candidate, DecisionExample

QUESTION = "Which way should the snake move?"


def to_example(g: SnakeGame, target: list[float], split: str, idx: str, meta: dict | None = None) -> DecisionExample:
    return DecisionExample(id=f"snake-{idx}", source_id=f"snake-{idx.split('-')[0]}", dataset=f"snake_{g.w}x{g.h}",
                           split=split, primitive="choice", state=g.text(), question=QUESTION,
                           candidates=[Candidate(d) for d in DIRS], target=list(target), meta=meta or {})


class FastEncoder:
    def __init__(self, dtok):
        self.dtok = dtok
        ids = lambda s: dtok.tok(s, add_special_tokens=False)["input_ids"]
        probe = ids("x\ny")
        self.nl = [t for t in probe if dtok.tok.decode([t]) == "\n"][0]  # the in-text newline, not a lone " \n"
        self.cell = {c: ids(c)[0] for c in [" .", " H", " F"] + [f" {k}" for k in range(1, 10)]}
        q = ids(QUESTION)[: dtok.max_question]
        cands, opt, spans = [], [], []
        for d in DIRS:
            opt.append(len(cands))
            cands.append(dtok.opt_id)
            start = len(cands)
            cands += ids(" " + d)[: dtok.max_candidate]
            spans.append((start, len(cands)))
        self.q, self.cands, self.opt, self.spans, self.decide = q, cands, opt, spans, len(cands)
        cands.append(dtok.decide_id)
        self._header = lru_cache(maxsize=4096)(lambda h: tuple(ids(h)))

    def state_ids(self, g: SnakeGame) -> list[int]:
        out = list(self._header(g.header()))
        for row in g.rows():
            out.append(self.nl)
            out += [self.cell[row[i : i + 2]] for i in range(0, len(row), 2)]
        return out[: self.dtok.max_state]

    def encode(self, g: SnakeGame):
        d = self.dtok
        prefix = d._cls() + self.state_ids(g) + d._sep() + self.q + d._sep()
        off = len(prefix)
        return (prefix + self.cands + d._sep(), [p + off for p in self.opt], [(a + off, b + off) for a, b in self.spans],
                self.decide + off)

    def batch(self, games: list[SnakeGame]) -> dict:
        """Joint-model inputs for a list of states. Equal lengths (one board size) need no padding mask."""
        enc = [self.encode(g) for g in games]
        T = max(len(e[0]) for e in enc)
        B, K = len(enc), len(DIRS)
        ids = np.full((B, T), self.dtok.pad_id, np.int32)
        for i, e in enumerate(enc):
            ids[i, : len(e[0])] = e[0]
        same = all(len(e[0]) == T for e in enc)
        out = {} if same else {"valid": mx.array(np.arange(T)[None, :] < np.array([len(e[0]) for e in enc])[:, None])}
        return {**out, "input_ids": mx.array(ids),
                "opt_positions": mx.array(np.array([e[1] for e in enc], np.int32)),
                "span_start": mx.array(np.array([[a for a, _ in e[2]] for e in enc], np.int32)),
                "span_end": mx.array(np.array([[b for _, b in e[2]] for e in enc], np.int32)),
                "level_index": mx.full((B, K), -1, dtype=mx.int32), "cand_mask": mx.ones((B, K), dtype=mx.bool_),
                "decide_positions": mx.array(np.array([e[3] for e in enc], np.int32))}


def check(enc: FastEncoder, games: list[SnakeGame]) -> int:
    """Number of states whose fast encoding differs from DecisionTokenizer's."""
    bad = 0
    for g in games:
        ref = enc.dtok.encode(to_example(g, [0.25] * 4, "test", "0-0"), "joint")
        ids, opt, spans, dec = enc.encode(g)
        bad += (ids != ref.input_ids) or (opt != ref.opt_positions) or (spans != ref.spans) or (dec != ref.decide_position)
    return bad
