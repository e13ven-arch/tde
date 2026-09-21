"""Tokenisation of (state, question, candidates) for the three readout modes.

Markers: every candidate is prefixed by [OPT]; a single [DECIDE] token closes
the candidate list. Scores are read from the [OPT] hidden states against the
[DECIDE] hidden state (pointer readout), so the number of candidates is free
at runtime (2..255) and no vocabulary is involved.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from tde.schema import DecisionExample

OPT_TOKEN = "[OPT]"
DECIDE_TOKEN = "[DECIDE]"


@dataclass
class EncodedExample:
    input_ids: list[int]          # joint: full sequence; branch: branch sequence; biencoder: state+question
    opt_positions: list[int]      # positions of [OPT] markers inside input_ids (joint/branch)
    decide_position: int          # position of [DECIDE] (joint/branch)
    state_ids: list[int]          # branch: state sequence; others: unused
    candidate_ids: list[list[int]]  # biencoder: one sequence per candidate
    level_index: list[int]        # per candidate: ordinal position for score, -1 otherwise
    target: list[float]
    primitive: str
    spans: list[tuple[int, int]] | None = None  # [start, end) of each candidate's text tokens (joint/branch)


class DecisionTokenizer:
    def __init__(self, tokenizer, max_state_tokens: int = 448, max_question_tokens: int = 64, max_candidate_tokens: int = 24,
                 max_total: int = 1024, marker: str = "mask"):
        """marker='mask' reuses the MLM [MASK] token as the [OPT]/[DECIDE] marker (its hidden state is pretrained to
        summarise context); marker='new' adds two fresh special tokens (v1 behaviour, weaker in Exp 001)."""
        self.tok = tokenizer
        self.marker = marker
        if marker == "mask" and tokenizer.mask_token_id is not None:
            self.added_tokens = 0
            self.opt_id = self.decide_id = tokenizer.mask_token_id
        elif marker == "eos" and tokenizer.eos_token_id is not None:
            self.added_tokens = 0
            self.opt_id = self.decide_id = tokenizer.eos_token_id
        else:
            self.added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": [OPT_TOKEN, DECIDE_TOKEN]})
            self.opt_id = tokenizer.convert_tokens_to_ids(OPT_TOKEN)
            self.decide_id = tokenizer.convert_tokens_to_ids(DECIDE_TOKEN)
        self.cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id  # may be None (Qwen)
        self.sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
        self.newline_ids = tokenizer("\n", add_special_tokens=False)["input_ids"]
        self.max_state = max_state_tokens
        self.max_question = max_question_tokens
        self.max_candidate = max_candidate_tokens
        self.max_total = max_total

    # ---------------------------------------------------------------- pieces
    def _ids(self, text: str, limit: int) -> list[int]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return ids[:limit]

    def _wrap(self, ids: list[int]) -> list[int]:
        return self._cls() + ids + self._sep()

    def _cls(self) -> list[int]:
        return [self.cls_id] if self.cls_id is not None else []

    def _sep(self) -> list[int]:
        """Segment separator: [SEP] for encoders; a newline for decoders whose only special token is eos (used as marker)."""
        if self.sep_id is not None and self.sep_id != self.opt_id:
            return [self.sep_id]
        return list(self.newline_ids)

    def _candidate_block(self, ex: DecisionExample) -> tuple[list[int], list[int], int, list[int], list[tuple[int, int]]]:
        ids, opt_pos, level_index, spans = [], [], [], []
        for i, c in enumerate(ex.candidates):
            opt_pos.append(len(ids))
            ids.append(self.opt_id)
            start = len(ids)
            ids += self._ids(" " + c.text(), self.max_candidate)
            spans.append((start, max(len(ids), start + 1)))
            level_index.append(i if ex.primitive == "score" else -1)
        decide_pos = len(ids)
        ids.append(self.decide_id)
        return ids, opt_pos, decide_pos, level_index, spans

    # ---------------------------------------------------------------- modes
    def encode(self, ex: DecisionExample, mode: str) -> EncodedExample:
        q_ids = self._ids(ex.question, self.max_question)
        cand_ids, opt_pos, decide_pos, level_index, spans = self._candidate_block(ex)
        if mode == "joint":
            # [CLS] state [SEP] question [SEP] [OPT] c1 ... [DECIDE] [SEP]
            budget = self.max_total - (len(q_ids) + len(cand_ids) + 4)
            s_ids = self._ids(ex.state, max(8, min(self.max_state, budget)))
            prefix = self._cls() + s_ids + self._sep() + q_ids + self._sep()
            input_ids = prefix + cand_ids + self._sep()
            offset = len(prefix)
            return EncodedExample(input_ids, [p + offset for p in opt_pos], decide_pos + offset, [], [], level_index, ex.target, ex.primitive,
                                  spans=[(a + offset, b + offset) for a, b in spans])
        if mode == "branch":
            state_ids = self._wrap(self._ids(ex.state, self.max_state))
            branch = self._cls() + q_ids + self._sep() + cand_ids + self._sep()
            offset = len(self._cls()) + len(q_ids) + len(self._sep())
            return EncodedExample(branch, [p + offset for p in opt_pos], decide_pos + offset, state_ids, [], level_index, ex.target, ex.primitive,
                                  spans=[(a + offset, b + offset) for a, b in spans])
        if mode == "biencoder":
            s_ids = self._ids(ex.state, self.max_state)
            sq = self._cls() + s_ids + self._sep() + q_ids + self._sep()
            cands = [self._wrap(self._ids(c.text(), self.max_candidate)) for c in ex.candidates]
            return EncodedExample(sq, [], -1, [], cands, level_index, ex.target, ex.primitive)
        raise ValueError(mode)


def _pad(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids, mask


def collate(encoded: list[EncodedExample], pad_id: int, mode: str) -> dict[str, torch.Tensor]:
    """Pad a list of EncodedExample into tensors. Targets are padded to Kmax with a candidate mask."""
    kmax = max(len(e.target) for e in encoded)
    target = torch.zeros((len(encoded), kmax))
    cand_mask = torch.zeros((len(encoded), kmax), dtype=torch.bool)
    level = torch.full((len(encoded), kmax), -1, dtype=torch.long)
    for i, e in enumerate(encoded):
        k = len(e.target)
        target[i, :k] = torch.tensor(e.target)
        cand_mask[i, :k] = True
        level[i, :k] = torch.tensor(e.level_index, dtype=torch.long)
    batch = {"target": target, "cand_mask": cand_mask, "level_index": level,
             "is_score": torch.tensor([e.primitive == "score" for e in encoded])}
    if mode in ("joint", "branch"):
        ids, mask = _pad([e.input_ids for e in encoded], pad_id)
        opt = torch.zeros((len(encoded), kmax), dtype=torch.long)
        for i, e in enumerate(encoded):
            opt[i, : len(e.opt_positions)] = torch.tensor(e.opt_positions, dtype=torch.long)
        span_start = torch.zeros((len(encoded), kmax), dtype=torch.long)
        span_end = torch.ones((len(encoded), kmax), dtype=torch.long)
        for i, e in enumerate(encoded):
            for j, (a, b) in enumerate(e.spans or []):
                span_start[i, j], span_end[i, j] = a, b
        batch.update(input_ids=ids, attention_mask=mask, opt_positions=opt, span_start=span_start, span_end=span_end,
                     decide_positions=torch.tensor([e.decide_position for e in encoded], dtype=torch.long))
        if mode == "branch":
            # de-duplicate identical states so a shared state is encoded once
            uniq: dict[tuple[int, ...], int] = {}
            state_index = []
            for e in encoded:
                key = tuple(e.state_ids)
                if key not in uniq:
                    uniq[key] = len(uniq)
                state_index.append(uniq[key])
            s_ids, s_mask = _pad([list(k) for k in uniq], pad_id)
            batch.update(state_ids=s_ids, state_mask=s_mask, state_index=torch.tensor(state_index, dtype=torch.long))
    elif mode == "biencoder":
        ids, mask = _pad([e.input_ids for e in encoded], pad_id)
        flat = [c for e in encoded for c in e.candidate_ids]
        c_ids, c_mask = _pad(flat, pad_id)
        owner = torch.tensor([i for i, e in enumerate(encoded) for _ in e.candidate_ids], dtype=torch.long)
        slot = torch.tensor([j for e in encoded for j in range(len(e.candidate_ids))], dtype=torch.long)
        batch.update(input_ids=ids, attention_mask=mask, cand_input_ids=c_ids, cand_attention_mask=c_mask,
                     cand_owner=owner, cand_slot=slot)
    return batch
