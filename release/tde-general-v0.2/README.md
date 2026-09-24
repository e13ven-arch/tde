---
license: apache-2.0
base_model: tdelab/tde-general-v0.1
language: [en]
tags: [decision-model, calibrated-classification, typed-decisions, choice, snake, mlx]
---

# TDE general v0.2 — the typed-decision encoder, now also playing Snake

The same 150M encoder as [tde-general-v0.1](https://huggingface.co/tdelab/tde-general-v0.1) (ModernBERT-base, pointer
readout; one forward pass over `state + question + candidates` returns a probability per candidate), fine-tuned further
so it plays Snake from a plain-text board while keeping its general decisions.

## Snake

The state is the board as text, one token per cell (` .` empty, ` F` food, ` H` head, ` 1`–` 9` body: moves until that
cell is free) under a line with the length and heading; the question is "Which way should the snake move?" with the
candidates up, down, left and right. The top candidate is the move. An anti-trap shield vetoes a move after which the
head can no longer reach its tail and takes the model's next choice instead.

| laya-mlx demo protocol: 24x16, length 6, seeds 101–104, 600 moves | food per game | survived |
|---|---|---|
| tde-general-v0.2 with the anti-trap shield | 39 · 38 · 40 · 38 (mean 38.75) | 4 of 4 |
| tde-general-v0.2, model alone | 27 · 18 · 13 · 29 (mean 21.75) | 0 of 4 |
| laya-mlx as published (planner features in the options, Hamiltonian-cycle shield) | 20 · 24 · 23 · 16 (mean 20.75) | 4 of 4 |

About 12 ms per move with MLX on an M5 Pro. A live page where the model plays game after game:
`python -m integrations.snake.demo` in the [code repository](https://github.com/e13ven-arch/tde) (tutorial:
[integrations/snake](https://github.com/e13ven-arch/tde/tree/main/integrations/snake)).

## General decisions

| In-distribution test (7 public datasets, 2,000-item sample) | accuracy | NLL |
|---|---|---|
| tde-general-v0.1 | 87.4% | 0.313 |
| tde-general-v0.2 | 86.4% | 0.336 |

JevBench and the typed-decisions test were not re-run for this checkpoint.

## Training

Initialised from tde-general-v0.1 and fine-tuned for 2 epochs (6,250 steps of 32) on 100k Snake positions from 8x8,
12x12 and 24x16 boards, each labelled with the moves of a shortest-path planner that keeps its tail reachable. Trained
with MLX on one Mac in about 80 minutes (`integrations/snake/build_sft.py`, then `tde/mlx/train.py`).

## Usage

```python
# pip install git+https://github.com/e13ven-arch/tde   (package name: tde)
from tde.inference import Decider
d = Decider.from_run("tdelab/tde-general-v0.2")   # or a local release folder
d.decide(board_text, {"type": "choice", "instructions": "Which way should the snake move?",
                      "criteria": {m: m for m in ("up", "down", "left", "right")}})
```
General questions work exactly as with v0.1.

## Limitations

Without the shield the snake eventually traps itself: all four protocol games end before move 600. General accuracy
is about one point below v0.1; the limitations listed for v0.1 apply unchanged.
