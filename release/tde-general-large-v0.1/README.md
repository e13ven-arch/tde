---
license: apache-2.0
language: en
library_name: tde
tags: [decision-model, typed-decisions, encoder, jevbench]
---

# tde-general-large-v0.1

A 395M encoder decision model: state + question + candidate options in, one forward pass, a probability over the options out. No text is generated.

- Backbone: ModernBERT-large architecture, initialised from the GLiClass-large encoder, fully fine-tuned.
- Readout: each option is read out of a `[MASK]` marker and scored against a decision token (pointer product); options are laid out with tied positions and a set attention mask, so the output is permutation-equivariant and independent of how many options are supplied.
- Question types: `noul` (yes/no), `choice` (pick one of K), `score` (pick one of K ordered levels). Up to 255 options, 1,024-token state budget.
- Training: one epoch over public classification sets recast as decisions, rule-generated worlds with program-computed labels, LegalBench rule-application tasks and the public typed-decisions training split. Loss: soft cross-entropy + ranked probability score.

## Use

```bash
pip install "git+https://github.com/e13ven-arch/tde@v0.1.1"
tde-serve --model tdelab/tde-general-large-v0.1 --port 8000
```

`POST /v1/systemone` takes `{state, questions: {id: {type, instructions, criteria}}}` and returns `{answers: {id: {type, noul | choice, probabilities, confidence}}}` (TypeSafe-compatible wire format). In Python:

```python
from tde.inference import Decider
d = Decider.from_run("tdelab/tde-general-large-v0.1")
d.decide("Customer: my card was declined at the shop.", {"type": "choice", "instructions": "Which intent?", "criteria": {"declined_card": "", "lost_card": "", "other": ""}})
```

## Evaluation (own runs)

| Set | Result |
|---|---|
| JevBench public items (231) | easy 48/48, standard 56/72, hard 41/111 |
| Full-label intent routing, one pass | clinc150 (150 labels) 0.928, banking77 (77) 0.853 |
| typed-decisions test (2,000), mixed-training mode | accuracy 0.695, Brier 0.094 |

Probabilities are the model's own softmax, no temperature. Public JevBench items were not used for training or checkpoint selection. Code, configs and experiment records: https://github.com/e13ven-arch/tde.
