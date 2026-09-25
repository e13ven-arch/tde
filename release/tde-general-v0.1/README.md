---
license: apache-2.0
base_model: answerdotai/ModernBERT-base
language: [en]
tags: [decision-model, calibrated-classification, typed-decisions, noul, choice, score]
---

# TDE general v0.1 — a 150M typed-decision encoder

One forward pass over `state + question + candidates` returns a probability distribution over the candidates. Three question types share one primitive: `noul` (yes/no probability), `choice` (2–255 named options), `score` (2–10 ordered levels, plus the expected level). No text is generated.

**What it is.** ModernBERT-base (149.6M) initialised from the GLiClass modern-base v3.0 encoder, fully fine-tuned with a pointer readout: every candidate is prefixed by a `[MASK]` token; the candidate representation is that `[MASK]` state plus the mean over the candidate's tokens; scores are dot products against a final `[MASK]` ("decide") state. Loss: soft cross-entropy + ranked probability score (ordinal) + permutation-consistency KL. Probabilities are used as trained (post-hoc temperatures ≈ 1).

**What it is not.** It is an open design derived from public material and small-scale experiments. It does not probe, call, distil from or otherwise use TypeSafe's Jev; all Jev figures quoted here are third-party publications.

## Results (this checkpoint, no task-specific stage)

| Evaluation | Result | Reference rows (third party) |
|---|---|---|
| JevBench v1.2 public items (231) | easy 100% · standard 61.1% · hard 34.2% · Brier 0.552 | openJev verdict hard 38.2% · Jev 1.13 hard 74.1% |
| typed-decisions test (2,000), mixed-training mode | 66.6% · Brier 0.109 | Jev 72.7% / 0.148 (generalist) |
| Full-label intent routing, one pass | clinc150 (150 labels) 86.5%, 82% of decisions auto-executable at ≤5% error · banking77 (77) 75.8%, 51% | — |
| In-distribution test (7 public datasets, 6,000) | 87.5%, ECE 0.014 (noise floor 0.008) | — |

A separate checkpoint `tde-typed-specialist-v0.1` (this model + 8 epochs on the typed-decisions train split) reaches 76.3% / Brier 0.063 on typed-decisions test (specialist mode; Jev 72.7 / 0.148, verdict-2.0 77.1 / 0.064 as self-reported).

## Usage

```python
# pip install git+https://github.com/e13ven-arch/tde   (package name: tde)
from tde.inference import Decider
d = Decider.from_run("path/to/tde-general-v0.1")
d.decide("Customer: my card was charged twice for one order.",
         {"type": "choice", "instructions": "Which team should handle this?",
          "criteria": {"billing": "charges and refunds", "technical": "app errors", "account": "login and identity"}})
# {'type': 'choice', 'probabilities': {...}, 'label': 'billing', 'confidence': ...}
```
Batch several questions over one state with `decide_batch`. For candidate sets larger than ~40, use chunked inference (`predict_chunked`, `--chunk_k`).

## Training data and provenance

Mixture v0.5 (≈186k examples used, 1 epoch): banking77 (CC-BY-4.0), clinc150 (CC-BY-3.0), AG News (research use), SST-5, GoEmotions (Apache-2.0, human vote distributions), BoolQ (CC-BY-SA-3.0), SNLI (CC-BY-SA-4.0), LegalBench rule tasks (CC-BY-4.0, 22 tasks held out), HotpotQA (CC-BY-SA-4.0), typed-decisions train split (LLM-teacher labels), and rule-generated policy / multi-hop / temporal worlds with programmatic labels (generator released with the code, Apache-2.0). Splits are assigned by source item before any rewriting; question templates are sampled independently of labels; yes/no questions are balanced by construction. A 13-gram overlap check between training states and the JevBench public items found no overlap; typed-decisions test items are never trained on.

## Limitations

Knowledge-heavy decisions (multi-hop, traps, temporal arithmetic) remain near chance on JevBench's hard tier; large fine-grained label sets (77+) lose accuracy versus small candidate subsets; ordinal `score` questions are the least accurate and least calibrated primitive; English only.
