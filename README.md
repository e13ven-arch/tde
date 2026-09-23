# TDE — Typed Decision Encoder (v0.1)

A research codebase for **non-autoregressive, calibrated decision models**: given a `state`, a natural-language `question` and a runtime list of `candidates`, the model returns a probability distribution over the candidates in one forward pass. Three question types share one primitive:

| type | candidates | output |
|---|---|---|
| `noul` | 2 (yes / no) | P(yes) |
| `choice` | 2–255 named options | distribution |
| `score` | 2–10 ordered levels | distribution + expected level |

This is a **matched-parameter study**, not a new architecture: encoder + candidate scorer designs already exist (GLiClass, TARS, openJev-verdict-2.0, Laya). What this repo adds is a controlled comparison of readout designs, encoder vs decoder at matched size and data, calibration objectives measured against human label distributions, and OOD / few-shot curves with pre-registered endpoints. See [docs/PLAN_v0.1.md](docs/PLAN_v0.1.md).

## Layout

```
tde/              model, data pipeline, losses, calibration, evaluation, inference API
scripts/          build_data.py · train.py · evaluate.py · run_jevbench.py · export_release.py · …
configs/release/  the two released models' training configs (general = Exp 010, specialist = Exp 014)
configs/          one YAML per experiment (ablations documented in docs/RESULTS_*.md)
integrations/     JevBench in-process adapter and tests
release/          model cards of tdelab/tde-general-v0.1 and tdelab/tde-typed-specialist-v0.1 (weights on Hugging Face)
tests/            split determinism, template leakage, metrics, model shapes, losses, synthetic data
docs/             plan, data inventories, results, release record
ops/              the authors' training-host scripts (not needed to use the model)
data/ runs/       generated (git-ignored)
```

## Released models

| Model | Hugging Face | Config | Use for |
|---|---|---|---|
| tde-general-v0.1 | `tdelab/tde-general-v0.1` | `configs/release/tde-general-v0.1.yaml` | zero-shot typed decisions (JevBench-style workloads) |
| tde-typed-specialist-v0.1 | `tdelab/tde-typed-specialist-v0.1` | `configs/release/tde-typed-specialist-v0.1.yaml` | the four typed-decisions workflows (specialist mode) |

```python
from tde.inference import Decider
d = Decider.from_run("tdelab/tde-general-v0.1")   # or a local run / release directory
d.decide(state, {"type": "choice", "instructions": "...", "criteria": {"a": "...", "b": "..."}})
```

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                   # no downloads needed
python scripts/smoke_test.py                # tiny synthetic end-to-end run
python scripts/build_data.py --stage 0      # downloads public datasets, writes data/v0.1/*.jsonl
python scripts/train.py --config configs/exp001_readout.yaml --readout joint
python scripts/evaluate.py --run runs/exp001_joint_full --split test
python scripts/run_baseline.py --baseline nli --limit 2000    # Track-B zero-shot baseline
```

Data inventory and licenses: [docs/DATA.md](docs/DATA.md).

## Rules this repo follows

- Splits are assigned by **source item id before any rewriting**; templates are sampled independently of labels; yes/no templates get negatives by construction.
- Every reported cell has ≥ 2,000 test items and paired bootstrap CIs.
- Calibration headlines use human label distributions or verifiable labels, never LLM-consensus labels.
- Jev (TypeSafe) outputs are never used for training, distillation, model selection or tuning; Jev numbers are cited from third-party publications only.
