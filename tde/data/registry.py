"""Dataset registry for v0.1.

Each entry records the Hugging Face id, license, which primitive(s) it feeds,
how it is used (training / OOD tier / calibration-only) and how to read a row.
Adapters live in build.py; this file is data only so tests and docs can import
it without pulling `datasets`.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_id: str
    hf_config: str | None
    license: str
    usage: str  # "train" | "ood_tier1" | "ood_tier2" | "calibration_only" | "comparability"
    adapter: str  # name of the adapter function in build.py
    stage: int  # first stage that pulls this dataset (0 = smoke/stage 0)
    hf_splits: tuple[str, ...] = ("train",)
    notes: str = ""
    max_rows: int | None = None  # cap per dataset for stage 0 builds


REGISTRY: dict[str, DatasetSpec] = {
    "banking77": DatasetSpec(
        name="banking77", hf_id="mteb/banking77", hf_config=None, license="CC-BY-4.0",
        usage="train", adapter="single_label", stage=0, hf_splits=("train", "test"),
        notes="77 banking intents; label names are snake_case and are rewritten to natural text.",
    ),
    "clinc150": DatasetSpec(
        name="clinc150", hf_id="clinc/clinc_oos", hf_config="plus", license="CC-BY-3.0",
        usage="train", adapter="clinc", stage=0, hf_splits=("train", "validation", "test"),
        notes="150 intents + out-of-scope; OOS rows become noul 'is this in scope?' negatives.",
    ),
    "ag_news": DatasetSpec(
        name="ag_news", hf_id="fancyzhx/ag_news", hf_config=None, license="research/non-commercial (AG corpus terms)",
        usage="train", adapter="single_label", stage=0, hf_splits=("train", "test"), max_rows=40000,
        notes="4 news topics.",
    ),
    "yahoo_topics": DatasetSpec(
        name="yahoo_topics", hf_id="community-datasets/yahoo_answers_topics", hf_config=None, license="research",
        usage="ood_tier1", adapter="yahoo", stage=1, hf_splits=("test",), max_rows=20000,
        notes="10 topics; OOD Tier 1 for ag_news (same family, unseen dataset). Never trained on.",
    ),
    "sst5": DatasetSpec(
        name="sst5", hf_id="SetFit/sst5", hf_config=None, license="research",
        usage="train", adapter="sst5", stage=0, hf_splits=("train", "validation", "test"),
        notes="5 ordered sentiment levels -> score primitive.",
    ),
    "go_emotions": DatasetSpec(
        name="go_emotions", hf_id="google-research-datasets/go_emotions", hf_config="raw", license="Apache-2.0",
        usage="train", adapter="go_emotions_raw", stage=0, hf_splits=("train",), max_rows=60000,
        notes="Raw per-rater annotations aggregated into a vote distribution over 28 labels (human soft labels).",
    ),
    "boolq": DatasetSpec(
        name="boolq", hf_id="google/boolq", hf_config=None, license="CC-BY-SA-3.0",
        usage="train", adapter="boolq", stage=0, hf_splits=("train", "validation"),
        notes="Passage as state, question as noul.",
    ),
    "snli": DatasetSpec(
        name="snli", hf_id="stanfordnlp/snli", hf_config=None, license="CC-BY-SA-4.0",
        usage="train", adapter="snli", stage=0, hf_splits=("train", "validation", "test"), max_rows=80000,
        notes="premise + hypothesis -> entailment / neutral / contradiction (rows with label -1 dropped).",
    ),
    "synth_policy": DatasetSpec(
        name="synth_policy", hf_id="", hf_config=None, license="generated in this repo (Apache-2.0)",
        usage="train", adapter="synth", stage=1, max_rows=8000,
        notes="Rule-generated policy compliance worlds with programmatic labels (tde/data/synth.py). Split by rule combination.",
    ),
    "synth_multihop": DatasetSpec(
        name="synth_multihop", hf_id="", hf_config=None, license="generated in this repo (Apache-2.0)",
        usage="train", adapter="synth", stage=1, max_rows=8000,
        notes="Rule-generated JSON worlds (users / departments / tickets) with 2-3 hop questions.",
    ),
    "synth_temporal": DatasetSpec(
        name="synth_temporal", hf_id="", hf_config=None, license="generated in this repo (Apache-2.0)",
        usage="train", adapter="synth", stage=1, max_rows=8000,
        notes="Rule-generated dates / durations / amounts in mixed formats with arithmetic labels.",
    ),
    "hotpot_qa": DatasetSpec(
        name="hotpot_qa", hf_id="hotpotqa/hotpot_qa", hf_config="distractor", license="CC-BY-SA-4.0",
        usage="train", adapter="hotpot", stage=1, hf_splits=("train", "validation"), max_rows=30000,
        notes="Multi-hop QA. yes/no comparison questions -> noul; bridge questions -> choice over the gold answer plus "
              "distractor titles/answers. State = supporting paragraphs first, then distractor paragraphs.",
    ),
    "legalbench": DatasetSpec(
        name="legalbench", hf_id="nguha/legalbench", hf_config=None, license="CC-BY-4.0 (per-task; see tde/data/legalbench_tasks.json)",
        usage="train", adapter="legalbench", stage=1, hf_splits=("train", "test"), max_rows=1500,
        notes="118 rule-application tasks (97 yes/no, 21 small multiclass) with the official task instruction as the question. "
              "~20% of tasks (hash-selected) are held out entirely as a task-level OOD test; max_rows caps rows per task.",
    ),
    "chaos_nli": DatasetSpec(
        name="chaos_nli", hf_id="", hf_config=None, license="research",
        usage="calibration_only", adapter="chaos_nli", stage=1,
        notes="Manual download: https://github.com/easonnie/ChaosNLI (100 human votes per item). "
              "Set TDE_CHAOSNLI_DIR to the extracted directory.",
    ),
    "typed_decisions": DatasetSpec(
        name="typed_decisions", hf_id="LocalLLaMA/typed-decisions", hf_config="all", license="see dataset card",
        usage="train_and_report", adapter="typed_decisions", stage=1, hf_splits=("train", "test"),
        notes="LLM-teacher labels. HF train split (6,000 decisions) enters training as verdict-2.0 did; HF test split "
              "(2,000 decisions) is written to eval_only/ and reported as a 'specialist' row, never merged into test.jsonl.",
    ),
}

# Natural-language label names and (optional) paraphrases used for held-out label-paraphrase tests.
LABEL_PARAPHRASES: dict[str, dict[str, list[str]]] = {
    "ag_news": {
        "World": ["international news", "global affairs"],
        "Sports": ["athletics", "sporting events"],
        "Business": ["finance and business", "commerce"],
        "Sci/Tech": ["science and technology", "tech"],
    },
    "sst5": {
        "very negative": ["strongly negative"],
        "negative": ["somewhat negative"],
        "neutral": ["mixed or neutral"],
        "positive": ["somewhat positive"],
        "very positive": ["strongly positive"],
    },
    "snli": {
        "entailment": ["follows from the premise"],
        "neutral": ["undetermined by the premise"],
        "contradiction": ["contradicts the premise"],
    },
}


def humanize_label(raw: str) -> str:
    """'card_arrival' -> 'card arrival'; 'Sci/Tech' stays."""
    return raw.replace("_", " ").strip()
