"""Question templates and label-independent rendering.

Design rules (see docs/PLAN_v0.1.md §3.3):
* The template for an item is sampled independently of its label.
* `noul` templates that ask about a specific label are applied to a label drawn
  50/50 from {true label, random other label}, so every template is balanced by
  construction and the answer cannot be read off the template.
* `choice` candidate sets always contain the true label plus a random subset of
  the other labels, in random order.
* ~20% of template ids are held out (hash-based) and only used for evaluation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from tde.data.splits import template_is_held_out
from tde.schema import Candidate, DecisionExample

YES_NO = [Candidate("yes"), Candidate("no")]


@dataclass(frozen=True)
class Template:
    id: str
    primitive: str  # noul | choice | score
    text: str  # may contain {label} (noul about a label) or {question} (dataset-provided question)
    kind: str = "generic"  # generic | label_noul | provided_question


# --- classification-style datasets ---------------------------------------
CLASSIFICATION_TEMPLATES: dict[str, list[Template]] = {
    "banking77": [
        Template("banking77.choice.1", "choice", "Which banking intent best matches the customer's message?"),
        Template("banking77.choice.2", "choice", "What is the customer asking about? Pick the closest intent."),
        Template("banking77.choice.3", "choice", "Route this support message to the most relevant intent."),
        Template("banking77.choice.4", "choice", "Classify the user's request."),
        Template("banking77.noul.1", "noul", "Is the customer's message about '{label}'?", "label_noul"),
        Template("banking77.noul.2", "noul", "Should this message be routed to the '{label}' handler?", "label_noul"),
        Template("banking77.noul.3", "noul", "Does the intent '{label}' describe this request?", "label_noul"),
    ],
    "clinc150": [
        Template("clinc150.choice.1", "choice", "Which intent does the user's utterance express?"),
        Template("clinc150.choice.2", "choice", "Select the assistant skill that should handle this request."),
        Template("clinc150.choice.3", "choice", "What does the user want?"),
        Template("clinc150.noul.1", "noul", "Is the user asking for '{label}'?", "label_noul"),
        Template("clinc150.noul.2", "noul", "Would the '{label}' skill correctly handle this utterance?", "label_noul"),
        Template("clinc150.noul.scope", "noul", "Is this request within the assistant's supported scope?", "scope_noul"),
    ],
    "ag_news": [
        Template("ag_news.choice.1", "choice", "Which section of the newspaper does this article belong to?"),
        Template("ag_news.choice.2", "choice", "What is the main topic of this news item?"),
        Template("ag_news.choice.3", "choice", "Classify the article by category."),
        Template("ag_news.choice.4", "choice", "Pick the desk that should have written this story."),
        Template("ag_news.noul.1", "noul", "Is this article about {label}?", "label_noul"),
        Template("ag_news.noul.2", "noul", "Would an editor file this story under {label}?", "label_noul"),
    ],
    "yahoo_topics": [
        Template("yahoo.choice.1", "choice", "Which topic category does this question belong to?"),
        Template("yahoo.choice.2", "choice", "Classify the question by subject."),
        Template("yahoo.noul.1", "noul", "Is this question about {label}?", "label_noul"),
    ],
    "snli": [
        Template("snli.choice.1", "choice", "Given the premise, what is the relationship of the hypothesis to it?"),
        Template("snli.choice.2", "choice", "Does the hypothesis follow from, contradict, or remain undetermined by the premise?"),
        Template("snli.choice.3", "choice", "Classify the logical relation between the two sentences."),
        Template("snli.noul.1", "noul", "Is the relation between premise and hypothesis best described as '{label}'?", "label_noul"),
        Template("snli.noul.entail", "noul", "Does the hypothesis necessarily follow from the premise?", "entail_noul"),
    ],
    "go_emotions": [
        Template("goemo.choice.1", "choice", "Which emotion is most strongly expressed in this comment?"),
        Template("goemo.choice.2", "choice", "What is the dominant emotion of the writer?"),
        Template("goemo.choice.3", "choice", "Label the emotional tone of the text."),
        Template("goemo.noul.1", "noul", "Does the comment express {label}?", "label_noul"),
        Template("goemo.noul.2", "noul", "Would a reader describe the writer as feeling {label}?", "label_noul"),
    ],
}

ORDINAL_TEMPLATES: dict[str, list[Template]] = {
    "sst5": [
        Template("sst5.score.1", "score", "How positive is the sentiment of this review sentence?"),
        Template("sst5.score.2", "score", "Rate the reviewer's overall opinion."),
        Template("sst5.score.3", "score", "On the scale below, how favorable is this text?"),
        Template("sst5.noul.1", "noul", "Is the sentiment of this sentence {label}?", "label_noul"),
        Template("sst5.noul.pos", "noul", "Is the overall sentiment positive (rather than negative or neutral)?", "positive_noul"),
    ],
}

PROVIDED_QUESTION_TEMPLATES: dict[str, list[Template]] = {
    "boolq": [
        Template("boolq.noul.1", "noul", "Based on the passage, {question}?", "provided_question"),
        Template("boolq.noul.2", "noul", "According to the text: {question}? Answer yes or no.", "provided_question"),
        Template("boolq.noul.3", "noul", "{question}?", "provided_question"),
    ],
}


def templates_for(dataset: str) -> list[Template]:
    for table in (CLASSIFICATION_TEMPLATES, ORDINAL_TEMPLATES, PROVIDED_QUESTION_TEMPLATES):
        if dataset in table:
            return table[dataset]
    raise KeyError(f"no templates for dataset {dataset!r}")


def split_templates(dataset: str) -> tuple[list[Template], list[Template]]:
    """(training templates, held-out templates)."""
    ts = templates_for(dataset)
    held = [t for t in ts if template_is_held_out(t.id)]
    train = [t for t in ts if not template_is_held_out(t.id)]
    if not train:  # tiny template sets: keep at least one for training
        train, held = ts[:1], ts[1:]
    return train, held


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _pick_template(rng: random.Random, templates: list[Template], primitive_weights: dict[str, float] | None) -> Template:
    if primitive_weights:
        weights = [primitive_weights.get(t.primitive, 1.0) for t in templates]
        return rng.choices(templates, weights=weights, k=1)[0]
    return rng.choice(templates)


def render_classification(
    *,
    rng: random.Random,
    dataset: str,
    source_id: str,
    split: str,
    state: str,
    label_idx: int,
    label_names: list[str],
    label_descriptions: dict[str, str] | None,
    templates: list[Template],
    n_rewrites: int,
    k_max: int = 12,
    target_dist: list[float] | None = None,
    primitive_weights: dict[str, float] | None = None,
    extra_meta: dict | None = None,
    k_fixed: int | None = None,
    id_suffix: str = "",
    p_full_k: float = 0.0,
) -> list[DecisionExample]:
    """Render up to `n_rewrites` examples for one single-label (or soft-label) row.

    `target_dist`, if given, is a distribution over all `label_names` (human votes);
    it is renormalised over the sampled candidate subset.
    """
    out: list[DecisionExample] = []
    n_labels = len(label_names)
    descs = label_descriptions or {}
    for r in range(n_rewrites):
        t = _pick_template(rng, templates, primitive_weights)  # independent of label
        meta = {"template_kind": t.kind}
        if extra_meta:
            meta.update(extra_meta)
        if t.primitive == "choice":
            if k_fixed:
                k = min(n_labels, k_fixed)
            elif n_labels > 2 and p_full_k and rng.random() < p_full_k:
                k = n_labels
            else:
                k = min(n_labels, rng.randint(2, k_max)) if n_labels > 2 else n_labels
            others = [i for i in range(n_labels) if i != label_idx]
            chosen = [label_idx] + rng.sample(others, k - 1)
            rng.shuffle(chosen)
            cands = [Candidate(label_names[i], descs.get(label_names[i], "")) for i in chosen]
            if target_dist is not None:
                mass = [max(target_dist[i], 0.0) for i in chosen]
                z = sum(mass)
                target = [m / z for m in mass] if z > 0 else [1.0 if i == label_idx else 0.0 for i in chosen]
            else:
                target = [1.0 if i == label_idx else 0.0 for i in chosen]
            question = t.text
        elif t.primitive == "noul" and t.kind == "label_noul":
            # 50/50 positive / negative by construction
            if rng.random() < 0.5 or n_labels < 2:
                asked = label_idx
            else:
                asked = rng.choice([i for i in range(n_labels) if i != label_idx])
            question = t.text.format(label=label_names[asked])
            cands = list(YES_NO)
            if target_dist is not None:
                p_yes = min(max(target_dist[asked], 0.0), 1.0)
            else:
                p_yes = 1.0 if asked == label_idx else 0.0
            target = [p_yes, 1.0 - p_yes]
            meta["asked_label"] = label_names[asked]
        else:
            raise ValueError(f"template {t.id} of kind {t.kind} cannot be rendered by render_classification")
        ex = DecisionExample(
            id=f"{source_id}#{t.id}#{r}{id_suffix}", source_id=source_id, dataset=dataset, split=split,
            primitive=t.primitive, state=state, question=question, candidates=cands, target=target,
            template_id=t.id, meta=meta,
        )
        ex.validate()
        out.append(ex)
    return out


def render_noul(
    *, rng: random.Random, dataset: str, source_id: str, split: str, state: str,
    template: Template, p_yes: float, question_fill: dict | None = None, extra_meta: dict | None = None, suffix: str = "",
) -> DecisionExample:
    question = template.text.format(**(question_fill or {}))
    ex = DecisionExample(
        id=f"{source_id}#{template.id}{suffix}", source_id=source_id, dataset=dataset, split=split,
        primitive="noul", state=state, question=question, candidates=list(YES_NO),
        target=[p_yes, 1.0 - p_yes], template_id=template.id, meta=dict(extra_meta or {}, template_kind=template.kind),
    )
    ex.validate()
    return ex


def render_score(
    *, rng: random.Random, dataset: str, source_id: str, split: str, state: str,
    template: Template, levels: list[Candidate], target: list[float], extra_meta: dict | None = None, suffix: str = "",
) -> DecisionExample:
    ex = DecisionExample(
        id=f"{source_id}#{template.id}{suffix}", source_id=source_id, dataset=dataset, split=split,
        primitive="score", state=state, question=template.text, candidates=list(levels), target=list(target),
        template_id=template.id, meta=dict(extra_meta or {}, template_kind=template.kind, ordered=True),
    )
    ex.validate()
    return ex
