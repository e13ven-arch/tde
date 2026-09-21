"""Leakage guards: the template must not reveal the answer, and noul templates must be balanced."""
import random
from collections import Counter

from tde.data.templates import CLASSIFICATION_TEMPLATES, render_classification, split_templates


def _render_many(n_items=600, n_labels=6, seed=0):
    rng = random.Random(seed)
    names = [f"label{i}" for i in range(n_labels)]
    train_t, _ = split_templates("ag_news")
    out = []
    for i in range(n_items):
        label = rng.randrange(n_labels)
        out += render_classification(rng=rng, dataset="ag_news", source_id=f"ag:{i}", split="train", state=f"state {i}",
                                     label_idx=label, label_names=names, label_descriptions=None, templates=train_t, n_rewrites=4)
    return out


def test_noul_templates_balanced():
    exs = [e for e in _render_many() if e.primitive == "noul"]
    by_t = Counter()
    yes_t = Counter()
    for e in exs:
        by_t[e.template_id] += 1
        yes_t[e.template_id] += int(e.target[0] > 0.5)
    for t, n in by_t.items():
        frac = yes_t[t] / n
        assert 0.4 < frac < 0.6, f"{t} yes-rate {frac:.2f} (template leaks the answer)"


def test_choice_contains_true_label_and_is_shuffled():
    exs = [e for e in _render_many() if e.primitive == "choice"]
    positions = Counter(e.label for e in exs)
    assert all(any(c.name == e.candidates[e.label].name for c in e.candidates) for e in exs)
    # true label must not sit at a fixed position
    assert max(positions.values()) / len(exs) < 0.6


def test_template_choice_independent_of_label():
    exs = _render_many()
    joint = Counter((e.template_id, e.meta.get("asked_label", e.candidates[e.label].name if e.primitive == "choice" else "")) for e in exs)
    # every training template is used for many different labels
    per_t = Counter()
    for (t, _), n in joint.items():
        per_t[t] += 1
    assert min(per_t.values()) >= 3


def test_all_datasets_have_templates():
    for ds in CLASSIFICATION_TEMPLATES:
        train_t, _ = split_templates(ds)
        assert train_t, ds
