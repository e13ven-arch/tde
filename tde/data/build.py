"""Build the v0.1 decision corpus from public datasets.

Usage (see scripts/build_data.py):
    build(datasets=["banking77", ...], out_dir="data/v0.1", stage=0, max_rows=None, seed=0)

Every source row gets a source_id and a split *before* rendering; rendering
then produces <= n_rewrites examples per row (train) or a fixed evaluation
rendering (calibration / test) using training templates plus held-out
templates for the held-out-template control.
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

from tde.data.registry import REGISTRY, DatasetSpec, humanize_label
from tde.data.splits import assign_split, make_source_id
from tde.data.templates import (
    YES_NO,
    Candidate,
    Template,
    render_classification,
    render_noul,
    render_score,
    split_templates,
)
from tde.schema import DecisionExample, write_jsonl

N_REWRITES_TRAIN = 4  # <= 8 per plan; 4 keeps stage 0 small
N_REWRITES_EVAL = 2   # one training-template rendering + one held-out-template rendering when available


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _load(spec: DatasetSpec, hf_split: str):
    from datasets import load_dataset  # lazy import so tests do not need it

    kwargs = {"split": hf_split}
    if spec.hf_config:
        return load_dataset(spec.hf_id, spec.hf_config, **kwargs)
    return load_dataset(spec.hf_id, **kwargs)


def _iter_rows(spec: DatasetSpec, max_rows: int | None, seed: int) -> Iterable[tuple[str, dict]]:
    """Yield (row_key, row) across the spec's HF splits, capped and shuffled deterministically."""
    rows: list[tuple[str, dict]] = []
    for hs in spec.hf_splits:
        ds = _load(spec, hs)
        for i, row in enumerate(ds):
            rows.append((f"{hs}/{i}", row))
    rng = random.Random(seed)
    rng.shuffle(rows)
    cap = max_rows or spec.max_rows
    if cap:
        rows = rows[:cap]
    return rows


def _eval_templates(dataset: str) -> tuple[list[Template], list[Template]]:
    return split_templates(dataset)


def _render_rows_classification(
    *, spec: DatasetSpec, rows: Iterable[tuple[str, dict]], read: Callable[[dict], tuple[str, int, list[float] | None] | None],
    label_names: list[str], label_descriptions: dict[str, str] | None, seed: int, k_max: int = 12,
) -> list[DecisionExample]:
    train_t, held_t = _eval_templates(spec.name)
    ok = lambda t: (t.primitive == "choice" and t.kind == "generic") or (t.primitive == "noul" and t.kind == "label_noul")
    train_t, held_t = [t for t in train_t if ok(t)], [t for t in held_t if ok(t)]
    out: list[DecisionExample] = []
    for row_key, row in rows:
        parsed = read(row)
        if parsed is None:
            continue
        state, label_idx, dist = parsed
        if not state or not state.strip():
            continue
        source_id = make_source_id(spec.name, row_key)
        split = assign_split(source_id)
        rng = random.Random(f"{seed}:{source_id}")
        if split == "train":
            out += render_classification(
                rng=rng, dataset=spec.name, source_id=source_id, split=split, state=state, label_idx=label_idx,
                label_names=label_names, label_descriptions=label_descriptions, templates=train_t,
                n_rewrites=N_REWRITES_TRAIN, k_max=k_max, target_dist=dist,
            )
        else:
            out += render_classification(
                rng=rng, dataset=spec.name, source_id=source_id, split=split, state=state, label_idx=label_idx,
                label_names=label_names, label_descriptions=label_descriptions, templates=train_t,
                n_rewrites=1, k_max=k_max, target_dist=dist, extra_meta={"template_group": "train_templates"},
            )
            if held_t:
                out += render_classification(
                    rng=rng, dataset=spec.name, source_id=source_id, split=split, state=state, label_idx=label_idx,
                    label_names=label_names, label_descriptions=label_descriptions, templates=held_t,
                    n_rewrites=1, k_max=k_max, target_dist=dist, extra_meta={"template_group": "held_out_templates"},
                )
            # full label set as candidates (K = all labels): tests candidate-count extrapolation; written to eval_only/
            choice_t = [t for t in train_t if t.primitive == "choice"]
            if choice_t and len(label_names) > 2:
                out += render_classification(
                    rng=rng, dataset=spec.name, source_id=source_id, split=split, state=state, label_idx=label_idx,
                    label_names=label_names, label_descriptions=label_descriptions, templates=choice_t,
                    n_rewrites=1, k_max=k_max, target_dist=dist, k_fixed=len(label_names),
                    extra_meta={"template_group": "full_k", "eval_group": "full_k"}, id_suffix="#fullk",
                )
    return out


# --------------------------------------------------------------------------
# adapters: each returns a list of DecisionExample for one DatasetSpec
# --------------------------------------------------------------------------

def _label_names(ds, label_col: str = "label", text_col: str = "label_text") -> list[str]:
    """ClassLabel names when available, else derive from a label_text column."""
    feat = ds.features.get(label_col)
    if feat is not None and hasattr(feat, "names"):
        return [humanize_label(n) for n in feat.names]
    if text_col in ds.column_names:
        mapping: dict[int, str] = {}
        for row in ds:
            mapping.setdefault(int(row[label_col]), str(row[text_col]))
        return [humanize_label(mapping[i]) for i in range(max(mapping) + 1)]
    raise ValueError(f"cannot derive label names for {ds}")


def adapter_single_label(spec: DatasetSpec, max_rows: int | None, seed: int) -> list[DecisionExample]:
    rows = list(_iter_rows(spec, max_rows, seed))
    if not rows:
        return []
    ds = _load(spec, spec.hf_splits[0])
    names = _label_names(ds)

    def read(row):
        return row["text"], int(row["label"]), None

    return _render_rows_classification(spec=spec, rows=rows, read=read, label_names=names, label_descriptions=None, seed=seed, k_max=spec.k_max)


def adapter_clinc(spec: DatasetSpec, max_rows: int | None, seed: int) -> list[DecisionExample]:
    rows = list(_iter_rows(spec, max_rows, seed))
    ds = _load(spec, spec.hf_splits[0])
    raw_names = ds.features["intent"].names
    oos_idx = raw_names.index("oos")
    in_scope = [i for i in range(len(raw_names)) if i != oos_idx]
    names = [humanize_label(raw_names[i]) for i in in_scope]
    remap = {orig: new for new, orig in enumerate(in_scope)}

    def read(row):
        if int(row["intent"]) == oos_idx:
            return None
        return row["text"], remap[int(row["intent"])], None

    out = _render_rows_classification(spec=spec, rows=rows, read=read, label_names=names, label_descriptions=None, seed=seed, k_max=spec.k_max)
    # scope noul: balanced yes (in-scope) / no (oos)
    scope_t = next(t for t in spec_templates(spec.name) if t.kind == "scope_noul")
    oos_rows = [(k, r) for k, r in rows if int(r["intent"]) == oos_idx]
    in_rows = [(k, r) for k, r in rows if int(r["intent"]) != oos_idx]
    rng = random.Random(f"{seed}:clinc-scope")
    in_rows = rng.sample(in_rows, min(len(in_rows), len(oos_rows)))
    for key, row in oos_rows + in_rows:
        source_id = make_source_id(spec.name, key)
        out.append(render_noul(
            rng=rng, dataset=spec.name, source_id=source_id, split=assign_split(source_id), state=row["text"],
            template=scope_t, p_yes=0.0 if int(row["intent"]) == oos_idx else 1.0, extra_meta={"scope_control": True},
        ))
    return out


def adapter_yahoo(spec: DatasetSpec, max_rows: int | None, seed: int) -> list[DecisionExample]:
    rows = list(_iter_rows(spec, max_rows, seed))
    ds = _load(spec, spec.hf_splits[0])
    names = [humanize_label(n) for n in ds.features["topic"].names]

    def read(row):
        state = " ".join(s for s in (row.get("question_title", ""), row.get("question_content", "")) if s)
        return state, int(row["topic"]), None

    return _render_rows_classification(spec=spec, rows=rows, read=read, label_names=names, label_descriptions=None, seed=seed)


def adapter_snli(spec: DatasetSpec, max_rows: int | None, seed: int) -> list[DecisionExample]:
    rows = list(_iter_rows(spec, max_rows, seed))
    names = ["entailment", "neutral", "contradiction"]
    descs = {
        "entailment": "the hypothesis must be true given the premise",
        "neutral": "the premise does not determine whether the hypothesis is true",
        "contradiction": "the hypothesis cannot be true if the premise is true",
    }

    def read(row):
        if int(row["label"]) < 0:
            return None
        state = f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}"
        return state, int(row["label"]), None

    out = _render_rows_classification(spec=spec, rows=rows, read=read, label_names=names, label_descriptions=descs, seed=seed, k_max=3)
    entail_t = next(t for t in spec_templates(spec.name) if t.kind == "entail_noul")
    rng = random.Random(f"{seed}:snli-entail")
    pos = [(k, r) for k, r in rows if int(r["label"]) == 0]
    neg = [(k, r) for k, r in rows if int(r["label"]) in (1, 2)]
    neg = rng.sample(neg, min(len(neg), len(pos)))
    for key, row in pos + neg:
        source_id = make_source_id(spec.name, key)
        out.append(render_noul(
            rng=rng, dataset=spec.name, source_id=source_id, split=assign_split(source_id),
            state=f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}", template=entail_t,
            p_yes=1.0 if int(row["label"]) == 0 else 0.0,
        ))
    return out


def adapter_go_emotions_raw(spec: DatasetSpec, max_rows: int | None, seed: int) -> list[DecisionExample]:
    """Aggregate per-rater rows into one vote distribution per comment (human soft labels)."""
    rows = list(_iter_rows(spec, None, seed))  # cap applied after aggregation
    ds = _load(spec, spec.hf_splits[0])
    emotion_cols = [c for c in ds.column_names if c not in {
        "text", "id", "author", "subreddit", "link_id", "parent_id", "created_utc", "rater_id", "example_very_unclear"}]
    votes: dict[str, list[int]] = defaultdict(lambda: [0] * len(emotion_cols))
    texts: dict[str, str] = {}
    n_raters: Counter = Counter()
    for _, row in rows:
        cid = row["id"]
        texts[cid] = row["text"]
        n_raters[cid] += 1
        for j, c in enumerate(emotion_cols):
            votes[cid][j] += int(row[c])
    ids = sorted(votes)
    rng = random.Random(seed)
    rng.shuffle(ids)
    cap = max_rows or spec.max_rows
    if cap:
        ids = ids[:cap]
    names = [humanize_label(c) for c in emotion_cols]
    agg_rows = []
    for cid in ids:
        v = votes[cid]
        tot = sum(v)
        if tot == 0 or n_raters[cid] < 2:
            continue
        dist = [x / tot for x in v]
        agg_rows.append((cid, {"text": texts[cid], "dist": dist, "label": max(range(len(v)), key=lambda i: v[i]), "n_raters": n_raters[cid]}))

    def read(row):
        return row["text"], int(row["label"]), row["dist"]

    return _render_rows_classification(spec=spec, rows=agg_rows, read=read, label_names=names, label_descriptions=None, seed=seed, k_max=8)


def adapter_boolq(spec: DatasetSpec, max_rows: int | None, seed: int) -> list[DecisionExample]:
    rows = list(_iter_rows(spec, max_rows, seed))
    train_t, held_t = _eval_templates(spec.name)
    out = []
    for key, row in rows:
        source_id = make_source_id(spec.name, key)
        split = assign_split(source_id)
        rng = random.Random(f"{seed}:{source_id}")
        q = row["question"].strip().rstrip("?")
        p_yes = 1.0 if bool(row["answer"]) else 0.0
        if split == "train":
            for r in range(min(N_REWRITES_TRAIN, len(train_t))):
                t = rng.choice(train_t)
                out.append(render_noul(rng=rng, dataset=spec.name, source_id=source_id, split=split, state=row["passage"],
                                       template=t, p_yes=p_yes, question_fill={"question": q}, suffix=f"#{r}"))
        else:
            t = rng.choice(train_t)
            out.append(render_noul(rng=rng, dataset=spec.name, source_id=source_id, split=split, state=row["passage"],
                                   template=t, p_yes=p_yes, question_fill={"question": q}, extra_meta={"template_group": "train_templates"}))
            if held_t:
                t2 = rng.choice(held_t)
                out.append(render_noul(rng=rng, dataset=spec.name, source_id=source_id, split=split, state=row["passage"],
                                       template=t2, p_yes=p_yes, question_fill={"question": q}, extra_meta={"template_group": "held_out_templates"}))
    return out


SST5_LEVELS = [
    Candidate("very negative", "the writer strongly dislikes it"),
    Candidate("negative", "the writer is somewhat unfavorable"),
    Candidate("neutral", "mixed or no clear opinion"),
    Candidate("positive", "the writer is somewhat favorable"),
    Candidate("very positive", "the writer strongly likes it"),
]


def adapter_sst5(spec: DatasetSpec, max_rows: int | None, seed: int) -> list[DecisionExample]:
    rows = list(_iter_rows(spec, max_rows, seed))
    train_t, held_t = _eval_templates(spec.name)
    names = [c.name for c in SST5_LEVELS]
    out = []
    for key, row in rows:
        source_id = make_source_id(spec.name, key)
        split = assign_split(source_id)
        rng = random.Random(f"{seed}:{source_id}")
        label = int(row["label"])
        target = [1.0 if i == label else 0.0 for i in range(5)]

        def render_with(templates: list[Template], n: int, group: str | None):
            res = []
            for r in range(n):
                t = rng.choice(templates)
                meta = {"template_group": group} if group else {}
                if t.primitive == "score":
                    res.append(render_score(rng=rng, dataset=spec.name, source_id=source_id, split=split, state=row["text"],
                                            template=t, levels=SST5_LEVELS, target=target, extra_meta=meta, suffix=f"#{r}"))
                elif t.kind == "label_noul":
                    asked = label if rng.random() < 0.5 else rng.choice([i for i in range(5) if i != label])
                    res.append(render_noul(rng=rng, dataset=spec.name, source_id=source_id, split=split, state=row["text"],
                                           template=t, p_yes=1.0 if asked == label else 0.0, question_fill={"label": names[asked]},
                                           extra_meta=dict(meta, asked_label=names[asked]), suffix=f"#{r}"))
                elif t.kind == "positive_noul":
                    res.append(render_noul(rng=rng, dataset=spec.name, source_id=source_id, split=split, state=row["text"],
                                           template=t, p_yes=1.0 if label >= 3 else 0.0, extra_meta=meta, suffix=f"#{r}"))
            return res

        if split == "train":
            out += render_with(train_t, N_REWRITES_TRAIN, None)
        else:
            out += render_with(train_t, 1, "train_templates")
            if held_t:
                out += render_with(held_t, 1, "held_out_templates")
    return out


HOTPOT_NOUL_T = ["{question}", "Answer from the passages: {question}", "Based only on the text, {question}"]
HOTPOT_CHOICE_T = ["{question}", "Using the passages, answer: {question}", "{question} Pick the correct answer."]


def adapter_hotpot(spec: DatasetSpec, max_rows: int | None, seed: int) -> list[DecisionExample]:
    """HotpotQA distractor: comparison yes/no -> noul; bridge/span -> choice (gold + distractors from context titles)."""
    rows = list(_iter_rows(spec, max_rows, seed))
    all_answers = [r["answer"] for _, r in rows if r["answer"].lower() not in ("yes", "no")]
    out = []
    for key, row in rows:
        source_id = make_source_id(spec.name, row["id"])
        split = assign_split(source_id)
        rng = random.Random(f"{seed}:{source_id}")
        titles = row["context"]["title"]; sents = row["context"]["sentences"]
        sup = set(row["supporting_facts"]["title"])
        paras = [(t, " ".join(ss)) for t, ss in zip(titles, sents)]
        sup_p = [p for p in paras if p[0] in sup]
        oth_p = [p for p in paras if p[0] not in sup]
        rng.shuffle(oth_p)
        chosen = sup_p + oth_p[: rng.randint(1, 3)]
        state = "\n\n".join(f"{t}: {txt}" for t, txt in chosen)
        q = row["question"].strip()
        ans = row["answer"].strip()
        meta = {"hotpot_type": row["type"], "hotpot_level": row["level"]}
        n_re = N_REWRITES_TRAIN if split == "train" else 1
        if ans.lower() in ("yes", "no"):
            for r in range(min(n_re, len(HOTPOT_NOUL_T))):
                t = rng.choice(HOTPOT_NOUL_T)
                e = DecisionExample(id=f"{source_id}#noul#{r}", source_id=source_id, dataset=spec.name, split=split, primitive="noul",
                                    state=state, question=t.format(question=q), candidates=list(YES_NO),
                                    target=[1.0, 0.0] if ans.lower() == "yes" else [0.0, 1.0], template_id="hotpot.noul", meta=meta)
                e.validate(); out.append(e)
        else:
            pool = [t for t in titles if t.lower() != ans.lower()]
            for r in range(min(n_re, len(HOTPOT_CHOICE_T))):
                k = rng.randint(3, 5)
                dis = rng.sample(pool, min(k - 1, len(pool)))
                while len(dis) < k - 1:
                    cand = rng.choice(all_answers)
                    if cand.lower() != ans.lower() and cand not in dis: dis.append(cand)
                names = [ans] + dis
                rng.shuffle(names)
                if len(set(n.lower() for n in names)) != len(names):
                    continue
                t = rng.choice(HOTPOT_CHOICE_T)
                e = DecisionExample(id=f"{source_id}#choice#{r}", source_id=source_id, dataset=spec.name, split=split, primitive="choice",
                                    state=state, question=t.format(question=q), candidates=[Candidate(n) for n in names],
                                    target=[1.0 if n == ans else 0.0 for n in names], template_id="hotpot.choice", meta=meta)
                e.validate(); out.append(e)
    return out


LEGALBENCH_WRAP = ["{ins}", "{ins} Answer based only on the text.", "Task: {ins}"]


def adapter_legalbench(spec: DatasetSpec, max_rows: int | None, seed: int) -> list[DecisionExample]:
    """LegalBench rule-application tasks from the vendored table; ~20% of tasks held out as task-level OOD test."""
    import csv
    from huggingface_hub import hf_hub_download
    from tde.data.splits import template_is_held_out

    table = json.load(open(Path(__file__).with_name("legalbench_tasks.json"), encoding="utf-8"))
    out = []
    cap = max_rows or spec.max_rows or 1500
    for task, t in table.items():
        held_task = template_is_held_out(f"legalbench:{task}", salt="tde-v0.1-legalbench-tasks", fraction=0.2)
        rows = []
        for hs in ("train", "test"):
            try:
                pth = hf_hub_download("nguha/legalbench", f"data/{task}/{hs}.tsv", repo_type="dataset")
                rows += [(f"{hs}/{i}", r) for i, r in enumerate(csv.DictReader(open(pth, encoding="utf-8"), delimiter="\t"))]
            except Exception as e:  # noqa: BLE001
                print(f"[legalbench] {task}/{hs}: {type(e).__name__}")
        rng = random.Random(f"{seed}:legalbench:{task}")
        rng.shuffle(rows)
        rows = rows[:cap]
        labels = t["labels"]
        for key, r in rows:
            ans = (r.get("answer") or "").strip()
            if not ans:
                continue
            source_id = make_source_id("legalbench", f"{task}/{key}")
            split = "test" if held_task else assign_split(source_id)
            parts = [f"{c.replace('_', ' ').capitalize()}: {r[c]}" if len(t["columns"]) > 1 else str(r[c]) for c in t["columns"] if r.get(c)]
            state = "\n".join(parts)
            if not state.strip():
                continue
            rr = random.Random(f"{seed}:{source_id}")
            question = rr.choice(LEGALBENCH_WRAP).format(ins=t["instruction"])
            meta = {"task": task, "task_held_out": held_task, "label_source": "human"}
            if t["kind"] == "noul":
                yes = ans.lower() == "yes"
                e = DecisionExample(id=f"{source_id}#noul", source_id=source_id, dataset="legalbench", split=split, primitive="noul",
                                    state=state, question=question, candidates=list(YES_NO), target=[1.0, 0.0] if yes else [0.0, 1.0],
                                    template_id=f"legalbench.{task}", meta=meta)
            else:
                if ans not in labels:
                    continue
                names = list(labels); rr.shuffle(names)
                e = DecisionExample(id=f"{source_id}#choice", source_id=source_id, dataset="legalbench", split=split, primitive="choice",
                                    state=state, question=question, candidates=[Candidate(n) for n in names],
                                    target=[1.0 if n == ans else 0.0 for n in names], template_id=f"legalbench.{task}", meta=meta)
            e.validate(); out.append(e)
    return out


def adapter_chaos_nli(spec: DatasetSpec, max_rows: int | None, seed: int) -> list[DecisionExample]:
    """ChaosNLI (Nie et al. 2020): 100 human votes per SNLI/MNLI item. Calibration-eval only."""
    root = os.environ.get("TDE_CHAOSNLI_DIR", "data/external/chaosnli")
    if not Path(root).exists():
        print(f"[chaos_nli] {root} not found; skipping. Download from https://github.com/easonnie/ChaosNLI "
              "and place chaosNLI_snli.jsonl / chaosNLI_mnli_m.jsonl there (or set TDE_CHAOSNLI_DIR).")
        return []
    names = ["entailment", "neutral", "contradiction"]
    key_map = {"e": 0, "n": 1, "c": 2}
    out = []
    rng = random.Random(seed)
    train_t, held_t = _eval_templates("snli")
    choice_t = [t for t in train_t + held_t if t.primitive == "choice"]
    for fn in ("chaosNLI_snli.jsonl", "chaosNLI_mnli_m.jsonl"):
        p = Path(root) / fn
        if not p.exists():
            continue
        for line in p.open():
            row = json.loads(line)
            counts = row["label_counter"]
            tot = sum(counts.values())
            dist = [counts.get(k, 0) / tot for k in key_map]
            ex = row["example"]
            state = f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}"
            source_id = make_source_id(spec.name, row["uid"])
            t = rng.choice(choice_t)
            e = DecisionExample(
                id=f"{source_id}#{t.id}", source_id=source_id, dataset=spec.name, split="test", primitive="choice",
                state=state, question=t.text, candidates=[Candidate(n) for n in names], target=dist, template_id=t.id,
                meta={"human_votes": tot, "label_source": "human_distribution"},
            )
            e.validate()
            out.append(e)
    return out


def adapter_typed_decisions(spec: DatasetSpec, max_rows: int | None, seed: int) -> list[DecisionExample]:
    """LocalLLaMA/typed-decisions: 400 test cases x 5 questions (LLM-teacher gold distributions).

    Row layout: `state` (JSON), `questions` = {qid: {type, instructions, criteria}}, `gold` = {qid: {type, label,
    probabilities, ...}}. noul criteria = {"true": ..., "false": ...}; choice criteria = {label: description};
    score criteria = [level descriptions] with gold keyed by level index. Kept for comparability with published
    rows only (usage="comparability" -> written to eval_only/, never trained on).
    """
    rows = list(_iter_rows(spec, max_rows, seed))
    out, skipped = [], 0
    for key, row in rows:
        try:
            state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
            qs = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
            gold = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
            hf_split = key.split("/")[0]
            split = "test" if hf_split == "test" else "train"
            source_id = make_source_id(spec.name, row.get("id") or key)
            for qid, q in qs.items():
                g = gold[qid]
                probs = g.get("probabilities") or {}
                qtype = q["type"]
                crit = q.get("criteria")
                if qtype == "noul":
                    crit = crit or {}
                    cands = [Candidate("yes", str(crit.get("true", ""))), Candidate("no", str(crit.get("false", "")))]
                    p_yes = float(probs.get("true", g.get("noul", 0.0)))
                    target = [p_yes, 1.0 - p_yes]
                    prim = "noul"
                elif qtype == "score":
                    levels = list(crit or [])
                    cands = [Candidate(str(i), str(d)) for i, d in enumerate(levels)]
                    target = [float(probs.get(str(i), 0.0)) for i in range(len(levels))]
                    prim = "score"
                else:
                    names = list(crit.keys()) if isinstance(crit, dict) else [str(c) for c in (crit or [])]
                    cands = [Candidate(n, str(crit[n]) if isinstance(crit, dict) and crit[n] else "") for n in names]
                    target = [float(probs.get(n, 0.0)) for n in names]
                    prim = "choice"
                z = sum(target)
                if z <= 0:
                    raise ValueError("empty gold")
                target = [t / z for t in target]
                e = DecisionExample(
                    id=f"{source_id}#{qid}", source_id=source_id, dataset=spec.name, split=split, primitive=prim,
                    state=state, question=str(q["instructions"]), candidates=cands, target=target,
                    template_id=f"typed_decisions.{row.get('workflow', 'unknown')}.{qid}",
                    meta={"label_source": "llm_teacher", "workflow": row.get("workflow"), "qid": qid,
                          "gold_label": g.get("label"), "eval_group": None},
                )
                e.validate()
                out.append(e)
        except Exception as ex:  # noqa: BLE001
            skipped += 1
            if skipped <= 3:
                print(f"[typed_decisions] skip {key}: {type(ex).__name__}: {ex}")
    if skipped:
        print(f"[typed_decisions] skipped {skipped} rows")
    for e in out:
        e.meta.pop("eval_group", None)
    return out


def adapter_synth(spec: DatasetSpec, max_rows: int | None, seed: int) -> list[DecisionExample]:
    """Rule-generated worlds; `max_rows` = number of worlds (each yields 4-6 questions)."""
    from tde.data.synth import worlds_to_examples
    family = spec.name.replace("synth_", "")
    n = max_rows or spec.max_rows or 20000
    return worlds_to_examples(family, n, seed=seed)


ADAPTERS: dict[str, Callable[[DatasetSpec, int | None, int], list[DecisionExample]]] = {
    "synth": adapter_synth,
    "single_label": adapter_single_label,
    "clinc": adapter_clinc,
    "yahoo": adapter_yahoo,
    "snli": adapter_snli,
    "go_emotions_raw": adapter_go_emotions_raw,
    "boolq": adapter_boolq,
    "sst5": adapter_sst5,
    "chaos_nli": adapter_chaos_nli,
    "hotpot": adapter_hotpot,
    "legalbench": adapter_legalbench,
    "typed_decisions": adapter_typed_decisions,
}


def spec_templates(dataset: str) -> list[Template]:
    from tde.data.templates import templates_for
    return templates_for(dataset)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def merge(out_dir: str | Path, seed: int = 0) -> dict[str, int]:
    """Concatenate data/<out>/by_dataset/*.<split>.jsonl into <split>.jsonl (deterministically shuffled)."""
    from tde.schema import read_jsonl
    out_dir = Path(out_dir)
    counts = {}
    (out_dir / "eval_only").mkdir(exist_ok=True)
    for split in ("train", "calibration", "test"):
        exs = []
        groups: dict[str, list] = {}
        for p in sorted((out_dir / "by_dataset").glob(f"*.{split}*.jsonl")):
            parts = p.name.split(".")
            name = parts[0]
            usage = REGISTRY[name].usage if name in REGISTRY else "train"
            grp = parts[2] if len(parts) == 4 else None  # <name>.<split>.<group>.jsonl
            items = list(read_jsonl(p))
            if grp:
                groups.setdefault(grp, []).extend(items)
            elif usage == "train" or (usage == "train_and_report" and split == "train"):
                exs += items
            else:  # OOD tiers, calibration-only, comparability sets never enter train/cal/test
                n = write_jsonl(out_dir / "eval_only" / f"{name}.{split}.jsonl", items)
                print(f"[merge] {name} ({usage}) {split}: {n} -> eval_only/")
        random.Random(seed).shuffle(exs)
        counts[split] = write_jsonl(out_dir / f"{split}.jsonl", exs)
        print(f"[merge] {split}: {counts[split]} examples -> {out_dir / f'{split}.jsonl'}")
        for grp, items in groups.items():
            n = write_jsonl(out_dir / "eval_only" / f"{grp}.{split}.jsonl", items)
            print(f"[merge] eval group {grp} {split}: {n} -> eval_only/")
    return counts


def build(datasets: list[str] | None, out_dir: str | Path, stage: int = 0, max_rows: int | None = None, seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "by_dataset").mkdir(parents=True, exist_ok=True)
    names = datasets or [n for n, s in REGISTRY.items() if s.stage <= stage]
    report_path = out_dir / "build_report.json"
    report: dict = json.loads(report_path.read_text()) if report_path.exists() else {"datasets": {}}
    for name in names:
        spec = REGISTRY[name]
        print(f"[build] {name} ({spec.hf_id} {spec.hf_config or ''}) license={spec.license} usage={spec.usage}")
        try:
            exs = ADAPTERS[spec.adapter](spec, max_rows, seed)
        except Exception as e:  # keep going; report the failure
            print(f"[build] FAILED {name}: {type(e).__name__}: {e}")
            report["datasets"][name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        c_split = Counter(e.split for e in exs)
        c_prim = Counter(e.primitive for e in exs)
        n_sources = len({e.source_id for e in exs})
        report["datasets"][name] = {"examples": len(exs), "source_items": n_sources, "splits": dict(c_split),
                                    "primitives": dict(c_prim), "usage": spec.usage, "license": spec.license}
        for split in ("train", "calibration", "test"):
            main = [e for e in exs if e.split == split and not e.meta.get("eval_group")]
            write_jsonl(out_dir / "by_dataset" / f"{name}.{split}.jsonl", main)
            for grp in sorted({e.meta.get("eval_group") for e in exs if e.meta.get("eval_group")}):
                write_jsonl(out_dir / "by_dataset" / f"{name}.{split}.{grp}.jsonl", [e for e in exs if e.split == split and e.meta.get("eval_group") == grp])
        print(f"[build]   {len(exs)} examples from {n_sources} source items: {dict(c_split)} {dict(c_prim)}")
    report["splits"] = merge(out_dir, seed)
    report["primitives"] = dict(Counter(p for d in report["datasets"].values() if "primitives" in d for p, n in d["primitives"].items() for _ in range(1)))
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report
