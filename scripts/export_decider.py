"""Export TDE decision jsonl files into decider's training pickle: (list[Example], {eval_name: list[Example]}).

    python scripts/export_decider.py --train data/v0.7/train.jsonl@synth_rubric:8000 data/v0.7/train.jsonl@legalbench:8000 ... \
        --eval rubric=data/v0.7/calibration.jsonl@synth_rubric:300 --out data/decider_mix.pkl

Mapping: context = state; one Q per example with options rendered as "name: description" (or the bare name) in the
example's candidate order; noul keeps its two candidates (yes / no) as options; score levels become options in level order.
gold = argmax of the target distribution. Requires decider (pip install decider-ai) on the path.
"""
from __future__ import annotations
import argparse, json, pickle, random
from decider.data.core import Example, Q


def load(path: str, cap: int | None, rng: random.Random, dataset: str | None = None) -> list[Example]:
    """path may be a merged split file; `dataset` keeps only rows of that dataset (spec syntax path@dataset:cap)."""
    out = []
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    if dataset:
        rows = [r for r in rows if r["dataset"] == dataset]
    if cap and len(rows) > cap:
        rows = rng.sample(rows, cap)
    for r in rows:
        opts = [(c["name"] + (": " + c["description"] if c.get("description") else "")) for c in r["candidates"]]
        gold = max(range(len(r["target"])), key=lambda i: r["target"][i])
        task = f"tde_{r['dataset']}_{r['primitive']}"
        out.append(Example(r["state"], [Q(r["question"], opts, gold)], task))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True, help="path[:cap] ...")
    ap.add_argument("--eval", nargs="*", default=[], help="name=path[:cap] ...")
    ap.add_argument("--replay", default=None, help="optional decider pickle (train list) to mix in as replay")
    ap.add_argument("--replay_n", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    train = []
    def parse(spec):
        path, _, cap = spec.partition(":"); path, _, ds = path.partition("@")
        return path, (ds or None), (int(cap) if cap else None)
    for spec in a.train:
        path, ds, cap = parse(spec)
        exs = load(path, cap, rng, ds); train += exs
        print(f"[export] {path}@{ds}: {len(exs)}")
    if a.replay and a.replay_n:
        rep = pickle.load(open(a.replay, "rb"))
        rep = rep[0] if isinstance(rep, tuple) else rep
        rng.shuffle(rep); train += rep[: a.replay_n]; print(f"[export] replay {min(a.replay_n, len(rep))} from {a.replay}")
    evals = {}
    for spec in a.eval:
        name, _, rest = spec.partition("="); path, ds, cap = parse(rest)
        evals[name] = load(path, cap, rng, ds); print(f"[export] eval {name}: {len(evals[name])}")
    rng.shuffle(train)
    pickle.dump((train, evals), open(a.out, "wb"))
    print(f"[export] wrote {a.out}: {len(train)} train examples, {len(evals)} eval sets")


if __name__ == "__main__":
    main()
