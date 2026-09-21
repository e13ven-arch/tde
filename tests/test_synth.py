"""Rule-generated data: validity, balance, split-by-world, and label re-derivation spot checks."""
import json
from collections import Counter

from tde.data.synth import worlds_to_examples


def _yes_rates(exs):
    n, y = Counter(), Counter()
    for e in exs:
        if e.primitive == "noul":
            n[e.template_id] += 1
            y[e.template_id] += int(e.target[0] > 0.5)
    return {t: y[t] / n[t] for t in n}


def test_all_families_valid_and_balanced():
    for fam in ("policy", "multihop", "temporal"):
        exs = worlds_to_examples(fam, 500, seed=1)
        assert exs and all(e.dataset == f"synth_{fam}" for e in exs)
        for t, r in _yes_rates(exs).items():
            assert 0.3 < r < 0.7, (fam, t, r)
        # deterministic
        again = worlds_to_examples(fam, 5, seed=1)
        assert [e.id for e in again] == [e.id for e in exs[: len(again)]]


def test_split_is_per_world():
    exs = worlds_to_examples("multihop", 300, seed=2)
    by_world = {}
    for e in exs:
        by_world.setdefault(e.source_id, set()).add(e.split)
    assert all(len(s) == 1 for s in by_world.values())


def test_multihop_labels_rederive():
    exs = [e for e in worlds_to_examples("multihop", 200, seed=3) if e.template_id.endswith("assignee_dept") and e.state.startswith("{")]
    for e in exs[:50]:
        w = json.loads(e.state)
        tid = e.question.split("ticket ")[-1].split("?")[0].split(" ")[0] if "ticket " in e.question else e.question.split("assigned to ")[-1].split(" ")[0]
        t = next(t for t in w["tickets"] if t["id"] == tid)
        asg = next(u for u in w["users"] if u["id"] == t["assignee"])
        assert e.candidates[e.label].name == asg["department"]


def test_policy_unmentioned_condition_blocks():
    exs = [e for e in worlds_to_examples("policy", 400, seed=4) if e.template_id.endswith("permitted")]
    # if the 'missing' question for the same world says yes, 'permitted' must say no
    perm = {e.source_id: e.target[0] > 0.5 for e in exs}
    miss = {e.source_id: e.target[0] > 0.5 for e in worlds_to_examples("policy", 400, seed=4) if e.template_id.endswith("missing")}
    assert all(not perm[s] for s, m in miss.items() if m)
