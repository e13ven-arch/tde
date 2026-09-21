import pytest

from tde.data.splits import assign_split, template_is_held_out
from tde.schema import Candidate, DecisionExample


def _ex(**kw):
    base = dict(id="x", source_id="ds:1", dataset="ds", split="train", primitive="choice", state="s",
                question="q?", candidates=[Candidate("a"), Candidate("b"), Candidate("c")], target=[1.0, 0.0, 0.0])
    base.update(kw)
    return DecisionExample(**base)


def test_validate_ok():
    _ex().validate()


def test_noul_needs_two():
    with pytest.raises(ValueError):
        _ex(primitive="noul").validate()


def test_score_range():
    with pytest.raises(ValueError):
        _ex(primitive="score", candidates=[Candidate(str(i)) for i in range(11)], target=[1.0] + [0.0] * 10).validate()


def test_target_sum():
    with pytest.raises(ValueError):
        _ex(target=[0.5, 0.0, 0.0]).validate()


def test_permutation_roundtrip():
    e = _ex(target=[0.2, 0.5, 0.3])
    p = e.with_candidate_order([2, 0, 1])
    assert [c.name for c in p.candidates] == ["c", "a", "b"]
    assert p.target == [0.3, 0.2, 0.5]


def test_split_is_deterministic_and_balanced():
    counts = {"train": 0, "calibration": 0, "test": 0}
    for i in range(20000):
        s = assign_split(f"ds:{i}")
        assert s == assign_split(f"ds:{i}")
        counts[s] += 1
    assert 0.78 < counts["train"] / 20000 < 0.82
    assert 0.08 < counts["calibration"] / 20000 < 0.12
    assert 0.08 < counts["test"] / 20000 < 0.12


def test_same_source_same_split_regardless_of_rewrite():
    # examples rendered from one source id all inherit its split; the id suffix never matters
    s = assign_split("ds:42")
    for suffix in ("#t1#0", "#t2#3", "#held#0"):
        assert assign_split("ds:42") == s  # rewrites never re-hash


def test_template_holdout_fraction():
    n = sum(template_is_held_out(f"t{i}") for i in range(5000))
    assert 0.16 < n / 5000 < 0.24
