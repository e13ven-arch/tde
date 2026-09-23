"""Tests for the tde_local adapter: label mapping, temperature choice, and strict probability handling.
No model download: the decider is stubbed."""
import math
from types import SimpleNamespace

import pytest

from jevbench.adapters.tde_local import TdeLocalAdapter


def _task(qtype, labels, criteria=None, state="s"):
    return SimpleNamespace(id="t", state=state, labels=labels,
                           question={"type": qtype, "instructions": "Q?", "criteria": criteria})


class _Stub:
    def __init__(self, probs):
        self.probs, self.device, self.temperature, self.cfg = probs, "cpu", None, {"readout": "joint", "backbone": "x"}

    def decide(self, state, q):
        return {"type": q["type"], "probabilities": dict(self.probs)}


def _adapter(probs):
    a = TdeLocalAdapter(endpoint="unused")
    a._decider = _Stub(probs)
    return a


def test_choice_mapping_follows_declared_label_order():
    a = _adapter({})
    q = a.build_question(_task("choice", ["b", "a"], {"a": "desc a", "b": "desc b"}))
    assert list(q["criteria"]) == ["b", "a"] and q["criteria"]["a"] == "desc a"


def test_score_mapping_and_level_count_check():
    a = _adapter({})
    q = a.build_question(_task("score", ["0", "1", "2"], ["low", "mid", "high"]))
    assert q["type"] == "score" and q["criteria"] == ["low", "mid", "high"]
    with pytest.raises(ValueError):
        a.build_question(_task("score", ["0", "1"], ["low", "mid", "high"]))


def test_noul_mapping_keeps_criteria():
    a = _adapter({})
    q = a.build_question(_task("noul", ["no", "yes"], {"true": "t", "false": "f"}))
    assert q == {"type": "noul", "instructions": "Q?", "criteria": {"true": "t", "false": "f"}}


def test_temperature_off_by_default_and_explicit():
    assert TdeLocalAdapter(endpoint="x").use_temperature is False
    assert TdeLocalAdapter(endpoint="x", use_temperature=True).use_temperature is True


def test_probabilities_pass_through_unmodified():
    a = _adapter({"a": 0.6, "b": 0.38})  # sums to 0.98: must NOT be renormalised here
    r = a.run(_task("choice", ["a", "b"], {"a": "", "b": ""}))
    assert r.ok and r.probs == {"a": 0.6, "b": 0.38} and r.probs_source == "native"


def test_noul_returns_yes_no_keys_in_declared_order():
    a = _adapter({"yes": 0.7, "no": 0.3})
    r = a.run(_task("noul", ["no", "yes"]))
    assert r.ok and list(r.probs) == ["no", "yes"] and r.probs["yes"] == 0.7


def test_missing_label_is_an_error_not_filled():
    a = _adapter({"a": 1.0})
    r = a.run(_task("choice", ["a", "b"], {"a": "", "b": ""}))
    assert not r.ok and r.probs is None and "label set mismatch" in r.error


def test_extra_label_is_an_error():
    a = _adapter({"a": 0.5, "b": 0.3, "c": 0.2})
    r = a.run(_task("choice", ["a", "b"], {"a": "", "b": ""}))
    assert not r.ok and "label set mismatch" in r.error


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, "0.5", None])
def test_malformed_probability_is_an_error(bad):
    a = _adapter({"a": bad, "b": 0.5})
    r = a.run(_task("choice", ["a", "b"], {"a": "", "b": ""}))
    assert not r.ok and "invalid probability" in r.error


def test_decider_exception_is_reported():
    class Boom(_Stub):
        def decide(self, state, q):
            raise RuntimeError("cuda gone")
    a = TdeLocalAdapter(endpoint="unused"); a._decider = Boom({})
    r = a.run(_task("noul", ["no", "yes"]))
    assert not r.ok and "cuda gone" in r.error
