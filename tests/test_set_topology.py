"""Tied candidate positions + set masks (joint topology pointwise / set) on a tiny random ModernBERT."""
import pytest
import torch

from tde.model.encoding import collate
from tde.model.factory import build_model
from tde.model.joint import set_masks
from tde.schema import Candidate, DecisionExample

STATE = "the sky is blue today and it rained on the table at sea"  # longer than the tiny sliding window (4 positions)


def _example(cands, question="What color is it?", state=STATE):
    k = len(cands)
    return DecisionExample("a", "s:1", "ds", "test", "choice", state, question, [Candidate(*c) for c in cands], [1.0] + [0.0] * (k - 1))


CANDS = [("blue", "like the sea"), ("red",), ("dark green", "not like the sky"), ("yellow",), ("white", "bright")]
OTHER = _example([("yes",), ("no",)], "Is it wet?", "it rained")


def _model(topology="pointwise", impl="sdpa"):
    torch.manual_seed(0)
    dtok, model = build_model("tiny-modernbert", "joint", tiny="modernbert", topology=topology)
    model.backbone.config._attn_implementation = impl
    return dtok, model.eval()


@torch.no_grad()
def _logits(model, dtok, exs):
    batch = collate([dtok.encode(e, "joint") for e in exs], dtok.pad_id, "joint")
    return model(batch)["logits"]


def test_tied_layout_starts_every_block_at_the_same_position():
    dtok, _ = _model()
    enc = dtok.encode(_example(CANDS), "joint")
    starts = {enc.tied_positions[p] for p in enc.opt_positions}
    assert starts == {enc.opt_positions[0]}  # = prefix length
    lens = [b - a for a, b in zip(enc.opt_positions, enc.opt_positions[1:] + [enc.decide_position])]
    assert len(set(lens)) > 1  # blocks differ in length, so the test exercises the tied layout
    assert enc.tied_positions[enc.decide_position] == enc.opt_positions[0] + 1 + dtok.max_candidate  # fixed, set-independent
    assert enc.block_ids[enc.decide_position] == -2 and enc.block_ids[0] == -1


@pytest.mark.parametrize("topology", ["pointwise", "set"])
@pytest.mark.parametrize("impl", ["sdpa", "eager"])
def test_tied_topologies_are_permutation_equivariant(topology, impl):
    dtok, model = _model(topology, impl)
    order = [3, 0, 4, 1, 2]
    z = _logits(model, dtok, [_example(CANDS), OTHER])
    zp = _logits(model, dtok, [_example(CANDS).with_candidate_order(order), OTHER])
    assert torch.allclose(zp[0, :5], z[0, order], atol=1e-5)
    assert torch.allclose(zp[1], z[1], atol=1e-5)


def test_seq_topology_is_order_sensitive():
    dtok, model = _model("seq")
    order = [3, 0, 4, 1, 2]
    z = _logits(model, dtok, [_example(CANDS)])
    zp = _logits(model, dtok, [_example(CANDS).with_candidate_order(order)])
    assert (zp[0] - z[0, order]).abs().max() > 1e-4  # negative control: the test above can fail


def test_pointwise_logits_ignore_other_candidates_and_set_does_not():
    dtok, model = _model("pointwise")
    small, full = _example(CANDS[:3]), _example(CANDS + [("black", "a very dark colour like the sky at night and not bright")])  # longest block last
    assert torch.allclose(_logits(model, dtok, [small])[0], _logits(model, dtok, [full])[0, :3], atol=1e-5)
    model.topology = "set"
    assert (_logits(model, dtok, [small])[0] - _logits(model, dtok, [full])[0, :3]).abs().max() > 1e-6  # fp32 noise is ~1e-7


@torch.no_grad()
def test_prefix_only_masks_reproduce_the_default_modernbert_masks():
    dtok, model = _model()
    batch = collate([dtok.encode(e, "joint") for e in (_example(CANDS), OTHER)], dtok.pad_id, "joint")
    ids, pad = batch["input_ids"], batch["attention_mask"]
    ref = model.backbone(input_ids=ids, attention_mask=pad).last_hidden_state
    flat = {"block_ids": torch.full_like(ids, -1), "position_ids": torch.arange(ids.size(1)).expand_as(ids), "attention_mask": pad}
    got = model.backbone(input_ids=ids, attention_mask=set_masks(flat, "pointwise", model.backbone.config),
                         position_ids=flat["position_ids"]).last_hidden_state
    valid = pad.bool()
    assert torch.allclose(got[valid], ref[valid], atol=1e-5)
