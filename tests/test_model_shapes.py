import torch

from tde.losses import permutation_kl, soft_cross_entropy, ranked_probability_score
from tde.model.encoding import collate
from tde.model.factory import build_model
from tde.schema import Candidate, DecisionExample


def _examples():
    return [
        DecisionExample("a", "s:1", "ds", "train", "choice", "the sky is blue today", "What color?",
                        [Candidate("blue", "like the sea"), Candidate("red"), Candidate("green")], [1.0, 0.0, 0.0]),
        DecisionExample("b", "s:2", "ds", "train", "noul", "it rained", "Is it wet?", [Candidate("yes"), Candidate("no")], [0.9, 0.1]),
        DecisionExample("c", "s:1", "ds", "train", "score", "the sky is blue today", "How bright?",
                        [Candidate("dark"), Candidate("dim"), Candidate("bright"), Candidate("blinding")], [0.0, 0.0, 1.0, 0.0]),
    ]


def _run(readout):
    dtok, model = build_model("tiny", readout, tiny=True, use_confidence_head=(readout == "joint"))
    exs = _examples()
    batch = collate([dtok.encode(e, readout) for e in exs], dtok.pad_id, readout)
    out = model(batch)
    logits = out["logits"]
    assert logits.shape == (3, 4)
    assert torch.isfinite(logits[batch["cand_mask"]]).all()
    assert (logits[~batch["cand_mask"]] < -1e3).all()
    loss = soft_cross_entropy(logits, batch["target"], batch["cand_mask"]) + ranked_probability_score(logits, batch["target"], batch["cand_mask"], batch["is_score"])
    loss.backward()
    return dtok, model, batch, out


def test_joint():
    _, _, batch, out = _run("joint")
    assert out["conf_logit"].shape == (3,)


def test_branch_shares_state():
    dtok, model, batch, _ = _run("branch")
    assert batch["state_ids"].shape[0] == 2  # examples a and c share a state
    assert batch["state_index"].tolist() == [0, 1, 0]


def test_biencoder():
    _run("biencoder")


def test_permutation_kl_zero_for_consistent_model():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1e4], [0.5, -0.5, -1e4, -1e4]])
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    perm = torch.tensor([[2, 0, 1, 3], [1, 0, 2, 3]])
    permuted = torch.stack([logits[0][perm[0]], logits[1][perm[1]]])
    assert permutation_kl(logits, permuted, perm, mask).abs() < 1e-5


def test_decision_tokenizer_marks_candidates():
    dtok, _ = build_model("tiny", "joint", tiny=True)
    e = _examples()[0]
    enc = dtok.encode(e, "joint")
    assert all(enc.input_ids[p] == dtok.opt_id for p in enc.opt_positions)
    assert enc.input_ids[enc.decide_position] == dtok.decide_id
    assert enc.spans and all(enc.input_ids[a] != dtok.opt_id for a, _ in enc.spans)
    assert enc.level_index == [-1, -1, -1]
