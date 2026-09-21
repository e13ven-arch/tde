"""The paired proper-reward policy gradient must have the same expected gradient as the direct Brier loss;
the correctness-only REINFORCE must not."""
import torch

from tde.losses import brier, correctness_pg, paired_brier_pg, soft_cross_entropy


def _setup(seed=0):
    torch.manual_seed(seed)
    logits = torch.tensor([[1.0, 0.2, -0.5, 0.0], [0.3, -0.3, 0.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    target = torch.tensor([[0.6, 0.3, 0.1, 0.0], [0.5, 0.5, 0.0, 0.0]])
    return logits, mask, target


def _mc_grad(fn, n_rounds=400, **kw):
    logits, mask, target = _setup()
    acc = torch.zeros_like(logits)
    for _ in range(n_rounds):
        if logits.grad is not None:
            logits.grad.zero_()
        fn(logits, target, mask, **kw).backward()
        acc += logits.grad
    return acc / n_rounds


def test_paired_pg_matches_brier_gradient():
    logits, mask, target = _setup()
    brier(logits, target, mask).backward()
    g_direct = logits.grad.clone()
    g_pg = _mc_grad(paired_brier_pg, n_samples=64)
    # expected reward is ||q||^2 - ||p-q||^2, so the surrogate's expected gradient is grad ||p-q||^2 = grad(brier)
    err = (g_pg - g_direct).abs().max().item()
    scale = g_direct.abs().max().item()
    assert err < 0.15 * scale + 0.01, (g_pg, g_direct)


def test_correctness_pg_is_not_proper():
    logits, mask, target = _setup()
    brier(logits, target, mask).backward()
    g_direct = logits.grad.clone()
    g_bad = _mc_grad(correctness_pg, n_samples=64)
    # direction must differ materially from the proper objective (cosine well below 1)
    cos = torch.nn.functional.cosine_similarity(g_bad.flatten(), g_direct.flatten(), dim=0).item()
    assert cos < 0.9, cos


def test_pg_losses_respect_mask():
    logits, mask, target = _setup()
    for fn in (paired_brier_pg, correctness_pg):
        if logits.grad is not None:
            logits.grad.zero_()
        fn(logits, target, mask, n_samples=16).backward()
        assert torch.all(logits.grad[~mask] == 0)
