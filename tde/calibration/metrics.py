"""Evaluation metrics for calibrated decision models.

Inputs are per-item probability vectors (variable K) and target distributions.
Conventions (stated in every report):
* Brier = sum_k (p_k - y_k)^2 per item (vector Brier; = 2x scalar Brier for K=2).
* NLL   = -sum_k y_k log max(p_k, 1e-6).
* ECE   = top-label ECE with 15 equal-mass bins, plus a Monte-Carlo noise floor
          (expected ECE of a perfectly calibrated model with the same confidences).
* AURC  = area under the risk-coverage curve when abstaining by confidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EPS = 1e-6


@dataclass
class Predictions:
    probs: list[np.ndarray]
    targets: list[np.ndarray]
    groups: list[str] | None = None  # source ids for grouped bootstrap
    ordinal: list[bool] | None = None

    def __len__(self) -> int:
        return len(self.probs)

    @property
    def conf(self) -> np.ndarray:
        return np.array([p.max() for p in self.probs])

    @property
    def pred(self) -> np.ndarray:
        return np.array([int(p.argmax()) for p in self.probs])

    @property
    def gold(self) -> np.ndarray:
        return np.array([int(t.argmax()) for t in self.targets])

    @property
    def correct(self) -> np.ndarray:
        return (self.pred == self.gold).astype(float)


# ----------------------------------------------------------------- pointwise
def accuracy(P: Predictions) -> float:
    return float(P.correct.mean())


def brier(P: Predictions) -> float:
    return float(np.mean([float(((p - t) ** 2).sum()) for p, t in zip(P.probs, P.targets)]))


def nll(P: Predictions) -> float:
    return float(np.mean([-(t * np.log(np.clip(p, EPS, 1))).sum() for p, t in zip(P.probs, P.targets)]))


def ece_equal_mass(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    n = len(conf)
    if n == 0:
        return float("nan")
    order = np.argsort(conf)
    bins = np.array_split(order, n_bins)
    e = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        e += len(b) / n * abs(conf[b].mean() - correct[b].mean())
    return float(e)


def ece_noise_floor(conf: np.ndarray, n_bins: int = 15, n_sim: int = 200, seed: int = 0) -> float:
    """Expected ECE if the model were perfectly calibrated: correctness ~ Bernoulli(conf)."""
    rng = np.random.default_rng(seed)
    vals = [ece_equal_mass(conf, (rng.random(len(conf)) < conf).astype(float), n_bins) for _ in range(n_sim)]
    return float(np.mean(vals))


def risk_coverage_curve(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-conf, kind="stable")
    err = 1.0 - correct[order]
    n = len(err)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    return coverage, risk


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    cov, risk = risk_coverage_curve(conf, correct)
    return float(risk.mean())


def e_aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Excess AURC over the oracle that orders errors last."""
    err_rate = 1.0 - correct.mean()
    n = len(correct)
    if err_rate == 0:
        return aurc(conf, correct)
    # oracle AURC for a classifier with error rate r
    k = int(round(err_rate * n))
    optimal = aurc(np.concatenate([np.ones(n - k), np.zeros(k)]), np.concatenate([np.ones(n - k), np.zeros(k)]))
    return float(aurc(conf, correct) - optimal)


def auroc_correct(conf: np.ndarray, correct: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if correct.min() == correct.max():
        return float("nan")
    return float(roc_auc_score(correct, conf))


def risk_at_coverage(conf: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    cov, risk = risk_coverage_curve(conf, correct)
    i = int(np.searchsorted(cov, coverage, side="left"))
    return float(risk[min(i, len(risk) - 1)])


def coverage_at_risk(conf: np.ndarray, correct: np.ndarray, risk_budget: float) -> float:
    cov, risk = risk_coverage_curve(conf, correct)
    ok = np.where(risk <= risk_budget)[0]
    return float(cov[ok[-1]]) if len(ok) else 0.0


def ordinal_metrics(P: Predictions) -> dict[str, float]:
    idx = [i for i, o in enumerate(P.ordinal or []) if o]
    if not idx:
        return {}
    mae, rps = [], []
    for i in idx:
        p, t = P.probs[i], P.targets[i]
        levels = np.arange(len(p))
        mae.append(abs(float((p * levels).sum()) - float(t.argmax())))
        rps.append(float(((np.cumsum(p) - np.cumsum(t)) ** 2).sum() / max(len(p) - 1, 1)))
    return {"ordinal_n": len(idx), "ordinal_mae": float(np.mean(mae)), "ordinal_rps": float(np.mean(rps))}


def summarize(P: Predictions, n_bins: int = 15) -> dict[str, float]:
    if len(P) == 0:
        return {"n": 0}
    conf, correct = P.conf, P.correct
    out = {
        "n": len(P),
        "accuracy": accuracy(P),
        "brier": brier(P),
        "nll": nll(P),
        "ece": ece_equal_mass(conf, correct, n_bins),
        "ece_noise_floor": ece_noise_floor(conf, n_bins),
        "aurc": aurc(conf, correct),
        "e_aurc": e_aurc(conf, correct),
        "auroc_correct": auroc_correct(conf, correct),
        "risk@cov50": risk_at_coverage(conf, correct, 0.5),
        "risk@cov80": risk_at_coverage(conf, correct, 0.8),
        "risk@cov90": risk_at_coverage(conf, correct, 0.9),
        "cov@risk1": coverage_at_risk(conf, correct, 0.01),
        "cov@risk5": coverage_at_risk(conf, correct, 0.05),
        "cov@risk10": coverage_at_risk(conf, correct, 0.10),
        "mean_conf": float(conf.mean()),
    }
    out.update(ordinal_metrics(P))
    return out


# ----------------------------------------------------------------- statistics
def paired_bootstrap(metric, P_a: Predictions, P_b: Predictions, n_resamples: int = 10000, seed: int = 0, groups: list[str] | None = None) -> dict[str, float]:
    """95% CI of metric(a) - metric(b), resampling items (or whole groups when `groups` is given)."""
    assert len(P_a) == len(P_b)
    n = len(P_a)
    rng = np.random.default_rng(seed)
    if groups:
        keys = sorted(set(groups))
        members = {k: [] for k in keys}
        for i, g in enumerate(groups):
            members[g].append(i)
        units = [np.array(members[k]) for k in keys]
    else:
        units = [np.array([i]) for i in range(n)]
    diffs = []
    for _ in range(n_resamples):
        pick = rng.integers(0, len(units), len(units))
        idx = np.concatenate([units[i] for i in pick])
        sa = Predictions([P_a.probs[i] for i in idx], [P_a.targets[i] for i in idx], ordinal=[P_a.ordinal[i] for i in idx] if P_a.ordinal else None)
        sb = Predictions([P_b.probs[i] for i in idx], [P_b.targets[i] for i in idx], ordinal=[P_b.ordinal[i] for i in idx] if P_b.ordinal else None)
        diffs.append(metric(sa) - metric(sb))
    diffs = np.array(diffs)
    return {"diff": float(metric(P_a) - metric(P_b)), "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5))}


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    """Exact two-sided McNemar test on discordant pairs."""
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return {"b": b, "c": c, "p_value": float(min(1.0, 2 * p))}


def min_detectable_difference(n: int, discordance: float = 0.2, alpha: float = 0.05, power: float = 0.8) -> float:
    """Approximate paired MDE in accuracy points given n items and a discordance rate."""
    from math import sqrt
    z_a, z_b = 1.96, 0.84
    return float((z_a + z_b) * sqrt(discordance / n))
