import numpy as np

from tde.calibration.metrics import Predictions, aurc, ece_equal_mass, ece_noise_floor, mcnemar_exact, summarize
from tde.calibration.temperature import BucketTemperature


def _calibrated(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    probs, targets = [], []
    for _ in range(n):
        k = rng.integers(2, 6)
        p = rng.dirichlet(np.ones(k) * 0.7)
        y = rng.choice(k, p=p)
        probs.append(p); targets.append(np.eye(k)[y])
    return Predictions(probs, targets)


def test_calibrated_model_ece_near_floor():
    P = _calibrated()
    e = ece_equal_mass(P.conf, P.correct)
    floor = ece_noise_floor(P.conf)
    assert e < floor * 2.5


def test_oracle_aurc_is_zero_excess():
    correct = np.array([1.0] * 90 + [0.0] * 10)
    conf = np.linspace(1, 0, 100)
    assert aurc(conf, correct) < aurc(conf[::-1], correct)


def test_summary_keys():
    s = summarize(_calibrated(500))
    for k in ("accuracy", "brier", "nll", "ece", "aurc", "cov@risk5", "risk@cov80"):
        assert k in s


def test_mcnemar():
    a = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1], dtype=float)
    b = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=float)
    r = mcnemar_exact(a, b)
    assert r["b"] == 6 and r["c"] == 0 and r["p_value"] < 0.05


def test_temperature_recovers_sharpening():
    rng = np.random.default_rng(1)
    logits, targets, prims = [], [], []
    for _ in range(2000):
        z = rng.normal(size=3) * 3.0  # over-confident logits
        p = np.exp(z / 2.5); p /= p.sum()  # true distribution is at T=2.5
        y = rng.choice(3, p=p)
        logits.append(z); targets.append(np.eye(3)[y]); prims.append("choice")
    T = BucketTemperature().fit(logits, targets, prims)
    assert 1.8 < T.temperature("choice", 3) < 3.3
