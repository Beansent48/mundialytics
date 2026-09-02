"""Half-time market model: structure, coherence and calibration against reality.

The model is a scaling of the engine's full-time lambdas rather than a fitted
model, because the per-team first-half share was measured to be noise
(split-half correlation -0.118). That makes these tests the real specification:
if the transformation is right, its marginals must reproduce the observed base
rates over 45,934 matches.

Run:  .venv/Scripts/python.exe -m pytest tests/test_half_time.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.props.half_time import HalfTimeModel  # noqa: E402

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"


@pytest.fixture(scope="module")
def model():
    return HalfTimeModel()


@pytest.fixture(scope="module")
def actuals():
    import pandas as pd
    if not FOUND.exists():
        pytest.skip("foundation not built")
    f = pd.read_csv(FOUND, low_memory=False)
    need = ["home_goals_ht", "away_goals_ht", "home_goals", "away_goals"]
    if any(c not in f.columns for c in need):
        pytest.skip("foundation has no half-time columns")
    return f.dropna(subset=need)


# ── structure ───────────────────────────────────────────────────────────────
def test_half_time_probabilities_are_a_distribution(model):
    ht = model.predict_half_time(1.8, 1.2)
    total = ht["p_home"] + ht["p_draw"] + ht["p_away"]
    assert total == pytest.approx(1.0, abs=1e-9)
    for v in (ht["p_home"], ht["p_draw"], ht["p_away"]):
        assert 0.0 <= v <= 1.0
    lines = sorted(ht["over"])
    ps = [ht["over"][ln] for ln in lines]
    assert all(ps[i] >= ps[i + 1] - 1e-12 for i in range(len(ps) - 1)), "over lines not monotone"


def test_ht_ft_is_a_distribution(model):
    d = model.predict_ht_ft(1.6, 1.3)
    assert len(d) == 9
    assert sum(d.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 <= v <= 1.0 for v in d.values())


def test_half_time_lambdas_are_a_fraction_of_full_time(model):
    ht = model.predict_half_time(2.0, 1.0)
    assert 0.35 < ht["lambda_home"] / 2.0 < 0.55
    assert 0.35 < ht["lambda_away"] / 1.0 < 0.55


def test_stronger_side_more_likely_to_lead_at_the_break(model):
    ht = model.predict_half_time(2.4, 0.7)
    assert ht["p_home"] > ht["p_away"]
    flipped = model.predict_half_time(0.7, 2.4)
    assert flipped["p_away"] > flipped["p_home"]


def test_comeback_paths_are_the_rarest(model):
    """Trailing at half-time and still winning must be the least likely routes."""
    d = model.predict_ht_ft(1.5, 1.3)
    assert d["1/2"] < d["1/1"] and d["1/2"] < d["X/X"]
    assert d["2/1"] < d["2/2"] and d["2/1"] < d["X/X"]


def test_ht_ft_margins_match_the_half_time_model(model):
    """Summing HT/FT over full-time outcomes must return the half-time result."""
    lh, la = 1.7, 1.1
    ht = model.predict_half_time(lh, la)
    d = model.predict_ht_ft(lh, la)
    for res, key in [("1", "p_home"), ("X", "p_draw"), ("2", "p_away")]:
        margin = sum(v for k, v in d.items() if k.startswith(f"{res}/"))
        assert margin == pytest.approx(ht[key], abs=1e-6), f"HT margin {res} disagrees"


# ── calibration against 26 seasons of real matches ──────────────────────────
def test_matches_real_half_time_base_rates(model, actuals):
    f = actuals
    ht = model.predict_half_time(f.home_goals.mean(), f.away_goals.mean())
    real = {"p_home": (f.home_goals_ht > f.away_goals_ht).mean(),
            "p_draw": (f.home_goals_ht == f.away_goals_ht).mean(),
            "p_away": (f.home_goals_ht < f.away_goals_ht).mean()}
    for k, v in real.items():
        assert ht[k] == pytest.approx(v, abs=0.01), f"{k}: {ht[k]:.3f} vs real {v:.3f}"

    tot = f.home_goals_ht + f.away_goals_ht
    assert ht["lambda_home"] + ht["lambda_away"] == pytest.approx(tot.mean(), abs=0.02)
    for ln, p in ht["over"].items():
        assert p == pytest.approx((tot > ln).mean(), abs=0.01), f"over {ln} miscalibrated"


def test_matches_real_ht_ft_frequencies(model, actuals):
    f = actuals
    import pandas as pd
    htr = np.where(f.home_goals_ht > f.away_goals_ht, "1",
                   np.where(f.home_goals_ht == f.away_goals_ht, "X", "2"))
    ftr = np.where(f.home_goals > f.away_goals, "1",
                   np.where(f.home_goals == f.away_goals, "X", "2"))
    combo = pd.Series([f"{a}/{b}" for a, b in zip(htr, ftr)])
    pred = model.predict_ht_ft(f.home_goals.mean(), f.away_goals.mean())
    errs = [abs(pred[k] - (combo == k).mean()) for k in pred]
    assert max(errs) < 0.01, f"worst HT/FT cell off by {max(errs):.4f}"
    assert float(np.mean(errs)) < 0.005, f"mean HT/FT error {np.mean(errs):.4f}"
