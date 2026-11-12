"""Analytics layer: endpoints that surface the model's internal data.

Verifies that the helpers produce the right shape and values, from the
low-level RPS vector to the full decomposition/error/calibration/profile
responses.  The Poisson GOF, Wilson intervals and Brier decomposition are
checked against hand-computed values.

Run:  .venv/Scripts/python.exe -m pytest tests/test_analytics.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from api.analytics import (  # noqa: E402
    _poisson_pmf,
    _rps,
    _wilson_ci,
    enhanced_calibration,
    prediction_errors,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


class TestRPS:
    def test_perfect_home(self):
        rps = _rps(
            np.array([1.0]), np.array([0.0]), np.array([0.0]),
            np.array([2]), np.array([0]),
        )
        assert rps[0] == pytest.approx(0.0)

    def test_perfect_draw(self):
        rps = _rps(
            np.array([0.0]), np.array([1.0]), np.array([0.0]),
            np.array([1]), np.array([1]),
        )
        assert rps[0] == pytest.approx(0.0)

    def test_perfect_away(self):
        rps = _rps(
            np.array([0.0]), np.array([0.0]), np.array([1.0]),
            np.array([0]), np.array([3]),
        )
        assert rps[0] == pytest.approx(0.0)

    def test_worst_case(self):
        rps = _rps(
            np.array([1.0]), np.array([0.0]), np.array([0.0]),
            np.array([0]), np.array([1]),
        )
        assert rps[0] == pytest.approx(1.0)

    def test_uniform(self):
        rps = _rps(
            np.array([1 / 3]), np.array([1 / 3]), np.array([1 / 3]),
            np.array([1]), np.array([0]),
        )
        expected = ((1 / 3 - 1) ** 2 + (2 / 3 - 1) ** 2) / 2
        assert rps[0] == pytest.approx(expected, rel=1e-6)

    def test_vectorised(self):
        rps = _rps(
            np.array([0.5, 0.0]),
            np.array([0.3, 0.0]),
            np.array([0.2, 1.0]),
            np.array([2, 0]),
            np.array([1, 3]),
        )
        assert rps.shape == (2,)
        assert rps[1] == pytest.approx(0.0)


class TestPoissonPMF:
    def test_zero_goals(self):
        assert _poisson_pmf(1.5, 0) == pytest.approx(math.exp(-1.5))

    def test_known_value(self):
        expected = math.exp(-2.0) * 4.0 / 2
        assert _poisson_pmf(2.0, 2) == pytest.approx(expected)

    def test_sum_to_one(self):
        total = sum(_poisson_pmf(1.3, k) for k in range(20))
        assert total == pytest.approx(1.0, abs=1e-9)


class TestWilsonCI:
    def test_zero_n(self):
        lo, hi = _wilson_ci(0.5, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_perfect_rate(self):
        lo, hi = _wilson_ci(1.0, 100)
        assert hi == pytest.approx(1.0, abs=1e-9)
        assert lo > 0.9

    def test_reasonable_interval(self):
        lo, hi = _wilson_ci(0.5, 100)
        assert lo < 0.5 < hi
        assert hi - lo < 0.25

    def test_symmetry(self):
        lo1, hi1 = _wilson_ci(0.3, 200)
        lo2, hi2 = _wilson_ci(0.7, 200)
        assert (hi1 - 0.3) == pytest.approx(0.7 - lo2, abs=1e-4)


# ── Full responses (need data files) ───────────────────────────────────────


WALKFORWARD = ROOT / "data/processed/enriched/understat_xg/walkforward_preds_deployed.csv"
COMPONENTS = ROOT / "data/processed/enriched/understat_xg/global_components.csv"


@pytest.fixture()
def foundation():
    pytest.importorskip("mundialytics")
    from api.engine import _club_engine, data_version
    _, _, df = _club_engine(data_version())
    return df


@pytest.mark.skipif(not WALKFORWARD.exists(), reason="walkforward CSV missing")
class TestPredictionErrors:
    def test_shape(self, foundation):
        result = prediction_errors(foundation)
        assert isinstance(result["total_rps"], float)
        assert 0.15 < result["total_rps"] < 0.25
        assert result["n"] > 5000
        assert len(result["by_season"]) >= 1
        assert len(result["by_league"]) >= 1
        assert len(result["by_tier"]) == 3

    def test_season_keys(self, foundation):
        result = prediction_errors(foundation)
        for s in result["by_season"]:
            assert set(s.keys()) == {"season", "rps", "n"}
            assert isinstance(s["rps"], float)

    def test_gof_structure(self, foundation):
        result = prediction_errors(foundation)
        gof = result["scoreline_gof"]
        assert gof is not None
        assert "chi2" in gof and "p" in gof and "df" in gof
        assert gof["df"] == 48  # (7*7) - 1
        assert len(gof["bins"]) <= 20
        for b in gof["bins"]:
            assert set(b.keys()) == {"score", "observed", "expected"}

    def test_tier_names(self, foundation):
        result = prediction_errors(foundation)
        tiers = {t["tier"] for t in result["by_tier"]}
        assert tiers == {"low", "mid", "high"}


@pytest.mark.skipif(not WALKFORWARD.exists(), reason="walkforward CSV missing")
class TestEnhancedCalibration:
    def test_shape(self):
        result = enhanced_calibration()
        assert result is not None
        assert len(result["bins"]) >= 10
        assert 0 < result["brier"] < 0.5

    def test_bin_keys(self):
        result = enhanced_calibration()
        for b in result["bins"]:
            assert set(b.keys()) == {"announced", "observed", "n", "ci_lo", "ci_hi"}
            assert b["ci_lo"] <= b["observed"] <= b["ci_hi"]

    def test_brier_decomposition(self):
        result = enhanced_calibration()
        assert result["reliability"] < result["brier"]
        assert result["uncertainty"] > 0
        assert result["resolution"] > 0

    def test_n_is_triple_matches(self):
        result = enhanced_calibration()
        w = pd.read_csv(WALKFORWARD)
        assert result["n"] == len(w) * 3


@pytest.mark.skipif(not COMPONENTS.exists(), reason="global_components CSV missing")
class TestDecomposition:
    def test_known_team(self, foundation):
        from api.analytics import decomposition
        result = decomposition("barcelona", foundation)
        assert result["team"] == "barcelona"
        assert len(result["matches"]) > 50

    def test_match_keys(self, foundation):
        from api.analytics import decomposition
        result = decomposition("barcelona", foundation)
        m = result["matches"][0]
        assert set(m.keys()) == {
            "date", "opponent", "venue", "goals", "conceded",
            "gl", "ad", "adx", "xr",
        }

    def test_lambdas_positive(self, foundation):
        from api.analytics import decomposition
        result = decomposition("barcelona", foundation)
        for m in result["matches"]:
            assert m["gl"] > 0
            assert m["ad"] > 0
            assert m["adx"] > 0
            assert m["xr"] > 0

    def test_nonexistent_team(self, foundation):
        from api.analytics import decomposition
        result = decomposition("zzz_no_team", foundation)
        assert result["matches"] == []


@pytest.mark.skipif(
    not COMPONENTS.exists() or not WALKFORWARD.exists(),
    reason="data files missing",
)
class TestTeamProfile:
    def test_shape(self, foundation):
        from api.analytics import team_profile
        from api.engine import _club_engine, data_version
        version = data_version()
        engine, _, _ = _club_engine(version)
        result = team_profile("barcelona", engine, foundation, version)
        assert result["team"] == "barcelona"
        assert result["elo"]["current"] is not None
        assert result["elo"]["peak"] >= result["elo"]["current"]
        assert len(result["elo"]["trajectory"]) > 100

    def test_profile_keys(self, foundation):
        from api.analytics import team_profile
        from api.engine import _club_engine, data_version
        version = data_version()
        engine, _, _ = _club_engine(version)
        result = team_profile("barcelona", engine, foundation, version)
        p = result["profile"]
        assert p is not None
        assert set(p.keys()) == {
            "attack", "defense", "strength", "attack_balance",
            "league_avg_attack", "league_avg_defense",
            "league_max_strength", "league_min_strength",
        }

    def test_h2h(self, foundation):
        from api.analytics import team_profile
        from api.engine import _club_engine, data_version
        version = data_version()
        engine, _, _ = _club_engine(version)
        result = team_profile("barcelona", engine, foundation, version)
        assert len(result["h2h"]) == 5
        for rival in result["h2h"]:
            assert "rival" in rival
            assert "total" in rival
            assert len(rival["matches"]) <= 10
