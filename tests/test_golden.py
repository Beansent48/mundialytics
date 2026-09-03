"""Golden regression tests — pure-math pins, data-independent.

These freeze the exact numerical behavior of the validated chains. If a
refactor, a pandas/scipy/sklearn upgrade, or an 'innocent' edit moves ANY of
these by more than 1e-9, something that was validated has changed and must be
re-validated before deploying. (The manual byte-identical audit of 2026-07-23,
automated forever.)

Run:  .venv/Scripts/python.exe -m pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.statistical_core.distributions import (  # noqa: E402
    outcome_probabilities, scoreline_distribution)


TOL = 1e-9


class TestScorelineDistribution:
    def test_dc_rho_matrix_cells(self):
        d = scoreline_distribution(1.5, 1.2, dixon_coles_rho=-0.17)
        assert d.matrix.values.sum() == pytest.approx(1.0, abs=TOL)
        assert d.matrix.iloc[0, 0] == pytest.approx(0.08777045352823995, abs=TOL)
        assert d.matrix.iloc[1, 1] == pytest.approx(0.14153489673083713, abs=TOL)
        assert d.matrix.iloc[2, 1] == pytest.approx(0.0907274979043828, abs=TOL)

    def test_outcome_probabilities_deployed_rho(self):
        p = outcome_probabilities(1.5, 1.2, dixon_coles_rho=-0.17)
        assert p["p_home_win"] == pytest.approx(0.4208999929, abs=1e-8)
        assert p["p_draw"] == pytest.approx(0.2959467331, abs=1e-8)
        assert p["p_away_win"] == pytest.approx(0.283153274, abs=1e-8)
        assert p["p_over_25"] == pytest.approx(0.5063752058, abs=1e-8)
        assert p["p_btts"] == pytest.approx(0.5634459396, abs=1e-8)

    def test_temper_default_is_identity(self):
        a = scoreline_distribution(1.6, 1.1, dixon_coles_rho=-0.07)
        b = scoreline_distribution(1.6, 1.1, dixon_coles_rho=-0.07, temper=1.0)
        assert np.allclose(a.matrix.values, b.matrix.values, atol=1e-15)

    def test_temper_active_pins(self):
        p = outcome_probabilities(1.5, 1.2, dixon_coles_rho=-0.17, temper=1.05)
        assert p["p_over_25"] == pytest.approx(0.4945477781, abs=1e-8)


class TestBookingPoints:
    def test_booking_grid_convolution(self):
        from mundialytics.props.team_props import _p_booking_over
        v = float(_p_booking_over(np.array([4.0]), 1.13, np.array([0.2]), 40.5)[0])
        assert v == pytest.approx(0.4676534628, abs=1e-8)


class TestEuroEvents:
    CAL = {"markets": {"corners": {"c": 1.468, "hfa": 0.2, "b": 0.424,
                                   "disp_total": 1.18, "disp_side": 1.64, "lines": [9.5]}}}

    def test_predict_euro_events_pins(self):
        from mundialytics.statistical_core.competition.european import predict_euro_events
        fx = predict_euro_events(1900, 1700, self.CAL)
        assert fx["corners"]["lambda_home"] == pytest.approx(6.55, abs=0.01)
        assert fx["corners"]["lambda_away"] == pytest.approx(3.51, abs=0.01)
        assert fx["corners"]["over"][9.5] == pytest.approx(0.5393, abs=1e-4)


class TestEuropeanTournament:
    def _mk(self, elos, fixtures=None, ko=None, seed=7):
        from mundialytics.statistical_core.competition.european import EuropeanTournament
        calib = {"c": 0.2043, "hfa": 0.2806, "b": 0.7394}
        t = EuropeanTournament("champions", elos, calib, fixtures, ko)
        t.rng = np.random.default_rng(seed)
        return t

    def test_equal_elos_near_uniform(self):
        elos = {f"T{i:02d}": 1700.0 for i in range(36)}
        res = self._mk(elos).simulate(800)
        assert res["p_champion"].sum() == pytest.approx(1.0, abs=1e-9)
        assert res["p_top24"].sum() == pytest.approx(24.0, abs=1e-6)
        # equal strengths -> nobody should stray far from 1/36
        assert res["p_champion"].max() < 3.5 / 36

    def test_strength_monotonicity(self):
        elos = {f"T{i:02d}": 2000.0 - i * 12 for i in range(36)}
        res = self._mk(elos).simulate(800).set_index("team")
        assert res.loc["T00", "p_champion"] > res.loc["T18", "p_champion"]
        assert res.loc["T00", "p_top24"] > 0.9
        # probabilities are monotone across rounds for every team
        assert (res["p_r16"] >= res["p_qf"] - 1e-9).all()
        assert (res["p_qf"] >= res["p_sf"] - 1e-9).all()
        assert (res["p_sf"] >= res["p_final"] - 1e-9).all()
        assert (res["p_final"] >= res["p_champion"] - 1e-9).all()


class TestDeployedCalibrations:
    """Deployed constants files exist and stay in their validated ranges."""

    def test_euro_lambda_constants(self):
        import json
        p = ROOT / "data/processed/elo_lambda_calibration_euro.json"
        if not p.exists():
            pytest.skip("euro calibration not generated")
        c = json.loads(p.read_text())
        assert 0.15 < c["c"] < 0.30
        assert 0.15 < c["hfa"] < 0.40
        assert 0.5 < c["b"] < 1.0

    def test_elo_event_constants(self):
        import json
        p = ROOT / "data/processed/elo_event_calibration.json"
        if not p.exists():
            pytest.skip("event calibration not generated")
        c = json.loads(p.read_text())
        # cards/fouls are STYLE, not strength: their Elo slope must stay zeroed
        assert c["markets"]["yellows"]["b"] == 0.0
        assert c["markets"]["fouls"]["b"] == 0.0
        assert c["markets"]["corners"]["b"] > 0.2
        assert c["markets"]["shots"]["b"] > 0.2


# ── date/season sanity ──────────────────────────────────────────────────────
def test_season_labels_agree_with_their_dates():
    """A season labelled 2025-2026 must contain 2025/2026 dates, not 1925's.

    Guards a whole class of silly-but-costly bug: two-digit season codes are
    ambiguous, and a library resolving "2627" to 1926/27 rather than 2026/27
    returns a full, plausible-looking dataset from the wrong century. That
    happened on a throwaway FBref probe; it must never reach the foundation.
    """
    import pandas as pd

    found = ROOT / "data/processed/foundation_big5_multi_season.csv"
    if not found.exists():
        pytest.skip("foundation not built")
    df = pd.read_csv(found, usecols=["season", "date"], low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "season"])

    bad = []
    for season, g in df.groupby("season"):
        y1, y2 = (int(p) for p in str(season).split("-"))
        years = set(g["date"].dt.year)
        if not years <= {y1, y2}:
            bad.append(f"{season}: {sorted(years - {y1, y2})}")
    assert not bad, f"season labels disagree with their dates -> {bad[:3]}"


def test_no_impossible_dates_in_the_foundation():
    import pandas as pd

    found = ROOT / "data/processed/foundation_big5_multi_season.csv"
    if not found.exists():
        pytest.skip("foundation not built")
    d = pd.to_datetime(pd.read_csv(found, usecols=["date"])["date"], errors="coerce")
    assert d.notna().all(), "unparseable dates"
    assert d.min() >= pd.Timestamp("1990-01-01"), f"date from the wrong century: {d.min()}"
    assert d.max() <= pd.Timestamp.now() + pd.Timedelta(days=7), f"future date: {d.max()}"
