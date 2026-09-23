"""No fixture may go without a prediction logged before kick-off.

A missed matchday is a test of the model that can never be run again. Before
2026-09-23 fixtures were lost silently: ten days with no calendar reported OK,
a Monday-night kick-off compared against midnight, clubs without a rating
skipped, and every debutant's opening matches dropped as "unknown team".

Run:  .venv/Scripts/python.exe -m pytest tests/test_prediction_coverage.py -q
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.serving.debutants import (  # noqa: E402
    predict_match_or_proxy, relegated_last_season)


# ── debutants ─────────────────────────────────────────────────────────────────
def _history():
    rows = []
    for season, teams in (("2025-2026", ["a", "b", "c", "old1", "old2"]),
                          ("2026-2027", ["a", "b", "c", "new"])):
        for h in teams:
            for aw in teams:
                if h != aw:
                    rows.append({"season": season, "competition": "L", "home_team": h,
                                 "away_team": aw})
    return pd.DataFrame(rows)


class FakeEngine:
    """lambda_home = strength of home, lambda_away = strength of away."""
    STRENGTH = {"a": 2.0, "b": 1.5, "c": 1.2, "old1": 0.8, "old2": 1.0}

    def predict_match(self, h, a, competition=None, neutral=False):
        ns = SimpleNamespace(lambda_home=self.STRENGTH[h], lambda_away=self.STRENGTH[a])
        for k in ("shots", "sot", "corners", "fouls", "yellows"):
            ns.__dict__[f"expected_{k}_home"] = 10.0
            ns.__dict__[f"expected_{k}_away"] = 8.0
        return ns

    def markets_from_lambdas(self, lh, la):
        from mundialytics.statistical_core.prediction_engine import markets_from_lambdas
        return markets_from_lambdas(lh, la)


def test_relegated_are_last_season_clubs_missing_from_this_calendar():
    df = _history()
    assert relegated_last_season(df, "L", "2026-2027", {"a", "b", "c", "new"}) == ["old1", "old2"]


def test_debutant_is_priced_from_the_relegated_average():
    df = _history()
    known = {"a", "b", "c", "old1", "old2"}
    p, stand_in = predict_match_or_proxy(FakeEngine(), df, "new", "a", "L", "2026-2027",
                                         {"a", "b", "c", "new"}, known=known)
    assert stand_in == {"new": ["old1", "old2"]}
    assert p.lambda_home == pytest.approx(0.9)      # mean of 0.8 and 1.0
    assert p.lambda_away == pytest.approx(2.0)
    assert p.p_home_win + p.p_draw + p.p_away_win == pytest.approx(1.0)


def test_known_teams_go_straight_to_the_engine():
    df = _history()
    p, stand_in = predict_match_or_proxy(FakeEngine(), df, "a", "b", "L", "2026-2027",
                                         {"a", "b"}, known={"a", "b"})
    assert stand_in == {} and p.lambda_home == 2.0


# ── the logger's coverage check ───────────────────────────────────────────────
@pytest.fixture
def logger(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "log_upcoming_round", ROOT / "scripts/log_upcoming_round.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "PRED_LOG", tmp_path / "predictions_log.csv")
    monkeypatch.setattr(mod, "COVERAGE_JSON", tmp_path / "coverage_last.json")
    mod.EXPECTED.clear()
    mod.MISSING_SOURCES.clear()
    return mod


def _args(**kw):
    return SimpleNamespace(days=8, dry_run=False, **kw)


def _log(path, rows):
    pd.DataFrame([{"partido": p, "fecha": f, "mercado": "1X2"} for p, f in rows]).to_csv(
        path, index=False)


def test_every_fixture_logged_passes(logger):
    _log(logger.PRED_LOG, [("a vs b", "2026-10-09")])
    logger.EXPECTED.append(("a vs b", "2026-10-09"))
    assert logger.check_coverage(_args()) == 0


def test_an_unlogged_fixture_fails_the_step(logger):
    _log(logger.PRED_LOG, [("a vs b", "2026-10-09")])
    logger.EXPECTED += [("a vs b", "2026-10-09"), ("c vs d", "2026-10-10")]
    assert logger.check_coverage(_args()) == logger.EXIT_UNCOVERED
    import json
    assert json.loads(logger.COVERAGE_JSON.read_text())["missing"] == ["c vs d (2026-10-10)"]


def test_a_rescheduled_fixture_logged_under_its_old_date_counts(logger):
    _log(logger.PRED_LOG, [("a vs b", "2026-10-09")])
    logger.EXPECTED.append(("a vs b", "2026-10-11"))
    assert logger.check_coverage(_args()) == 0


def test_a_missing_calendar_in_season_fails(logger, monkeypatch):
    # no log file at all, and one league whose calendar could not be fetched
    logger.MISSING_SOURCES.append("LaLiga")
    monkeypatch.setattr(logger.pd.Timestamp, "now", classmethod(
        lambda cls, *a, **k: pd.Timestamp("2026-10-01")))
    assert logger.check_coverage(_args()) == logger.EXIT_NO_CALENDAR
