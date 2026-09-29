"""The market drift report (scripts/report_market_drift.py).

It exists because the 2026/27 shots bias sat in the log for three weeks: a
calibrated market must stay quiet, a market whose every line misses the same
way must raise an alert, and the per-match counts must compare like with like.

Run:  .venv/Scripts/python.exe -m pytest tests/test_market_drift.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import report_market_drift as rmd  # noqa: E402


def _ev(p: float, happened: float, n: int = 400, market: str = "shots") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    y = rng.random(n) < happened
    return pd.DataFrame({"mercado_id": market, "ambito": "Total", "linea": "24.5", "es_jugador": False,
                         "prob": p, "lado": "OVER", "acierto": y.astype(float)})


def test_calibrated_market_is_quiet_and_a_biased_one_alerts():
    quiet = rmd.calibration_drift(_ev(0.6, 0.6)).iloc[0]
    loud = rmd.calibration_drift(_ev(0.6, 0.77)).iloc[0]
    assert abs(quiet["z"]) < rmd.Z_ALERT
    assert loud["z"] >= rmd.Z_ALERT and loud["actual"] > loud["expected"]


def test_pick_one_markets_are_left_to_the_track_record():
    assert rmd.calibration_drift(_ev(0.5, 0.5, market="1X2")).empty


def test_under_picks_count_as_the_over_not_happening():
    ev = _ev(0.3, 0.0).assign(lado="UNDER", acierto=1.0)   # every UNDER right = no over
    row = rmd.calibration_drift(ev).iloc[0]
    assert row["actual"] == 0.0 and row["expected"] == 0.3


def test_count_drift_compares_the_logged_expectation_with_the_result():
    log = pd.DataFrame({"mercado": "1X2", "logged_at": "2026-10-01", "home": [f"h{i}" for i in range(30)],
                        "away": [f"a{i}" for i in range(30)], "fecha": "2026-10-09",
                        "exp_shots_home": 12.0, "exp_shots_away": 10.0})
    found = pd.DataFrame({"home_team": log["home"], "away_team": log["away"],
                          "date": pd.Timestamp("2026-10-09"), "competition": "LaLiga",
                          "home_shots": 14.0 + np.arange(30) % 3, "away_shots": 11.0})
    c = rmd.count_drift(log, found)
    shots = c[(c["market"] == "shots") & (c["scope"] == "all")].iloc[0]
    assert shots["expected"] == 22.0 and shots["actual"] == 26.0 and shots["z"] > rmd.Z_ALERT
    assert set(c["market"]) == {"shots"}          # no expectation logged -> not reported
