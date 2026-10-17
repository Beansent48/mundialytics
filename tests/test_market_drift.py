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


def _log(partido, mercado, ambito, linea, fp, logged_at):
    return pd.DataFrame({"partido": partido, "mercado": mercado, "ambito": ambito,
                         "linea": linea, "model_fp": fp, "logged_at": logged_at})


def test_era_join_survives_a_market_with_no_line():
    """The player markets carry no line: NaN on one side, "" on the other.

    Before this was canonicalised through a float, 45% of the log failed to join
    and landed in the oldest era by default -- which is the opposite of the point.
    """
    log = _log(["p1", "p2"], "jug_goleador", ["mbappe", "kane"], [np.nan, np.nan],
               "fp_new", "2026-10-05")
    ev = pd.DataFrame({"partido": ["p1", "p2"], "mercado_id": "jug_goleador",
                       "ambito": ["mbappe", "kane"], "linea": ["", ""],
                       "acierto": [1.0, 0.0]})
    out = rmd.attach_era(ev, log)
    assert list(out["era"]) == ["fp_new", "fp_new"]


def test_a_rows_era_is_the_model_that_priced_it_not_its_month():
    """October fixtures logged before PR #25 are priced by the September model."""
    log = pd.concat([
        _log(["p1"], "shots", "Total", [22.5], None, "2026-09-18"),
        _log(["p2"], "shots", "Total", [22.5], "fp_new", "2026-10-05")])
    ev = pd.DataFrame({"partido": ["p1", "p2"], "mercado_id": "shots", "ambito": "Total",
                       "linea": ["22.5", "22.5"], "acierto": [1.0, 1.0]})
    assert list(rmd.attach_era(ev, log)["era"]) == [rmd.ERA_NONE, "fp_new"]
    assert rmd.era_order(rmd.attach_era(ev, log)) == [rmd.ERA_NONE, "fp_new"]


def test_a_pooled_alert_is_never_blamed_on_a_superseded_model():
    """With too few results on the newest era the verdict is undecided, not a
    reading taken from the era before it."""
    n_old, n_new = 400, 10
    ev = pd.concat([
        _ev(0.6, 0.85, n=n_old).assign(partido=[f"old{i}" for i in range(n_old)]),
        _ev(0.6, 0.60, n=n_new).assign(partido=[f"new{i}" for i in range(n_new)])])
    log = pd.concat([
        _log([f"old{i}" for i in range(n_old)], "shots", "Total", 24.5, None, "2026-09-18"),
        _log([f"new{i}" for i in range(n_new)], "shots", "Total", 24.5, "fp_new", "2026-10-05")])
    pooled = rmd.calibration_drift(ev).to_dict("records")
    pooled[0]["kind"] = "calibration"
    v = rmd.era_verdicts(pooled, ev, log, pd.DataFrame())[0]
    assert v["era"] == "fp_new" and v["n"] == n_new
    assert v["status"] == "undecided"          # not "live" off the 400 old rows


def test_an_alert_still_there_on_the_newest_model_stays_live():
    n = 400
    ev = _ev(0.6, 0.85, n=n).assign(partido=[f"m{i}" for i in range(n)])
    log = _log([f"m{i}" for i in range(n)], "shots", "Total", 24.5, "fp_new", "2026-10-05")
    pooled = rmd.calibration_drift(ev).to_dict("records")
    pooled[0]["kind"] = "calibration"
    v = rmd.era_verdicts(pooled, ev, log, pd.DataFrame())[0]
    assert v["era"] == "fp_new" and v["status"] == "live" and v["z"] >= rmd.Z_ALERT
