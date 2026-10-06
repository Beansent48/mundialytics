"""The log, the drift report and the played page must all name the same model.

The board's event markets are priced by TeamPropsModel, but the pre-kickoff log
only ever carried `exp_*`, the engine's own EventLambdaModel -- a different and
measurably worse model (scripts/validate_expected_stats_source.py: team props
wins 5/5 seasons in all five markets). So the record, the drift report built on
it and the played page's "what we said" were all describing a model we do not
price with.

The log now carries `tp_*` alongside, and every reader prefers it. `exp_*` is
left exactly as it was: it is all the rows logged before 2026-10-06 have, and
rewriting its meaning would break the drift series built on it.

Run:  .venv/Scripts/python.exe -m pytest tests/test_logged_served_model.py -q
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import report_market_drift as rmd  # noqa: E402
from mundialytics.serving.logged_prediction import STAT_KEYS, expected_from_row  # noqa: E402


def _log_row(**over) -> pd.Series:
    row = {"logged_at": "2026-10-06T06:40:00", "seleccion": "1", "prob": 0.5,
           "lambda_home": 1.7, "lambda_away": 1.1,
           "p_home": 0.5, "p_draw": 0.25, "p_away": 0.25,
           "model_fp": "abc", "train_cutoff": "2026-09-20"}
    for k in STAT_KEYS:
        for side in ("home", "away"):
            row[f"exp_{k}_{side}"] = 9.0
    row.update(over)
    return pd.Series(row)


# ── what the played page reads back ───────────────────────────────────────────

def test_the_served_model_wins_over_the_engines_number():
    r = _log_row(tp_shots_home=14.5, tp_shots_away=10.25)
    got = expected_from_row(r)
    assert got["shots_home"] == 14.5
    assert got["shots_away"] == 10.25


def test_markets_without_a_served_number_keep_the_engines():
    """Rows logged before 2026-10-06, and any market team props dropped."""
    got = expected_from_row(_log_row(tp_shots_home=14.5, tp_shots_away=10.25))
    assert got["corners_home"] == 9.0
    assert got["yellows_away"] == 9.0


def test_a_row_with_neither_carries_no_expectation_rather_than_a_zero():
    row = _log_row()
    for k in STAT_KEYS:
        for side in ("home", "away"):
            row[f"exp_{k}_{side}"] = None
    assert expected_from_row(row) == {}


# ── what the drift report measures ────────────────────────────────────────────

def _m(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


def test_drift_coalesces_onto_the_served_model():
    m = _m(tp_shots_home=[12.0, None], exp_shots_home=[16.0, 15.0])
    got = rmd._coalesce(m, ("tp_shots_home", "exp_shots_home")).tolist()
    assert got == [12.0, 15.0], "served number first, engine's only where it is missing"


def test_drift_falls_back_when_the_served_column_does_not_exist_at_all():
    m = _m(exp_shots_home=[16.0, 15.0])
    assert rmd._coalesce(m, ("tp_shots_home", "exp_shots_home")).tolist() == [16.0, 15.0]


def test_drift_skips_a_count_it_has_no_column_for():
    assert rmd._coalesce(_m(other=[1.0]), ("tp_shots_home", "exp_shots_home")) is None


def test_every_count_in_the_report_asks_for_the_served_column_first():
    """A new count added without its tp_* name would silently measure the engine."""
    for name, ((hnames, anames), _) in rmd.COUNTS.items():
        if name == "goals":          # goals have one model, the lambdas
            continue
        assert hnames[0].startswith("tp_"), name
        assert anames[0].startswith("tp_"), name
        assert hnames[-1].startswith("exp_"), name


def test_count_drift_runs_on_a_log_that_mixes_both_vintages():
    """The series spans the change: old rows carry exp_* only, new ones both."""
    log = pd.DataFrame({
        "mercado": ["1X2", "1X2"], "logged_at": ["2026-09-15", "2026-10-06"],
        "home": ["a", "c"], "away": ["b", "d"], "fecha": ["2026-09-16", "2026-10-07"],
        "lambda_home": [1.5, 1.6], "lambda_away": [1.2, 1.1],
        "exp_shots_home": [13.0, 13.0], "exp_shots_away": [11.0, 11.0],
        "tp_shots_home": [None, 16.0], "tp_shots_away": [None, 10.0],
    })
    found = pd.DataFrame({
        "date": ["2026-09-16", "2026-10-07"], "home_team": ["a", "c"], "away_team": ["b", "d"],
        "competition": ["LaLiga", "LaLiga"], "home_goals": [1, 2], "away_goals": [1, 0],
        "home_shots": [14, 16], "away_shots": [10, 8],
    })
    out = rmd.count_drift(log, found)
    shots = out[(out["market"] == "shots") & (out["scope"] == "all")]
    assert len(shots) == 1
    # Both rows must be counted -- the old one must not be dropped for a missing
    # tp_* column -- and the new one must be measured on its tp_* (16+10=26), not
    # on the exp_* sitting right next to it (13+11=24). Mean 25, not 24.
    assert shots.iloc[0]["n"] == 2
    assert shots.iloc[0]["expected"] == 25.0
    assert shots.iloc[0]["actual"] == 24.0
