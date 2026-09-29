"""A played match is graded against what was logged BEFORE kick-off.

The match page used to re-run the current engine, fitted on that very match,
and present the result as the pre-match call. These pin the replacement: the
log is read back, only pre-kickoff rows count, and a full row rebuilds the
exact numbers that were logged.

Run:  .venv/Scripts/python.exe -m pytest tests/test_logged_prediction.py -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.serving import logged_prediction as lp  # noqa: E402
from mundialytics.statistical_core.prediction_engine import (  # noqa: E402
    deployed_markets_from_lambdas, markets_from_lambdas)

LH, LA = 2.0943, 1.0258


def _row(**kw):
    base = dict(logged_at="2026-10-01T10:00:00", season="2026-2027", jornada=8,
                partido="dortmund vs werder bremen", fecha="2026-10-09",
                home="dortmund", away="werder bremen", ambito="Total")
    base.update(kw)
    return base


@pytest.fixture
def log(tmp_path, monkeypatch):
    def write(rows):
        p = tmp_path / "predictions_log.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        monkeypatch.setattr(lp, "LOG", p)
        lp._cache.update(mtime=None, df=None)
        return p
    return write


def _full_rows(logged_at="2026-10-01T10:00:00", kickoff="2026-10-09T18:30Z"):
    probs, _ = deployed_markets_from_lambdas(LH, LA)
    common = dict(logged_at=logged_at, kickoff_utc=kickoff, train_cutoff="2026-09-20",
                  model_fp="abc123def456", lambda_home=LH, lambda_away=LA,
                  p_home=round(probs["p_home_win"], 4), p_draw=round(probs["p_draw"], 4),
                  p_away=round(probs["p_away_win"], 4), exp_shots_home=14.2)
    return [
        _row(**common, mercado="1X2", linea="", prob=round(probs["p_home_win"], 4),
             seleccion="1"),
        _row(**common, mercado="Goles", linea=2.5, prob=round(probs["p_over_25"], 4),
             seleccion="OVER"),
    ]


def test_full_row_rebuilds_the_logged_numbers(log):
    log(_full_rows())
    got = lp.find_logged("dortmund", "werder bremen", "2026-10-09")
    assert got is not None and got.full
    assert got.model_fp == "abc123def456" and got.train_cutoff == "2026-09-20"
    pred = lp.as_prediction(got, fallback=None)
    probs, dist = deployed_markets_from_lambdas(LH, LA)
    assert pred.p_home_win == pytest.approx(probs["p_home_win"], abs=1e-4)
    assert pred.p_over_25 == pytest.approx(probs["p_over_25"], abs=1e-4)
    assert pred.expected_shots_home == pytest.approx(14.2)
    assert pred.top_scorelines == dist.top_scorelines(8)


def test_row_logged_after_kickoff_is_never_used(log):
    # logged at 20:45 Madrid = 18:45 UTC, fifteen minutes after kick-off
    log(_full_rows(logged_at="2026-10-09T20:45:00"))
    assert lp.find_logged("dortmund", "werder bremen", "2026-10-09") is None


def test_row_logged_on_match_day_before_kickoff_counts(log):
    log(_full_rows(logged_at="2026-10-09T10:00:00"))
    assert lp.find_logged("dortmund", "werder bremen", "2026-10-09") is not None


def test_old_row_is_partial_and_grades_only_what_was_said(log):
    log([
        _row(mercado="1X2", linea="", prob=0.5383, seleccion="1"),
        _row(mercado="Goles", linea=2.5, prob=0.5878, seleccion="UNDER"),
    ])
    got = lp.find_logged("dortmund", "werder bremen", "2026-10-09")
    assert got is not None and not got.full
    src = got.source()
    assert src["kind"] == "logged-partial"
    assert src["pick"] == "1" and src["pickProbability"] == 0.5383
    # UNDER at 58.78% is an over probability of 41.22%
    assert src["over25"] == pytest.approx(0.4122)


def test_earliest_pre_kickoff_row_wins(log):
    early = _row(logged_at="2026-10-01T10:00:00", mercado="1X2", linea="", prob=0.51,
                 seleccion="1")
    late = _row(logged_at="2026-10-05T10:00:00", mercado="1X2", linea="", prob=0.61,
                seleccion="1")
    log([late, early])
    assert lp.find_logged("dortmund", "werder bremen", "2026-10-09").pick_prob == 0.51


def test_no_log_means_no_logged_prediction(log):
    log([_row(mercado="1X2", linea="", prob=0.5, seleccion="1", home="x", away="y")])
    assert lp.find_logged("dortmund", "werder bremen", "2026-10-09") is None


def test_markets_from_lambdas_keeps_matrix_raw_and_sharpens_only_the_trio():
    raw, dist = markets_from_lambdas(2.4, 0.7, rho=-0.06, temper=1.05, sharpen_gamma=1.0)
    sharp, _ = markets_from_lambdas(2.4, 0.7, rho=-0.06, temper=1.05, sharpen_gamma=1.3)
    m = dist.matrix.to_numpy()
    import numpy as np
    assert raw["p_home_win"] == pytest.approx(float(np.tril(m, -1).sum()), abs=1e-9)
    assert sharp["p_home_win"] > raw["p_home_win"]
    assert sharp["p_over_25"] == raw["p_over_25"]
