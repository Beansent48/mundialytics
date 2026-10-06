"""The match page's expected shots/corners/cards must come from the model we price with.

`_expected_stats` used to read the engine's own EventLambdaModel (rolling form +
Elo) while the team-props card right below it read TeamPropsModel (the validated
recipe: EWMA windows, ASYM supremacy on the engine's lambdas, MLE-strengths
prior, league level, running residual, referee). The page therefore showed one
quantity twice with two different values, and a played match was compared
against a model we never serve. The point estimate now prefers the team-props
lambda and keeps the engine's as the fallback.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from api.main import _expected_stats  # noqa: E402


class _Pred:
    """Only the fields _expected_stats reads off a MatchPrediction."""
    expected_shots_home, expected_shots_away = 12.4, 9.1
    expected_sot_home, expected_sot_away = 4.2, 3.0
    expected_corners_home, expected_corners_away = 5.5, 4.1
    expected_fouls_home, expected_fouls_away = 11.0, 12.5
    expected_yellows_home, expected_yellows_away = 1.8, 2.1


FX = {
    "shots": {"lambda_home": 14.9, "lambda_away": 10.2, "dispersion": 1.39},
    "corners": {"lambda_home": 6.1, "lambda_away": 4.4, "dispersion": 1.18},
}


def _by_key(rows):
    return {r["key"]: r for r in rows}


def test_prefers_team_props_lambdas():
    s = _by_key(_expected_stats(_Pred(), FX))
    assert s["shots"]["source"] == "team_props"
    assert (s["shots"]["home"], s["shots"]["away"]) == (14.9, 10.2)
    assert s["corners"]["source"] == "team_props"
    assert (s["corners"]["home"], s["corners"]["away"]) == (6.1, 4.4)


def test_falls_back_to_the_engine_per_market():
    """A market the props model dropped keeps the engine's number, not a hole."""
    s = _by_key(_expected_stats(_Pred(), FX))
    for key, home, away in (("shotsOnTarget", 4.2, 3.0), ("fouls", 11.0, 12.5),
                            ("yellows", 1.8, 2.1)):
        assert s[key]["source"] == "engine"
        assert (s[key]["home"], s[key]["away"]) == (home, away)


def test_a_logged_number_is_never_replaced_by_a_recomputed_one():
    """The regression this file exists for.

    On a played match `pred` is rebuilt from the pre-kickoff log, while `fx` is
    computed now by a model that has already seen the result. Preferring `fx`
    there would quietly turn the record into hindsight -- the same mistake the
    track record was fixed for in 2026-09.
    """
    p = _Pred()
    p.expected_from_log = frozenset({"shots_home", "shots_away"})
    s = _by_key(_expected_stats(p, FX))
    assert s["shots"]["source"] == "logged"
    assert (s["shots"]["home"], s["shots"]["away"]) == (12.4, 9.1)
    # a market the log didn't carry is still free to use the better model
    assert s["corners"]["source"] == "team_props"
    assert (s["corners"]["home"], s["corners"]["away"]) == (6.1, 4.4)


def test_one_side_logged_is_not_treated_as_logged():
    """Half a logged pair can't be mixed with half a recomputed one."""
    p = _Pred()
    p.expected_from_log = frozenset({"shots_home"})
    s = _by_key(_expected_stats(p, FX))
    assert s["shots"]["source"] == "team_props"


def test_as_prediction_marks_what_came_from_the_log():
    """_expected_stats reads `expected_from_log`; the rebuild must set it."""
    from mundialytics.serving.logged_prediction import LoggedPrediction, as_prediction

    lp = LoggedPrediction(
        logged_at="2026-09-19T08:00:00", full=True, pick="1", pick_prob=0.5,
        over25=0.55, trio={"home": 0.5, "draw": 0.25, "away": 0.25},
        lambdas=(1.6, 1.1), model_fp=None, train_cutoff=None,
        expected={"shots_home": 15.0, "shots_away": 8.0},
    )
    ns = as_prediction(lp, fallback=_Pred())
    assert ns.expected_from_log == frozenset({"shots_home", "shots_away"})
    assert ns.expected_shots_home == 15.0
    assert ns.expected_corners_home == _Pred.expected_corners_home  # fallback
    assert _by_key(_expected_stats(ns, FX))["shots"]["source"] == "logged"


def test_no_props_models_at_all_still_serves_every_stat():
    s = _by_key(_expected_stats(_Pred(), None))
    assert set(s) == {"shots", "shotsOnTarget", "corners", "fouls", "yellows"}
    assert all(r["source"] == "engine" for r in s.values())
    assert (s["shots"]["home"], s["shots"]["away"]) == (12.4, 9.1)


def test_range_brackets_the_point_estimate_it_is_shown_with():
    """The range is the reader's calibration of the number next to it, so it must
    wrap the served point estimate -- the bug would be a range built round the
    engine's lambda under a team-props number.

    One count of slack on the top end: the range is the MODE-centred interval of a
    skewed count distribution, so for a small lambda it can legitimately close just
    below the mean (yellows 2.1 prices as 1-2).
    """
    for fx in (FX, None):
        for r in _expected_stats(_Pred(), fx):
            for side in ("home", "away"):
                rng = r[f"{side}Range"]
                assert rng is not None, (r["key"], side)
                assert rng["lo"] <= r[side] <= rng["hi"] + 1, (r["key"], side, rng)
                assert 0.0 < rng["p"] <= 1.0
