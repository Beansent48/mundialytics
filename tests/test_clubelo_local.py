"""Local ClubElo roll-forward: update maths, conservation, and provenance.

Exists because the European layer lost its rating source — the ClubElo API
returned 502 for three days with the Champions League five days out, and the
newest cached snapshot was six weeks stale. Rather than depend on it, the
snapshot is seeded once and advanced locally from results we hold.

The update rule's constants are FITTED (scripts/fit_local_elo.py: K=16,
home advantage=65, and rolling forward beat freezing in 5/5 cutoffs), so what
these tests pin is the maths and the invariants, not the fitted values.

Run:  .venv/Scripts/python.exe -m pytest tests/test_clubelo_local.py -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.ratings.clubelo_local import (  # noqa: E402
    EloParams, expected_home, goal_diff_multiplier, roll_forward, to_frame)


def _match(h, a, hg, ag, date="2026-08-01"):
    return pd.DataFrame([{"date": pd.Timestamp(date), "home_team": h, "away_team": a,
                          "home_goals": hg, "away_goals": ag, "neutral": 0}])


# ── the expectation curve ───────────────────────────────────────────────────
def test_equal_teams_expect_more_than_half_at_home():
    assert expected_home(1500, 1500, hfa=65) > 0.5


def test_neutral_venue_removes_the_home_edge():
    assert expected_home(1500, 1500, hfa=65, neutral=True) == pytest.approx(0.5)


def test_expectation_is_monotone_in_rating():
    a = expected_home(1500, 1500, 65)
    b = expected_home(1700, 1500, 65)
    c = expected_home(1300, 1500, 65)
    assert c < a < b


def test_four_hundred_points_is_the_classic_ten_to_one():
    # the defining property of the 400-point scale
    assert expected_home(1900, 1500, hfa=0) == pytest.approx(10 / 11, abs=1e-9)


# ── margin weighting ────────────────────────────────────────────────────────
def test_bigger_margins_weigh_more_and_one_goal_is_the_baseline():
    assert goal_diff_multiplier(1) == 1.0
    assert goal_diff_multiplier(0) == 1.0          # a draw carries no margin
    ms = [goal_diff_multiplier(m) for m in (1, 2, 3, 4, 5, 6)]
    assert all(ms[i] < ms[i + 1] for i in range(len(ms) - 1))


def test_margin_weight_ignores_direction():
    assert goal_diff_multiplier(-3) == goal_diff_multiplier(3)


# ── roll-forward invariants ─────────────────────────────────────────────────
def test_updates_are_zero_sum_so_the_scale_cannot_drift():
    """The mean rating must be untouched — otherwise the validated
    elo_lambda_calibration_euro.json, fitted on ClubElo's scale, stops applying."""
    seed = {"A": 1700.0, "B": 1500.0, "C": 1600.0}
    elo, _ = roll_forward(seed, _match("A", "B", 3, 0), EloParams(16, 65))
    assert sum(elo.values()) == pytest.approx(sum(seed.values()), abs=1e-9)


def test_winner_gains_and_loser_loses():
    seed = {"A": 1500.0, "B": 1500.0}
    elo, _ = roll_forward(seed, _match("A", "B", 2, 0), EloParams(16, 65))
    assert elo["A"] > 1500.0 > elo["B"]


def test_beating_a_stronger_side_moves_more_than_beating_a_weaker_one():
    p = EloParams(16, 65)
    upset, _ = roll_forward({"A": 1500.0, "B": 1900.0}, _match("A", "B", 1, 0), p)
    routine, _ = roll_forward({"A": 1500.0, "B": 1100.0}, _match("A", "B", 1, 0), p)
    assert (upset["A"] - 1500.0) > (routine["A"] - 1500.0)


def test_a_bigger_win_moves_the_rating_further():
    p = EloParams(16, 65)
    narrow, _ = roll_forward({"A": 1500.0, "B": 1500.0}, _match("A", "B", 1, 0), p)
    rout, _ = roll_forward({"A": 1500.0, "B": 1500.0}, _match("A", "B", 5, 0), p)
    assert rout["A"] > narrow["A"]


def test_unknown_clubs_are_skipped_not_invented():
    seed = {"A": 1500.0}
    elo, last = roll_forward(seed, _match("A", "GHOST", 1, 0), EloParams(16, 65))
    assert elo == seed and last == {}


def test_empty_fixture_list_is_a_no_op():
    seed = {"A": 1500.0}
    elo, last = roll_forward(seed, pd.DataFrame(), EloParams(16, 65))
    assert elo == seed and last == {}


# ── provenance ──────────────────────────────────────────────────────────────
def test_untouched_clubs_are_flagged_stale_with_the_seed_date():
    seed = {"A": 1500.0, "B": 1500.0, "IDLE": 1400.0}
    elo, last = roll_forward(seed, _match("A", "B", 1, 0, "2026-08-05"), EloParams(16, 65))
    df = to_frame(elo, last, "2026-07-23").set_index("club")
    assert df.loc["IDLE", "stale"] and df.loc["IDLE", "last_updated"] == "2026-07-23"
    assert not df.loc["A", "stale"] and df.loc["A", "last_updated"] == "2026-08-05"


def test_generated_table_is_usable_if_present():
    f = ROOT / "data/processed/clubelo_local.csv"
    if not f.exists():
        pytest.skip("local table not built")
    d = pd.read_csv(f)
    assert {"club", "elo", "last_updated", "stale", "seed_date"} <= set(d.columns)
    # ClubElo's real span runs from ~700 (Andorran/Gibraltarian minnows) to
    # ~2100 (the European champion), so this is a sanity bound, not a tight one.
    assert d["elo"].between(600, 2600).all(), "ratings outside any plausible range"
    assert d["club"].is_unique
