"""
Tests for the competition layer (statistical_core/competition).

Fast, mostly synthetic — the LeagueState / standings / xPoints / resume-simulator
are exercised without training the full PredictionEngine (that's covered by one
optional integration check at the end, skipped if the foundation CSV is absent).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mundialytics.statistical_core.competition import (
    compute_standings,
    expected_points_table,
    simulate_rest_of_season,
)
from mundialytics.statistical_core.competition.state import LeagueState

FOUNDATION = Path("data/processed/foundation_big5_multi_season.csv")


def _played(rows):
    return pd.DataFrame(
        [{"home_team": h, "away_team": a, "home_goals": hg, "away_goals": ag} for h, a, hg, ag in rows]
    )


def _state(teams, played_rows, remaining_rows, cutoff="2025-01-01"):
    played = _played(played_rows)
    remaining = pd.DataFrame(
        [{"home_team": h, "away_team": a} for h, a in remaining_rows]
    )
    return LeagueState(
        competition="TestLiga", season="2024-2025",
        cutoff_date=pd.to_datetime(cutoff), teams=list(teams),
        played=played, remaining=remaining,
    )


# ── Standings ──────────────────────────────────────────────────────────────────

def test_standings_basic_order_and_points():
    # A beats B, A beats C, B draws C.
    played = _played([("A", "B", 2, 0), ("A", "C", 1, 0), ("B", "C", 1, 1)])
    table = compute_standings(played, teams=["A", "B", "C"])
    # B and C both on 1 pt, but C's GD (-1) beats B's (-2) -> a, c, b.
    assert list(table["team"]) == ["a", "c", "b"]
    assert dict(zip(table["team"], table["points"])) == {"a": 6, "b": 1, "c": 1}
    a = table[table["team"] == "a"].iloc[0]
    assert (a["won"], a["drawn"], a["lost"], a["goal_diff"]) == (2, 0, 0, 3)


def test_standings_goal_difference_tiebreak():
    # A and B both 3 pts; A has better GD.
    played = _played([("A", "C", 5, 0), ("B", "C", 1, 0)])
    table = compute_standings(played, teams=["A", "B", "C"])
    assert list(table["team"])[:2] == ["a", "b"]


def test_standings_includes_teams_with_no_matches():
    played = _played([("A", "B", 1, 0)])
    table = compute_standings(played, teams=["A", "B", "Z"])
    assert set(table["team"]) == {"a", "b", "z"}
    assert table[table["team"] == "z"].iloc[0]["played"] == 0


# ── xPoints ────────────────────────────────────────────────────────────────────

def _lambda_frame(fixtures_with_lams):
    return pd.DataFrame(
        [{"home_team": h, "away_team": a, "lambda_home": lh, "lambda_away": la}
         for h, a, lh, la in fixtures_with_lams]
    )


def test_xpoints_never_below_current_and_counts_matches():
    st = _state(["A", "B"], [("A", "B", 1, 0)], [("B", "A"), ("A", "B")])
    lam = _lambda_frame([("B", "A", 1.2, 1.4), ("A", "B", 1.6, 1.0)])
    xp = expected_points_table(lam, st)
    for _, r in xp.iterrows():
        assert r["projected_points"] >= r["current_points"] - 1e-9
        assert r["matches_remaining"] == 2


# ── Resume simulator ─────────────────────────────────────────────────────────────

def test_probabilities_normalise():
    teams = [f"T{i}" for i in range(6)]
    played = [(teams[0], teams[1], 1, 0)]
    remaining = [(teams[i], teams[j]) for i in range(6) for j in range(6) if i != j]
    st = _state(teams, played, remaining)
    lam = _lambda_frame([(h, a, 1.4, 1.2) for h, a in remaining])
    fc = simulate_rest_of_season(lam, st, n_sims=3000, relegation_places=2, top_places=(2, 4))
    assert abs(fc.team_probs["p_champion"].sum() - 1.0) < 1e-9
    assert abs(fc.team_probs["p_relegation"].sum() - 2.0) < 1e-9
    # Each team's position row is a proper distribution.
    assert np.allclose(fc.position_matrix.sum(axis=1), 1.0)


def test_locked_lead_dominates_when_few_games_left():
    # A has a massive lead with only one round left -> near-certain champion.
    teams = ["A", "B", "C"]
    played = [("A", "B", 5, 0), ("A", "C", 5, 0), ("B", "C", 0, 0),
              ("A", "B", 5, 0), ("A", "C", 5, 0)]
    remaining = [("B", "A"), ("C", "B")]
    st = _state(teams, played, remaining)
    lam = _lambda_frame([("B", "A", 1.0, 1.5), ("C", "B", 1.2, 1.2)])
    fc = simulate_rest_of_season(lam, st, n_sims=3000)
    champ = fc.team_probs[fc.team_probs["team"] == "a"].iloc[0]
    assert champ["p_champion"] > 0.95


def test_deterministic_with_seed():
    teams = ["A", "B", "C", "D"]
    remaining = [(teams[i], teams[j]) for i in range(4) for j in range(4) if i != j]
    st = _state(teams, [("A", "B", 1, 1)], remaining)
    lam = _lambda_frame([(h, a, 1.3, 1.1) for h, a in remaining])
    a = simulate_rest_of_season(lam, st, n_sims=2000, random_seed=7)
    b = simulate_rest_of_season(lam, st, n_sims=2000, random_seed=7)
    pd.testing.assert_frame_equal(a.team_probs, b.team_probs)


def test_completed_season_returns_current_table():
    teams = ["A", "B", "C"]
    played = [("A", "B", 3, 0), ("A", "C", 2, 0), ("B", "C", 1, 0)]
    st = _state(teams, played, [], cutoff=None)
    fc = simulate_rest_of_season(_lambda_frame([]), st, n_sims=5)
    champ = fc.team_probs.iloc[0]
    assert champ["team"] == "a" and champ["p_champion"] == 1.0


# ── Integration (needs the real foundation CSV) ─────────────────────────────────

def _snap(md, teams, champ):
    return {
        "matchday": md, "fingerprint": {"n_played": md * len(teams) // 2}, "n_remaining": 10,
        "standings": [{"team": teams[0], "points": 40, "rank": 1}],
        "fixtures": {"played": [], "remaining": []},
        "forecast": {"team_probs": [{"team": t, "p_champion": c, "p_relegation": 0.0, "exp_points": 70}
                                     for t, c in zip(teams, champ)],
                     "position_matrix": {"teams": teams, "positions": [1, 2], "values": [[0.6, 0.4], [0.4, 0.6]]}},
    }


def test_cache_bundle_roundtrip_and_slug(tmp_path):
    from mundialytics.statistical_core.competition import forecast_cache as fc
    bundle = {
        "meta": {"schema": fc.BUNDLE_SCHEMA, "competition": "LaLiga", "season": "2024-2025",
                 "matchdays": [10, 20], "current_matchday": 20},
        "snapshots": {"10": _snap(10, ["a", "b"], [0.5, 0.5]), "20": _snap(20, ["a", "b"], [0.7, 0.3])},
    }
    path = fc.save_bundle(bundle, cache_dir=tmp_path)
    assert path.exists() and path.name == "LaLiga__2024-2025.json"
    assert fc.load_bundle("LaLiga", "2024-2025", cache_dir=tmp_path) == bundle
    assert fc.load_bundle("Nope", "2020-2021", cache_dir=tmp_path) is None


def test_matchday_grid_caps_and_includes_top():
    from mundialytics.statistical_core.competition.forecast_cache import _matchday_grid
    assert _matchday_grid(38, 5, None) == [5, 10, 15, 20, 25, 30, 35, 37]
    assert _matchday_grid(38, 10, 25) == [10, 20, 25]
    assert _matchday_grid(34, 5, None) == [5, 10, 15, 20, 25, 30, 33]


def test_full_grid_adds_final_point_only_when_season_complete():
    from mundialytics.statistical_core.competition.forecast_cache import _full_grid

    class _Probe:
        def __init__(self, complete): self.is_complete = complete

    complete = _full_grid(_Probe(True), 38, 5, None)
    assert complete == [5, 10, 15, 20, 25, 30, 35, 37, 38]

    in_progress = _full_grid(_Probe(False), 38, 5, None)
    assert 38 not in in_progress and in_progress == [5, 10, 15, 20, 25, 30, 35, 37]

    # A caller explicitly capping below total_rounds never gets the final point,
    # even on a complete season (e.g. a deliberate historical-backtest slice).
    capped = _full_grid(_Probe(True), 38, 5, 25)
    assert capped == [5, 10, 15, 20, 25]


def test_snapshot_helpers_snap_and_timeline():
    from mundialytics.statistical_core.competition import forecast_cache as fc
    bundle = {
        "meta": {"schema": fc.BUNDLE_SCHEMA, "matchdays": [10, 20]},
        "snapshots": {"10": _snap(10, ["a", "b"], [0.5, 0.5]), "20": _snap(20, ["a", "b"], [0.7, 0.3])},
    }
    assert fc.available_matchdays(bundle) == [10, 20]
    used, snap = fc.snapshot_for(bundle, 13)   # nearest to 13 is 10
    assert used == 10 and snap["matchday"] == 10
    used2, _ = fc.snapshot_for(bundle, 17)     # nearest to 17 is 20
    assert used2 == 20
    tl = fc.build_timeline(bundle)
    assert len(tl) == 4 and {r["matchday"] for r in tl} == {10, 20}


@pytest.mark.skipif(not FOUNDATION.exists(), reason="foundation CSV not present")
def test_cutoff_split_and_no_result_leak():
    from mundialytics.statistical_core.competition import load_league_state_from_foundation
    found = pd.read_csv(FOUNDATION, low_memory=False)
    st = load_league_state_from_foundation("LaLiga", "2024-2025", cutoff_matchday=19, foundation=found)
    # played + remaining must equal the full season, and remaining must not leak results.
    assert st.n_played + st.n_remaining == 380
    assert "home_goals" not in st.remaining.columns
    assert st.n_played > 0 and st.n_remaining > 0
    # Final-table sanity: complete season -> Barcelona champion.
    full = load_league_state_from_foundation("LaLiga", "2024-2025", foundation=found)
    assert "barcelona" in full.standings.iloc[0]["team"].lower()
