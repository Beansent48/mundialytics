"""SquadLab's live match and the competition around it, pinned where they rot.

The live engine replaced a scoreline decorated after the fact. Its whole claim
is that every rule it plays by is measured, so these tests hold it to the
measurements: the team model's goals survive the minute-by-minute play, the
rarer events happen about as often as they do in real matches, and the rules
around a match (bans, injuries, the bench, the shoot-out) do what UEFA's do.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mundialytics.statistical_core.squadlab import match_engine as E
from mundialytics.statistical_core.squadlab.match_engine import MatchPlayer, Side, play_match

# The measured rules live in data/processed (not in git); CI plays by a trimmed
# copy of them, so the engine is exercised on a clean checkout too.
FIXTURE = Path(__file__).parent / "fixtures" / "squadlab_match_dynamics_small.json"
DYN = E.load_dynamics(str(E.DYNAMICS if E.DYNAMICS.exists() else FIXTURE))

SHAPE = ["Goalkeeper"] + ["Defender"] * 4 + ["Midfielder"] * 3 + ["Forward"] * 3
BENCH = ["Goalkeeper", "Defender", "Defender", "Midfielder", "Midfielder", "Forward", "Forward"]


def _eleven(tag: str, ovr: float = 75.0) -> list[MatchPlayer]:
    return [MatchPlayer(f"{tag}{i}", f"{tag}{i}", p, ovr,
                        g=0.45 if p == "Forward" else 0.12 if p == "Midfielder" else 0.05,
                        a=0.15, yc=0.2, taker=1.0 if i == 9 else 0.0)
            for i, p in enumerate(SHAPE)]


def _bench(tag: str, ovr: float = 72.0) -> list[MatchPlayer]:
    return [MatchPlayer(f"{tag}b{i}", f"{tag}b{i}", p, ovr) for i, p in enumerate(BENCH)]


def _many(n: int, lam=(1.6, 1.1), seed: int = 3, **kw):
    rng = np.random.default_rng(seed)
    return [play_match(Side("H", _eleven("h"), _bench("h"), lam=lam[0], yellows_lam=2.0),
                       Side("A", _eleven("a"), _bench("a"), lam=lam[1], yellows_lam=2.2),
                       rng, dyn=DYN, **kw) for _ in range(n)]


def test_the_team_models_goals_survive_the_live_match():
    """Game state and red cards move goals around a match; they must not move
    the total off the team model's expectation."""
    logs = _many(3000)
    h = np.mean([m.home_goals for m in logs])
    a = np.mean([m.away_goals for m in logs])
    assert abs(h - 1.6) < 0.08, h
    assert abs(a - 1.1) < 0.07, a


def test_rare_events_happen_about_as_often_as_in_real_matches():
    dyn = DYN
    logs = _many(2500, lam=(1.35, 1.35))
    per = 2 * len(logs)

    def count(*kinds):
        return sum(1 for m in logs for e in m.events if e.kind in kinds) / per

    pens = count("pen_goal", "pen_saved", "pen_missed")
    assert abs(pens - dyn.pen["awarded_per_team_match"]) < 0.03, pens
    reds = count("red", "second_yellow")
    measured = (dyn.raw["red_cards"]["direct_per_team_match"]
                + dyn.raw["red_cards"]["second_yellow_per_team_match"])
    assert abs(reds - measured) < 0.03, (reds, measured)
    og = count("own_goal") / max(count("goal", "pen_goal", "own_goal"), 1e-9)
    assert 0.01 < og < 0.05, og
    subs = count("sub")
    assert 3.5 < subs <= 5.0, subs


def test_every_goal_has_a_name_and_the_log_adds_up():
    for m in _many(300):
        for side, goals in (("home", m.home_goals), ("away", m.away_goals)):
            ev = [e for e in m.events if e.side == side and e.kind in ("goal", "pen_goal", "own_goal")]
            assert len(ev) == goals
            assert all(e.player for e in ev)
            scored = sum(line.goals for line in m.lines[side].values())
            own = sum(1 for e in ev if e.kind == "own_goal")
            assert scored + own == goals


def test_a_keeper_never_scores_and_nobody_plays_past_the_whistle():
    for m in _many(300):
        for side in ("home", "away"):
            for line in m.lines[side].values():
                if line.position == "Goalkeeper":
                    assert line.goals == 0
                assert 1 <= line.minutes <= 90


def test_no_side_makes_more_than_five_changes():
    for m in _many(300):
        for side in ("home", "away"):
            subs = [e for e in m.events if e.kind == "sub" and e.side == side]
            assert len(subs) <= E.MAX_SUBS


def test_a_sending_off_costs_the_side_goals():
    """Down to ten, a side scores about half of its rate: the measured factor."""
    dyn = DYN
    assert dyn.red_factor(10, 11) < 0.7 < 1.3 < dyn.red_factor(11, 10)
    assert dyn.red_factor(10, 10) == 1.0


def test_a_level_knockout_goes_to_extra_time_and_a_shootout():
    logs = _many(400, lam=(0.4, 0.4), extra_time_if_level=True, shootout_if_level=True)
    level_after_et = [m for m in logs if m.home_goals == m.away_goals]
    assert level_after_et, "no level match in 400 low-scoring games"
    for m in level_after_et:
        assert m.extra_time and m.shootout is not None
        assert m.shootout[0] != m.shootout[1]
        assert m.winner in ("H", "A")
    assert any(m.extra_time for m in logs)


def test_a_second_leg_is_decided_on_aggregate_not_on_the_night():
    rng = np.random.default_rng(1)
    for _ in range(80):
        m = play_match(Side("H", _eleven("h"), _bench("h"), lam=1.2),
                       Side("A", _eleven("a"), _bench("a"), lam=1.2), rng, dyn=DYN,
                       extra_time_if_level=True, shootout_if_level=True, start_score=(3, 0))
        # three up from the first leg: only a level aggregate after 90 plays on
        at90 = {"home": 0, "away": 0}
        for e in m.events:
            if e.kind in ("goal", "pen_goal", "own_goal") and e.order < 200:
                at90[e.side] += 1
        assert m.extra_time == (at90["home"] + 3 == at90["away"])


def test_a_weak_bench_costs_more_than_a_typical_one():
    """The goal clock already contains ordinary changes; only a drop beyond the
    real clubs' typical gap should cost the side anything."""
    gap = {"Defender": 2.5, "Midfielder": 3.0, "Forward": 2.5, "Goalkeeper": 3.0}

    def mean_goals(bench_ovr):
        rng = np.random.default_rng(9)
        tot = 0
        for _ in range(1500):
            m = play_match(Side("H", _eleven("h", 85), _bench("h", bench_ovr), lam=1.5, elo_per_ovr=40.0),
                           Side("A", _eleven("a"), _bench("a"), lam=1.2), rng, dyn=DYN,
                           typical_gap=gap)
            tot += m.home_goals
        return tot / 1500

    assert mean_goals(50.0) < mean_goals(83.0)


# ── the competition around the match ───────────────────────────────────────────
def _state():
    from mundialytics.statistical_core.squadlab.cards import best_eleven, load_cards
    from mundialytics.statistical_core.squadlab.champions import SquadState

    df = load_cards()
    if df.empty:
        pytest.skip("no card catalogue")
    xi = best_eleven(df, "real madrid")
    rest = df[(df["kind"] == "actual") & (df["club"] == "real madrid")
              & ~df["card_id"].isin([c.card_id for c in xi])]
    from mundialytics.statistical_core.squadlab.cards import cards_from_frame
    bench = cards_from_frame(rest.nlargest(7, "overall"))
    return SquadState(xi, bench, DYN), xi


class _Log:
    def __init__(self, lines):
        self.lines = {"home": lines}


def _line(card, **kw):
    line = E.PlayerLine(card.card_id, card.player, card.position, minutes=90)
    for k, v in kw.items():
        setattr(line, k, v)
    return line


def test_the_third_booking_bans_a_player_for_one_match():
    st, xi = _state()
    p = xi[5]
    rng = np.random.default_rng(0)
    for _ in range(2):
        st.after_match(_Log({p.card_id: _line(p, yellows=1)}), "home", "liga", None, rng)
        assert st.available(p.card_id, None) is None
    st.after_match(_Log({p.card_id: _line(p, yellows=1)}), "home", "liga", None, rng)
    assert st.available(p.card_id, None) == "sancion"
    starters, _, absences = st.lineup(None)
    assert p.card_id not in {c.card_id for c in starters}
    assert any(a.reason == "sancion" for a in absences)
    # the ban is served by missing the next match
    st.after_match(_Log({}), "home", "liga", None, rng)
    assert st.available(p.card_id, None) is None


def test_two_bookings_in_one_match_are_a_red_not_part_of_the_tally():
    st, xi = _state()
    p = xi[6]
    rng = np.random.default_rng(0)
    st.after_match(_Log({p.card_id: _line(p, yellows=2, red=True)}), "home", "liga", None, rng)
    assert st.available(p.card_id, None) == "sancion"
    assert st.yellows[p.card_id] == 0


def test_an_injured_starter_is_covered_from_the_bench_in_his_position():
    st, xi = _state()
    p = next(c for c in xi if c.position == "Defender")
    when = pd.Timestamp("2026-10-01")
    st.injured_until[p.card_id] = when + pd.Timedelta(days=10)
    starters, bench, absences = st.lineup(when)
    assert len(starters) == 11
    assert p.card_id not in {c.card_id for c in starters}
    assert [c.position for c in starters].count("Defender") == 4
    assert any(a.reason == "lesion" for a in absences)
    # back once the days have passed
    starters, _, _ = st.lineup(when + pd.Timedelta(days=11))
    assert p.card_id in {c.card_id for c in starters}


def test_injuries_last_as_long_as_measured():
    dyn = DYN
    rng = np.random.default_rng(4)
    days = [dyn.injury_days(rng) for _ in range(4000)]
    curve = dyn.raw["injury_absence"]["still_out_after_days"]
    # the share still out after a week matches the Kaplan-Meier curve
    at7 = next(s for d, s in curve if d >= 7)
    assert abs(np.mean([d > 7 for d in days]) - at7) < 0.05


def test_the_squad_shape_rewards_defence_and_attack_the_right_way():
    from mundialytics.statistical_core.squadlab.champions import SHAPE_COEF

    assert SHAPE_COEF["att"] > 0          # more attack: more goals scored
    assert SHAPE_COEF["def"] < 0          # more defence: fewer conceded
    assert SHAPE_COEF["gk"] < 0           # a better keeper: fewer conceded


def test_every_card_has_match_rates():
    from mundialytics.statistical_core.squadlab.cards import load_cards
    from mundialytics.statistical_core.squadlab.match_players import CARD_RATES

    if not CARD_RATES.exists():
        pytest.skip("card rates not built")
    cards = load_cards()
    rates = pd.read_csv(CARD_RATES)
    assert set(cards["card_id"]) <= set(rates["card_id"])
    for col in ("g", "a", "yc"):
        assert rates[col].notna().all() and (rates[col] >= 0).all()
    fw = rates.loc[rates["position"] == "Forward", "g"].mean()
    df_ = rates.loc[rates["position"] == "Defender", "g"].mean()
    assert fw > 2 * df_
    assert (rates.loc[rates["position"] == "Goalkeeper", "g"] == 0).all()


# ── la Champions histórica ─────────────────────────────────────────────────────
def _historic():
    from mundialytics.statistical_core.squadlab import historic as H

    if not H.TEAMS_DIR.exists() or not any(H.TEAMS_DIR.glob("*.csv")):
        pytest.skip("ClubElo histories not downloaded")
    return H


def test_every_historic_field_is_thirty_six_sides():
    H = _historic()
    for pool in H.POOLS:
        e = H.entrants(pool, 5)
        assert len(e) == 36, pool
        assert e["label"].is_unique, pool


def test_a_historic_side_is_rated_by_its_own_seasons_elo():
    """Milan 1988/89 is Milan's ClubElo on 1 June 1989, not a player bridge."""
    H = _historic()
    c = H.champions()
    row = c[(c["club"] == "Milan") & (c["year"] == 1989)]
    assert len(row) == 1
    assert row["elo"].iloc[0] == pytest.approx(H.elo_on("Milan", pd.Timestamp("1989-06-01")))
    assert row["label"].iloc[0] == "Milan 1988/89"


def test_the_historic_tournament_is_reproducible_and_complete():
    H = _historic()
    a = H.play("campeones", 11, n_sims=50)
    b = H.play("campeones", 11, n_sims=50)
    assert a.champion == b.champion and a.runner_up == b.runner_up
    assert len(a.table) == 36
    assert a.champion in set(a.entrants["label"])
    assert len(a.rounds["r16"]) == 8 and len(a.rounds["final"]) == 1
