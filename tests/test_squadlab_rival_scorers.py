"""The other 35 clubs score under their own names.

A Champions run used to attribute goals only for the drafted eleven: every other
club in the tournament scored anonymously, and the top-scorer table at the end
was your own players and nobody else. The squad table stopped at the big five,
so two thirds of the field had no players to name.

These tests hold the arithmetic (attributed goals must add up to the goals the
table says were scored) and the one rule that would otherwise go unnoticed: a
man you drafted is not also playing for his real club.
"""
from __future__ import annotations

import numpy as np
import pytest

from mundialytics.statistical_core.squadlab.cards import (
    cards_from_frame, load_cards,
)
from mundialytics.statistical_core.squadlab.champions import (
    SQUAD_TEAM_NAME, ChampionsRun, load_field,
)
from mundialytics.statistical_core.squadlab.rival_squads import load_rival_squads

pytestmark = pytest.mark.skipif(load_cards().empty, reason="no card catalogue")


@pytest.fixture(scope="module")
def field():
    return load_field()


@pytest.fixture(scope="module")
def run(field):
    df = load_cards()
    squad = []
    for pos, n in (("Goalkeeper", 1), ("Defender", 4), ("Midfielder", 3),
                   ("Forward", 3)):
        at = df[(df["position"] == pos) & (df["kind"] == "actual")].nlargest(n, "overall")
        squad += [c.to_profile() for c in cards_from_frame(at)]
    r = ChampionsRun(SQUAD_TEAM_NAME, squad, 1750.0, field,
                     rng=np.random.default_rng(5))
    return r, r.play()


def test_every_club_in_the_field_has_players(field) -> None:
    """A club with no squad scores anonymously, which is the bug being fixed."""
    teams = sorted(field["elo"])
    got = load_rival_squads(teams)
    missing = [t for t in teams if t not in got]
    assert not missing, f"no squad for {missing}"


def test_attributed_goals_add_up(run) -> None:
    """The scorers have to reconstruct the scoreline, club by club. This is the
    check that catches a whole class of quiet corruption at once."""
    _, res = run
    scored: dict[str, int] = {}
    named: dict[str, int] = {}
    for m in res.matches:
        for team, goals in ((m.home, m.home_goals), (m.away, m.away_goals)):
            if team == SQUAD_TEAM_NAME:
                continue
            scored[team] = scored.get(team, 0) + int(goals)
        for team, events in (m.rival_events or {}).items():
            named[team] = named.get(team, 0) + len(events)

    for team, total in scored.items():
        assert named.get(team, 0) == total, \
            f"{team}: {named.get(team, 0)} attributed vs {total} scored"


def test_a_drafted_man_does_not_also_play_for_his_old_club(run) -> None:
    """He took a club's slot in the draw; he cannot be scoring against you."""
    from mundialytics.identity.current_squads import short_key

    r, _ = run
    drafted = {short_key(p.player) for p in r.squad}
    for team, players in r.rival_squads.items():
        clash = [p.player for p in players if short_key(p.player) in drafted]
        assert not clash, f"{team} still fields drafted player(s): {clash}"


def test_the_scorer_table_is_the_tournaments(run) -> None:
    """Not just yours: the whole point of giving the field real players."""
    _, res = run
    sc = res.scorers
    assert len(sc), "nobody scored at all"
    assert "equipo" in sc.columns
    teams = set(sc["equipo"])
    assert len(teams - {SQUAD_TEAM_NAME}) >= 5, f"only these clubs scored: {teams}"


def test_forwards_outscore_defenders(run) -> None:
    """The attribution is weighted, not uniform. If this fails the weights are
    not reaching the rival squads and every goal is a coin flip."""
    r, res = run
    pos = {p.player: p.position
           for players in r.rival_squads.values() for p in players}
    by_pos: dict[str, int] = {}
    for m in res.matches:
        for events in (m.rival_events or {}).values():
            for scorer, _ in events:
                p = pos.get(scorer)
                if p:
                    by_pos[p] = by_pos.get(p, 0) + 1
    assert by_pos.get("Forward", 0) > by_pos.get("Defender", 0), by_pos
    assert by_pos.get("Goalkeeper", 0) == 0, "a keeper scored"


def test_a_one_cap_wonder_does_not_top_the_charts() -> None:
    """One appearance and one goal is a rate of 1.0 taken raw, which would make
    a youth-team debutant the most dangerous player in Europe."""
    import pandas as pd

    from mundialytics.statistical_core.squadlab.rival_squads import _profiles

    grp = pd.DataFrame({
        "player": ["Debutant", "Real Striker"],
        "pos_group": ["Forward", "Forward"],
        "apps": [1.0, 30.0], "goals": [1.0, 20.0], "assists": [0.0, 5.0],
        "uefa_apps": [0.0, 0.0], "uefa_goals": [0.0, 0.0], "uefa_assists": [0.0, 0.0],
    })
    out = {p.player: p.offensive_strength for p in _profiles(grp)}
    assert out["Real Striker"] > out["Debutant"]
