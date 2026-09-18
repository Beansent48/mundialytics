"""Real players for the 35 clubs you are NOT managing.

WHY THIS EXISTS. A Champions run could name only the drafted eleven. Every goal
by every other club in the tournament was anonymous, and the top-scorer table at
the end listed your own players and nobody else, because the squad table stopped
at the big five and two thirds of the field play outside it. With the table now
covering all 108 participants, the rest of the field can score under its own
names.

WHAT IT DOES NOT DO. It does not invent ratings. Clubs outside the big five have
never been measured here -- no xG, no duel data, nothing the card builder uses --
so the only honest signal is what the man has actually done this season: goals
and assists per appearance, off ESPN. That is thin for a club six matches into a
Conference League campaign and empty for one yet to play, so the rate is pulled
towards a positional prior rather than trusted raw. A teenager with one
appearance and one goal must not end the tournament as top scorer.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

# Two clubs no amount of normalising will join up. ClubElo's "Atletico" matches
# both Madrid and Bilbao, and the squad table spells the club the way the
# football-data feed does ("ath madrid"), which shares no prefix with it;
# ClubElo's "oe" transliteration will not meet "bodo glimt" either.
FIELD_TO_SQUAD = {
    "Atletico": "ath madrid",
    "Bodoe Glimt": "bodo glimt",
}

# How much a man's own record counts against the prior for his position. Six
# appearances is roughly when a rate starts to mean something; below it the
# prior dominates, which is what stops a one-cap, one-goal youngster from
# topping the scoring charts.
SHRINK_APPS = 6.0

# Goals and assists per appearance by position, the level the shrink pulls
# towards. Read off the same table, so they move with the data rather than
# being a constant somebody once typed.
FALLBACK_GOAL_RATE = {"Forward": 0.28, "Midfielder": 0.10,
                      "Defender": 0.04, "Goalkeeper": 0.001}
FALLBACK_ASSIST_RATE = {"Forward": 0.12, "Midfielder": 0.11,
                        "Defender": 0.05, "Goalkeeper": 0.002}

# The strength scale attribute_goals was fitted on: it reads
# max(strength - 30, 1), so this range keeps a rival squad in the same
# arithmetic as a drafted one instead of a parallel one.
STRENGTH_LO, STRENGTH_HI = 34.0, 88.0

# How many of a club's players can realistically score in one match. The whole
# registered squad includes three keepers and a youth bench; letting all 32 into
# the draw quietly flattens it towards the reserves.
SQUAD_DEPTH = 18


@dataclass(frozen=True)
class RivalPlayer:
    """The shape attribute_goals reads: a name, a position and two strengths."""

    player: str
    position: str
    offensive_strength: float
    creation_strength: float


def _strength(rate: float, ceiling: float) -> float:
    """A per-appearance rate placed on the profile scale, not rescaled per club.

    Anchoring on a competition-wide ceiling rather than on each club's own best
    is what keeps Bayern's forwards stronger than Sabah's: a club-relative
    scaling would hand every side an equally dangerous striker.
    """
    if ceiling <= 0:
        return STRENGTH_LO
    share = min(max(rate / ceiling, 0.0), 1.0) ** 0.5
    return STRENGTH_LO + share * (STRENGTH_HI - STRENGTH_LO)


def _profiles(grp: pd.DataFrame) -> list[RivalPlayer]:
    apps = grp["apps"].fillna(0) + grp["uefa_apps"].fillna(0)
    goals = grp["goals"].fillna(0) + grp["uefa_goals"].fillna(0)
    assists = grp["assists"].fillna(0) + grp["uefa_assists"].fillna(0)
    pos = grp["pos_group"].fillna("Midfielder")

    gp = pos.map(FALLBACK_GOAL_RATE).fillna(0.08)
    ap = pos.map(FALLBACK_ASSIST_RATE).fillna(0.08)
    g_rate = (goals + gp * SHRINK_APPS) / (apps + SHRINK_APPS)
    a_rate = (assists + ap * SHRINK_APPS) / (apps + SHRINK_APPS)

    out = pd.DataFrame({
        "player": grp["player"].astype(str), "position": pos.astype(str),
        "apps": apps, "g": g_rate, "a": a_rate,
    })
    # Keepers are not in the draw at all. The engine's attack weight for them is
    # 0.0667 rather than zero, which over a whole tournament is enough to have
    # one or two goalkeepers on the scoresheet -- about twenty times life. There
    # is no model here of the corner a keeper goes up for, so the honest number
    # is none.
    out = out[out["position"] != "Goalkeeper"]
    # deepest squads first, so the draw is over players who actually feature
    out = out.sort_values("apps", ascending=False).head(SQUAD_DEPTH)
    return [
        RivalPlayer(player=r.player, position=r.position,
                    offensive_strength=_strength(r.g, 0.80),
                    creation_strength=_strength(r.a, 0.45))
        for r in out.itertuples()
    ]


@lru_cache(maxsize=4)
def _squads_by_team(fingerprint: str = "") -> dict[str, list[RivalPlayer]]:
    from mundialytics.identity.current_squads import load_current_squads

    cs = load_current_squads()
    if cs is None or cs.empty:
        return {}
    for col in ("uefa_apps", "uefa_goals", "uefa_assists"):
        if col not in cs.columns:
            cs[col] = 0.0
    return {str(team): _profiles(grp) for team, grp in cs.groupby("team")}


def load_rival_squads(field_teams) -> dict[str, list[RivalPlayer]]:
    """Field club name (ClubElo's spelling) -> the players who can score for it.

    A club that cannot be resolved is simply absent, and its goals stay
    anonymous. That is the honest failure: inventing a name for a club we could
    not identify would be worse than leaving the scorer blank.
    """
    from mundialytics.statistical_core.competition.european import make_resolver

    by_team = _squads_by_team()
    if not by_team:
        return {}
    resolve = make_resolver(sorted(by_team))

    out: dict[str, list[RivalPlayer]] = {}
    for team in field_teams:
        key = FIELD_TO_SQUAD.get(str(team)) or resolve(str(team))
        if key and by_team.get(key):
            out[str(team)] = by_team[key]
    return out
