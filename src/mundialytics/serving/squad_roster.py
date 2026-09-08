"""
Turning a chosen eleven into rosters the season simulator can play with.

Shared by the Streamlit app and the HTTP API. Both need the same two things and
both would get them subtly wrong on their own — in particular the cloning rule,
which is load-bearing rather than cosmetic.
"""
from __future__ import annotations

import dataclasses

from mundialytics.identity.current_squads import NameIndex
from mundialytics.statistical_core.schemas import canonical_name

# Marks a drafted copy. A signed player and the original at his old club are two
# identities with the same stats: without the mark, one name sits in two rosters
# and his goals are counted twice in the league's scorer race.
SQUAD_CLONE_MARK = " ✦"

XI_SLOTS = {"Goalkeeper": 1, "Defender": 4, "Midfielder": 3, "Forward": 3}


def clone_for_squad(profile, squad_team_name: str):
    """Same-stats copy of a player under the user's club, with its own identity."""
    if str(profile.player).endswith(SQUAD_CLONE_MARK):
        return profile
    return dataclasses.replace(
        profile,
        player=f"{profile.player}{SQUAD_CLONE_MARK}",
        team=squad_team_name,
    )


def strip_clone(name: str) -> str:
    return str(name).replace(SQUAD_CLONE_MARK, "")


def is_clone(name: str) -> bool:
    return str(name).endswith(SQUAD_CLONE_MARK)


def _pools_from_current_squads(model, squads) -> dict[str, list]:
    """Club -> the profiles of the players actually at that club right now.

    The profiles carry the club a player is REMEMBERED for, which is not the
    club he plays for: grouping by it fielded Xavi, Puyol and Messi for
    Barcelona and left every club promoted since the profile data was built
    with nobody at all. The squad table answers the membership question, the
    profiles answer the ability question, and they meet on the player's name.

    A squad member with no profile is dropped rather than invented, so a club
    whose players have never been measured falls below the eleven-a-side bar
    and keeps scoreline-only detail — the same outcome as before, reached
    honestly.
    """
    if squads is None or getattr(squads, "empty", True):
        return {}
    idx = NameIndex()
    for p in model.profiles_.values():
        # better-sampled profile wins a name collision: a career regular is a
        # likelier match for a squad member than a one-cameo namesake
        idx.add(p.player, p, rank=float(getattr(p, "matches", 0) or 0))

    pools: dict[str, list] = {}
    if "pos_group" not in squads.columns:
        squads = squads.assign(pos_group="Unknown")
    for team, grp in squads.groupby("team"):
        found = []
        for who, pos in zip(grp["player"], grp["pos_group"]):
            prof, kind = idx.lookup_detail(who)
            if prof is None:
                continue
            # A short-key or containment match dropped name parts to get here,
            # so it has to agree about the position too. Without this Alavés'
            # goalkeeper Adrián Rodríguez becomes the retired forward Adrián
            # López Rodríguez — and then keeps goal.
            if kind != "full" and pos in XI_SLOTS and prof.position != pos:
                continue
            # slot him where he plays NOW; the profile's own position is a
            # career summary and goes stale with the rest of it
            found.append((prof, pos if pos in XI_SLOTS else prof.position))
        if found:
            pools[canonical_name(team)] = found
    return pools


def real_team_rosters(model, real_teams: list[str], current_squads=None) -> dict[str, list]:
    """A best XI per real club, so every match produces player-level events.

    Real clubs keep their players even when the user has drafted one — the
    drafted version is a separate identity. A club with too little coverage is
    skipped rather than faked; its matches simply stay scoreline-only.

    `current_squads` (see mundialytics.identity.current_squads) picks today's
    squad. Clubs it does not cover — historical sides, European opponents
    outside the big five — fall back to the career-profile grouping, which is
    the right answer for them: a 2011 Barcelona XI *should* be Xavi and Messi.
    """
    from mundialytics.identity.current_squads import load_current_squads

    squads = load_current_squads() if current_squads is None else current_squads
    current = _pools_from_current_squads(model, squads)

    by_team: dict[str, list] = {}
    for p in model.profiles_.values():
        by_team.setdefault(canonical_name(p.team), []).append(p)

    out: dict[str, list] = {}
    for team in real_teams:
        key = canonical_name(team)
        pool = current.get(key)
        if pool is None:
            pool = [(p, p.position) for p in by_team.get(key, [])]
        if not pool:
            continue
        xi: list = []
        for pos, n in XI_SLOTS.items():
            xi += sorted(
                (p for p, at in pool if at == pos), key=lambda p: -p.overall
            )[:n]
        if len(xi) >= 8:  # enough to attribute events meaningfully
            out[team] = xi
    return out
