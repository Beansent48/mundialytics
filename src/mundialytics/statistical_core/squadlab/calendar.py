"""Round-robin fixture calendar generation.

The real match-results dataset (data/processed/foundation_big5_multi_season.csv)
contains only completed historical matches — zero future fixtures — so a
season's calendar has to be generated, not read off existing data. This is a
standard double round-robin (circle method): every pair of teams plays each
other twice, once at each venue.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fixture:
    matchday: int
    home: str
    away: str


_BYE = "__BYE__"


def generate_double_round_robin(teams: list[str], shuffle_seed: int | None = None) -> list[Fixture]:
    """Standard circle-method double round-robin.

    Every team plays every other team twice (once home, once away).
    For n teams this produces 2*(n-1) matchdays with n//2 fixtures each
    (n odd: one team has a bye each matchday, dropped from the output).

    shuffle_seed, if given, randomises initial team order (so the same
    team list doesn't always produce the same fixture pattern) while
    keeping the round-robin guarantees intact.
    """
    if len(teams) < 2:
        return []

    names = list(teams)
    if shuffle_seed is not None:
        import random
        random.Random(shuffle_seed).shuffle(names)

    odd = len(names) % 2 == 1
    if odd:
        names = names + [_BYE]
    n = len(names)
    half = n // 2

    fixtures: list[Fixture] = []
    rotation = names[:]
    for round_idx in range(n - 1):
        for i in range(half):
            home, away = rotation[i], rotation[n - 1 - i]
            if round_idx % 2 == 1:
                home, away = away, home
            if home != _BYE and away != _BYE:
                fixtures.append(Fixture(matchday=round_idx + 1, home=home, away=away))
        # Rotate all but the first element
        rotation = [rotation[0]] + [rotation[-1]] + rotation[1:-1]

    first_leg_matchdays = n - 1
    second_leg = [
        Fixture(matchday=f.matchday + first_leg_matchdays, home=f.away, away=f.home)
        for f in fixtures
    ]
    return fixtures + second_leg
