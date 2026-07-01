"""Per-match player events (goal/assist/card attribution) and Sofascore-style
0-10 match ratings for a SquadLab squad.

No existing precedent in the codebase — PlayerStrengthModel only produces a
static season-aggregate 0-100 `overall`, never a per-match rating. Only
computed for squad players: real historical-team players aren't tracked
(out of scope for V1, see plan).

Heuristic, not ML-fit: there's no ground-truth match-rating dataset to
calibrate against, same epistemic status as the squad event-lambda
calibration constants. Calibrated only by "does the distribution look sane".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mundialytics.statistical_core.player_strength import (
    POSITION_ATTACK_WEIGHT,
    POSITION_DEFENSE_WEIGHT,
    PlayerStrengthProfile,
)

# Solo-goal rate: a goal has no assist with this fixed probability,
# consistent with real-world unassisted-goal rates.
P_UNASSISTED = 0.20

# Per-goal rating bonus, diminishing after the 3rd in a single match.
GOAL_BONUS_SCHEDULE = [0.9, 0.7, 0.5, 0.3]
GOAL_BONUS_TAIL = 0.2

ASSIST_BONUS = 0.5
YELLOW_CARD_PENALTY = 0.25
CLEAN_SHEET_BONUS = {"Goalkeeper": 0.5, "Defender": 0.35, "Midfielder": 0.15, "Forward": 0.05}
CONCESSION_PENALTY_PER_GOAL = 0.15
CONCESSION_PENALTY_CAP = 1.0
RATING_BASE = 6.5
RATING_MIN, RATING_MAX = 2.0, 10.0

# Flat, position-tilted card-attribution weights — there's no per-player
# card-rate field on PlayerStrengthProfile beyond what's already baked into
# defensive_strength (which is wrong-signed for "who gets carded": more
# defensive doesn't mean more cards).
CARD_WEIGHT = {"Defender": 1.0, "Midfielder": 1.0, "Forward": 0.5, "Goalkeeper": 0.1}


@dataclass
class PlayerMatchEvent:
    player: str
    goals: int = 0
    assists: int = 0
    yellow_cards: int = 0
    rating: float = RATING_BASE


def _scorer_weights(squad: list[PlayerStrengthProfile]) -> np.ndarray:
    return np.array([
        POSITION_ATTACK_WEIGHT.get(p.position, 0.45) * max(p.offensive_strength - 30.0, 1.0)
        for p in squad
    ])


def _assist_weights(squad: list[PlayerStrengthProfile]) -> np.ndarray:
    return np.array([
        (0.5 * POSITION_ATTACK_WEIGHT.get(p.position, 0.45)
         + 0.5 * POSITION_DEFENSE_WEIGHT.get(p.position, 0.40))
        * max(p.offensive_strength - 30.0, 1.0)
        for p in squad
    ])


def attribute_goals(
    squad: list[PlayerStrengthProfile], n_goals: int, rng: np.random.Generator
) -> list[tuple[str, str | None]]:
    """Returns [(scorer, assister_or_None), ...] for n_goals events.

    Own goals are explicitly out of scope for V1 — no data signal grounds
    who concedes one, so every goal is attributed to the scoring squad.
    """
    if n_goals <= 0 or not squad:
        return []

    names = [p.player for p in squad]
    scorer_w = _scorer_weights(squad)
    scorer_p = scorer_w / scorer_w.sum() if scorer_w.sum() > 0 else np.full(len(squad), 1.0 / len(squad))

    events: list[tuple[str, str | None]] = []
    for _ in range(n_goals):
        scorer_idx = rng.choice(len(squad), p=scorer_p)
        scorer = names[scorer_idx]

        if rng.random() < P_UNASSISTED or len(squad) < 2:
            events.append((scorer, None))
            continue

        assist_w = _assist_weights(squad).astype(float)
        assist_w[scorer_idx] = 0.0
        total = assist_w.sum()
        if total <= 0:
            events.append((scorer, None))
            continue
        assist_p = assist_w / total
        assister_idx = rng.choice(len(squad), p=assist_p)
        events.append((scorer, names[assister_idx]))

    return events


def attribute_cards(
    squad: list[PlayerStrengthProfile], n_cards: int, rng: np.random.Generator
) -> list[str]:
    """Returns a list of player names, one per card event (a player can
    appear more than once — second-yellow/red is out of V1 scope, season
    tallies just sum however many a player picks up)."""
    if n_cards <= 0 or not squad:
        return []

    names = [p.player for p in squad]
    weights = np.array([CARD_WEIGHT.get(p.position, 0.5) for p in squad])
    probs = weights / weights.sum() if weights.sum() > 0 else np.full(len(squad), 1.0 / len(squad))
    idxs = rng.choice(len(squad), size=n_cards, p=probs, replace=True)
    return [names[i] for i in idxs]


def _goal_bonus(n: int) -> float:
    total = 0.0
    for i in range(n):
        total += GOAL_BONUS_SCHEDULE[i] if i < len(GOAL_BONUS_SCHEDULE) else GOAL_BONUS_TAIL
    return total


def compute_match_ratings(
    squad: list[PlayerStrengthProfile],
    goal_events: list[tuple[str, str | None]],
    card_players: list[str],
    goals_conceded: int,
    rng: np.random.Generator,
) -> dict[str, PlayerMatchEvent]:
    """Every squad player gets an entry, even with 0 goals/assists/cards."""
    goals_by_player: dict[str, int] = {}
    assists_by_player: dict[str, int] = {}
    for scorer, assister in goal_events:
        goals_by_player[scorer] = goals_by_player.get(scorer, 0) + 1
        if assister:
            assists_by_player[assister] = assists_by_player.get(assister, 0) + 1

    cards_by_player: dict[str, int] = {}
    for player in card_players:
        cards_by_player[player] = cards_by_player.get(player, 0) + 1

    clean_sheet = goals_conceded == 0

    out: dict[str, PlayerMatchEvent] = {}
    for p in squad:
        goals = goals_by_player.get(p.player, 0)
        assists = assists_by_player.get(p.player, 0)
        cards = cards_by_player.get(p.player, 0)

        rating = RATING_BASE
        rating += _goal_bonus(goals)
        rating += ASSIST_BONUS * assists
        if clean_sheet:
            rating += CLEAN_SHEET_BONUS.get(p.position, 0.0)
        rating -= YELLOW_CARD_PENALTY * cards
        concession_penalty = CONCESSION_PENALTY_PER_GOAL * goals_conceded * POSITION_DEFENSE_WEIGHT.get(p.position, 0.40)
        rating -= min(concession_penalty, CONCESSION_PENALTY_CAP)

        sigma = max(0.45 - 0.0015 * p.overall, 0.15)
        rating += float(rng.normal(0.0, sigma))

        rating = float(np.clip(rating, RATING_MIN, RATING_MAX))
        out[p.player] = PlayerMatchEvent(
            player=p.player, goals=goals, assists=assists, yellow_cards=cards, rating=rating,
        )
    return out
