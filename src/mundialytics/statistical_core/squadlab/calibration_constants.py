"""Squad strength -> AttackDefenseModel-scale calibration constants.

These map PlayerStrengthModel.team_strength()'s attack_index/defense_index
(0-100) onto AttackDefenseModel's log-scale attack/defense parameters, and
onto real per-market event rates (shots/sot/corners/fouls/yellow_cards), so
a fictional squad can play real opponents through AttackDefenseModel's own
formula (lh = exp(mu + ha + attack[home] - defense[away])).

WHY THIS IS A RANGE-BASED MAP, NOT A PER-CLUB REGRESSION:

The original plan was to reconstruct each real club's actual best-11 from
player_profiles_with_positions.csv and regress team_strength() output
directly against that same club's real AttackDefenseModel.team_params()
row. Two attempts at this failed:

  1. Matching against the LIVE model (fit on foundation_big5_multi_season.csv,
     seasons 2021/22-2025/26): R^2 ~ 0.003-0.08 across every filtering
     attempt (minimum matches per player, minimum AD sample per team).
  2. Matching against a temporary model fit on StatsBomb's older,
     denser-coverage Big5 seasons (La Liga 2004-2021, Premier League/Serie A
     2003-2016, etc.): still R^2 ~ 0.03-0.06, occasionally 0.3 with very
     small samples (n<20 teams).

Root cause: player_profiles_with_positions.csv has no season column — it
aggregates a player's ENTIRE career into one row tagged to a single team.
"Reconstruct Lyon's current best-11" can silently mix a 2016 Lyon player
with a 2022 one under the same team tag, with no way to tell. This is the
same gap already documented as deferred in [[project_player_rating_data]]
memory (career-aggregated data, no season/match split) — it just turned out
to be a hard blocker here, not just a nice-to-have for season awards.

Given that, this module does NOT claim per-club correspondence. Instead it
maps the OBSERVED RANGE of achievable squad strength (the weakest and
strongest XIs draftable from the current player pool) onto the OBSERVED
RANGE of real AttackDefenseModel parameters (5th-95th percentile across all
130 fitted teams), via a simple two-point linear map. This guarantees:
  - correct ORDERING (a stronger squad always gets a higher attack_param)
  - a REALISTIC SCALE (the spread matches what real teams actually span)
but does NOT guarantee that a specific attack_index value corresponds to
any specific real club's true strength — that level of precision needs
season-tagged player data this project doesn't have yet.

Recomputed by: scripts/fit_squad_lambda_calibration.py
"""
from __future__ import annotations

# Two-point linear map: GOAL_ATTACK_SLOPE * attack_index + GOAL_ATTACK_INTERCEPT
# Anchored at the weakest draftable XI (attack_index ~45.4 -> AD 5th pct)
# and the strongest draftable all-star XI (attack_index ~71.9 -> AD 95th pct).
GOAL_ATTACK_SLOPE: float = 0.062548
GOAL_ATTACK_INTERCEPT: float = -3.540336
GOAL_DEFENSE_SLOPE: float = 0.043257
GOAL_DEFENSE_INTERCEPT: float = -2.531200

# Safety clip: never let an extreme squad extrapolate past the real
# attack/defense parameter range AttackDefenseModel actually produced
# across its 130 fitted teams.
ATTACK_PARAM_CLIP: tuple[float, float] = (-1.1593, 1.3526)
DEFENSE_PARAM_CLIP: tuple[float, float] = (-0.5581, 0.6256)

# Per-market event-rate calibration: (basis_index, slope, intercept, clip_min, clip_max).
# basis_index is "attack" for attack-driven markets (shots/sot/corners) or
# "defense" for discipline markets (fouls/yellow_cards — more aggressive
# defending tends to mean more fouls, not fewer).
EVENT_CALIBRATION: dict[str, dict[str, float | str]] = {
    "shots":        {"basis": "attack",  "slope": 0.2283, "intercept": -0.6143, "clip_min": 8.39, "clip_max": 18.98},
    "sot":          {"basis": "attack",  "slope": 0.0959, "intercept": -1.1423, "clip_min": 2.34, "clip_max": 7.89},
    "corners":      {"basis": "attack",  "slope": 0.0872, "intercept": -0.3299, "clip_min": 2.97, "clip_max": 7.05},
    "fouls":        {"basis": "defense", "slope": 0.2432, "intercept": -2.2654, "clip_min": 8.46, "clip_max": 15.71},
    "yellow_cards": {"basis": "defense", "slope": 0.0611, "intercept": -1.5298, "clip_min": 1.37, "clip_max": 3.16},
}
