"""Squad strength -> AttackDefenseModel-scale calibration constants.

These map PlayerStrengthModel.team_strength()'s attack_index/defense_index
(0-100) onto AttackDefenseModel's log-scale attack/defense parameters, and
onto real per-market event rates (shots/sot/corners/fouls/yellow_cards), so
a fictional squad can play real opponents through AttackDefenseModel's own
formula (lh = exp(mu + ha + attack[home] - defense[away])).

STATUS AS OF 2026-07-01 (second pass): BOTH ATTACK AND DEFENSE ARE NOW
PRECISE PER-CLUB REGRESSIONS — read on for how defense got fixed.

The original plan was to reconstruct each real club's actual best-11 and
regress team_strength() output directly against that same club's real
AttackDefenseModel.team_params() row. Two attempts at this failed against
CAREER-aggregated player data (player_profiles_with_positions.csv has no
season column, so "reconstruct Lyon's current best-11" silently mixed
players from incompatible eras — R^2 ~ 0.003-0.08 no matter how it was
filtered; see git history / [[project_player_rating_data]] for the full
account of both failed attempts).

A third attempt, once data/processed/player_profiles_by_season.csv existed
(real per-season rosters, built via scripts/build_player_profiles_by_season.py),
regressed season-scoped rosters against an AttackDefenseModel fit on ONLY
that season's matches (scripts/fit_squad_lambda_calibration_season_scoped.py):
  - attack_idx -> attack_param: R^2 = 0.643 (99 matched team-seasons) — a
    real, precise per-club fit.
  - defense_idx -> defense_param: R^2 = 0.012 — still noise, even with
    season-scoped rosters AND after fixing goalkeepers to use the real
    save%-based GK score. Confirmed hypothesis (2026-07-01, dedicated
    investigation): tackles_per_match/pressures_per_match are workrate
    (VOLUME) stats that run BACKWARDS for team quality — a 99-team-season
    correlation check showed both stats NEGATIVELY correlated with real
    attack_param AND defense_param alike (-0.24 to -0.60), while passing
    volume/completion (a possession-dominance proxy) correlated POSITIVELY
    with both (+0.5 to +0.88). A team/player under constant defensive
    pressure racks up more tackles/presses out of necessity; a dominant one
    barely needs to. Same confound found at individual level: legendary
    center-backs (Ramos, Piqué, Van Dijk, Puyol) ranked in the hundreds out
    of ~2200 defenders under the old formula.

FIX (2026-07-01): extended the StatsBomb adapter
(src/mundialytics/data/adapters/statsbomb.py) to extract QUALITY signals
instead of volume — duel win/loss outcome, "Dribbled Past" (opponent beat
this player), Clearance, Block — and rebuilt player_strength.py's defensive
axis around duel_win_rate as the dominant stat (confirmed the ONLY defensive
metric here positively correlated with real quality, +0.46 to +0.55) plus a
new "creation" axis (key_passes_per_match/pass_completion) for chance
creation through passing, which fixed a related problem (elite deep-lying
playmakers like Xavi/Modric/Kroos ranking below destroyer-type midfielders).
Clearances/blocks turned out to still be volume-confounded (-0.46/-0.56, same
flaw as tackles/pressures, just measured deeper in the defensive third) so
they're kept at low weight rather than dropped outright. See
player_strength.py's DEFENSIVE_STATS/CREATION_STATS docstrings for the full
account, and [[project_player_rating_data]] memory.

Re-running scripts/fit_squad_lambda_calibration_season_scoped.py against the
rebuilt formula: attack_idx -> attack_param R^2 = 0.681 (up from 0.643),
defense_idx -> defense_param R^2 = 0.354 (up from 0.012) — both now precise,
human-reviewed per-club fits. GOAL_ATTACK_SLOPE/INTERCEPT and
GOAL_DEFENSE_SLOPE/INTERCEPT below are this fit's coefficients. Defense is
still a weaker fit than attack (0.354 vs 0.681) — the available data only
supports a modest defensive-quality signal (duel win rate has real but
limited separating power vs. goals-per-match for attack), not a data bug to
chase further without new stats.

The range-based fallback logic still exists in the codebase for markets
where no precise regression was attempted: map the OBSERVED RANGE of
achievable squad strength (weakest/strongest draftable XI) onto the OBSERVED
RANGE of real AttackDefenseModel parameters/event rates via a two-point
linear map — correct ordering and a realistic scale, not per-club precision.
This still governs EVENT_CALIBRATION (shots/sot/corners/fouls/yellow_cards)
below, re-run 2026-07-01 to reflect the rebuilt formula's shifted
attack_index/defense_index achievable range (defense_index in particular
compressed from ~72 max to ~63 max, since duel-win-rate has less dynamic
range than raw tackle/pressure volume did).

Recomputed by: scripts/fit_squad_lambda_calibration.py (range-based,
EVENT_CALIBRATION + param clips) and
scripts/fit_squad_lambda_calibration_season_scoped.py (precise attack+defense fit).
"""
from __future__ import annotations

# GOAL_ATTACK_SLOPE/INTERCEPT, GOAL_DEFENSE_SLOPE/INTERCEPT: precise per-club
# regressions from scripts/fit_squad_lambda_calibration_season_scoped.py —
# attack_idx/defense_idx of a season-reconstructed real roster -> that team's
# real AttackDefenseModel attack_param/defense_param for that same season, 99
# matched team-seasons. R^2 = 0.681 (attack) / 0.354 (defense).
GOAL_ATTACK_SLOPE: float = 0.097868
GOAL_ATTACK_INTERCEPT: float = -5.347848
GOAL_DEFENSE_SLOPE: float = 0.159608
GOAL_DEFENSE_INTERCEPT: float = -8.792863

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
    "shots":        {"basis": "attack",  "slope": 0.2372, "intercept": -1.2817, "clip_min": 8.39, "clip_max": 18.98},
    "sot":          {"basis": "attack",  "slope": 0.0996, "intercept": -1.4226, "clip_min": 2.34, "clip_max": 7.89},
    "corners":      {"basis": "attack",  "slope": 0.0906, "intercept": -0.5847, "clip_min": 2.97, "clip_max": 7.05},
    "fouls":        {"basis": "defense", "slope": 0.2759, "intercept": -3.3737, "clip_min": 8.46, "clip_max": 15.71},
    "yellow_cards": {"basis": "defense", "slope": 0.0694, "intercept": -1.8084, "clip_min": 1.37, "clip_max": 3.16},
}
