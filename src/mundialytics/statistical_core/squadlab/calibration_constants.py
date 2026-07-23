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
human-reviewed per-club fits. Defense is still a weaker fit than attack
(0.354 vs 0.681) — the available data only supports a modest defensive-
quality signal (duel win rate has real but limited separating power vs.
goals-per-match for attack), not a data bug to chase further without new stats.

THIRD PASS (2026-07-01, same day): the user asked for individual "overall"
star ratings to land ~90-92 (see [[project_player_rating_data]]). Fixing that
required two changes that shifted this calibration again:
  1. A NEW max-weighted "def_score_primary" was added for the individual
     overall rating ONLY (a player's single best defensive facet dominates,
     matching how off_score already lets a pure poacher's goals alone carry
     them) -- but team_strength()/defensive_strength deliberately kept the
     OLD smooth DEF_WEIGHTS average, since the max-weighted version collapsed
     this exact calibration's defense R^2 to 0.002 when tried here (11
     different players' 11 different "best facets" average into a much
     noisier team signal than everyone's duel_win_rate specifically).
  2. The defensive-quality ANCHOR_* curves were recalibrated using ONLY
     established (100+ Big5-match) players' real ceiling as the top
     reference point, rather than all-players percentiles (which let small-
     sample noise set an unreachable top end) -- this changed the absolute
     scores feeding BOTH defensive_strength and def_score_primary, and
     nudged this calibration's defense R^2 from 0.354 to 0.251 (still far
     above the original 0.012, just not quite back to the second-pass peak).
Re-fit result this pass: attack R^2 = 0.676, defense R^2 = 0.251.

Also re-ran against the SEPARATE historical-data-completion work done the
same session: data/processed/foundation_big5_multi_season.csv was extended
from 5 seasons (2021-2026, 8907 matches) to 26 seasons (2000/01-2025/26,
45841 matches, all via football-data.co.uk) -- AttackDefenseModel now fits
219 teams instead of 130, which is why ATTACK_PARAM_CLIP/DEFENSE_PARAM_CLIP
shifted slightly below (independent of the player-rating changes above,
these come purely from the real fitted model's own parameter range).

The range-based fallback logic still exists in the codebase for markets
where no precise regression was attempted: map the OBSERVED RANGE of
achievable squad strength (weakest/strongest draftable XI) onto the OBSERVED
RANGE of real AttackDefenseModel parameters/event rates via a two-point
linear map — correct ordering and a realistic scale, not per-club precision.
This still governs EVENT_CALIBRATION (shots/sot/corners/fouls/yellow_cards)
below, re-run 2026-07-01 against both the extended 26-season match dataset
and the recalibrated player-rating formula.

Recomputed by: scripts/fit_squad_lambda_calibration.py (range-based,
EVENT_CALIBRATION + param clips) and
scripts/fit_squad_lambda_calibration_season_scoped.py (precise attack+defense fit).
"""
from __future__ import annotations

# GOAL_ATTACK_SLOPE/INTERCEPT, GOAL_DEFENSE_SLOPE/INTERCEPT: precise per-club
# attack_idx/defense_idx of a season-reconstructed real roster -> that team's
# real AttackDefenseModel attack_param/defense_param.
#
# VARIANCE-MATCHED (2026-07-23), not least-squares. The end-to-end realism
# validation (scripts/validate_squadlab_realism.py) proved the old LS-regression
# constants COMPRESSED every reconstructed squad toward mid-table: a least-
# squares line with R^2<1 attenuates its output range (predicted std =
# corr * real std), so defense (R^2=0.25 -> corr 0.5) came out at only HALF the
# real spread, and title-winning XIs simulated to ~50 pts instead of ~90.
# These constants instead map the index onto the REAL param distribution's
# spread (same mean+std, ordering preserved): slope = std_param/std_idx.
# Result on the 99-club validation: points-spread 14.5 -> 18.1 (real 18.0),
# correlation 0.72 -> 0.76, MAE 10.7 -> 10.0 pts. The right trade for a
# SIMULATION (realistic spread) over per-club precision. Residual ~6pt error vs
# the engine's 4pt ceiling is irreducible player-data noise (defense signal).
GOAL_ATTACK_SLOPE: float = 0.100588
GOAL_ATTACK_INTERCEPT: float = -5.545813
GOAL_DEFENSE_SLOPE: float = 0.247168
GOAL_DEFENSE_INTERCEPT: float = -14.187625

# Safety clip: never let an extreme squad extrapolate past the real
# attack/defense parameter range AttackDefenseModel actually produced —
# now across 219 teams / 26 seasons (2000/01-2025/26) after the historical
# data-completion pass, up from 130 teams / 5 seasons.
ATTACK_PARAM_CLIP: tuple[float, float] = (-1.1565, 1.3858)
DEFENSE_PARAM_CLIP: tuple[float, float] = (-0.4436, 0.6626)

# Per-market event-rate calibration: (basis_index, slope, intercept, clip_min, clip_max).
# basis_index is "attack" for attack-driven markets (shots/sot/corners) or
# "defense" for discipline markets (fouls/yellow_cards — more aggressive
# defending tends to mean more fouls, not fewer).
EVENT_CALIBRATION: dict[str, dict[str, float | str]] = {
    "shots":        {"basis": "attack",  "slope": 0.2017, "intercept": 0.2651, "clip_min": 8.19, "clip_max": 17.21},
    "sot":          {"basis": "attack",  "slope": 0.1105, "intercept": -1.8139, "clip_min": 2.68, "clip_max": 7.20},
    "corners":      {"basis": "attack",  "slope": 0.0777, "intercept": 0.3454, "clip_min": 3.50, "clip_max": 6.58},
    "fouls":        {"basis": "defense", "slope": 0.4279, "intercept": -9.9959, "clip_min": 9.55, "clip_max": 22.34},
    "yellow_cards": {"basis": "defense", "slope": 0.0666, "intercept": -1.7220, "clip_min": 1.24, "clip_max": 2.98},
}
