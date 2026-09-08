#!/usr/bin/env python3
"""Build the SquadLab CARD CATALOGUE — every card the draft can deal.

WHY A CATALOGUE AND NOT THE RATING MODEL DIRECTLY. SquadLab used to draft
straight out of PlayerStrengthModel.profiles_, which is a career-aggregated
research artefact: it covers whoever StatsBomb's open data covers (59% of the
players actually playing this season), files each player under the club he is
remembered for, and has exactly one version of him. A draft game needs the
opposite of all three — every squad member of every club we simulate, filed
under his real club, in three flavours:

  ACTUAL  the player as he is now. The bulk of the pool.
  PRIME   one standout SEASON of a player, carrying that season's label and
          that season's rating ("SUÁREZ 15/16"). Rarer and stronger.
  ICONO   a retired generational name, no year, the strongest and rarest.

Nothing here modifies the rating model or anything the betting side reads —
this file is written for the game layer and only the game layer uses it.

COVERAGE IS THE POINT. A draft that cannot deal a card for a third of a squad
is broken, so every current squad member gets a card. Where his own history
exists the rating comes from it; where it does not (a promoted club's academy,
a signing from outside the big five) the card carries a PROJECTION from the
level of the club he plays for, how much he plays and what he has produced this
season — R^2 0.18 against known ratings, which is weak and is why such cards are
marked `projected` rather than passed off as measured.

THE LEVEL WAS REBUILT ON 2026-09-06, AND VALIDATED BEFORE SHIPPING.
Target: goals + assists per 90 in 2024/25, predicted from 2023/24 inputs, over
the 644 players present in both. Nothing about the outcome season is used to
build the predictor.

                                   n      R2 vieja -> nueva   Spearman
    todas las posiciones         644      +0.260  +0.316    +0.486 -> +0.524
    donde el término aplica      354      +0.233  +0.304    +0.512 -> +0.578
    donde NO aplica (DF+GK)      290      -0.044  -0.044    +0.186 -> +0.186

The last row is the point: defenders and keepers are untouched, and the target
does not describe their job anyway (R2 -0.044 for the deployed rating itself).

Three routes were tried and REJECTED, recorded here so they are not retried:

  - GOALKEEPER SHOT-STOPPING DOES NOT PERSIST. 23/24 -> 24/25 over the 96
    keepers with 8+ matches in both: R2 -0.101 for PSxG+/- per 90, -0.070 for
    save%. The raw persistence is r +0.06 and +0.17. There is no measurable
    goalkeeping to rate, so no keeper model was built; the fix for keepers was
    a name-matching defect (below), not a new number.
  - "THE LEVEL HE IS TRUSTED WITH" ADDS NOTHING. Target = 24/25 minutes-weighted
    club Elo, over 1,743 big-5 players: club Elo + minutes + age alone reach
    R2 +0.430, and adding every measured performance stat gives +0.401. Worse.
    Per position the lift is negative everywhere. Dropped.
  - THE LEAGUE-RELATIVE ATTACKING Z WINS THE TEST AND IS STILL WRONG. It takes
    R2 from +0.260 to +0.554 — but the target is an attacking outcome, so it
    would rank every striker above every defender by construction. The
    within-position z is the smaller, correctly-shaped signal, and is what
    ships. A test that a bad answer wins is a bad test for that answer.

Run with the project venv:
    .venv/Scripts/python.exe scripts/build_squadlab_cards.py
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.identity.current_squads import (  # noqa: E402
    NameIndex, full_key, load_current_squads, short_key,
)
from mundialytics.identity.display_names import display_name  # noqa: E402

OUT = ROOT / "data/processed/squadlab_cards.csv"
ROLES = ROOT / "data/processed/player_ratings_roles.csv"
SEASONS = ROOT / "data/processed/player_ratings_seasons.csv"
MODERN = ROOT / "data/processed/player_profiles_modern.csv"
CLUBELO = ROOT / "data/processed/clubelo_local.csv"
ICONS = ROOT / "data/curated/icon_players.csv"
CURATED_PRIMES = ROOT / "data/curated/prime_players.csv"
PROFILES = ROOT / "data/processed/player_profiles_with_positions.csv"
DEFENSE_FBREF = ROOT / "data/processed/player_defense_fbref.csv"
ATTACK_FBREF = ROOT / "data/processed/player_attack_fbref.csv"
UNDERSTAT_PM = ROOT / "data/external/advanced/understat/understat_player_match.csv"

# A prime is a great season, not a fluke: blending the season rating back toward
# the player's career level keeps a one-year hot streak from out-rating Messi
# while still rewarding it. Pure season rating had Bas Dost and a centre-back
# with 28 games at 94.5 — the rating model's known forward bias, amplified.
PRIME_SEASON_W = 0.58
PRIME_MIN_MATCHES = 25
# Per-position quantiles, not absolute ratings. The rating model rewards
# attacking output, so an absolute cut of 87 leaves the prime pool with 47
# forwards, 10 defenders and NO goalkeepers — a draft that then falls back to
# ordinary cards for exactly the slots it fills first. Asking for the top slice
# OF EACH POSITION reads the same thing off a scale each position actually
# reaches.
PRIME_SEASON_Q = 0.90     # his season must be a top-decile season for his role
PRIME_CAREER_Q = 0.85     # and he must be a top-sixth player in that role
# A prime nobody recognises is a wasted rare card. The strongest notability
# signal the data carries is the club he had that season, so a prime either
# comes from a side of real European weight or from a player whose whole career
# rates highly. Without this the pool filled with 89-rated full-backs from
# Gazelec Ajaccio — the rating model's role bias, dealt out as a prize.
PRIME_MIN_CLUB_ELO = 1720.0
# From 2014/15 the modern Understat+FBref pipeline covers the same seasons on
# far more evidence, so a StatsBomb row for one of them is the same season
# measured worse — and measured with a coverage bias, since StatsBomb released
# some clubs' whole 2015/16 and not others'. Left in, the defender primes filled
# up with 2015/16 squad players. Before 2014/15 StatsBomb is the ONLY source we
# have, and it is where Henry 03/04 and Puyol 09/10 come from, so it stays.
SB_ERA_CUTOFF = 2014
PRIME_CLUB_EXEMPT_Q = 0.98    # a truly elite career needs no famous club
PRIME_MAX = 96.0
PRIME_FLOOR = 82.0
# Primes are picked per POSITION, in the proportion an eleven needs them. A
# single global cut starves defenders and keepers: the rating model rewards
# attacking output, so the best 200 seasons overall are almost all forwards, and
# a draft would keep falling back to ordinary cards for the back line.
PRIMES_PER_POSITION = {"Goalkeeper": 18, "Defender": 72, "Midfielder": 56, "Forward": 56}
# How much measured axis detail a profile must have before it pulls the card
# away from what its rating implies. 60 matches ~ half weight.
AXIS_SHRINK_MATCHES = 60.0
# A profile has to have been seen this often before it helps define what the
# axis distribution even looks like.
AXIS_MIN_MATCHES = 20
MAX_PRIMES_PER_PLAYER = 1

ICON_MIN = 86.0

POS_ORDER = ["Goalkeeper", "Defender", "Midfielder", "Forward"]

# ── the LEVEL of a card, not its shape ────────────────────────────────────────
# A card's `overall` used to be `player_ratings_roles.csv:base_ovr` and nothing
# else. Two things are wrong with reading that file literally, both measured on
# 2026-09-06 and both fixed here rather than in the ratings file, which feeds
# the betting side (see the protected-baseline rule).
#
# 1. MORE THAN HALF OF IT IS A FLOOR. The display curves map a raw score onto
#    an OVR with `np.interp`, which HOLDS FLAT below its first anchor — so every
#    player under it lands on exactly the same number. 7,152 of 13,297 ratings
#    (53.8%) sit precisely on their curve's bottom anchor: 69.7% of strikers on
#    66.0, 65.8% of keepers on 66.0, 40.3% of centre backs on 64.0. Nick Pope
#    with 496 matches, ter Stegen with 460 — both 66.0. That is not a low
#    rating, it is the absence of one, and a draft that ranks by it is ranking
#    by nothing. A floor-pinned rating is therefore treated as MISSING and goes
#    down the projection path, which is weak (R² 0.18) but at least varies.
CURVE_FLOOR = {"STR": 66.0, "WNG": 66.0, "CRE": 66.0,
               "PIV": 64.0, "CB": 64.0, "FB": 64.0, "GK": 66.0}
_ROLE_GRP = {
    "Central de salida": "CB", "Central stopper": "CB",
    "Lateral defensivo": "FB", "Lateral ofensivo": "FB",
    "Destructor": "PIV", "Pivote organizador": "PIV", "Box-to-box": "PIV",
    "Creador": "CRE", "Mediapunta": "CRE",
    "Extremo": "WNG", "Extremo interior": "WNG",
    "Killer": "STR", "Target man": "STR", "Delantero completo": "STR",
    "Falso 9": "STR", "Portero": "GK",
}
FLOOR_EPS = 0.05

# 2. IT KNOWS NOTHING ABOUT THE LAST TWO SEASONS. `base_ovr` is a career
#    aggregate of a pipeline whose modern half died; the validated attacking
#    measurement is allowed to move the level, and the size of the move is
#    FITTED, not chosen. Regressing 2024/25 goals+assists per 90 on
#    (base_ovr, att_z) over 644 players who appear in both seasons gives
#    0.0649 per sd of att_z against 0.0147 per rating point — so one standard
#    deviation of measured attacking is worth **4.4 rating points**. Adding the
#    term lifts the temporal R² from **+0.260 to +0.304**.
# THE LEVEL USES A DIFFERENT Z FROM THE AXIS, and the difference is the whole
# point. `att_z` divides by each position's own spread, which is right for the
# card FACE ("how good is he for a winger") and wrong for the LEVEL, which is
# compared across positions: Honorat and Kane both came out +2.48 and got the
# same bump, on 0.46 and 1.24 goals+assists per 90. `att_level_z` centres on the
# position mean but divides by the POOLED spread, so positions stay unbiased
# while the fact that forwards vary more in attacking output survives. Kane goes
# to +3.00 and Honorat to +1.47.
#
# With that fixed the old trade-off disappears — 8 points per sd is better than
# the previous 4.4 at BOTH jobs, measured on the same 705 players:
#
#     peso   R² predice 24/25   corr describe 23/24
#      4.4        +0.363              +0.590
#      6.0        +0.366              +0.615
#      8.0        +0.361              +0.640   <- desplegado
#     12.0        +0.340              +0.664
ATT_LEVEL_POINTS = 8.0
ATT_LEVEL_Z_CAP = 3.0
# ...and only where attacking output IS the job. The validation target is goals
# and assists, so letting the term touch defenders rewards the ones who attack:
# the first run put Clauss, Frimpong, Dimarco and five other full-backs above
# van Dijk, who FELL from 88.8 to 87.6. Same trap as the league-relative z, one
# level down — a signal validated on an attacking outcome may only set the level
# of players judged by attacking outcomes. Defenders and keepers keep the
# rating they had; for them there is no validated target and no pretending.
ATT_LEVEL_POSITIONS = {"Forward", "Midfielder"}

# 3. AND IT PAYS YOU FOR HAVING BEEN PARSED. `player_profiles_with_positions.csv`
#    carries appearances and summary stats (shots, goals, assists, fouls, cards)
#    from one parse and everything event-derived (duels, tackles, passes,
#    interceptions, xG) from another — and for **63% of profiles the second one
#    produced nothing**: `defense_creation_matches == 0`, every event stat left
#    on its neutral default (duel_win_rate 0.500, pass_completion 0.750, the
#    rest 0.0). The rating then feeds those placeholders in WITH THE WEIGHT OF
#    HIS FULL MATCH COUNT. Rodri: 365 matches, not one event. Lewis Dunk: 299.
#    Andy Robertson: 275. Among players with 100+ matches, 92.4% are blank.
#
#    Measured, on players who also have an independent FBref row: the parsed
#    group is rated +9.6 (defenders) / +8.6 (midfielders) / +9.0 (forwards)
#    points higher while PRODUCING THE SAME OR LESS (+0.10 / -0.21 / -0.34 sd).
#    Nine points for having been parsed.
#
#    The size shipped is not that nine, and the difference matters. Regressing
#    the outcome on (base_ovr, has-no-events) asks for **+4.2 points**, and
#    sweeping the correction peaks at exactly +4: R2 +0.2435 -> +0.2746,
#    Spearman +0.4661 -> +0.4930, falling away again by +8. So part of that nine
#    is real ability and part is the artefact; the temporal fit separates them
#    and only the artefact is removed. A quantile-matching version that equalised
#    the two groups outright was tried first and REJECTED — it failed this same
#    gate (R2 +0.260 -> +0.249).
COVERAGE_BLIND_POINTS = 4.0
# ...and in proportion to how much of the rating the blind half actually is.
# The first version paid the full +4 to anyone whose profile had no events, but
# `base_ovr` blends the StatsBomb half with the modern one weighted by matches:
# Rodri is 365 blind matches out of 658, Lautaro Martínez is 12 out of 279. Paying
# them the same put Lautaro top of the whole game, one place above Mbappé, on a
# correction for contamination he barely has. Against the temporal gate the two
# are a TIE (R2 +0.352 vs +0.349, Spearman +0.549 vs +0.556) — this is not
# shipped because it scores better, it is shipped because paying a 4%-blind
# player a 100% correction is wrong, and the tie says the data does not object.

# 4. AND WHEN THE PROFILE IS BLIND, THE ROLE IS A GUESS TOO. The role is picked
#    by best fit over the same profile, so a player whose events were never
#    parsed gets his role chosen from placeholders. Rodri came out "Destructor",
#    which puts **86% of the rating on interceptions and tackles won** — where
#    he is ordinary — and reads none of the four columns where he is 98th-99th
#    percentile among 2,477 midfielders: pass completion 92.0, passes 112.1/90,
#    progressive passes 11.5/90, progressive distance 483.6/90. His real role,
#    "Pivote organizador", weights the organising block at 64% and reads exactly
#    those columns.
#
#    So the role is re-picked from what IS measured — but only where the stored
#    one was a guess (blind profile) AND the measurement is decisive. Those two
#    guards matter: without them 78% of midfielders change role, which is not a
#    correction, it is a different model. With them it is 289 players (33% of
#    midfielders with a role); Kimmich, Modrić, Kroos and Xhaka keep theirs
#    because their profiles are genuinely measured, and Bellingham and Casemiro
#    keep theirs because the measurement does not separate the candidates.
#
#    These passing signals are the steadiest thing in the whole dataset —
#    season-to-season r **+0.83 to +0.90** over 1,224 players, against +0.06 for
#    goalkeeper shot-stopping, which was rejected for exactly that reason.
ROLE_REPICK_MARGIN = 0.5
ROLE_REPICK_MIN_90S = 8.0
# The level needs the same soft ceiling the axes have, and for the same reason:
# a hard clip stacks everyone who reaches it on one number. Retuned once the
# bump grew — at max 96 / compression 0.45 the top FORTY-THREE cards sat on 96.0
# together. Measured over the whole pool: max 99 with compression 0.35 leaves
# exactly two cards within half a point of the top, which is the "a 99 should be
# two or three cards in the whole game" the design asks for.
LEVEL_KNEE = 88.0
# Retuned with the ceiling at 94: at 0.35 the whole elite clipped flat on 94.0
# together. 0.18 means reaching 94 needs a pre-ceiling 121, which nobody does,
# so the top approaches the ceiling and separates instead of stacking on it.
# 0.18 was an over-correction and it wrecked the top of the game. It squashed
# everything above the knee by a factor of five and a half: the best fifty
# outfield cards spanned **1.3 points** and the best twenty spanned 0.9, with
# 106 cards sitting between 88 and 89. Kane and a 2. Bundesliga striker differ
# by fifteen points before the ceiling and by eight tenths after it — at that
# compression the order of the elite is noise, which is exactly what it looked
# like. Swept over the whole pool: 0.55 gives the top fifty ~3.9 points and the
# top twenty ~2.7, with nobody stacked on the cap.
LEVEL_COMPRESS = 0.55
# An ordinary card must not out-rank a rare one. Primes reach 96 and icons 97,
# and with the level bump grown, NINE actual cards were passing the best prime
# and 62 the median icon. 94 is the ceiling for ACTUAL; the scarcity of the
# other two families is the whole point of them.
LEVEL_MAX = 94.0
# NOT USED, and the reason matters. The LEAGUE-relative attacking z wins the
# same test by a mile (+0.260 → +0.554) — but the test's target is goals and
# assists, an attacking outcome, so of course it does. Putting it in the level
# would rank every striker above every defender by construction, which is the
# exact bias the role system exists to remove. The within-position z is the
# smaller, correctly-shaped signal, and it is the one that ships.


def _blind_share(profile, rating_matches: float) -> float:
    """What fraction of a rating rests on the blind StatsBomb half.

    `base_ovr` blends the two eras weighted by matches, so a profile with no
    parsed events contaminates the rating only in proportion to its share of
    them. No profile at all means the whole thing is unknown, hence 1.0.
    """
    if profile is None:
        return 1.0
    sb = float(getattr(profile, "matches", 0) or 0)
    tot = float(rating_matches or 0)
    if tot <= 0:
        return 1.0
    return float(min(sb / tot, 1.0))


def soft_level(v: float) -> float:
    """Diminishing returns at the top of the LEVEL scale (see LEVEL_KNEE)."""
    v = float(v)
    if v <= LEVEL_KNEE:
        return v
    return min(LEVEL_MAX, LEVEL_KNEE + (v - LEVEL_KNEE) * LEVEL_COMPRESS)


def is_floor_pinned(role: object, ovr: float) -> bool:
    """True when this rating is the bottom of its display curve, i.e. absent."""
    f = CURVE_FLOOR.get(_ROLE_GRP.get(str(role), ""))
    return f is not None and abs(float(ovr) - f) < FLOOR_EPS


def measured_roles() -> NameIndex:
    """player -> the midfield role his own FBref numbers actually describe.

    Three candidate profiles, each a z-blend within midfielders: CONTROL (pass
    completion, pass volume, progressive passes, progressive distance),
    DESTRUCTION (interceptions, tackles won) and CREATION (key passes,
    progressive carries). The winner takes it only if it clears the runner-up by
    ROLE_REPICK_MARGIN — see the note at COVERAGE_BLIND_POINTS.
    """
    src = ROOT / "data/external/advanced/fbref_kaggle"
    need = ["2324_Passing.csv", "2324_Defense.csv", "2324_Possession.csv"]
    if not all((src / f).exists() for f in need):
        return NameIndex()
    pa = pd.read_csv(src / "2324_Passing.csv")
    de = pd.read_csv(src / "2324_Defense.csv")
    po = pd.read_csv(src / "2324_Possession.csv")
    n90 = pd.to_numeric(pa["minutes_90s"], errors="coerce")

    def col(df, name):
        return dict(zip(df["player"], pd.to_numeric(df[name], errors="coerce")))

    d = pd.DataFrame({
        "player": pa["player"],
        "pos": pa["position"].astype(str).str.split(",").str[0].str.upper(),
        "n90": n90,
        "pass_pct": pd.to_numeric(pa["passes_pct"], errors="coerce"),
        "passes_p90": pd.to_numeric(pa["passes"], errors="coerce") / n90.clip(lower=0.5),
        "prog_p90": pd.to_numeric(pa["prog_passes"], errors="coerce") / n90.clip(lower=0.5),
        "progdist_p90": pd.to_numeric(pa["passes_prog_dist"], errors="coerce") / n90.clip(lower=0.5),
        "int_p90": pa["player"].map(col(de, "intr")) / n90.clip(lower=0.5),
        "tklw_p90": pa["player"].map(col(de, "tklw")) / n90.clip(lower=0.5),
        "kp_p90": pd.to_numeric(pa["KP"], errors="coerce") / n90.clip(lower=0.5),
        "carries_p90": pa["player"].map(col(po, "prog_carries")) / n90.clip(lower=0.5),
    })
    d = d[d["n90"] >= ROLE_REPICK_MIN_90S].copy()
    if len(d) < 200:
        return NameIndex()
    if ATTACK_FBREF.exists():
        att = pd.read_csv(ATTACK_FBREF).drop_duplicates("player").set_index("player")
        for c in ("npxg_p90", "xag_p90", "sca_p90"):
            d[c] = d["player"].map(att[c]) if c in att.columns else np.nan
    else:
        for c in ("npxg_p90", "xag_p90", "sca_p90"):
            d[c] = np.nan
    if DEFENSE_FBREF.exists():
        dfn = pd.read_csv(DEFENSE_FBREF).drop_duplicates("player").set_index("player")
        for c in ("aerial_pct", "challenge_pct"):
            d[c] = d["player"].map(dfn[c]) if c in dfn.columns else np.nan
    else:
        d["aerial_pct"] = d["challenge_pct"] = np.nan

    # Each candidate role is a contrast of the stats its own blend weights, so a
    # player lands where his numbers point. This is a CLASSIFICATION, i.e. a
    # design choice about what the role names mean — the taxonomy is not
    # measured, only his position within it is.
    CANDIDATES = {
        "DF": {"Central stopper": ("aerial_pct", "challenge_pct"),
               "Central de salida": ("pass_pct", "progdist_p90"),
               "Lateral ofensivo": ("xag_p90", "sca_p90")},
        "MF": {"Pivote organizador": ("pass_pct", "passes_p90", "prog_p90", "progdist_p90"),
               "Destructor": ("int_p90", "tklw_p90"),
               "Creador": ("kp_p90", "carries_p90"),
               "Mediapunta": ("npxg_p90", "xag_p90")},
        "FW": {"Killer": ("npxg_p90", "npxg_p90"),
               "Target man": ("aerial_pct", "aerial_pct"),
               "Extremo": ("sca_p90", "xag_p90"),
               "Falso 9": ("kp_p90", "pass_pct")},
    }
    idx = NameIndex()
    for pos, cands in CANDIDATES.items():
        sub = d[d["pos"] == pos]
        if len(sub) < 60:
            continue

        def z(c, frame=sub):
            v = pd.to_numeric(frame[c], errors="coerce")
            return ((v - v.mean()) / (v.std() or 1.0)).fillna(0.0)

        prof = pd.DataFrame({name: sum(z(c) for c in cols) / len(cols)
                             for name, cols in cands.items()}, index=sub.index)
        vals = prof.to_numpy()
        order = np.sort(vals, axis=1)
        margin = order[:, -1] - order[:, -2]
        pick = np.array(prof.columns)[vals.argmax(axis=1)]
        for name, role, mg, n in zip(sub["player"], pick, margin, sub["n90"]):
            if mg >= ROLE_REPICK_MARGIN:
                idx.add(name, str(role), rank=float(n))
    return idx


def gk_level_index() -> NameIndex:
    """Career goalkeeper score, matched by NAME INDEX rather than by `.lower()`.

    The deployed builder joins `_build_gk_scores()` on a bare lowercased name,
    and the two source files disagree about spelling: StatsBomb writes "marc
    andré ter stegen", the profiles write "Marc-André ter Stegen". The join
    misses **29% of keepers**, who then take the 50.0 default, which the display
    curve maps to exactly 66.0. ter Stegen has a real score of 70.2 — worth
    about 85 — and his card said 66.0 because of a hyphen. Same defect lives in
    `player_strength.py:714`, on the betting side, and is reported rather than
    changed here.

    Caveat kept in the open: season-to-season goalkeeper shot-stopping barely
    persists (r +0.06 for PSxG+/- per 90, +0.17 for save%), so even a correct
    score is a weak measurement. These are CAREER aggregates over hundreds of
    matches, which is far steadier than one season, but a keeper rating is the
    least trustworthy number in the catalogue.
    """
    from mundialytics.statistical_core.player_strength import _build_gk_scores

    idx = NameIndex()
    for name, score in _build_gk_scores().items():
        idx.add(name, float(score), rank=float(score))
    return idx


GKX = [50.0, 60.0, 66.0, 70.0, 74.0, 78.0, 85.0]
GKY = [66.0, 76.0, 82.0, 85.0, 87.5, 89.0, 91.0]


def _norm(s: object) -> str:
    t = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in t if not unicodedata.combining(c)).lower().strip()


def person_tokens(name: object) -> frozenset:
    """Token set used to decide whether two spellings are the same person.

    The season ratings carry both eras' spellings — "Lionel Messi" from the
    modern pipeline and "Lionel Andrés Messi Cuccittini" from StatsBomb — and a
    first-plus-surname key does not join them, so Messi was dealt two separate
    prime cards. Token SETS do join them: the same man written at two lengths
    always has one set contained in the other.
    """
    from mundialytics.identity.current_squads import _fold
    return frozenset(_fold(name).split())


def dedupe_people(df: pd.DataFrame, name_col: str, keep: int = 1) -> pd.DataFrame:
    """Keep at most `keep` rows per person, in the frame's existing order.

    Greedy rather than a groupby because "same person" here is a containment
    test, not an equality on some key — there is no single key that both
    spellings of a name produce.
    """
    seen: list[frozenset] = []
    counts: list[int] = []
    keep_mask = []
    for name in df[name_col]:
        toks = person_tokens(name)
        hit = None
        for i, prev in enumerate(seen):
            if toks and prev and (toks <= prev or prev <= toks):
                hit = i
                break
        if hit is None:
            seen.append(toks)
            counts.append(1)
            keep_mask.append(True)
        else:
            counts[hit] += 1
            keep_mask.append(counts[hit] <= keep)
    return df[pd.Series(keep_mask, index=df.index)]


def _season_label(code: object) -> str:
    """'1516' or '2015/2016' -> '15/16'."""
    s = str(code).strip()
    if "/" in s:
        a, b = s.split("/")[0][-2:], s.split("/")[-1][-2:]
        return f"{a}/{b}"
    if len(s) == 4 and s.isdigit():
        return f"{s[:2]}/{s[2:]}"
    return s


# ── sources ────────────────────────────────────────────────────────────────────
def axis_samples() -> dict[str, tuple[float, float]]:
    """player -> (attacking sample, defensive/creative sample).

    Two numbers, not one, because the data has two. `matches` counts every
    appearance StatsBomb released; `defense_creation_matches` counts only the
    matches whose raw events were parsed for duels, clearances and blocks — and
    that is zero for 62% of defenders.
    """
    prof = pd.read_csv(PROFILES, usecols=lambda c: c in {
        "player", "matches", "defense_creation_matches", "shots_sample"})
    out = {}
    for r in prof.itertuples(index=False):
        n_att = float(getattr(r, "shots_sample", 0) or 0) or float(getattr(r, "matches", 0) or 0)
        out[str(r.player)] = (n_att, float(getattr(r, "defense_creation_matches", 0) or 0))
    return out


def defense_index() -> tuple:
    """FBref's defensive quality per player, and how much of it there is.

    This is the measurement StatsBomb could not give: `defense_creation_matches`
    is zero for 62% of defenders, so the axis fell back to "what his role
    implies" for two thirds of them. FBref's published season tables carry
    challenge success rate, aerial win rate and times dribbled past for 72% of
    the players in our squads — including the entire Championship, Segunda and
    2. Bundesliga, which is where the promoted clubs' defenders come from.

    Returns an index of z-scores (already standardised within position, so it
    plugs straight in as "how far from what his rating implies") plus the 90s
    behind each, which is what the credibility weighting needs.
    """
    if not DEFENSE_FBREF.exists():
        return NameIndex(), 0
    return _fbref_index(pd.read_csv(DEFENSE_FBREF), "def_level_z", squad_positions())


def attack_level_index() -> tuple:
    """The cross-position attacking z, for the LEVEL — see ATT_LEVEL_POINTS.

    Kept as a Z, not a percentile: the 8-points-per-sd weight was fitted and
    validated in those units. The AXES use percentiles because they are placed
    on a fixed card scale; the level is a fitted adjustment, which is different.
    """
    if not ATTACK_FBREF.exists():
        return NameIndex(), 0
    d = pd.read_csv(ATTACK_FBREF)
    if "att_level_z" not in d.columns:
        return NameIndex(), 0
    d = d[d["att_level_z"].notna()]
    idx = NameIndex()
    for r in d.itertuples(index=False):
        season = _season_code(getattr(r, "season", ""))
        n90 = float(getattr(r, "n90", 0) or 0)
        rank = (float(season) if season.isdigit() else 0.0) + min(n90, 999.0) / 1000.0
        idx.add(r.player, (float(r.att_level_z), n90, season), rank=rank)
    return idx, len(d)


def attack_index() -> tuple:
    """FBref's attacking quality per player, and how much of it there is.

    The defensive twin of this could not be validated — a defender's measured
    actions predict nothing about what his team concedes. This one could, and it
    is the reason to trust it: the weights behind `att_z` were fitted on
    2023/24 features against 2024/25 goals + assists per 90, out of sample,
    **R² +0.51 over 1,615 players**. What a forward did last season really does
    say what he will do next, and that is exactly the claim a card face makes.

    Same convention as the defensive index: a within-position z-score, so it
    reads as "how far from what his position implies", already credibility-
    weighted by minutes.
    """
    if not ATTACK_FBREF.exists():
        return NameIndex(), 0
    return _fbref_index(pd.read_csv(ATTACK_FBREF), "att_level_z", squad_positions())


def creation_index() -> tuple:
    """FBref's creation quality per player — the axis that had no measurement.

    A card's creation was its rating plus what the role implies and nothing
    else, so Olise (0.46 expected assists per 90, among the best in Europe) read
    like any other winger of his rating. `crea_z` is fitted the same way the
    attacking one is: 2023/24 features against **2024/25 assists per 90**,
    R² +0.290 over 1,615 players, led by xAG, key passes and shot-creating
    actions. Standardised within position, so it says how far above his own
    position he creates.
    """
    if not ATTACK_FBREF.exists():
        return NameIndex(), 0
    d = pd.read_csv(ATTACK_FBREF)
    if "crea_z" not in d.columns:
        return NameIndex(), 0
    col = "crea_level_z" if "crea_level_z" in d.columns else "crea_z"
    d = d[d[col].notna()]
    return _fbref_index(d, col, squad_positions())


# "Creation" is not one thing. For a winger it is the chance he makes; for a
# centre back it is playing out — and Pau Cubarsí is 99th percentile among
# defenders on pass accuracy (93.5%) and progressive distance (550 per 90) while
# the chance-creation axis reads him at -0.04, because it counts key passes and
# expected assists, which a centre back does not produce. The roles below get
# their creation from BUILD-UP instead.
BUILDUP_ROLES = {"Central stopper", "Central de salida", "Lateral defensivo",
                 "Destructor", "Pivote organizador"}


def buildup_index() -> tuple:
    """Playing-out quality per player: accuracy, progression, volume."""
    from mundialytics.identity.fbref_quality import load as _fbq

    q = _fbq()
    if q.empty:
        return NameIndex(), 0
    d = pd.read_csv(DEFENSE_FBREF) if DEFENSE_FBREF.exists() else pd.DataFrame()
    pos = dict(zip(d.get("player", []), d.get("position", []))) if len(d) else {}
    q = q.copy()
    q["position"] = [pos.get(w, "Midfielder") for w in q["player"]]
    q["n90"] = q["fbq_90s"]
    cols = {"pass_pct": 0.40, "prog_passes_p90": 0.34, "prog_dist_p90": 0.26}
    z = np.zeros(len(q))
    for c, w in cols.items():
        v = pd.to_numeric(q[c], errors="coerce")
        zz = v.groupby(q["position"]).transform(
            lambda x: (x - x.mean()) / (x.std() if x.std() else 1.0))
        z += np.nan_to_num(zz.to_numpy()) * w
    q["build_z"] = z / sum(cols.values())
    q = q[q["build_z"].notna()]
    return _fbref_index(q, "build_z", squad_positions())


def _season_code(x: object) -> str:
    """'2015/2016', '15/16', '2324' -> a comparable four-digit code."""
    s = "".join(ch for ch in str(x) if ch.isdigit())
    if len(s) == 8:      # 20152016
        return s[2:4] + s[6:8]
    if len(s) == 4:      # 2324
        return s
    return s[-4:] if len(s) > 4 else s


def squad_positions() -> NameIndex:
    """player -> the position his CARD will use, from the live squad file."""
    sq = load_current_squads()
    idx = NameIndex()
    if sq.empty:
        return idx
    for who, pos in zip(sq["player"], sq["pos_group"]):
        if str(pos) in POS_ORDER:
            idx.add(who, str(pos), rank=1.0)
    return idx


def _fbref_index(d: pd.DataFrame, zcol: str, squad_pos: NameIndex | None = None) -> tuple:
    # the PERCENTILE within his position, because the z's spread is not 1 and
    # a scale written in standard deviations silently truncates itself.
    #
    # And within the position the CARD uses, not FBref's. They disagree for 184
    # players (12% of those in both), and the disagreement is not random: FBref
    # files an attacking wing back under DF, so Franck Honorat's attacking z was
    # computed against DEFENDERS — where being attacking is rare — and then read
    # off the MIDFIELDER scale. He came out level with Olise on 0.46 goals+
    # assists per 90 against Olise's 1.04. Saka is filed the same way.
    d = d.copy()
    if squad_pos is not None:
        d["position"] = [
            (lambda h, f: str(h) if h else str(f))(squad_pos.lookup(w), f)
            for w, f in zip(d["player"], d.get("position", pd.Series("", index=d.index)))]
    d["pct_pos"] = (pd.to_numeric(d[zcol], errors="coerce")
                 .groupby(d["position"] if "position" in d.columns else 0)
                 .rank(pct=True))
    idx = NameIndex()
    for r in d.itertuples(index=False):
        season = _season_code(getattr(r, "season", ""))
        n90 = float(getattr(r, "n90", 0) or 0)
        # the LATEST season wins a tie, with minutes as the tiebreak — for an
        # "actual" card the current measurement is the one that describes him
        rank = (float(season) if season.isdigit() else 0.0) + min(n90, 999.0) / 1000.0
        pct = getattr(r, "pct_pos", np.nan)
        if not np.isfinite(pct):
            continue
        idx.add(r.player, (float(pct), n90, season), rank=rank)
    return idx, len(d)


def fb_axis(hit: object, season: str = "") -> tuple | None:
    """`(z, n90)` from an FBref index hit, or None if it describes another year.

    FBref only reaches back to 2023/24 here, and a PRIME card is a specific
    past season. Scoring "Suárez 15/16" off Suárez's 2024/25 output would print
    a peak card measured on a player eight years past it — the same class of
    mistake as pricing Hull with its 2016/17 eleven. So a prime takes the
    measurement only when it is a measurement OF THAT SEASON; otherwise the
    axis falls back to what his rating and role imply, which is what a card for
    an unmeasured year honestly is.
    """
    if hit is None:
        return None
    z, n90, row_season = hit
    if season and _season_code(season) != row_season:
        return None
    return (z, n90)


def load_sources() -> dict:
    from mundialytics.statistical_core.player_strength import PlayerStrengthModel

    print("cargando fuentes...", flush=True)
    model = PlayerStrengthModel().fit()
    roles = pd.read_csv(ROLES)
    seasons = pd.read_csv(SEASONS)
    elo = pd.read_csv(CLUBELO)
    squads = load_current_squads()
    if squads.empty:
        raise SystemExit("falta current_squads.csv — ejecuta scripts/build_current_squads.py")
    print(f"  perfiles {len(model.profiles_):,} · roles {len(roles):,} · "
          f"temporadas {len(seasons):,} · plantillas {len(squads):,}")
    return {"model": model, "roles": roles, "seasons": seasons, "elo": elo, "squads": squads}


def club_strength(elo: pd.DataFrame) -> dict[str, float]:
    """Accent-folded club name -> Elo, for judging how notable a season was."""
    return {_norm(c): float(e) for c, e in zip(elo["club"], elo["elo"])}


def club_elo_map(squads: pd.DataFrame, elo: pd.DataFrame) -> dict[str, float]:
    from mundialytics.statistical_core.competition.european import make_resolver

    res = make_resolver(list(elo["club"]))
    lut = dict(zip(elo["club"], elo["elo"]))
    out = {}
    for t in squads["team"].unique():
        hit = res(t)
        if hit:
            out[t] = float(lut[hit])
    return out


# ── axis estimation ────────────────────────────────────────────────────────────
# How far each axis sits from the RATING, per role. This is what a card face
# actually claims: a Killer's attack is roughly his rating, and his defending is
# nowhere near it. The numbers are a design decision, stated here rather than
# hidden in a fit, because the underlying measurement cannot answer the question
# — see fit_axis_residuals for why.
ROLE_OFFSET = {
    "Killer":             {"attack": +2, "defense": -34, "creation": -16},
    "Target man":         {"attack": +1, "defense": -29, "creation": -18},
    "Delantero completo": {"attack": +1, "defense": -28, "creation": -10},
    "Extremo":            {"attack": 0, "defense": -31, "creation": -10},
    "Extremo interior":   {"attack": 0, "defense": -30, "creation": -8},
    "Mediapunta":         {"attack": -3, "defense": -32, "creation": 0},
    "Creador":            {"attack": -14, "defense": -22, "creation": 0},
    "Pivote organizador": {"attack": -19, "defense": -10, "creation": -1},
    "Box-to-box":         {"attack": -10, "defense": -9, "creation": -6},
    "Destructor":         {"attack": -25, "defense": -1, "creation": -15},
    "Central stopper":    {"attack": -34, "defense": 0, "creation": -21},
    "Central de salida":  {"attack": -30, "defense": -1, "creation": -11},
    "Lateral ofensivo":   {"attack": -17, "defense": -7, "creation": -11},
    "Lateral defensivo":  {"attack": -25, "defense": -2, "creation": -17},
    "Portero":            {"attack": -28, "defense": 0, "creation": -20},
}
POS_OFFSET = {
    "Forward":    {"attack": +1, "defense": -31, "creation": -12},
    "Midfielder": {"attack": -11, "defense": -14, "creation": -2},
    "Defender":   {"attack": -30, "defense": -1, "creation": -14},
    "Goalkeeper": {"attack": -28, "defense": 0, "creation": -20},
}
# How far a player's own measurement may pull an axis off his role's baseline,
# at full credibility. One standard deviation of the residual is worth this many
# rating points, capped at 1.5 sd.
AXIS_SPREAD = 6.0
AXIS_Z_CAP = 1.5
# Above the knee the scale compresses. Without it every elite forward printed 99
# in attack — the offsets put a 96-rated Killer at 98 before his own residual is
# even added, so the whole top of the game flattened into one number and the 99
# stopped meaning anything. With these values a 99 needs a pre-ceiling 112,
# which two or three cards in the catalogue reach and nobody else does.
AXIS_KNEE = 88.0
AXIS_COMPRESS = 0.5
# FBref counts full matches, not appearances; twelve of them is half weight.
FBREF_SHRINK_90S = 12.0
# A per-position ceiling on the defensive axis, and the one number here that is
# a stated rule rather than a measurement. The axis is anchored on the rating,
# so a 97-rated forward printed 74 in defending purely because he is a 97 — the
# card was reading his greatness as tackling. Nobody who plays up front defends
# at 74. Read what it does honestly: it is a cap, it binds on 1.3% of forwards,
# and every one it binds on is a 91+ icon or prime.
POSITION_DEF_CAP = {"Forward": 65.0}


# ── the card face, rebuilt 2026-09-07 ─────────────────────────────────────────
# AN AXIS IS AN ABSOLUTE STATEMENT, NOT A RESTATEMENT OF THE LEVEL. It used to
# be `level + ROLE_OFFSET[axis] + a nudge`, so every axis moved with the level:
# measured across the pool, a forward's DEFENCE correlated **+0.946** with his
# own overall. It was not describing his defending, it was describing how good
# he is. The icons, whose axes are set by hand, sit at +0.190 — independent, as
# an axis should be, and that is why their cards read right and these did not.
#
# So each axis is now placed by ITS OWN measurement: the within-position z from
# FBref, put on a scale per position and axis. The centres and spreads below are
# a design decision — the card scale always is — and they are calibrated so the
# ACTUAL distribution has the same SHAPE as the icons and primes and sits below
# them, because an ordinary card should not out-read a rare one.
#
#   (value at z = 0, rating points per standard deviation)
# THE Z DOES NOT HAVE UNIT VARIANCE, and assuming it did is what put van Dijk
# on 80 in defence. After credibility shrinkage the within-position z has a
# standard deviation of ~0.63 and the best defender in the whole file reaches
# +1.62, so a scale written as "8 points per standard deviation, up to ±2.5"
# only ever used a third of its intended range: van Dijk sits at the 94th
# percentile of defenders and came out 80.3, and nobody could pass 85.
#
# So the axis is placed by PERCENTILE, which does not care what the z's spread
# happens to be. Each entry is (value at the 5th percentile, at the 50th, at
# the 99th) for that position and axis, interpolated through. The targets are
# read off the icons and primes — the hand-set cards whose faces read right —
# and set below them, because an ordinary card should not out-read a rare one.
AXIS_SCALE = {
    "Forward":    {"attack": (52.0, 74.0, 93.0), "defense": (28.0, 42.0, 58.0),
                   "creation": (46.0, 63.0, 86.0)},
    "Midfielder": {"attack": (38.0, 58.0, 86.0), "defense": (40.0, 58.0, 80.0),
                   "creation": (48.0, 68.0, 92.0)},
    "Defender":   {"attack": (28.0, 42.0, 64.0), "defense": (52.0, 74.0, 92.0),
                   "creation": (42.0, 60.0, 82.0)},
    "Goalkeeper": {"attack": (30.0, 40.0, 55.0), "defense": (58.0, 72.0, 89.0),
                   "creation": (38.0, 50.0, 66.0)},
}
# A full season is ~30 of these. Credibility pulls an axis toward its own
# median rather than shrinking a z, which is the same idea one layer up: a
# ten-match cameo lands near the middle instead of at an extreme.
# Twelve, not six. A dozen of the top forty outfield cards rested on fewer than
# twenty measured matches — Daka on 12.1, Akpom on 11.7 — and at six a half
# season already kept two thirds of its spread. At twelve a full season keeps
# 72% and a twelve-match cameo only half.
AXIS_CRED_90S = 12.0

# ── and the OVERALL follows the axes ─────────────────────────────────────────
# The level and the axes were computed by two independent routes and could
# disagree on the same card — Haaland read 92.1 overall next to 84.2 in attack.
# A card is one object. The role says how its axes combine, which is the whole
# idea of a role: a fixed weighting of the statistics. These are the same A/O/D
# splits the rating builder scores with.
ROLE_AXIS_WEIGHT = {                     # (attack, defense, creation)
    "Killer":             (0.94, 0.02, 0.04),
    "Target man":         (0.55, 0.35, 0.10),
    "Delantero completo": (0.50, 0.08, 0.42),
    "Falso 9":            (0.32, 0.06, 0.62),
    # A winger who only creates used to tie one who creates AND scores, because
    # the card was 72% creation. Modern wingers score: roughly half and half.
    "Extremo":            (0.42, 0.06, 0.52),
    "Extremo interior":   (0.55, 0.03, 0.42),
    # a little more weight on control (defence) across the midfield roles: a
    # pure creator with a monster assist season used to top the line while
    # complete midfielders sat below, because defending counted for almost
    # nothing. Now that 25/26 measures their duels and tackles, it should.
    "Mediapunta":         (0.32, 0.16, 0.52),
    "Creador":            (0.10, 0.16, 0.74),
    "Box-to-box":         (0.30, 0.38, 0.32),
    "Pivote organizador": (0.08, 0.38, 0.54),
    "Destructor":         (0.06, 0.86, 0.08),
    "Central de salida":  (0.05, 0.60, 0.35),
    "Central stopper":    (0.03, 0.90, 0.07),
    "Lateral ofensivo":   (0.20, 0.30, 0.50),
    "Lateral defensivo":  (0.10, 0.60, 0.30),
}
POS_AXIS_WEIGHT = {
    "Forward":    (0.60, 0.06, 0.34),
    "Midfielder": (0.24, 0.38, 0.38),
    "Defender":   (0.10, 0.62, 0.28),
}
# How far the axes may pull the level from the career rating, at full
# measurement. Below 1.0 because the career rating carries seasons the two
# FBref years do not.
# ── current form (26/27) ──────────────────────────────────────────────────────
# "Nivel actual" — a card is what a player IS now, not his best measured year.
# The advanced stats are one to two seasons old, so the only current signal is
# ESPN's basic counting from this season: is he playing, and is he producing.
# It is deliberately bounded below the recent measurement (the user's rule) and
# ramped by how much of the season has actually happened, because three games is
# noise. A highly-rated player who is not starting loses part of the gap between
# his rating and a starter's floor; one who plays and scores is nudged up.
# ── the honest scale ──────────────────────────────────────────────────────────
# The old scale compressed every good outfielder between 88 and 94, where noise
# reordered them: Kane (99th percentile of forward production) came out below
# Wissa (88th). The user's rule is blunt and correct — most players are 75-80,
# only the ELITE of each position nears 90, and a normal (ACTUAL) card caps at
# 92; primes and icons live above that. So the blended strength orders players
# WITHIN their position and this curve gives the distribution its shape. It is
# per position, which also ends the forwards' monopoly on the top: each position
# now has its own 90s.
ACTUAL_MAX = 92.0
# The overall used to be hard-clipped at 92, which flattened every elite into one
# wall — nine cards read EXACTLY 92.0, so Malen sat level with Mbappé. The user
# asked for the opposite: keep the separation at the top, just bring the ceiling
# down to 92. So above a knee the excess is compressed through a tanh, which is
# monotonic (order and gaps survive, squeezed) and saturating (a single freak
# value lands at the ceiling without dragging everyone else down with it). Below
# the knee nothing moves. Only ACTUAL cards use this; primes and icons keep their
# own ceilings, above 92, as before.
ACTUAL_KNEE = 86.0
ACTUAL_ASYMPTOTE = 93.5   # the tanh approaches this; the hard cap at 92 still holds


def soft_cap_actual(v: float) -> float:
    v = float(v)
    if v <= ACTUAL_KNEE:
        return v
    span = ACTUAL_ASYMPTOTE - ACTUAL_KNEE
    return min(ACTUAL_MAX, ACTUAL_KNEE + span * float(np.tanh((v - ACTUAL_KNEE) / span)))
# Percentile against MEASURED players only, never the padded catalogue. 258 of
# 706 forwards are projected backups, so ranking against all of them put Cutrone
# — a 0.53 G+A/90 Serie A starter, genuinely the 68th percentile of measured
# forwards — at the 96th and therefore 88. The reference is the players we
# actually have data on; the projected ones are then placed against that ruler,
# and land low because they are unknowns.
ANCHOR_PCT_XS = [0.02, 0.25, 0.50, 0.75, 0.90, 0.97, 0.995, 1.00]
ANCHOR_PCT_YS = [56.0, 70.0, 76.0, 81.0, 86.0, 89.0, 91.5, 92.0]


FORM_FLOOR = 74.0          # a typical starter; nobody is penalised below it
FORM_PEN = 0.45            # fraction of the excess-over-floor lost at zero play
FORM_PROD_MAX = 2.5        # most that current production can add
FORM_RAMP_MATCHES = 8.0    # full weight only once the club has played this many
FORM_MIN_MINUTES = 180.0   # two full games before current production counts


AXES_LEVEL_WEIGHT = 0.55
# ...and the split follows how much EVIDENCE each side has, not a fixed number.
# A fixed 0.55 gave a career built on 170 matches the same standing as one built
# on 700, which is what holds a young player down: Cubarsí is 99th percentile
# among defenders on pass accuracy and progressive distance, and his card was
# anchored to a career that has barely started. Career credibility is
# matches/(matches+CAREER_K); measurement credibility is the FBref sample; the
# weight is their ratio, capped so a long career never disappears entirely.
CAREER_K = 300.0
AXES_LEVEL_WEIGHT_MAX = 0.75


def overall_from_axes(role: str, pos: str, ax: dict) -> float | None:
    """What the role's own weighting of these axes comes to."""
    w = ROLE_AXIS_WEIGHT.get(str(role)) or POS_AXIS_WEIGHT.get(str(pos))
    if not w:
        return None
    num = den = 0.0
    for wi, v in zip(w, (ax.get("attack"), ax.get("defense"), ax.get("creation"))):
        if v is None or not np.isfinite(v):
            continue
        num += wi * float(v)
        den += wi
    return num / den if den > 1e-9 else None
def measured_axis(pos: str, ax: str, hit: tuple | None) -> float | None:
    """An axis placed by its own measurement, or None when nobody measured it."""
    if hit is None:
        return None
    scale = AXIS_SCALE.get(str(pos))
    if not scale or ax not in scale:
        return None
    lo, mid, hi = scale[ax]
    pct, n90 = float(hit[0]), float(hit[1])
    # the last stretch keeps rising instead of flattening on `hi`. Everyone
    # above the 99th percentile used to collapse onto the same number: Honorat
    # and Olise both read 85.1 in creation, so the card could not say which of
    # them creates more. The top 1% is exactly where a card game needs to
    # separate people.
    v = float(np.interp(pct, [0.05, 0.50, 0.99, 1.0],
                        [lo, mid, hi, hi + (hi - mid) * 0.15]))
    cred = n90 / (n90 + AXIS_CRED_90S)
    return float(np.clip(mid + (v - mid) * cred, 25.0, 99.0))


def soft_ceiling(v: float) -> float:
    """Diminishing returns at the top of the axis scale."""
    v = float(v)
    if v <= AXIS_KNEE:
        return v
    return min(99.0, AXIS_KNEE + (v - AXIS_KNEE) * AXIS_COMPRESS)


def cap_defense(pos: str, v: float) -> float:
    """Apply the position ceiling on the defensive axis (see POSITION_DEF_CAP)."""
    lim = POSITION_DEF_CAP.get(str(pos))
    return float(v) if lim is None or not np.isfinite(v) else float(min(float(v), lim))


def fit_axis_residuals(model) -> dict:
    """Per position, what the rating already predicts about each axis.

    THE MEASUREMENT CANNOT CARRY THE CARD, and it is worth being exact about
    why. The axes come from StatsBomb's free data, whose coverage is not a
    sample — it is a release policy. Barcelona's whole modern history is in it,
    so the ten best-measured defenders in the pool are ten Barcelona defenders;
    Messi has 602 matches and Cristiano 76. Worse, `defense_creation_matches` —
    the sample behind duels, clearances and blocks — is **zero for 62% of
    defenders**, so Lewis Dunk's 299 Premier League matches carry a duel win
    rate of exactly 0.50, the neutral fallback. Read literally, the best centre
    backs in the world cap out at 75 while two thirds of them sit on the
    placeholder.

    So the rating sets the level and the role sets the shape, and the
    measurement is allowed only to say how a player differs from what his own
    rating already implies — the residual. Where there is no measurement the
    residual is not zero-as-a-number, it is absent, and the axis falls back to
    exactly what the role implies. That is the honest reading of "we have never
    measured this man defending".
    """
    rows = []
    for p in model.profiles_.values():
        if p.overall <= 0:
            continue
        rows.append({"pos": p.position, "ovr": p.overall,
                     "attack": p.offensive_strength, "defense": p.defensive_strength,
                     "creation": p.creation_strength})
    df = pd.DataFrame(rows)
    out = {}
    for pos, grp in df.groupby("pos"):
        if len(grp) < 40:
            continue
        f = {}
        for ax in ("attack", "defense", "creation"):
            b, a = np.polyfit(grp["ovr"], grp[ax], 1)
            res = grp[ax] - (a + b * grp["ovr"])
            f[ax] = (float(a), float(b), float(res.std() or 1.0))
        out[str(pos)] = f
    return out


def axes_for_card(fits: dict, pos: str, role: str, card_ovr: float,
                  profile=None, n_att: float = 0.0, n_def: float = 0.0,
                  fbref_def: tuple | None = None,
                  fbref_att: tuple | None = None,
                  fbref_crea: tuple | None = None,
                  fbref_build: tuple | None = None) -> dict[str, float]:
    """The four numbers printed on a card face.

    `n_att` / `n_def` are the sample sizes behind the attacking and the
    defensive/creative measurements. They are separate because the data is:
    a player can have 299 matches of shooting and none at all of defending.
    """
    # The role is a career label and the position is this week's team sheet, so
    # when they disagree the team sheet wins: Götze's role is still an attacking
    # midfielder's, ESPN lines him up in defence, and the role offset printed a
    # defender with 35 in defending.
    off = POS_OFFSET.get(str(pos), POS_OFFSET["Midfielder"])
    if _role_pos(str(role)) == str(pos):
        off = ROLE_OFFSET.get(str(role), off)
    fit = fits.get(str(pos)) or fits.get("Midfielder")
    measured = {"attack": getattr(profile, "offensive_strength", None),
                "defense": getattr(profile, "defensive_strength", None),
                "creation": getattr(profile, "creation_strength", None)}
    own_ovr = float(getattr(profile, "overall", 0) or 0)

    out: dict[str, float] = {}
    for ax in ("attack", "defense", "creation"):
        base = float(card_ovr) + off[ax]
        n = n_att if ax == "attack" else n_def
        val = measured[ax]
        # Measured first, and on its own scale — see AXIS_SCALE. The level does
        # not enter here at all, which is the point: a winger who does not
        # defend reads low in defence however good he is.
        fb = (fbref_def if ax == "defense"
              else fbref_att if ax == "attack"
              else (fbref_build if str(role) in BUILDUP_ROLES and fbref_build is not None
                    else fbref_crea))
        m = measured_axis(pos, ax, fb)
        if m is not None:
            out[ax] = m
            continue
        if fit and val is not None and own_ovr > 0 and n > 0:
            a, b, sd = fit[ax]
            z = (float(val) - (a + b * own_ovr)) / max(sd, 1e-6)
            cred = n / (n + AXIS_SHRINK_MATCHES)
            base += AXIS_SPREAD * cred * float(np.clip(z, -AXIS_Z_CAP, AXIS_Z_CAP))
        out[ax] = float(np.clip(soft_ceiling(base), 25.0, 99.0))
    out["defense"] = cap_defense(pos, out["defense"])
    if pos == "Goalkeeper":
        # A keeper has ONE rating and it is shot-stopping. The attacking and
        # creative axes were being computed for him out of symmetry and meant
        # nothing — every keeper in the pool sat on the same placeholder,
        # printed next to a real number as if it were one too.
        # No ceiling here: it exists to stop the DERIVED axes from all piling up
        # at 99, and a keeper's shot-stopping is not derived from his rating, it
        # IS his rating. Compressing it printed a 94 keeper with 91 in saves.
        out["gk"] = out["defense"] = float(np.clip(card_ovr, 25.0, 99.0))
        out["attack"] = out["creation"] = float("nan")
    else:
        out["gk"] = 50.0
    return out


def raw_axes(profile, pos: str, card_ovr: float) -> dict[str, float]:
    """The axes the SIMULATOR reads, left on the scale it was calibrated on.

    The squad-to-lambda bridge and the scorer weights were fitted against the
    measured distribution (median ~50, ceiling ~89). Feeding them the card-face
    numbers instead would hand them a squad unlike anything they were fitted on
    and flatten the gap between a striker and a full-back. So a card carries
    both: what it shows, and what the engine plays with.
    """
    if profile is not None and getattr(profile, "overall", 0) > 0:
        return {"attack": float(profile.offensive_strength),
                "defense": float(profile.defensive_strength),
                "creation": float(profile.creation_strength),
                "gk": float(profile.gk_strength)}
    # no profile: place him on the measured scale at the same standing his
    # rating gives him, so the engine sees one population, not two
    lo, hi = (40.0, 88.0) if pos == "Forward" else (42.0, 72.0)
    frac = float(np.clip((card_ovr - 60.0) / 35.0, 0.0, 1.0))
    v = lo + frac * (hi - lo)
    return {"attack": v if pos == "Forward" else 0.75 * v,
            "defense": v if pos in ("Defender", "Goalkeeper") else 0.8 * v,
            "creation": 0.85 * v, "gk": card_ovr if pos == "Goalkeeper" else 50.0}


# ── projection for players with no history at all ──────────────────────────────
def fit_projection(squads: pd.DataFrame, known: pd.Series, elo_map: dict) -> tuple:
    """Ridge on (club Elo, share of the season played, output) -> rating.

    Deliberately reported rather than hidden: cross-validated R^2 is 0.18 and the
    residual spread barely narrows (5.5 vs 6.2 rating points). Four matches of a
    new season simply do not tell you how good a player is. What it does capture
    is the one thing that genuinely separates an unknown Bayern squad member from
    an unknown Le Mans one — the level of the club that signed him.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    X, feats = projection_features(squads, elo_map)
    mask = known.notna() & X.notna().all(axis=1)
    if mask.sum() < 200:
        return None, feats, float("nan")
    m = Ridge(alpha=1.0).fit(X[mask], known[mask].astype(float))
    r2 = float(cross_val_score(Ridge(alpha=1.0), X[mask], known[mask].astype(float),
                               cv=5, scoring="r2").mean())
    return m, feats, r2


def projection_features(squads: pd.DataFrame, elo_map: dict) -> tuple[pd.DataFrame, list[str]]:
    d = squads.copy()
    d["club_elo"] = d["team"].map(elo_map)
    d["club_elo"] = d["club_elo"].fillna(d["club_elo"].median())
    mins = 90 * d["starts"] + 25 * (d["apps"] - d["starts"]).clip(lower=0)
    for c in ("goals", "shots", "assists"):
        d[c + "_p90"] = np.where(mins > 0, d[c] / mins.clip(lower=1) * 90.0, 0.0)
    d["start_share"] = d["starts"] / d["team_matches"].clip(lower=1)
    base = ["club_elo", "start_share", "squad_share", "goals_p90", "shots_p90",
            "assists_p90", "apps"]
    X = pd.get_dummies(d[base + ["pos_group"]], columns=["pos_group"]).astype(float)
    return X, list(X.columns)


# ── the three card families ────────────────────────────────────────────────────
def _level_index(path, col: str) -> NameIndex:
    """NameIndex of (level_z, n90) for a spread-preserving level column."""
    idx = NameIndex()
    if not path.exists():
        return idx
    d = pd.read_csv(path)
    if col not in d.columns:
        return idx
    for r in d.itertuples(index=False):
        v = getattr(r, col)
        n = float(getattr(r, "n90", 0) or 0)
        if pd.notna(v):
            idx.add(r.player, (float(v), n), rank=n)
    return idx


def buildup_level_index() -> NameIndex:
    """Playing-out quality as a spread-preserving level (for defensive roles)."""
    from mundialytics.identity.fbref_quality import load as _fbq
    q = _fbq()
    if q.empty:
        return NameIndex()
    d = pd.read_csv(DEFENSE_FBREF) if DEFENSE_FBREF.exists() else pd.DataFrame()
    pos = dict(zip(d.get("player", []), d.get("position", []))) if len(d) else {}
    q = q.copy()
    q["position"] = [pos.get(w, "Midfielder") for w in q["player"]]
    z = np.zeros(len(q))
    for c, w in (("pass_pct", 0.40), ("prog_passes_p90", 0.34), ("prog_dist_p90", 0.26)):
        v = pd.to_numeric(q[c], errors="coerce")
        zz = v.groupby(q["position"]).transform(
            lambda x: (x - x.mean()) / (x.std() if x.std() else 1.0))
        z += np.nan_to_num(zz.to_numpy()) * w
    q["bl_z"] = (z - z.mean()) / (z.std() or 1.0)
    idx = NameIndex()
    for r in q.itertuples(index=False):
        n = float(getattr(r, "fbq_90s", 0) or 0)
        idx.add(r.player, (float(r.bl_z), n), rank=n)
    return idx


# The overall maps the role-weighted level signal to a card through a percentile
# WITHIN POSITION — but of the CLEAN strength (the level z's), never the old
# padded pool. Per position because attacking output separates its elite far
# more than defending does in free data; an absolute curve then caps defenders
# and keepers ~4 points below forwards. Per-position gives every line its own
# 90s tier (the user's rule) while the clean signal keeps mid players honest
# (Cutrone is the 72nd percentile of forward strength, so ~78, not 88).
# A Z-SCORE within position, not a percentile. Percentile saturates the top —
# everyone above the 97th lands 89-92 in a flat wall, so Saka (level +1.85) tied
# Mbappé (+3.51) and the true best could not separate. A z keeps the elite's
# real spread: mapped through this curve, +2 -> 88, +2.7 -> 90.5, +3.5 -> 92, so
# only a player far above his position mean nears the cap and the very best sit
# clearly above the merely-very-good. Per position still, so each line has its
# own elite and defenders are judged against defenders.
STRENGTH_Z_XS = [-2.0, -1.0, 0.0, 1.0, 2.0, 2.7, 3.5]
STRENGTH_Z_YS = [56.0, 66.0, 74.0, 82.0, 88.0, 90.5, 92.0]
STRENGTH_CRED_90S = 8.0
STRENGTH_CAREER_W = 0.85   # measurement leads; career stabilises and covers gaps
UNMEASURED_MAX = 86.0      # no current measurement -> cannot reach the elite tier
# For keepers the shot-stopping score is nearly noise (it does not persist
# season to season, r +0.06), so which elite club trusts a keeper is a better
# quality signal than his save% — Courtois faces few, easy shots at Madrid and
# reads modest, Trapp over-performed at Paris FC. This much of a keeper's
# strength is his club's Elo; the rest is his shot-stopping.
KEEPER_CLUB_W = 0.72
# Defenders have the same disease in a milder form: the raw defensive stats
# reward VOLUME, so a relegation centre-back under constant siege files more
# tackles and clearances than Koundé does in a side that dominates the ball,
# and the weak-team wall crowds out Barcelona and Madrid. The defensive stats
# do carry real signal (unlike a keeper's save%), so the club only gets a
# minority share here — enough to surface the elite-club defenders the volume
# metric buries, not enough to erase what a defender actually does.
DEF_CLUB_W = 0.33


def build_current(src: dict, fits: dict, samples: dict, fb_def=None,
                  fb_att=None, fb_crea=None, fb_lvl=None, fb_build=None) -> pd.DataFrame:
    squads, model, roles = src["squads"], src["model"], src["roles"]
    elo_map = club_elo_map(squads, src["elo"])

    role_idx = NameIndex()
    for r in roles.itertuples(index=False):
        role_idx.add(r.player, r, rank=float(r.matches or 0))
    prof_idx = NameIndex()
    for p in model.profiles_.values():
        prof_idx.add(p.player, p, rank=float(p.matches or 0))

    gk_idx = gk_level_index()
    role_idx_measured = measured_roles()
    known, source, role_lbl, matches = [], [], [], []
    n_repick = 0
    prof_hit = []
    n_floor = n_gk_fixed = 0
    # the position he plays TODAY is evidence the name index does not have, and
    # with it a longer name drop is safe: "Bruno Fernandes" reaches "Bruno
    # Miguel Borges Fernandes" and "Gabriel Jesus" reaches "Gabriel Fernando de
    # Jesus", 44 squad members who were falling through to the projection.
    def _same_pos(pos):
        if str(pos) not in POS_ORDER:
            return None
        return lambda rec: _role_pos(str(getattr(rec, "role", ""))) == str(pos)

    def _same_pos_prof(pos):
        if str(pos) not in POS_ORDER:
            return None
        return lambda pr: str(getattr(pr, "position", "")) == str(pos)

    for who, pos in zip(squads["player"], squads["pos_group"]):
        r, kind = role_idx.lookup_detail(who, _same_pos(pos))
        p, pkind = prof_idx.lookup_detail(who, _same_pos_prof(pos))
        # a short-key match dropped a name part to get here; make it agree on
        # position before it lends a rating (see identity.current_squads)
        if (p is not None and pkind not in ("full", "confirmed")
                and pos in POS_ORDER and p.position != pos):
            p = None
        prof_hit.append(p)
        role_ok = r is not None and (kind in ("full", "confirmed") or pos not in POS_ORDER
                                     or _role_pos(str(r.role)) in (pos, ""))
        role = str(r.role) if role_ok else ""
        # a rating sitting exactly on its curve's bottom anchor is not a low
        # rating, it is the absence of one — over half the file is there
        floored = role_ok and is_floor_pinned(r.role, r.base_ovr)
        if floored:
            n_floor += 1
        if role_ok and not floored:
            known.append(float(r.base_ovr))
            source.append("rating")
            role_lbl.append(role)
            matches.append(int(r.matches or 0))
            continue
        # keepers: their own career score, matched by name INDEX. The deployed
        # join is a bare `.lower()` and misses 29% of them onto the 66.0 floor.
        gk = gk_idx.lookup(who) if str(pos) == "Goalkeeper" else None
        if gk is not None:
            n_gk_fixed += 1
            known.append(float(np.interp(float(gk), GKX, GKY)))
            source.append("portero")
            role_lbl.append(role or "Portero")
            matches.append(int(r.matches or 0) if role_ok else 0)
        elif p is not None and p.overall > 0 and not is_floor_pinned(
                getattr(p, "role", ""), p.overall):
            known.append(float(p.overall))
            source.append("perfil")
            role_lbl.append(str(getattr(p, "role", "")))
            matches.append(int(p.matches or 0))
        else:
            known.append(np.nan)
            source.append("")
            # the role label survives even when the rating does not: it still
            # shapes the axes, and it is the only thing the old row carried
            role_lbl.append(role or str(getattr(p, "role", "")) if p is not None else role)
            matches.append(int(r.matches or 0) if role_ok else 0)
    print(f"  ratings pegados al suelo de su curva (tratados como ausentes): {n_floor:,}")
    print(f"  porteros recuperados por indice de nombres: {n_gk_fixed:,}")

    # the role is picked off the same profile as the rating, so a blind profile
    # gives a guessed role. Re-pick it from what IS measured, where the stored
    # one was a guess and the measurement is decisive (see ROLE_REPICK_MARGIN).
    ns_pre = [samples.get(getattr(p, "player", ""), (0.0, 0.0)) if p is not None else (0.0, 0.0)
              for p in prof_hit]
    n_given = 0
    for i, (who, pos, n) in enumerate(zip(squads["player"], squads["pos_group"], ns_pre)):
        if str(pos) == "Goalkeeper":
            if not role_lbl[i]:
                role_lbl[i] = "Portero"
                n_given += 1
            continue
        hit = role_idx_measured.lookup(who)
        if not hit:
            continue
        if not role_lbl[i]:
            # no rating means no role; his own numbers can still say what kind
            # of player he is
            role_lbl[i] = str(hit)
            n_given += 1
        elif n[1] <= 0 and str(hit) != role_lbl[i]:
            role_lbl[i] = str(hit)
            n_repick += 1
    print(f"  roles reasignados desde la medicion (perfil ciego): {n_repick:,}")
    print(f"  roles asignados a cartas que no tenian: {n_given:,}")

    known = pd.Series(known, index=squads.index)
    model_r, feats, r2 = fit_projection(squads, known, elo_map)
    X, _ = projection_features(squads, elo_map)
    X = X.reindex(columns=feats, fill_value=0.0)
    proj = pd.Series(model_r.predict(X), index=squads.index) if model_r is not None \
        else pd.Series(known.mean(), index=squads.index)
    print(f"  proyeccion para jugadores sin historial: R2 {r2:.3f} "
          f"({int(known.isna().sum())} cartas proyectadas)")

    # The measured attacking signal moves the LEVEL, not only the card face, at
    # the size the temporal fit gave it: 4.4 rating points per standard
    # deviation, within position, and only for forwards and midfielders — see
    # ATT_LEVEL_POSITIONS for why a defender must not be moved by it.
    fa = [fb_axis(fb_att.lookup(who)) if fb_att is not None else None
          for who in squads["player"]]
    fl = [fb_axis(fb_lvl.lookup(who)) if fb_lvl is not None else None
          for who in squads["player"]]
    ns = [samples.get(getattr(p, "player", ""), (0.0, 0.0)) if p is not None else (0.0, 0.0)
          for p in prof_hit]
    # a rating built on a profile whose events were never parsed is a rating
    # built on placeholders — see COVERAGE_BLIND_POINTS. Projected cards do not
    # come from that profile at all, so they are not corrected.
    blind = pd.Series(
        [COVERAGE_BLIND_POINTS * _blind_share(p, m) if (k and n[1] <= 0) else 0.0
         for k, n, p, m in zip(known.notna(), ns, prof_hit, matches)],
        index=squads.index)
    print(f"  cartas sin eventos parseados en su perfil: {int((blind > 0).sum()):,} "
          f"(correccion media +{blind[blind > 0].mean():.1f}, maxima "
          f"+{blind.max():.1f})")
    bump = pd.Series(
        [0.0 if a is None or str(pos) not in ATT_LEVEL_POSITIONS
         else ATT_LEVEL_POINTS * float(np.clip(a[0], -ATT_LEVEL_Z_CAP, ATT_LEVEL_Z_CAP))
         for a, pos in zip(fl, squads["pos_group"])],
        index=squads.index)
    print(f"  nivel movido por el ataque medido: {(bump != 0).sum():,} cartas, "
          f"media {bump[bump != 0].mean():+.1f} pts, rango "
          f"{bump.min():+.1f}..{bump.max():+.1f}")
    ovr = (known.fillna(proj) + bump + blind).map(soft_level).clip(50, LEVEL_MAX)
    out = pd.DataFrame({
        "kind": "actual",
        "player": squads["player"].values,
        "display": [display_name(x) for x in squads["player"]],
        "club": squads["team"].values,
        "league": squads["competition"].values,
        "position": squads["pos_group"].values,
        "role": role_lbl,
        "season_label": "",
        "overall": ovr.round(1).values,
        "matches": matches,
        "source": np.where(known.notna(), source, "proyectado"),
    })
    fb = [fb_axis(fb_def.lookup(who)) if fb_def is not None else None
          for who in squads["player"]]
    # blend the club's level into the defense AXIS (which the simulator reads),
    # by the same DEF_CLUB_W that lifts the overall — both must agree, or the
    # card would show Koundé as elite while the sim still defends like the stat
    d_elo = pd.Series([float(elo_map.get(t, np.nan)) for t in squads["team"]],
                      index=squads.index)
    d_is = pd.Series(list(out["position"]), index=squads.index) == "Defender"
    club_pct = d_elo[d_is].rank(pct=True)
    fb = list(fb)
    for i, hit in enumerate(fb):
        idx = squads.index[i]
        if (d_is.loc[idx] and hit is not None and idx in club_pct.index
                and np.isfinite(club_pct.loc[idx])):
            p, n = hit
            fb[i] = ((1 - DEF_CLUB_W) * p + DEF_CLUB_W * float(club_pct.loc[idx]), n)
    fc = [fb_axis(fb_crea.lookup(who)) if fb_crea is not None else None
          for who in squads["player"]]
    fbu = [fb_axis(fb_build.lookup(who)) if fb_build is not None else None
           for who in squads["player"]]
    axes = [axes_for_card(fits, str(pos), rl, float(o), p, n[0], n[1], f, a, cc, bu)
            for p, pos, rl, o, n, f, a, cc, bu in zip(prof_hit, out["position"], out["role"],
                                                      out["overall"], ns, fb, fa, fc, fbu)]
    # ── the overall, rebuilt from the clean level signals (2026-09-07) ─────────
    # Every earlier layer (career+axes blend, spread-restore, percentile over a
    # pool padded with backups) inflated and re-ordered the top — Wissa 98.3
    # above Kane, Cutrone at the 96th percentile. The level z's are clean and
    # correctly ordered, so the overall is now a role-weighted blend of them
    # through one ABSOLUTE curve. Career only stabilises and covers the
    # unmeasured; measurement leads.
    att_lvl = _level_index(ATTACK_FBREF, "att_level_z")
    crea_lvl = _level_index(ATTACK_FBREF, "crea_level_z")
    def_lvl = _level_index(DEFENSE_FBREF, "def_level_z")
    build_lvl = buildup_level_index()
    career = pd.Series([float(x) for x in out["overall"]], index=squads.index)
    strength, mcred = [], []
    gk_mask = pd.Series(list(out["position"]), index=squads.index) == "Goalkeeper"
    gk_str = pd.Series(np.nan, index=squads.index)
    if gk_mask.any():
        gk_car = pd.Series([float(out["overall"][i]) for i in range(len(squads))],
                           index=squads.index)[gk_mask]
        gk_elo = pd.Series([float(elo_map.get(t, np.nan)) for t in squads["team"]],
                           index=squads.index)[gk_mask]
        zc = (gk_car - gk_car.mean()) / (gk_car.std() or 1.0)
        ze = ((gk_elo - gk_elo.mean()) / (gk_elo.std() or 1.0)).fillna(0.0)
        gk_str.loc[gk_mask] = (1 - KEEPER_CLUB_W) * zc + KEEPER_CLUB_W * ze
    def_mask = pd.Series(list(out["position"]), index=squads.index) == "Defender"
    def_club_z = pd.Series(0.0, index=squads.index)
    if def_mask.any():
        de = pd.Series([float(elo_map.get(t, np.nan)) for t in squads["team"]],
                       index=squads.index)[def_mask]
        def_club_z.loc[def_mask] = ((de - de.mean()) / (de.std() or 1.0)).fillna(0.0)

    for who, ps, rl in zip(squads["player"], out["position"], out["role"]):
        if str(ps) == "Goalkeeper":
            # a keeper's strength blends his shot-stopping with the level of the
            # club that trusts him (KEEPER_CLUB_W), because the raw stat is
            # nearly noise. This is what lifts Courtois over Trapp.
            gv = gk_str.loc[squads.index[len(strength)]]
            strength.append(float(gv) if np.isfinite(gv) else None)
            mcred.append(0.9); continue
        av = att_lvl.lookup(who)
        cv = crea_lvl.lookup(who)
        dv = def_lvl.lookup(who)
        if str(ps) == "Defender" and dv is not None and np.isfinite(dv[0]):
            cz = float(def_club_z.loc[squads.index[len(strength)]])
            dv = ((1 - DEF_CLUB_W) * dv[0] + DEF_CLUB_W * cz, dv[1])
        bv = build_lvl.lookup(who)
        crea_sig = bv if (str(rl) in BUILDUP_ROLES and bv is not None) else cv
        wa, wd, wc = (ROLE_AXIS_WEIGHT.get(str(rl))
                      or POS_AXIS_WEIGHT.get(str(ps), (0.4, 0.3, 0.3)))
        num = den = n90 = 0.0
        for w, v in ((wa, av), (wd, dv), (wc, crea_sig)):
            if v is not None and np.isfinite(v[0]):
                num += w * v[0]; den += w; n90 = max(n90, v[1])
        if den < 1e-9:
            strength.append(None); mcred.append(0.0)
        else:
            strength.append(num / den)
            mcred.append(min(n90 / (n90 + STRENGTH_CRED_90S), 1.0))
    # z-score of the clean strength within each position (measured only), then
    # the absolute curve. Small samples are shrunk toward the mean by mcred.
    st_ser = pd.Series([np.nan if x is None else float(x) for x in strength],
                       index=squads.index)
    mc_ser = pd.Series(mcred, index=squads.index)
    posser = pd.Series(list(out["position"]), index=squads.index)
    meas_over = pd.Series(np.nan, index=squads.index)
    for pg, idx in posser.groupby(posser).groups.items():
        vals = st_ser.loc[idx].dropna()
        if len(vals) < 10:
            continue
        mu, sd = float(vals.mean()), float(vals.std() or 1.0)
        for i in vals.index:
            z = (st_ser.loc[i] - mu) / sd * float(mc_ser.loc[i])
            meas_over.loc[i] = float(np.interp(z, STRENGTH_Z_XS, STRENGTH_Z_YS))
    bl_vals = []
    for i, (st, mc, car) in enumerate(zip(strength, mcred, career)):
        idx = squads.index[i]
        mo = meas_over.loc[idx]
        if st is None or not np.isfinite(mo):
            bl_vals.append(min(float(car), UNMEASURED_MAX)); continue
        w = STRENGTH_CAREER_W * mc
        bl_vals.append((1 - w) * float(car) + w * float(mo))
    bl = pd.Series(bl_vals, index=squads.index)
    n_meas = int(sum(x is not None for x in strength))
    print(f"  overall desde señales de nivel: {n_meas:,} cartas medidas, "
          f"curva absoluta (mediana provisional {bl.median():.0f})")

    # current form from this season's ESPN counting stats
    tm = pd.to_numeric(squads["team_matches"], errors="coerce").fillna(0)
    starts = pd.to_numeric(squads["starts"], errors="coerce").fillna(0)
    apps = pd.to_numeric(squads["apps"], errors="coerce").fillna(0)
    play_share = (starts / tm.clip(lower=1)).clip(0, 1)
    ramp = (tm / FORM_RAMP_MATCHES).clip(0, 1)
    penalty = (FORM_PEN * (bl - FORM_FLOOR).clip(lower=0) * (1 - play_share) * ramp)
    mins = 90 * starts + 25 * (apps - starts).clip(lower=0)
    cur_ga90 = np.where(mins > 0,
                        (pd.to_numeric(squads["goals"], errors="coerce").fillna(0)
                         + pd.to_numeric(squads["assists"], errors="coerce").fillna(0))
                        / mins.clip(lower=1) * 90.0, np.nan)
    cur = pd.Series(cur_ga90, index=squads.index)
    played = mins >= FORM_MIN_MINUTES
    prod_z = pd.Series(0.0, index=squads.index)
    for _pos, grp in squads.groupby("pos_group"):
        m = grp.index[played.loc[grp.index]]
        if len(m) < 15:
            continue
        v = cur.loc[m]
        prod_z.loc[m] = ((v - v.mean()) / (v.std() or 1.0)).clip(-1.5, 1.5)
    bonus = FORM_PROD_MAX * prod_z * ramp
    bl = bl - penalty + bonus
    print(f"  forma actual: penalizacion media {penalty[penalty > 0].mean():.1f} pts a "
          f"{int((penalty > 0.1).sum()):,} cartas; bonus max {bonus.max():.1f}")

    out["overall"] = [round(float(np.clip(soft_cap_actual(v), 50.0, ACTUAL_MAX)), 1) for v in bl]
    print(f"  escala honesta: mediana {np.median(out['overall']):.0f}, "
          f"90+ = {int((np.array(out['overall']) >= 90).sum()):,} cartas "
          f"({np.mean(np.array(out['overall']) >= 90) * 100:.1f}%)")
    axes = [axes_for_card(fits, str(pos), rl, float(o), p, n[0], n[1], f, a, cc, bu)
            for p, pos, rl, o, n, f, a, cc, bu in zip(prof_hit, out["position"], out["role"],
                                                      out["overall"], ns, fb, fa, fc, fbu)]
    out["def_source"] = ["fbref" if f is not None else "rol" for f in fb]
    out["att_source"] = ["fbref" if a is not None else "rol" for a in fa]
    out["crea_source"] = ["fbref" if x is not None else "rol" for x in fc]
    for k in ("attack", "defense", "creation", "gk"):
        out[k] = [round(float(a[k]), 1) for a in axes]
    rawax = [raw_axes(p, str(pos), float(o))
             for p, pos, o in zip(prof_hit, out["position"], out["overall"])]
    for k in ("attack", "defense", "creation", "gk"):
        out["raw_" + k] = [round(float(a[k]), 1) for a in rawax]
    out["n_att"] = [n[0] for n in ns]
    out["n_def"] = [n[1] for n in ns]
    return out


_ROLE_POS = {
    "Portero": "Goalkeeper",
    "Central stopper": "Defender", "Central de salida": "Defender",
    "Lateral ofensivo": "Defender", "Lateral defensivo": "Defender",
    "Destructor": "Midfielder", "Pivote organizador": "Midfielder",
    "Creador": "Midfielder", "Box-to-box": "Midfielder", "Mediapunta": "Midfielder",
    "Extremo": "Forward", "Extremo interior": "Forward", "Killer": "Forward",
    "Target man": "Forward", "Delantero completo": "Forward",
}


def _role_pos(role: str) -> str:
    return _ROLE_POS.get(role, "")


def season_teams() -> dict:
    """(player key, season code) -> the club he played most for that season."""
    if not UNDERSTAT_PM.exists():
        return {}
    pm = pd.read_csv(UNDERSTAT_PM, usecols=["player", "season", "team", "minutes"])
    pm["minutes"] = pd.to_numeric(pm["minutes"], errors="coerce").fillna(0)
    g = (pm.groupby(["player", "season", "team"])["minutes"].sum()
         .reset_index().sort_values("minutes", ascending=False)
         .drop_duplicates(["player", "season"]))
    out = {}
    for r in g.itertuples(index=False):
        out[(full_key(r.player), str(r.season))] = r.team
        out.setdefault((short_key(r.player), str(r.season)), r.team)
    return out


def build_primes(src: dict, fits: dict, samples: dict, fb_def=None,
                 fb_att=None) -> pd.DataFrame:
    seasons, roles, model = src["seasons"], src["roles"], src["model"]
    base = dict(zip(roles["player"].map(full_key), roles["base_ovr"]))

    s = seasons[seasons["matches"] >= PRIME_MIN_MATCHES].copy()
    start_year = pd.to_numeric(s["season"].astype(str).str.slice(0, 4), errors="coerce")
    start_year = start_year.where(start_year > 1900, 2000 + pd.to_numeric(
        s["season"].astype(str).str.slice(0, 2), errors="coerce"))
    s = s[~((s["era"] == "SB") & (start_year >= SB_ERA_CUTOFF))]
    s["fk"] = s["player"].map(full_key)
    s["base"] = s["fk"].map(base)
    s["base"] = s["base"].fillna(s["version_ovr"] - 4)
    s["pos"] = [_role_pos(str(r)) or "Midfielder" for r in s["role"]]
    q_season = s.groupby("pos")["version_ovr"].quantile(PRIME_SEASON_Q)
    q_base = s.groupby("pos")["base"].quantile(PRIME_CAREER_Q)
    q_exempt = s.groupby("pos")["base"].quantile(PRIME_CLUB_EXEMPT_Q)
    s = s[(s["version_ovr"] >= s["pos"].map(q_season))
          & (s["base"] >= s["pos"].map(q_base))]
    s["prime_ovr"] = (PRIME_SEASON_W * s["version_ovr"]
                      + (1 - PRIME_SEASON_W) * s["base"]).clip(PRIME_FLOOR, PRIME_MAX)

    teams = season_teams()
    s["club"] = [
        (r.team if isinstance(r.team, str) and r.team
         else teams.get((full_key(r.player), str(r.season)))
         or teams.get((short_key(r.player), str(r.season))) or "")
        for r in s.itertuples(index=False)
    ]
    elo_lut = club_strength(src["elo"])
    s["club_elo"] = [elo_lut.get(_norm(c), np.nan) for c in s["club"]]
    s = s[(s["club_elo"] >= PRIME_MIN_CLUB_ELO) | (s["base"] >= s["pos"].map(q_exempt))]

    # one card per player: the season standing furthest above his own level,
    # which is what "prime" means — not simply his highest-rated year
    s["lift"] = s["version_ovr"] - s["base"]
    s = s.sort_values(["prime_ovr", "lift"], ascending=False)
    s = dedupe_people(s, "player", keep=MAX_PRIMES_PER_PLAYER)
    s = pd.concat([grp.head(PRIMES_PER_POSITION.get(str(pos), 40))
                   for pos, grp in s.groupby("pos", sort=False)], ignore_index=True)
    prof_idx = NameIndex()
    for p in model.profiles_.values():
        prof_idx.add(p.player, p, rank=float(p.matches or 0))

    rows = []
    for r in s.itertuples(index=False):
        pos = str(r.pos)
        club = r.club
        prof = prof_idx.lookup(r.player)
        n_att, n_def = samples.get(getattr(prof, "player", ""), (0.0, 0.0))
        yr = str(r.season)
        fdef = fb_axis(fb_def.lookup(r.player), yr) if fb_def is not None else None
        fatt = fb_axis(fb_att.lookup(r.player), yr) if fb_att is not None else None
        ax = axes_for_card(fits, pos, str(r.role), float(r.prime_ovr), prof, n_att, n_def,
                           fdef, fatt)
        rax = raw_axes(prof, pos, float(r.prime_ovr))
        rows.append({
            **{"raw_" + k: round(float(v), 1) for k, v in rax.items()},
            "n_att": n_att, "n_def": n_def,
            "def_source": "fbref" if fdef is not None else "rol",
            "att_source": "fbref" if fatt is not None else "rol",
            "kind": "prime", "player": r.player, "display": display_name(r.player),
            "club": str(club).title(), "league": "", "position": pos, "role": str(r.role),
            "season_label": _season_label(r.season), "overall": round(float(r.prime_ovr), 1),
            "matches": int(r.matches), "source": "temporada",
            **{k: round(float(v), 1) for k, v in ax.items()},
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        gk = out["position"] == "Goalkeeper"
        out.loc[gk, "gk"] = out.loc[gk, "overall"]
        out.loc[gk, "defense"] = out.loc[gk, "overall"]
    return out


def build_curated_primes(src: dict, fits: dict) -> pd.DataFrame:
    """The marquee seasons the data cannot produce, stated as a design choice.

    Defenders and keepers are the hole. The per-season rating file has NO
    goalkeeper rows at all, and for defenders the only rows from 2015/16 onward
    are StatsBomb's, whose coverage is a release policy rather than a sample —
    so the one Van Dijk season on record is his 2015/16 at Southampton, not the
    Liverpool year everybody means. Rather than ship "Van Dijk 15/16 ·
    Southampton" as if it were his peak, those cards are authored here and
    marked `curado`, exactly as the icons whose sample is too thin already are.
    """
    if not CURATED_PRIMES.exists():
        return pd.DataFrame()
    # season as text: "0708" read as a number is 708, and the card said so
    cur = pd.read_csv(CURATED_PRIMES, dtype={"season": str})
    rows = []
    for r in cur.itertuples(index=False):
        pos = str(r.position)
        ovr = float(r.ovr)
        ax = axes_for_card(fits, pos, str(r.role), ovr, None, 0.0, 0.0)
        rax = raw_axes(None, pos, ovr)
        rows.append({
            "kind": "prime", "player": str(r.player), "display": display_name(str(r.player)),
            "club": str(r.club), "league": "", "position": pos, "role": str(r.role),
            "season_label": _season_label(r.season), "overall": round(ovr, 1),
            "matches": 0, "source": "curado", "n_att": 0.0, "n_def": 0.0,
            **{k: round(float(v), 1) for k, v in ax.items()},
            **{"raw_" + k: round(float(v), 1) for k, v in rax.items()},
        })
    return pd.DataFrame(rows)


def build_icons(src: dict, fits: dict, samples: dict) -> pd.DataFrame:
    if not ICONS.exists():
        return pd.DataFrame()
    cur = pd.read_csv(ICONS)
    model, roles, seasons = src["model"], src["roles"], src["seasons"]
    squads = src["squads"]

    active = NameIndex()
    for t, p in zip(squads["team"], squads["player"]):
        active.add(p, t)
    prof_idx = NameIndex()
    for p in model.profiles_.values():
        prof_idx.add(p.player, p, rank=float(p.matches or 0))
    role_by_name = dict(zip(roles["player"], zip(roles["role"], roles["peak_ovr"], roles["matches"])))
    best_season = (seasons[seasons["matches"] >= 15].sort_values("version_ovr", ascending=False)
                   .drop_duplicates("player"))
    peak_season = dict(zip(best_season["player"], best_season["version_ovr"]))

    rows, skipped = [], []
    for r in cur.itertuples(index=False):
        name = str(r.player)
        club_now = active.lookup(name)
        if club_now:
            skipped.append(f"{name} (en activo en {club_now})")
            continue
        p = prof_idx.lookup(name)
        role, peak, m = role_by_name.get(name, ("", np.nan, 0))
        ovr = float(r.ovr) if pd.notna(r.ovr) else float(
            max(peak or 0, peak_season.get(name, 0), ICON_MIN))
        ovr = float(np.clip(ovr, ICON_MIN, 99.0))
        pos = str(r.position) if isinstance(r.position, str) and r.position else (
            p.position if p is not None else "Midfielder")
        n_att, n_def = samples.get(getattr(p, "player", ""), (0.0, 0.0))
        if pd.notna(r.attack):
            ax = {"attack": soft_ceiling(float(r.attack)),
                  "defense": cap_defense(pos, soft_ceiling(float(r.defense))),
                  "creation": soft_ceiling(float(r.creation)),
                  "gk": soft_ceiling(float(r.gk)) if pd.notna(r.gk) else 50.0}
            stat_src = "curado"
        else:
            ax = axes_for_card(fits, pos, str(r.role or role), ovr, p, n_att, n_def)
            stat_src = "datos" if p is not None else "estimado"
        if pos == "Goalkeeper":
            ax["gk"] = ax["defense"] = ovr
            ax["attack"] = ax["creation"] = float("nan")
        rax = raw_axes(p, pos, ovr)
        rows.append({
            **{"raw_" + k: round(float(v), 1) for k, v in rax.items()},
            "n_att": n_att, "n_def": n_def,
            "kind": "icono", "player": name,
            "display": display_name(name),
            "club": str(r.icon_club), "league": "", "position": pos,
            "role": str(r.role) if isinstance(r.role, str) and r.role else role,
            "season_label": "", "overall": round(ovr, 1),
            "matches": int(m or 0), "source": stat_src,
            **{k: round(float(v), 1) for k, v in ax.items()},
        })
    if skipped:
        print(f"  iconos descartados por seguir en activo: {'; '.join(skipped)}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    src = load_sources()
    fits = fit_axis_residuals(src["model"])
    samples = axis_samples()
    fb_def, n_fb = defense_index()
    fb_att, n_fa = attack_index()
    fb_crea, n_fc = creation_index()
    fb_lvl, _ = attack_level_index()
    fb_build, n_fb2 = buildup_index()
    print(f"  defensa medida por FBref: {n_fb:,} jugadores", flush=True)
    print(f"  ataque medido por FBref:  {n_fa:,} jugadores", flush=True)

    print("cartas ACTUAL...", flush=True)
    print(f"  creacion medida por FBref: {n_fc:,} jugadores", flush=True)
    print(f"  construccion medida por FBref: {n_fb2:,} jugadores", flush=True)
    cur = build_current(src, fits, samples, fb_def, fb_att, fb_crea, fb_lvl, fb_build)
    print("cartas PRIME...", flush=True)
    pri = build_primes(src, fits, samples, fb_def, fb_att)
    hand = build_curated_primes(src, fits)
    if not hand.empty:
        # a curated season wins over a measured one for the same player: it is
        # there precisely because the measured one is the wrong year
        pri = pri[~pri["player"].map(person_tokens).isin(
            set(hand["player"].map(person_tokens)))]
        pri = pd.concat([pri, hand], ignore_index=True)
    print("cartas ICONO...", flush=True)
    ico = build_icons(src, fits, samples)

    cards = pd.concat([cur, pri, ico], ignore_index=True)
    cards.insert(0, "card_id", [f"{k[:3]}_{i:05d}" for i, k in enumerate(cards["kind"])])
    for c in ("raw_attack", "raw_defense", "raw_creation", "raw_gk", "n_att", "n_def"):
        if c not in cards.columns:
            cards[c] = np.nan
    cards[["n_att", "n_def"]] = cards[["n_att", "n_def"]].fillna(0.0)
    for c in ("def_source", "att_source", "crea_source"):
        if c not in cards.columns:
            cards[c] = "rol"
        cards[c] = cards[c].fillna("rol")
    # ── make the SIMULATOR axes follow the card ───────────────────────────────
    # raw_* is what the engine plays with (who scores, who assists), and it used
    # to come straight from the old StatsBomb profiles — so Haaland's raw attack
    # was 68 while his card shows 90, and the sim rewarded whoever StatsBomb
    # happened to cover. The FACE now reflects the player honestly, so raw_* is
    # re-ordered to match it. The trick keeps the sim's calibration untouched:
    # per position the SET of raw values is left exactly as it was (the bridge
    # and scorer weights were fitted on that distribution), only reassigned so
    # the highest-face player gets the highest raw. Order follows the card;
    # scale does not move.
    def _match_raw_to_face(df, face_col, raw_col):
        out = df[raw_col].astype(float).copy()
        for _pos, idx in df.groupby("position").groups.items():
            sub = df.loc[idx]
            m = sub.index[sub[face_col].notna() & sub[raw_col].notna()]
            if len(m) < 5:
                continue
            ranks = sub.loc[m, face_col].rank(method="first").astype(int).to_numpy() - 1
            sorted_raw = np.sort(sub.loc[m, raw_col].astype(float).to_numpy())
            out.loc[m] = sorted_raw[ranks]
        return out

    for face_col, raw_col in (("attack", "raw_attack"), ("defense", "raw_defense"),
                              ("creation", "raw_creation"), ("gk", "raw_gk")):
        cards[raw_col] = _match_raw_to_face(cards, face_col, raw_col)
    _fw = cards[cards["position"] == "Forward"]
    print(f"  raw del simulador realineado con la cara: "
          f"corr(overall, raw_attack) delanteros "
          f"{_fw['overall'].corr(_fw['raw_attack']):+.3f}")

    cols = ["card_id", "kind", "player", "display", "club", "league", "position", "role",
            "season_label", "overall", "attack", "defense", "creation", "gk",
            "raw_attack", "raw_defense", "raw_creation", "raw_gk",
            "n_att", "n_def", "matches", "source", "def_source", "att_source",
            "crea_source"]
    cards = cards[cols]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cards.to_csv(args.out, index=False)

    print(f"\nESCRITO {args.out}")
    for kind, grp in cards.groupby("kind"):
        print(f"  {kind:7} {len(grp):5d} cartas · media {grp['overall'].mean():.1f} · "
              f"max {grp['overall'].max():.0f} · min {grp['overall'].min():.0f}")
    print(f"  fuentes: {cards['source'].value_counts().to_dict()}")
    fb_share = (cards["def_source"] == "fbref").mean()
    print(f"  eje defensivo medido (FBref): {fb_share:.1%} de las cartas")
    for pos in ("Defender", "Goalkeeper"):
        sub = cards[cards["position"] == pos]
        if len(sub):
            print(f"     {pos:11} {(sub['def_source'] == 'fbref').mean():.1%}")
    print(f"  eje ofensivo medido (FBref):  {(cards['att_source'] == 'fbref').mean():.1%} "
          "de las cartas")
    for pos in ("Forward", "Midfielder"):
        sub = cards[cards["position"] == pos]
        if len(sub):
            print(f"     {pos:11} {(sub['att_source'] == 'fbref').mean():.1%}")
    fw = cards[cards["position"] == "Forward"]
    print(f"  tope de defensa en delanteros: max {fw['defense'].max():.1f} "
          f"(limite {POSITION_DEF_CAP['Forward']:.0f}, afecta a "
          f"{(fw['defense'] >= POSITION_DEF_CAP['Forward'] - 1e-6).sum()} cartas)")
    a = cards[cards["kind"] == "actual"]
    thin = a.groupby("club").size()
    thin = thin[thin < 16]
    print(f"  clubes con menos de 16 cartas: {len(thin)}"
          + (f" ({', '.join(thin.index)})" if len(thin) else ""))
    for pos in POS_ORDER:
        n = int((a["position"] == pos).sum())
        print(f"    {pos:11} {n:5d}")


if __name__ == "__main__":
    main()
