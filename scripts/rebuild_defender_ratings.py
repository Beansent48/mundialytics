#!/usr/bin/env python3
"""Rebuild the DEFENDER rating from measurement alone.

WHY. 34% of the rating pool sits at exactly 66.0 and 61% of defenders at 67 or
below, because the defensive axis they are computed from is the neutral 0.50
placeholder for two thirds of them. The symptom: Cubarsí (170 matches), Dunk
(598), Tarkowski (661), Mosquera (204) and Marquinhos (543) all come out between
68.7 and 68.8. Five different centre backs, one number. That is not a rating
that is low, it is a rating that was never computed.

WHAT WAS TRIED FIRST, AND FAILED. The obvious "purely from data" design is to
fit the weights against what a defence is for — conceding fewer goals. Three
targets, all recorded here because the negative result is the useful part:

  club-season goals conceded     R² 0.28, but blocks correlate **+0.50** and
                                 clearances **+0.40** — the wrong sign. A side
                                 under siege blocks and clears more, so the fit
                                 was learning team weakness.
  on-pitch xG against            R² 0.17, same two stats driving it, same reason.
  the same, within club          R² **-0.000**. Every correlation collapses
                                 (max |r| 0.12).
  On-Off (with him vs without)   R² **-0.014**.

Once the team is differenced out, a defender's measured actions do not predict
what his team concedes. So there is no outcome to fit weights against, and any
rating claiming to be validated against results would be team strength wearing a
player's name.

WHAT THE DATA DOES SUPPORT. It says clearly which stats to THROW AWAY: blocks,
clearances, tackles and interceptions carry the wrong sign at club level, which
is the volume confound this project already documented in 2026-07. What is left
are the true rates — share of dribblers stopped, share of aerials won, how often
he is beaten — and those carry the right sign, weakly. So:

  * only rate stats, never volume;
  * each weighted by |r| against club goals conceded, so the weighting is
    measured rather than chosen;
  * each shrunk toward its position mean by ITS OWN denominator;
  * the 0-100 scale is a quantile map onto the ratings of players who were
    actually measured — no anchor curve, no blanket uplift.

The honest claim is therefore narrow and worth stating plainly: this ORDERS
defenders by measured defending and spreads them over a real range. It is not
validated against results, because nothing here can be.

Run: .venv/Scripts/python.exe scripts/rebuild_defender_ratings.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.identity.current_squads import NameIndex  # noqa: E402

DEF_FBREF = ROOT / "data/processed/player_defense_fbref.csv"
DEF_RAW = ROOT / "data/external/advanced/fbref_kaggle"
ROLES = ROOT / "data/processed/player_ratings_roles.csv"
CLUBELO = ROOT / "data/processed/clubelo_local.csv"
OUT = ROOT / "data/processed/player_ratings_defenders.csv"
# Leagues with fewer rated clubs than this cannot anchor their own level.
MIN_CLUBS_FOR_LEAGUE = 4

DEFENDER_ROLES = {"Central stopper", "Central de salida",
                  "Lateral ofensivo", "Lateral defensivo"}

# |r| against club-season goals conceded, measured in the experiment recorded in
# the docstring. Volume stats are absent on purpose: theirs pointed the other
# way. These are transcribed from the measurement and must stay that way —
# dribbled_past was carrying 0.100 here against a measured 0.052, which is a
# number I chose while claiming the file did not contain any.
RATE_WEIGHTS = {"aerial_pct": 0.205, "challenge_pct": 0.138, "dribbled_past_p90": 0.052}
# Sign, also measured: being beaten more often went WITH conceding more (+0.052),
# so it counts against him. Weakly, which is why its weight is the smallest.
DRIBBLED_IS_BAD = True

def elo_per_rating_point() -> float:
    """How many Elo points one rating point is worth, from the real clubs.

    Already measured elsewhere and reused rather than re-chosen: the squad-to-Elo
    line in squadlab/champions.py is fitted on 96 clubs' own best elevens, and
    its slope is exactly this conversion. Inverting it turns a league's Elo gap
    into rating points — so the Premier League being 347 Elo above Serie B
    becomes a defined number of rating points, not a knob.

    The first attempt fitted the league weight against the OLD ratings and got
    0.14, i.e. almost nothing. That was circular: those ratings are mostly the
    plateau this script exists to remove, so of course they barely move with
    league. A rating system cannot be asked how much it knows about leagues when
    the answer is "nothing".
    """
    try:
        from mundialytics.statistical_core.squadlab.cards import load_cards
        from mundialytics.statistical_core.squadlab.champions import squad_elo_scale

        elo = pd.read_csv(CLUBELO).dropna(subset=["club", "elo"])
        scale = squad_elo_scale(None, load_cards(), dict(zip(elo["club"], elo["elo"])))
        if scale.slope_mean > 1.0:
            return float(scale.slope_mean)
    except Exception:
        pass
    return 29.4      # the value that fit measured, kept as a documented fallback


# A rating plateau is a rating that was never computed. These are the values the
# old pipeline parks unmeasured players on; they define who to REPLACE, and they
# are excluded from the target distribution so the new scale is not fitted to
# the very artefact it exists to remove.
def plateaus(series: pd.Series, min_share: float = 0.01) -> set[float]:
    counts = series.round(1).value_counts(normalize=True)
    return set(counts[counts >= min_share].index)


def league_strength() -> dict[str, float]:
    """Mean ClubElo of each league's clubs — how hard its duels actually are.

    Without this the first build put Serie B, MLS and Segunda defenders at 92:
    winning 71% of your duels in the Championship is not the same act as winning
    71% in the Premier League, and a z-score pooled across seventeen leagues
    cannot tell the difference. The correction is measured, not chosen — it is
    the Elo of the clubs those players actually faced.
    """
    from mundialytics.statistical_core.competition.european import make_resolver

    if not CLUBELO.exists():
        return {}
    elo = pd.read_csv(CLUBELO).dropna(subset=["club", "elo"])
    lut = dict(zip(elo["club"].astype(str), elo["elo"].astype(float)))
    res = make_resolver(list(lut))
    out = {}
    for path, col_lg, col_club in ((DEF_RAW / "2324_Defense.csv", "league", "club"),
                                   (DEF_RAW / "fbref_players_2425.csv", "Comp", "Squad")):
        if not path.exists():
            continue
        d = pd.read_csv(path, low_memory=False)
        for lg, g in d.groupby(col_lg):
            vals = [lut[res(c)] for c in g[col_club].dropna().unique() if res(c)]
            if len(vals) >= MIN_CLUBS_FOR_LEAGUE:
                out.setdefault(str(lg), []).append(float(np.mean(vals)))
    return {k: float(np.mean(v)) for k, v in out.items()}


def level_beta(fb: pd.DataFrame, ref_sd: float) -> float:
    """How much of a rating the level he plays at is worth, measured.

    Fitted on the ratings the old pipeline DID compute — the plateau rows are
    excluded, so this is not the circular fit that returned 0.14 earlier: those
    players have real, varied ratings and real, varied clubs.
    """
    roles = pd.read_csv(ROLES)
    bad = plateaus(roles["base_ovr"])
    idx = NameIndex()
    for r in roles.itertuples(index=False):
        if round(float(r.base_ovr), 1) in bad:
            continue
        idx.add(r.player, float(r.base_ovr), rank=float(r.matches or 0))
    known = pd.DataFrame({"ovr": [idx.lookup(p) for p in fb["player"]],
                          "elo": fb["level_elo"].to_numpy()}).dropna()
    if len(known) < 200 or known["elo"].std() == 0:
        return 0.0
    slope = np.polyfit(known["elo"], known["ovr"], 1)[0]      # rating pts per Elo
    per_sd = slope * known["elo"].std()                        # pts per sd of level
    print(f"  ajustado sobre {len(known):,} defensas con valoración real: "
          f"{per_sd:.1f} puntos por desviación de nivel")
    return float(np.clip(per_sd / ref_sd, 0.0, 1.5))


def club_elo(clubs: pd.Series) -> pd.Series:
    """Each club's own Elo — the level that player actually plays at."""
    from mundialytics.statistical_core.competition.european import make_resolver

    if not CLUBELO.exists():
        return pd.Series(np.nan, index=clubs.index)
    elo = pd.read_csv(CLUBELO).dropna(subset=["club", "elo"])
    lut = dict(zip(elo["club"].astype(str), elo["elo"].astype(float)))
    res = make_resolver(list(lut))
    cache: dict[str, float] = {}
    out = []
    for c in clubs.astype(str):
        if c not in cache:
            hit = res(c)
            cache[c] = float(lut[hit]) if hit else np.nan
        out.append(cache[c])
    return pd.Series(out, index=clubs.index)


def league_of(fb: pd.DataFrame) -> pd.Series:
    """Each row's league, from whichever dump it came out of."""
    lg = {}
    for path, col_lg, col_club in ((DEF_RAW / "2324_Defense.csv", "league", "club"),
                                   (DEF_RAW / "fbref_players_2425.csv", "Comp", "Squad")):
        if not path.exists():
            continue
        d = pd.read_csv(path, low_memory=False)
        for club, l in zip(d[col_club], d[col_lg]):
            lg.setdefault(str(club), str(l))
    return fb["club"].astype(str).map(lg)


def build() -> pd.DataFrame:
    if not DEF_FBREF.exists():
        raise SystemExit("falta player_defense_fbref.csv — ejecuta "
                         "scripts/build_player_defense_fbref.py")
    fb = pd.read_csv(DEF_FBREF)
    roles = pd.read_csv(ROLES)

    fb = fb[fb["position"] == "Defender"].copy()
    have = [c for c in RATE_WEIGHTS if c in fb.columns]
    if not have:
        raise SystemExit("el volcado de FBref no trae ninguna tasa utilizable")

    fb["league"] = league_of(fb)
    strength = league_strength()
    fb["league_elo"] = fb["league"].map(strength)
    # A league mean is a blunt instrument: it put 43 of the top 50 in the
    # Premier League, because that division's Elo is lifted by its depth and
    # every one of its players then inherits the same large step. The club's OWN
    # Elo is the level he actually plays at, we have it for all of them, and it
    # replaces the step with a continuum.
    # ...but only inside the leagues ClubElo actually covers. Argentina's Arsenal
    # de Sarandí resolved onto Arsenal FC and put three of its defenders in the
    # world top five.
    european = fb["league"].isin(set(strength))
    fb["club_elo"] = club_elo(fb["club"]).where(european)
    fb["level_elo"] = fb["club_elo"].fillna(fb["league_elo"])

    # composite: each rate standardised WITHIN ITS LEAGUE, signed so higher is
    # better, weighted by how strongly it moved goals conceded at club level
    z_total = np.zeros(len(fb))
    w_total = 0.0
    for stat in have:
        col = fb[stat].astype(float)
        z = col.groupby(fb["league"]).transform(
            lambda s: (s - s.mean()) / (s.std() if s.std() else 1.0))
        z = z.fillna((col - col.mean()) / (col.std() or 1.0))
        if stat == "dribbled_past_p90" and DRIBBLED_IS_BAD:
            z = -z
        w = RATE_WEIGHTS[stat]
        z_total += np.nan_to_num(z.to_numpy()) * w
        w_total += w
    fb["def_score"] = z_total / w_total

    # credibility: a defender seen for ten full matches has earned less movement
    # than one seen for thirty
    cred = fb["n90"] / (fb["n90"] + 10.0)
    fb["def_score"] = fb["def_score"] * cred

    # A league we cannot place, we cannot compare against: better to leave those
    # players to the role baseline than to park them on the median and pretend.
    unrated = fb["level_elo"].isna().sum()
    if unrated:
        print(f"  descartados {unrated} sin Elo de club ni de liga (no se pueden situar)")
    fb = fb[fb["level_elo"].notna()].copy()
    print(f"  nivel por club: {fb['club_elo'].notna().mean():.0%} de las filas; "
          f"el resto cae a la media de su liga")

    # ── the target distribution: what a MEASURED rating looks like ────────────
    bad = plateaus(roles["base_ovr"])
    measured = roles[~roles["base_ovr"].round(1).isin(bad)]
    ref = np.sort(measured["base_ovr"].to_numpy())
    print(f"  mesetas descartadas del patrón: {sorted(round(b, 1) for b in bad)}")
    print(f"  patrón = {len(ref):,} valoraciones que sí se calcularon "
          f"(p10 {np.quantile(ref, .1):.0f} · mediana {np.median(ref):.0f} · "
          f"p99 {np.quantile(ref, .99):.0f})")

    # Combine BEFORE mapping, not after. Adding the league on top of a value
    # already drawn from an all-leagues distribution counts the division twice —
    # it put five Premier League centre backs on the 96 ceiling together.
    per_pt = elo_per_rating_point()
    ref_sd = float(np.std(ref)) or 1.0
    # How much the level he plays at is worth is itself measured: regress the
    # ratings that WERE computed on their clubs' Elo. Converting the full Elo
    # gap instead made the club worth 27 rating points, so five Arsenal
    # defenders outranked every other club's best.
    beta = level_beta(fb, ref_sd)
    lz = (fb["level_elo"] - fb["level_elo"].mean()) / (fb["level_elo"].std() or 1.0)
    league_z = beta * lz
    combined = fb["def_score"] + league_z
    print(f"  peso del nivel del club: beta = {beta:.2f} sd "
          f"({(league_z.max() - league_z.min()) * ref_sd:.1f} puntos entre el mejor y el peor)")
    pct = combined.rank(pct=True)
    fb["new_ovr"] = np.quantile(ref, pct.clip(0.001, 0.999)).round(1)

    # keep his best-sampled season, then attach what he had before
    fb = fb.sort_values("n90", ascending=False).drop_duplicates("player")
    idx = NameIndex()
    for r in roles.itertuples(index=False):
        idx.add(r.player, (float(r.base_ovr), str(r.role), float(r.matches or 0)),
                rank=float(r.matches or 0))
    hit = [idx.lookup(p) for p in fb["player"]]
    fb["old_ovr"] = [h[0] if h else np.nan for h in hit]
    fb["role"] = [h[1] if h else "" for h in hit]
    fb["career_matches"] = [h[2] if h else 0 for h in hit]

    cols = ["player", "club", "season", "role", "n90", "career_matches",
            "def_score", "old_ovr", "new_ovr",
            "challenge_pct", "aerial_pct", "dribbled_past_p90"]
    return fb[[c for c in cols if c in fb.columns]].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = build()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    known = out.dropna(subset=["old_ovr"])
    print(f"\nESCRITO {args.out}  ({len(out):,} defensas)")
    print(f"  antes: mediana {known['old_ovr'].median():.1f}  sd {known['old_ovr'].std():.1f}  "
          f"rango {known['old_ovr'].min():.0f}-{known['old_ovr'].max():.0f}")
    print(f"  ahora: mediana {known['new_ovr'].median():.1f}  sd {known['new_ovr'].std():.1f}  "
          f"rango {known['new_ovr'].min():.0f}-{known['new_ovr'].max():.0f}")
    plateau = known["old_ovr"].round(1).value_counts(normalize=True).iloc[0]
    print(f"  el valor más repetido pasa de {plateau:.0%} de los defensas a "
          f"{known['new_ovr'].round(1).value_counts(normalize=True).iloc[0]:.0%}")
    print()
    print("  top 12 por la nueva valoración:")
    for r in out.nlargest(12, "new_ovr").itertuples(index=False):
        print(f"     {str(r.player)[:24]:26} {r.new_ovr:5.1f}  (antes "
              f"{r.old_ovr if pd.notna(r.old_ovr) else float('nan'):.1f})  "
              f"duelo {r.challenge_pct:.0f}%  aéreo {r.aerial_pct:.0f}%  ({r.club})")


if __name__ == "__main__":
    main()
