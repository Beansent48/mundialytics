#!/usr/bin/env python3
"""A real defensive rating per player, from FBref's published season tables.

WHAT IT REPLACES. The card's defensive axis came from StatsBomb, where
`defense_creation_matches` is zero for 62% of defenders — Lewis Dunk had 299
Premier League matches and a duel win rate of exactly 0.50, the neutral
placeholder. The axis was measuring exposure, not defending.

QUALITY, NOT VOLUME. This keeps the distinction the rating model was rebuilt
around in 2026-07: tackles and pressures are workrate stats that run BACKWARDS
against team quality, because a side under siege racks them up out of
necessity. So the score leans on the rate stats FBref publishes and the anchor
model always wanted but never had for most players:

    challenge %   share of dribblers he actually tackles
    aerial %      share of aerial duels won
    dribbled past times beaten per 90, inverted
    blocks, interceptions, clearances per 90, at low weight — still
                  volume-confounded, kept as texture rather than as the verdict

Each is turned into a within-position z-score so a centre back is compared with
centre backs, then blended. The output is a z-score, not a rating: the card
builder uses it as "how far from what his overall already implies", which is the
only thing a measurement should be allowed to say once the rating exists.

Run: .venv/Scripts/python.exe scripts/build_player_defense_fbref.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SRC = ROOT / "data/external/advanced/fbref_kaggle"
OUT = ROOT / "data/processed/player_defense_fbref.csv"

# weight, and whether a high value is GOOD
SIGNALS = {
    "challenge_pct": (0.30, True),
    "aerial_pct": (0.26, True),
    "dribbled_past_p90": (0.18, False),
    "interceptions_p90": (0.10, True),
    "blocks_p90": (0.08, True),
    "clearances_p90": (0.08, True),
}
MIN_90S = 5.0        # under five full matches the rates are noise
# Every rate here is a ratio with its own denominator, and the denominators are
# small: a defender who was dribbled at three times and won all three shows a
# 100% challenge rate. Each rate is shrunk toward its position mean by its OWN
# count, which is what the first pass got wrong — it gated on minutes played and
# then trusted a 3-of-3 as if it were a 60-of-90.
SHRINK_CHALLENGES = 25.0
SHRINK_AERIALS = 30.0
POSITIONS = {"GK": "Goalkeeper", "DF": "Defender", "MF": "Midfielder", "FW": "Forward"}


def _pos(raw: object) -> str:
    """FBref writes "DF,MF"; the first token is where he mostly plays."""
    tok = str(raw).replace("-", ",").split(",")[0].strip().upper()
    return POSITIONS.get(tok, "Midfielder")


def load_2425() -> pd.DataFrame:
    p = SRC / "fbref_players_2425.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p, low_memory=False)
    if "Tkl%" not in d.columns:
        return pd.DataFrame()
    n90 = pd.to_numeric(d.get("90s_stats_defense", d.get("90s")), errors="coerce")
    out = pd.DataFrame({
        "player": d["Player"], "season": "2425", "club": d.get("Squad"),
        "position": d["Pos"].map(_pos), "n90": n90,
        "challenge_pct": pd.to_numeric(d["Tkl%"], errors="coerce"),
        "n_challenges": pd.to_numeric(d.get("Att_stats_defense"), errors="coerce"),
        "aerial_pct": pd.to_numeric(d.get("Won%"), errors="coerce"),
        "n_aerials": (pd.to_numeric(d.get("Won"), errors="coerce").fillna(0)
                      + pd.to_numeric(d.get("Lost_stats_misc"), errors="coerce").fillna(0)),
        "dribbled_past": pd.to_numeric(d.get("Tkld"), errors="coerce"),
        "interceptions": pd.to_numeric(d.get("Int"), errors="coerce"),
        "blocks": pd.to_numeric(d.get("Blocks_stats_defense", d.get("Blocks")), errors="coerce"),
        "clearances": pd.to_numeric(d.get("Clr"), errors="coerce"),
    })
    return out


def load_2324() -> pd.DataFrame:
    dp, mp = SRC / "2324_Defense.csv", SRC / "2324_Misc.csv"
    if not dp.exists():
        return pd.DataFrame()
    d = pd.read_csv(dp)
    out = pd.DataFrame({
        "player": d["player"], "season": "2324", "club": d.get("club"),
        "position": d["position"].map(_pos),
        "n90": pd.to_numeric(d["minutes_90s"], errors="coerce"),
        "challenge_pct": pd.to_numeric(d.get("challenge_tackles_pct"), errors="coerce"),
        "n_challenges": pd.to_numeric(d.get("challenges"), errors="coerce"),
        "aerial_pct": np.nan,
        "n_aerials": np.nan,
        "dribbled_past": pd.to_numeric(d.get("challenges_lost"), errors="coerce"),
        "interceptions": pd.to_numeric(d.get("intr"), errors="coerce"),
        "blocks": pd.to_numeric(d.get("blocks"), errors="coerce"),
        "clearances": pd.to_numeric(d.get("clearances", d.get("clr")), errors="coerce"),
    })
    if mp.exists():
        m = pd.read_csv(mp)
        won = pd.to_numeric(m.get("aerials_won"), errors="coerce").fillna(0)
        lost = pd.to_numeric(m.get("aerials_lost"), errors="coerce").fillna(0)
        pct = pd.to_numeric(m.get("aerials_won_pct"), errors="coerce")
        out["aerial_pct"] = out["player"].map(dict(zip(m["player"], pct)))
        out["n_aerials"] = out["player"].map(dict(zip(m["player"], won + lost)))
    return out


def load_2526() -> pd.DataFrame:
    """25/26 defending, from the SofaScore half of the scouting merge.

    FBref publishes no defensive tables for 25/26 yet, so the duel and tackle
    rates come from SofaScore. Its win PERCENTAGES need no denominator; the
    counts are divided by SofaScore's own minutes. Like the Understat block, a
    transferred player carries the same season totals on each club row, so he is
    collapsed to one. Provider definitions differ from FBref's (tackle success
    vs challenge success) but each season is standardised within itself before
    blending, so only the RANK within the season crosses over — see build().
    """
    p = SRC / "scouting_2526.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p, low_memory=False)
    d = d.sort_values("90s_", ascending=False).drop_duplicates("player")
    n90 = pd.to_numeric(d["minutesPlayed"], errors="coerce") / 90.0
    aw = pd.to_numeric(d["aerialDuelsWon"], errors="coerce").fillna(0)
    al = pd.to_numeric(d["aerialLost"], errors="coerce").fillna(0)
    blocks = (pd.to_numeric(d["blockedShots"], errors="coerce").fillna(0)
              + pd.to_numeric(d["outfielderBlocks"], errors="coerce").fillna(0))
    return pd.DataFrame({
        "player": d["player"], "season": "2526", "club": d["team"],
        "position": d["pos_"].map(_pos), "n90": n90,
        "challenge_pct": pd.to_numeric(d["tacklesWonPercentage"], errors="coerce"),
        "n_challenges": pd.to_numeric(d["tackles"], errors="coerce"),
        "aerial_pct": pd.to_numeric(d["aerialDuelsWonPercentage"], errors="coerce"),
        "n_aerials": aw + al,
        "dribbled_past": pd.to_numeric(d["dribbledPast"], errors="coerce"),
        "interceptions": pd.to_numeric(d["interceptions"], errors="coerce"),
        "blocks": blocks,
        "clearances": pd.to_numeric(d["clearances"], errors="coerce"),
    })


def align_positions(d: pd.DataFrame) -> pd.DataFrame:
    """Use the position the SQUAD says, not the one FBref filed him under.

    They disagree for 184 players — 12% of those in both — and not at random:
    FBref files an attacking wing back under DF. Every z here is standardised
    within position, so Franck Honorat's attacking score was computed against
    DEFENDERS, where 0.06 non-penalty xG per 90 is above average: he came out
    +1.12 while Olise, on 0.38, came out +0.67 among forwards. Standardising
    against the wrong population is not a small error, it inverts the answer.
    """
    from mundialytics.identity.current_squads import NameIndex, load_current_squads
    sq = load_current_squads()
    if sq.empty or "position" not in d.columns:
        return d
    idx = NameIndex()
    for who, pos in zip(sq["player"], sq["pos_group"]):
        if str(pos) in {"Goalkeeper", "Defender", "Midfielder", "Forward"}:
            idx.add(who, str(pos), rank=1.0)
    d = d.copy()
    hit = [idx.lookup(w) for w in d["player"]]
    d["position"] = [str(h) if h else str(f) for h, f in zip(hit, d["position"])]
    print(f"  posicion alineada con la plantilla en {sum(h is not None for h in hit):,} "
          f"de {len(d):,} jugadores")
    return d


def build() -> pd.DataFrame:
    frames = [f for f in (load_2526(), load_2425(), load_2324()) if not f.empty]
    if not frames:
        raise SystemExit(
            "faltan los volcados de FBref — ejecuta scripts/download_fbref_kaggle.py")
    full = pd.concat(frames, ignore_index=True)
    full = full[pd.to_numeric(full["n90"], errors="coerce").fillna(0) >= MIN_90S].copy()
    full = align_positions(full)
    ns = {k: int(v) for k, v in full["season"].value_counts().items()}

    for col in ("dribbled_past", "interceptions", "blocks", "clearances"):
        full[col + "_p90"] = full[col] / full["n90"].clip(lower=0.5)

    # shrink each ratio toward its (season, position) mean by its own count, so
    # a 3-of-3 challenge cannot read as elite AND a provider whose numbers run
    # hot is shrunk against ITS OWN season's mean, never a pooled one
    for stat, n_col, k in (("challenge_pct", "n_challenges", SHRINK_CHALLENGES),
                           ("aerial_pct", "n_aerials", SHRINK_AERIALS)):
        n = pd.to_numeric(full.get(n_col), errors="coerce").fillna(0.0)
        prior = full.groupby(["season", "position"])[stat].transform("mean")
        cred = n / (n + k)
        full[stat] = (cred * full[stat].fillna(prior) + (1 - cred) * prior)

    # composite z WITHIN (season, position): a centre back judged against the
    # centre backs OF THAT SEASON. This is what makes the providers mix safely —
    # FBref's challenge% (23/24, 24/25) and SofaScore's tackle% (25/26) never
    # share a scale, only the within-season RANK carries across the blend.
    num = pd.Series(0.0, index=full.index)
    den = pd.Series(0.0, index=full.index)
    key = [full["season"], full["position"]]
    for stat, (w, higher_better) in SIGNALS.items():
        col = stat if stat in full.columns else stat.replace("_p90", "") + "_p90"
        if col not in full.columns:
            continue
        z = full.groupby(key)[col].transform(
            lambda s: (s - s.mean()) / (s.std() if s.std() else 1.0))
        z = z if higher_better else -z
        m = z.notna().astype(float)
        num = num + z.fillna(0.0) * w * m
        den = den + w * m
    full["_comp"] = num / den.replace(0.0, np.nan)
    full["def_signals"] = (den / sum(w for w, _ in SIGNALS.values())).round(2)

    # display rates: one row per player, recency-blended (25/26 leads)
    from build_player_attack_fbref import blend_level, recency_collapse
    full["league"] = ""      # blend_level carries a league; the defence axis has none
    rate_cols = ["challenge_pct", "aerial_pct", "dribbled_past_p90",
                 "interceptions_p90", "blocks_p90", "clearances_p90",
                 "n_challenges", "n_aerials"]
    rate_cols = [c for c in rate_cols if c in full.columns]
    both = int(full["player"].duplicated().sum())
    d = recency_collapse(full, rate_cols)
    print(f"  recencia 25/26>24/25>23/24 — filas/temporada {ns} ({both:,} fusionados)")

    # the SCORE: the per-season composite, recency-blended then re-standardised.
    bl = blend_level(full, "_comp")
    z = d["player"].map(dict(zip(bl["player"], bl["lvl"]))).astype(float)
    cred = d["n90"] / (d["n90"] + 10.0)
    d["def_z"] = np.clip((z * cred).fillna(0.0), -6, 6)
    # spread-preserving level twin — re-standardised to unit so the card can
    # combine it with the attacking level on one scale
    zc = (z - z.mean()) / (z.std() or 1.0)
    d["def_level_z"] = np.clip((zc * cred).fillna(0.0), -6, 6)
    d["def_signals"] = d["player"].map(
        full.groupby("player")["def_signals"].max()).fillna(0.0).round(2)

    cols = ["player", "season", "club", "position", "n90", "def_z", "def_level_z", "def_signals",
            "challenge_pct", "n_challenges", "aerial_pct", "n_aerials",
            "dribbled_past_p90", "interceptions_p90", "blocks_p90", "clearances_p90"]
    return d[cols].round(3).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = build()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"ESCRITO {args.out}")
    print(f"  {len(out):,} jugadores · {out['season'].value_counts().to_dict()}")
    print(f"  señales disponibles de media: {out['def_signals'].mean():.0%}")
    print()
    for pos in ("Defender", "Midfielder"):
        sub = out[out["position"] == pos]
        if sub.empty:
            continue
        best = sub.nlargest(6, "def_z")
        print(f"  mejores {pos.lower()}s por z defensiva:")
        for r in best.itertuples(index=False):
            print(f"     {r.player[:26]:28} {r.def_z:+.2f}  "
                  f"duelo {r.challenge_pct or float('nan'):.0f}%  "
                  f"aereo {r.aerial_pct if pd.notna(r.aerial_pct) else float('nan'):.0f}%  "
                  f"({r.club})")


if __name__ == "__main__":
    main()
