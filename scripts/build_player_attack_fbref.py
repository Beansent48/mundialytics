#!/usr/bin/env python3
"""An attacking rating per player, with weights validated ACROSS SEASONS.

WHY THIS ONE CAN BE VALIDATED AND THE DEFENSIVE ONE COULD NOT. A defender's job
leaves no player-level trace in free data: once the team is differenced out, his
measured actions predict nothing about what the team concedes (R² -0.000, see
rebuild_defender_ratings.py). An attacker's job does. Goals and assists are his
own, and the question "does what he did last season predict what he does next
season" is answerable with the two seasons on disk.

So the weights here are not transcribed correlations, they are fitted on a real
out-of-sample task: features from 2023/24, target = goals + assists per 90 in
2024/25, for the players who appear in both. Nothing about the current season is
used to score the current season, which is the whole point.

WHAT GOES IN. The rate stats FBref publishes that describe attacking output
rather than volume of touches: non-penalty xG per 90, expected assists per 90,
shot- and goal-creating actions per 90, chance quality (npxG per shot), and
finishing over expectation. Penalties are stripped — a penalty taker is not a
better forward, he is the designated taker.

Run: .venv/Scripts/python.exe scripts/build_player_attack_fbref.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SRC = ROOT / "data/external/advanced/fbref_kaggle"
OUT = ROOT / "data/processed/player_attack_fbref.csv"

FEATURES = ["npxg_p90", "xag_p90", "sca_p90", "gca_p90", "npxg_per_shot", "finishing_p90"]
# The CREATION half, fitted the same way against the same kind of target. There
# was no measured creation axis at all: a card's creation was its rating plus
# what the role implies, so Olise — 0.46 expected assists per 90, among the best
# in Europe — read like any other winger of his rating. These are the features
# that describe making the chance rather than taking it, and the target is next
# season's assists per 90.
# `assists_p90` is here on purpose. The attacking half already contains the
# goals a player actually scored, hidden in plain sight — `finishing_p90` is
# (goals − penalties − npxG), so npxG + finishing IS his real scoring — but the
# creation half had no equivalent and ran entirely on expected numbers. Adding
# the assists he actually made costs nothing in prediction (−0.001) and moves
# how well the axis DESCRIBES the season he played from +0.779 to **+0.807**,
# which is what a card is for. Last season's assists predicting next season's
# is a lagged feature, not leakage: they are different seasons.
CREA_FEATURES = ["xag_p90", "sca_p90", "gca_p90", "kp_p90", "prog_passes_p90",
                 "assists_p90"]
MIN_90S = 5.0
CRED_90S = 10.0
# The clip used to be ±3 and it was binding on exactly the players it matters
# for: Honorat and Olise both came out at +3.00 and +2.98 on creation, so the
# card could not tell 0.46 goals+assists per 90 from 1.04. Eleven players hit
# it. Credibility already shrinks a thin sample toward zero; the clip only
# exists to stop a freak rate running away, so it sits far out of the way.
Z_CLIP = 6.0
# ── league strength ──────────────────────────────────────────────────────────
# Standardising within league removes the scoring environment, which is right
# for the SHAPE of a player and exactly wrong for his LEVEL: it erases how hard
# the league is. Measured on the 728 players in both seasons, the effect is not
# small and it runs BACKWARDS — those measured outside the big five score +0.27
# standard deviations against +0.08 for those inside, and then produce 0.169
# goals+assists per 90 the next season against 0.212. We were rating the lower
# leagues higher and they went on to do less.
#
# So a fraction of the league's own strength is added back. ClubElo gives it —
# 1,817 for the Premier League down to 1,449 for Ligue 2, 368 points of spread —
# and the weight is the fitted ratio of the two coefficients: one standard
# deviation of league strength is worth 0.15 of measurement.
# 0.15 was what a regression asked for, and that regression cannot see the
# effect properly: its sample is players who ended up in the big five, so the
# lower-league group in it is the selected few who earned a move. Calibrated
# instead against the group difference that IS measurable — those measured
# outside produce 0.043 goals+assists per 90 less the following season, which is
# 0.20 of a standard deviation — the weight comes out at 0.24.
LEAGUE_W = 0.24


# 25/26 is the season that describes a player NOW, so it leads; 24/25 and 23/24
# are backup that keep a longer career in view (and cover whoever the recent
# season missed). A player with both seasons used to keep whichever had more
# minutes, so Gyökeres' 23/24 Sporting explosion counted in full two years
# later. Blended instead with the recent season weighted far more; a player with
# only an old season keeps it but his current form (added in the card layer) is
# what then decides whether he still rates.
SEASON_RECENCY = {"2526": 1.0, "2425": 0.25, "2324": 0.08}
BLEND_RATE_COLS = ["npxg_p90", "xag_p90", "sca_p90", "gca_p90", "npxg_per_shot",
                   "finishing_p90", "goals_p90", "kp_p90", "prog_passes_p90",
                   "assists_p90", "ga_p90"]


def recency_collapse(d: pd.DataFrame, rate_cols: list[str],
                     carry: tuple = ("club", "league", "position")) -> pd.DataFrame:
    """One row per player: rate_cols recency-weighted, `carry` from the recent row.

    Shared by the attacking and defensive builders so both weight 24/25 far
    above 23/24 the same way.
    """
    d = d.copy()
    d["_sw"] = d["season"].astype(str).map(SEASON_RECENCY).fillna(0.5)
    d["_w"] = d["_sw"] * pd.to_numeric(d["n90"], errors="coerce").clip(lower=0.1)
    rows = []
    for player, g in d.groupby("player", sort=False):
        g = g.sort_values("season")            # 2324 first, 2425 last
        recent = g.iloc[-1]
        w = g["_w"].to_numpy(dtype=float)
        row = {"player": player, "season": recent["season"]}
        for c in carry:
            if c in g.columns:
                row[c] = recent[c]
        for c in rate_cols:
            if c not in g.columns:
                continue
            v = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(v) & (w > 0)
            row[c] = float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan
        row["n90"] = float((g["_sw"] * pd.to_numeric(g["n90"], errors="coerce")).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def recency_blend(d: pd.DataFrame) -> pd.DataFrame:
    """One row per player, the recent season weighted far above the old one."""
    return recency_collapse(d, BLEND_RATE_COLS)


def league_elo() -> dict:
    """FBref league name -> the mean ClubElo of the clubs it actually contains."""
    from mundialytics.statistical_core.competition.european import make_resolver
    path = ROOT / "data/processed/clubelo_local.csv"
    if not path.exists():
        return {}
    elo = pd.read_csv(path)
    res = make_resolver(list(elo["club"]))
    lut = dict(zip(elo["club"], elo["elo"]))
    cache: dict[str, float] = {}

    def celo(c):
        c = str(c)
        if c not in cache:
            h = res(c)
            cache[c] = float(lut[h]) if h else np.nan
        return cache[c]

    rows = []
    sh = pd.read_csv(SRC / "2324_Shooting.csv")
    rows.append(sh[["club", "league"]])
    w = pd.read_csv(SRC / "fbref_players_2425.csv", low_memory=False)
    rows.append(w[["Squad", "Comp"]].rename(columns={"Squad": "club", "Comp": "league"}))
    d = pd.concat(rows, ignore_index=True).drop_duplicates(["club", "league"])
    d["elo"] = [celo(c) for c in d["club"]]
    g = (d.dropna(subset=["elo"]).groupby("league")
         .agg(elo=("elo", "mean"), clubs=("club", "nunique")))
    return {k: float(v) for k, v in g.loc[g["clubs"] >= 5, "elo"].items()}
POSITIONS = {"GK": "Goalkeeper", "DF": "Defender", "MF": "Midfielder", "FW": "Forward"}


def _pos(raw: object) -> str:
    tok = str(raw).replace("-", ",").split(",")[0].strip().upper()
    return POSITIONS.get(tok, "Midfielder")


def load_2425() -> pd.DataFrame:
    d = pd.read_csv(SRC / "fbref_players_2425.csv", low_memory=False)
    n90 = pd.to_numeric(d["90s"], errors="coerce")
    npxg = pd.to_numeric(d["npxG"], errors="coerce")
    goals = pd.to_numeric(d["Gls"], errors="coerce")
    pens = pd.to_numeric(d["PK"], errors="coerce").fillna(0)
    return pd.DataFrame({
        "player": d["Player"], "club": d["Squad"], "league": d["Comp"], "season": "2425",
        "position": d["Pos"].map(_pos), "n90": n90,
        "birth_year": pd.to_numeric(d["Born"], errors="coerce"),
        "npxg_p90": npxg / n90.clip(lower=0.5),
        "xag_p90": pd.to_numeric(d["xAG"], errors="coerce") / n90.clip(lower=0.5),
        "sca_p90": pd.to_numeric(d["SCA90"], errors="coerce"),
        "gca_p90": pd.to_numeric(d["GCA90"], errors="coerce"),
        "npxg_per_shot": pd.to_numeric(d["npxG/Sh"], errors="coerce"),
        "finishing_p90": (goals - pens - npxg) / n90.clip(lower=0.5),
        # the goals he actually scored, penalties out. Expected metrics predict
        # better and describe worse, and a card is a description.
        "goals_p90": (goals - pens) / n90.clip(lower=0.5),
        "kp_p90": pd.to_numeric(d["KP"], errors="coerce") / n90.clip(lower=0.5),
        "prog_passes_p90": pd.to_numeric(d["PrgP"], errors="coerce") / n90.clip(lower=0.5),
        "assists_p90": pd.to_numeric(d["Ast"], errors="coerce") / n90.clip(lower=0.5),
        "ga_p90": ((goals - pens) + pd.to_numeric(d["Ast"], errors="coerce")) / n90.clip(lower=0.5),
        # possession craft: keeping the ball and beating a man — see possession_level
        "pass_pct": pd.to_numeric(d["Cmp%"], errors="coerce"),
        "dribble_pct": pd.to_numeric(d["Succ%"], errors="coerce"),
        "dribble_att": pd.to_numeric(d["Att_stats_possession"], errors="coerce"),
        "dispossessed_p90": pd.to_numeric(d["Dis"], errors="coerce") / n90.clip(lower=0.5),
    })


def load_2324() -> pd.DataFrame:
    one = lambda d: d.sort_values("minutes_90s", ascending=False).drop_duplicates(  # noqa: E731
        subset=["player", "club"])
    sh = one(pd.read_csv(SRC / "2324_Shooting.csv"))
    pa = one(pd.read_csv(SRC / "2324_Passing.csv"))
    gs = one(pd.read_csv(SRC / "2324_GSC.csv"))
    n90 = pd.to_numeric(sh["minutes_90s"], errors="coerce")
    npxg = pd.to_numeric(sh["npxg"], errors="coerce")
    goals = pd.to_numeric(sh["goals"], errors="coerce")
    pens = pd.to_numeric(sh["pens_scored"], errors="coerce").fillna(0)
    xag = dict(zip(pa["player"], pd.to_numeric(pa["xAG"], errors="coerce")))
    ast = dict(zip(pa["player"], pd.to_numeric(pa["assists"], errors="coerce")))
    sca = dict(zip(gs["player"], pd.to_numeric(gs["sca_per90"], errors="coerce")))
    gca = dict(zip(gs["player"], pd.to_numeric(gs["gca_per90"], errors="coerce")))
    return pd.DataFrame({
        "player": sh["player"], "club": sh["club"], "league": sh["league"], "season": "2324",
        "position": sh["position"].map(_pos), "n90": n90,
        "birth_year": np.nan,          # 23/24 dump carries only age; recent seasons cover it
        "npxg_p90": npxg / n90.clip(lower=0.5),
        "xag_p90": sh["player"].map(xag) / n90.clip(lower=0.5),
        "sca_p90": sh["player"].map(sca),
        "gca_p90": sh["player"].map(gca),
        "npxg_per_shot": pd.to_numeric(sh["npxg_per_shot"], errors="coerce"),
        "finishing_p90": (goals - pens - npxg) / n90.clip(lower=0.5),
        "goals_p90": (goals - pens) / n90.clip(lower=0.5),
        "kp_p90": sh["player"].map(dict(zip(pa["player"], pd.to_numeric(pa["KP"], errors="coerce")))) / n90.clip(lower=0.5),
        "prog_passes_p90": sh["player"].map(dict(zip(pa["player"], pd.to_numeric(pa["prog_passes"], errors="coerce")))) / n90.clip(lower=0.5),
        "assists_p90": sh["player"].map(ast) / n90.clip(lower=0.5),
        "ga_p90": ((goals - pens) + sh["player"].map(ast)) / n90.clip(lower=0.5),
        # 23/24 has pass completion but no take-on table here; the recent seasons
        # carry the dribble half (this is the least-weighted season anyway)
        "pass_pct": sh["player"].map(dict(zip(pa["player"], pd.to_numeric(pa["passes_pct"], errors="coerce")))),
        "dribble_pct": np.nan,
        "dribble_att": np.nan,
        "dispossessed_p90": np.nan,
    })


# the 25/26 dump is a different provenance — a FBref + Understat + SofaScore
# merge (chuongtrinh on Kaggle) — because FBref's own 25/26 tables are not
# published yet. Its league labels differ; map them to the FBref names the rest
# of the pipeline (and league_elo) already speak.
LEAGUE_2526 = {
    "ENG-Premier League": "eng Premier League",
    "ESP-La Liga": "es La Liga",
    "FRA-Ligue 1": "fr Ligue 1",
    "GER-Bundesliga": "de Bundesliga",
    "ITA-Serie A": "it Serie A",
}


def load_2526() -> pd.DataFrame:
    """25/26 attacking rates, from the Understat half of the scouting merge.

    Only the basics FBref itself publishes for 25/26 would be — goals, assists,
    shots — so the expected numbers (npxG, xAG) come from Understat, which is
    internally consistent (npxG pairs with npGoals, xA with assists). SCA, GCA
    and progressive passes have no equivalent in this dump; they are left blank
    and the older FBref season carries them. Provider scale does not matter: the
    level standardises each season within itself before blending.
    """
    p = SRC / "scouting_2526.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p, low_memory=False)
    # The Understat block is a per-player SEASON TOTAL, duplicated onto every
    # FBref club-row a transferred player has. Dividing those totals by a single
    # club's minutes invents rates — Malen's Villa row read 2.31 npxG/90. So the
    # denominator is Understat's OWN minutes, and the player is collapsed to one
    # row (keeping the club he played most, which decides the league term).
    d["_n90u"] = pd.to_numeric(d["minutes_under"], errors="coerce") / 90.0
    d["_fb90"] = pd.to_numeric(d["90s_"], errors="coerce")
    d = d.sort_values("_fb90", ascending=False).drop_duplicates("player")
    den = d["_n90u"].clip(lower=0.5)
    npg = pd.to_numeric(d["np_goals_under"], errors="coerce")
    npxg = pd.to_numeric(d["np_xg_under"], errors="coerce")
    ast = pd.to_numeric(d["assists_under"], errors="coerce")
    shots = pd.to_numeric(d["shots_under"], errors="coerce")
    return pd.DataFrame({
        "player": d["player"], "club": d["team"],
        "league": d["league"].map(LEAGUE_2526).fillna(d["league"]),
        "season": "2526", "position": d["pos_"].map(_pos), "n90": d["_n90u"],
        "birth_year": pd.to_numeric(d["born_"], errors="coerce"),
        "npxg_p90": npxg / den,
        "xag_p90": pd.to_numeric(d["xa_under"], errors="coerce") / den,
        "sca_p90": np.nan,          # no SCA here -> the 24/25 FBref season carries it
        "gca_p90": np.nan,
        "npxg_per_shot": npxg / shots.replace(0, np.nan),
        "finishing_p90": (npg - npxg) / den,
        "goals_p90": npg / den,
        "kp_p90": pd.to_numeric(d["key_passes_under"], errors="coerce") / den,
        "prog_passes_p90": np.nan,  # no PrgP here -> the 24/25 FBref season carries it
        "assists_p90": ast / den,
        "ga_p90": (npg + ast) / den,
        # possession craft from SofaScore (percentages need no denominator)
        "pass_pct": pd.to_numeric(d["accuratePassesPercentage"], errors="coerce"),
        "dribble_pct": pd.to_numeric(d["successfulDribblesPercentage"], errors="coerce"),
        "dribble_att": pd.to_numeric(d["totalContest"], errors="coerce"),
        "dispossessed_p90": pd.to_numeric(d["dispossessed"], errors="coerce") / den,
    })


def season_std_composite(full: pd.DataFrame, wdict: dict) -> pd.Series:
    """A per-row level composite, standardised WITHIN each season.

    Each feature is turned into a z-score within its own (season, league), the
    weighted mean is taken over the features present in that row (weights
    renormalised, so a season missing SCA is not penalised), then the composite
    is centred per (season, position) and divided by that SEASON's own spread.

    Standardising each season within itself is what makes mixing providers safe:
    Understat's npxG runs ~20% hotter than FBref's, but once each season is on
    its own z-scale that offset cannot lift whoever happened to be measured in
    the hotter year. The spread is kept (divide by the pooled, not the
    within-position, std) so an elite still reads elite across positions.
    """
    num = pd.Series(0.0, index=full.index)
    den = pd.Series(0.0, index=full.index)
    key = [full["season"], full["league"]]
    for stat, w in wdict.items():
        if stat not in full.columns:
            continue
        col = pd.to_numeric(full[stat], errors="coerce")
        z = col.groupby(key).transform(lambda s: (s - s.mean()) / (s.std() if s.std() else 1.0))
        m = z.notna().astype(float)
        num = num + z.fillna(0.0) * w * m
        den = den + w * m
    comp = num / den.replace(0.0, np.nan)
    centred = comp - comp.groupby([full["season"], full["position"]]).transform("mean")
    sd = centred.groupby(full["season"]).transform(lambda s: s.std() or 1.0)
    return centred / sd


def blend_level(full: pd.DataFrame, sig_col: str) -> pd.DataFrame:
    """Recency-weighted per-player mean of a per-season level signal.

    One row per player: the signal averaged over his seasons weighted by
    recency x minutes (so 25/26 leads and a thin recent season defers to a full
    older one), plus the recency-weighted minutes for credibility and the most
    recent league/position.
    """
    sw = full["season"].astype(str).map(SEASON_RECENCY).fillna(0.3)
    n90 = pd.to_numeric(full["n90"], errors="coerce")
    w = sw * n90.clip(lower=0.1)
    g = pd.DataFrame({"player": full["player"], "v": pd.to_numeric(full[sig_col], errors="coerce"),
                      "w": w, "swn": sw * n90, "season": full["season"].astype(str),
                      "league": full["league"], "position": full["position"]})
    rows = []
    for player, gg in g.groupby("player", sort=False):
        gg = gg.sort_values("season")
        v, ww = gg["v"].to_numpy(float), gg["w"].to_numpy(float)
        ok = np.isfinite(v) & (ww > 0)
        rows.append({
            "player": player,
            "lvl": float(np.average(v[ok], weights=ww[ok])) if ok.any() else np.nan,
            "n90": float(np.nansum(gg["swn"].to_numpy(float))),
            "league": gg["league"].iloc[-1], "position": gg["position"].iloc[-1]})
    return pd.DataFrame(rows)


# Possession craft — keeping the ball and beating a man with a short dribble.
# This is what a deep playmaker (Pedri, Rodri, Frenkie de Jong) is elite at and
# the chance-creation numbers miss entirely: their value is retention and
# progression, not assists. Quality over volume — dribble SUCCESS rate, shrunk
# by attempts so a two-of-two does not read as elite, never dribbles attempted,
# or Cherki's wasteful high-volume dribbling (48% success) would score here too.
POSS_SIGNALS = {"pass_pct": (0.45, True), "dribble_pct": (0.35, True),
                "dispossessed_p90": (0.20, False)}
POSS_SHRINK_DRIBBLE = 20.0


def possession_level(full: pd.DataFrame) -> pd.Series:
    """Per-player possession/control level (player -> z), standardised per season
    then recency-blended, on the same scale as att_level_z."""
    f = full.copy()
    if "dribble_pct" in f.columns:                 # shrink the rate by attempts
        n = pd.to_numeric(f.get("dribble_att"), errors="coerce").fillna(0.0)
        prior = f.groupby(["season", "position"])["dribble_pct"].transform("mean")
        cr = n / (n + POSS_SHRINK_DRIBBLE)
        f["dribble_pct"] = cr * f["dribble_pct"].fillna(prior) + (1 - cr) * prior
    num = pd.Series(0.0, index=f.index)
    den = pd.Series(0.0, index=f.index)
    key = [f["season"], f["position"]]
    for stat, (w, hb) in POSS_SIGNALS.items():
        if stat not in f.columns:
            continue
        z = f.groupby(key)[stat].transform(lambda s: (s - s.mean()) / (s.std() if s.std() else 1.0))
        z = z if hb else -z
        m = z.notna().astype(float)
        num = num + z.fillna(0.0) * w * m
        den = den + w * m
    f["_poss"] = num / den.replace(0.0, np.nan)
    bl = blend_level(f, "_poss")
    zc = (bl["lvl"] - bl["lvl"].mean()) / (bl["lvl"].std() or 1.0)
    cr = bl["n90"] / (bl["n90"] + CRED_90S)
    return pd.Series(dict(zip(bl["player"], np.clip(zc.fillna(0.0) * cr, -Z_CLIP, Z_CLIP))))


def fit_weights(a: pd.DataFrame, b: pd.DataFrame, features=None,
                target: str = "ga_p90") -> tuple[dict, float, int]:
    """Weights from 23/24 features against a 24/25 outcome, for players in both."""
    FEATURES_ = features or FEATURES
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    prev = a[a["n90"] >= MIN_90S].drop_duplicates("player").set_index("player")
    nxt = b[b["n90"] >= MIN_90S].drop_duplicates("player").set_index("player")
    both = prev.index.intersection(nxt.index)
    X = prev.loc[both, FEATURES_].astype(float)
    y = nxt.loc[both, target].astype(float)
    ok = X.notna().all(axis=1) & y.notna()
    X, y = X[ok], y[ok]
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
    r2 = float(cross_val_score(pipe, X, y, cv=5, scoring="r2").mean())
    pipe.fit(X, y)
    coef = dict(zip(FEATURES_, pipe[-1].coef_))
    return coef, r2, int(len(X))


def align_positions(d: pd.DataFrame) -> pd.DataFrame:
    """Use the position the SQUAD says, not the one FBref filed him under.

    They disagree for 184 players — 12% of those in both — and not at random:
    FBref files an attacking wing back under DF. Every z here is standardised
    within position, so Franck Honorat's attacking score was computed against
    DEFENDERS, where 0.06 non-penalty xG per 90 is above average: he came out
    +1.12 while Olise, on 0.38, came out +0.67 among forwards. Standardising
    against the wrong population is not a small error, it inverts the answer.
    """
    from mundialytics.identity.current_squads import NameIndex, full_key, load_current_squads
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
    # curated position overrides win over BOTH the squad and FBref — for players
    # a name mismatch leaves unmatched (Grimaldo, a left back FBref files as MF)
    # or that the feeds simply get wrong. Hand-maintained from football knowledge.
    ov = ROOT / "data/curated/position_overrides.csv"
    n_ov = 0
    if ov.exists():
        omap = {}
        for r in pd.read_csv(ov).itertuples(index=False):
            if str(getattr(r, "position", "")) in {"Goalkeeper", "Defender", "Midfielder", "Forward"}:
                omap[full_key(r.player)] = str(r.position)
        if omap:
            pos = d["position"].tolist()
            for i, who in enumerate(d["player"]):
                o = omap.get(full_key(who))
                if o:
                    pos[i] = o
                    n_ov += 1
            d["position"] = pos
    print(f"  posicion alineada con la plantilla en {sum(h is not None for h in hit):,} "
          f"de {len(d):,} jugadores ({n_ov} por correccion curada)")
    return d


def build() -> pd.DataFrame:
    a, b = load_2324(), load_2425()
    # EACH AXIS ANSWERS ITS OWN QUESTION. The attacking axis used to be fitted
    # against goals AND assists, so assists pulled xAG, SCA and GCA into it and
    # a creator scored high on a bar labelled ATAQUE — 33% of the axis was
    # creation, and Honorat (0.06 npxG per 90) out-attacked Olise (0.38).
    # Fitted against goals alone that falls to 11%, and how well the axis
    # describes the goals he actually scored rises from +0.824 to +0.867.
    # The LEVEL keeps the goals+assists fit below: total contribution is the
    # right question for how good a player is, just not for a bar called attack.
    coef, r2, n = fit_weights(a, b, target="goals_p90")
    print(f"  validación temporal: features 23/24 -> GOLES/90 de 24/25 "
          f"sobre {n:,} jugadores, R² = {r2:+.3f}")
    print("  pesos aprendidos (desviaciones típicas):")
    for k, v in sorted(coef.items(), key=lambda t: -abs(t[1])):
        print(f"     {k:16} {v:+.4f}")
    # a negative weight would say "doing this makes you worse next season", which
    # for attacking output means noise, not signal — dropped rather than trusted
    weights = {k: v for k, v in coef.items() if v > 0}
    if not weights:
        raise SystemExit("ningún rasgo ofensivo predice la producción futura")

    c = load_2526()
    full = pd.concat([a, b, c], ignore_index=True)
    full = full[full["n90"].fillna(0) >= MIN_90S].copy()
    full = align_positions(full)
    ns = {k: int(v) for k, v in full["season"].value_counts().items()}
    both = int(full["player"].duplicated().sum())
    d = recency_blend(full)
    print(f"  recencia: 25/26 pesa {SEASON_RECENCY['2526']}, 24/25 {SEASON_RECENCY['2425']}, "
          f"23/24 {SEASON_RECENCY['2324']} — filas/temporada {ns} ({both:,} fusionados)")

    total = sum(weights.values())
    score = np.zeros(len(d))
    for stat, w in weights.items():
        col = d[stat].astype(float)
        z = col.groupby(d["league"]).transform(
            lambda s: (s - s.mean()) / (s.std() if s.std() else 1.0))
        z = z.fillna((col - col.mean()) / (col.std() or 1.0))
        score += np.nan_to_num(z.to_numpy()) * (w / total)
    d["att_league_z"] = np.clip(score, -Z_CLIP, Z_CLIP)

    # ── the LEVEL keeps the goals + assists fit ───────────────────────────────
    lcoef, lr2, ln = fit_weights(a, b, target="ga_p90")
    print(f"  nivel: features 23/24 -> G+A/90 de 24/25 sobre {ln:,}, R² = {lr2:+.3f}")
    lw = {k: v for k, v in lcoef.items() if v > 0}

    # ...and then WITHIN POSITION, which is the number a card can use. Against
    # the league a centre back scores -0.29 and a forward +0.55 for doing their
    # jobs, and the card already knows which position he plays — adding that in
    # again would count it twice. Comparing him with his own position leaves
    # only what this tool can actually add: is he better going forward than
    # others in his role. Same convention as def_z.
    # the league goes into the AXES too, not just the level. Without it the axis
    # was flat across leagues (+0.014 sd for the big five) and creation actually
    # ran backwards (-0.020): a striker who dominated the 2. Bundesliga printed
    # the same attack as one who dominated the Premier League.
    strength = league_elo()
    lg_z = pd.Series(0.0, index=d.index)
    if strength:
        e = d["league"].map(strength).astype(float)
        e = e.fillna(e.median())
        lg_z = (e - e.mean()) / (e.std() or 1.0)
        d["league_elo"] = e
    pos_z = pd.Series(score + LEAGUE_W * lg_z.to_numpy(), index=d.index).groupby(
        d["position"]).transform(
        lambda s: (s - s.mean()) / (s.std() if s.std() else 1.0))
    cred = d["n90"] / (d["n90"] + CRED_90S)
    d["att_z"] = np.clip(pos_z.fillna(0.0) * cred, -Z_CLIP, Z_CLIP)
    # A third form, and the one a LEVEL needs. `att_z` divides by each
    # position's own spread, so "elite for a midfielder" and "elite for a
    # forward" both come out +2.5 — Honorat (0.46 goals+assists per 90) and Kane
    # (1.24) landed on the same number and therefore the same rating bump. A
    # level is compared ACROSS positions, so it must keep the spread (see
    # season_std_composite), and it is standardised WITHIN each season before
    # blending so 25/26's hotter Understat xG cannot lift whoever it measured.
    print(f"  fuerza de liga aplicada a nivel y ejes: peso {LEAGUE_W} sd por sd")

    def finalize_level(sig_col: str) -> pd.Series:
        """per-player level: recency-blended season signal + league + credibility."""
        bl = blend_level(full, sig_col)
        if strength:
            e = bl["league"].map(strength).astype(float)
            e = e.fillna(e.median())
            lgz = (e - e.mean()) / (e.std() or 1.0)
        else:
            lgz = pd.Series(0.0, index=bl.index)
        cr = bl["n90"] / (bl["n90"] + CRED_90S)
        val = np.clip((bl["lvl"].fillna(0.0) + LEAGUE_W * lgz) * cr, -Z_CLIP, Z_CLIP)
        return pd.Series(dict(zip(bl["player"], val)))

    full["_att_lvl"] = season_std_composite(full, lw)
    d["att_level_z"] = d["player"].map(finalize_level("_att_lvl")).fillna(0.0)
    d["att_signals"] = d[list(weights)].notna().mean(axis=1).round(2)

    # ── the creation half, same construction, its own target ──────────────────
    ccoef, cr2, cn = fit_weights(a, b, CREA_FEATURES, "assists_p90")
    print(f"  validación temporal: features 23/24 -> asistencias/90 de 24/25 "
          f"sobre {cn:,} jugadores, R² = {cr2:+.3f}")
    for k, v in sorted(ccoef.items(), key=lambda t: -abs(t[1])):
        print(f"     {k:16} {v:+.4f}")
    cw = {k: v for k, v in ccoef.items() if v > 0}
    if cw:
        ctot = sum(cw.values())
        cscore = np.zeros(len(d))
        for stat, w in cw.items():
            col = d[stat].astype(float)
            z = col.groupby(d["league"]).transform(
                lambda s: (s - s.mean()) / (s.std() if s.std() else 1.0))
            z = z.fillna((col - col.mean()) / (col.std() or 1.0))
            cscore += np.nan_to_num(z.to_numpy()) * (w / ctot)
        cpos = pd.Series(cscore + LEAGUE_W * lg_z.to_numpy(), index=d.index).groupby(
            d["position"]).transform(
            lambda s: (s - s.mean()) / (s.std() if s.std() else 1.0))
        d["crea_z"] = np.clip(cpos.fillna(0.0) * cred, -Z_CLIP, Z_CLIP)
        # spread-preserving LEVEL twin, standardised per season then blended
        # exactly like att_level_z, so a modest creator stays modest and a hot
        # 25/26 provider cannot inflate whoever it measured.
        full["_crea_lvl"] = season_std_composite(full, cw)
        d["crea_level_z"] = d["player"].map(finalize_level("_crea_lvl")).fillna(0.0)
    else:
        d["crea_z"] = np.nan
        d["crea_level_z"] = np.nan

    # birth year (invariant per player) — the disambiguator the card matcher uses
    # to tell two same-named players apart. Taken from whichever season has it.
    bmap = full.dropna(subset=["birth_year"]).groupby("player")["birth_year"].first()
    d["birth_year"] = d["player"].map(bmap)

    # ── possession / control level (Pedri, Rodri: retention + short dribbling) ──
    d["poss_level_z"] = d["player"].map(possession_level(full)).fillna(0.0)
    print(f"  nivel de posesión (pase + regate exitoso): "
          f"{(d['poss_level_z'] != 0).sum():,} jugadores")

    cols = ["player", "club", "league", "season", "position", "n90", "birth_year", "att_z",
            "att_level_z", "att_league_z", "crea_z", "crea_level_z", "poss_level_z",
            "league_elo", "att_signals"] + FEATURES + [
            c for c in CREA_FEATURES if c not in FEATURES] + ["assists_p90", "goals_p90", "ga_p90"]
    return d[cols].round(3).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = build()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nESCRITO {args.out}  ({len(out):,} jugadores)")
    for pos in ("Forward", "Midfielder"):
        sub = out[out["position"] == pos].nlargest(6, "att_z")
        print(f"  mejores {pos.lower()}s:")
        for r in sub.itertuples(index=False):
            print(f"     {str(r.player)[:24]:26} z={r.att_z:+.2f}  npxG/90 {r.npxg_p90:.2f}  "
                  f"xAG/90 {r.xag_p90:.2f}  ({r.club})")


if __name__ == "__main__":
    main()
