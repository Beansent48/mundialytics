"""FBref's measured player quality, keyed for the rating builder.

WHY THIS EXISTS. The unified rating scores a defensive midfielder with
`DEFW = 0.55 * interceptions + 0.45 * tackles won` — volume and nothing else,
because the modern pipeline (Understat) never carried a duel-win rate. The
builder's own comment says what that costs: "journeymen match Piqué/Van Dijk",
and it names Mittelstädt. The StatsBomb side was fixed years ago by weighting
`duel_win_rate` at 0.42 in DEFG; the modern side never was, because there was
nothing to weight.

There is now. FBref's published season tables (free, via Kaggle — see
`scripts/download_fbref_kaggle.py`) carry, for 7,000+ players across 17 leagues:

  * challenge success rate and how often he is dribbled past — DEFENDING quality
  * aerial win rate — the other half of a duel
  * pass completion, pass volume, progressive passes and progressive distance —
    ORGANISING, which is the thing a deep-lying midfielder is actually good at
    and which no block in the rating reads. Rodri is 98th-99th percentile on all
    four among 2,477 midfielders and the rating had him as an ordinary
    "Destructor" because it only counted his tackles.

Both are steady enough to describe a player rather than a season: the passing
measures persist year-to-year at **r +0.83 to +0.90** over 1,224 players, and the
duel rates at r +0.5 to +0.6 — against +0.06 for goalkeeper shot-stopping, which
was rejected for exactly that reason.

Everything is returned on its natural units (percentages, per-90 counts) so the
builder's own anchor curves place it, and every rate is shrunk by ITS OWN
denominator — a 100% duel record from three attempts is not a duel record.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "data/external/advanced/fbref_kaggle"

# Each rate is pulled toward the population mean by its own denominator, not by
# minutes: a player with three aerials and three wins is not a 100% header.
SHRINK_CHALLENGES = 25.0
SHRINK_AERIALS = 30.0
MIN_90S = 5.0
# "dribbled past" is a bad thing, and the builder's anchor curves are monotonic
# increasing over non-negative values, so it ships as its complement.
BEATEN_CEILING = 5.0

COLUMNS = ["duel_pct", "aerial_pct", "beaten_rarely",
           "pass_pct", "passes_p90", "prog_passes_p90", "prog_dist_p90",
           # shot- and goal-creating actions: the two that actually predict
           # next season's assists (R² +0.32 and +0.25 alone), and the reason
           # the winger block no longer runs on cross volume.
           "sca_p90", "gca_p90", "fbq_90s"]


def _shrink(pct: pd.Series, n: pd.Series, k: float) -> pd.Series:
    prior = pct.mean()
    w = n.fillna(0) / (n.fillna(0) + k)
    return prior + (pct.fillna(prior) - prior) * w


def _from_2324() -> pd.DataFrame:
    """The 2023/24 tables, joined on (player, CLUB).

    Not on the name alone: 891 of the 10,578 rows share a name with another
    row, and a name-keyed dict silently keeps whichever came last. Manchester
    City's Rodri (92.0% passing, 3,656 passes) was being handed the numbers of
    Villarreal B's Rodri in the Segunda. The three tables are the same
    player-season rows, so club makes the join exact.
    """
    def _one_per_club(df: pd.DataFrame) -> pd.DataFrame:
        """(player, club) is ALMOST unique — three pairs in 10,578 rows are two
        different men with the same name at the same club (two "João Pedro" at
        Grêmio). Left alone, the merge is many-to-many: it inflated the table to
        10,596 rows and blended the two namesakes' numbers. The fuller season
        wins; the other is lost, which is the honest cost of a name-only key."""
        return (df.sort_values("minutes_90s", ascending=False)
                  .drop_duplicates(subset=["player", "club"]))

    de = _one_per_club(pd.read_csv(SRC / "2324_Defense.csv"))
    mi = _one_per_club(pd.read_csv(SRC / "2324_Misc.csv"))
    pa = _one_per_club(pd.read_csv(SRC / "2324_Passing.csv"))
    gs = _one_per_club(pd.read_csv(SRC / "2324_GSC.csv"))
    keys = ["player", "club"]
    d = (de[keys + ["minutes_90s", "challenge_tackles_pct", "challenges", "challenges_lost"]]
         .merge(mi[keys + ["aerials_won", "aerials_lost", "aerials_won_pct"]],
                on=keys, how="left")
         .merge(pa[keys + ["passes_pct", "passes", "prog_passes", "passes_prog_dist"]],
                on=keys, how="left")
         .merge(gs[keys + ["sca_per90", "gca_per90"]], on=keys, how="left"))
    n90 = pd.to_numeric(d["minutes_90s"], errors="coerce")
    num = lambda c: pd.to_numeric(d[c], errors="coerce")  # noqa: E731
    return pd.DataFrame({
        "player": d["player"], "fbq_90s": n90,
        "duel_pct": num("challenge_tackles_pct"),
        "n_challenges": num("challenges"),
        "aerial_pct": num("aerials_won_pct"),
        "n_aerials": num("aerials_won").fillna(0) + num("aerials_lost").fillna(0),
        "beaten": num("challenges_lost") / n90.clip(lower=0.5),
        "pass_pct": num("passes_pct"),
        "passes_p90": num("passes") / n90.clip(lower=0.5),
        "prog_passes_p90": num("prog_passes") / n90.clip(lower=0.5),
        "prog_dist_p90": num("passes_prog_dist") / n90.clip(lower=0.5),
        "sca_p90": num("sca_per90"),
        "gca_p90": num("gca_per90"),
    })


def _from_2425() -> pd.DataFrame:
    w = pd.read_csv(SRC / "fbref_players_2425.csv", low_memory=False)
    n90 = pd.to_numeric(w.get("90s_stats_defense", w["90s"]), errors="coerce")
    won = pd.to_numeric(w.get("Won"), errors="coerce").fillna(0)
    lost = pd.to_numeric(w.get("Lost_stats_misc"), errors="coerce").fillna(0)
    n90p = pd.to_numeric(w["90s"], errors="coerce")
    return pd.DataFrame({
        "player": w["Player"], "fbq_90s": n90,
        "duel_pct": pd.to_numeric(w.get("Tkl%"), errors="coerce"),
        "n_challenges": pd.to_numeric(w.get("Att_stats_defense"), errors="coerce"),
        "aerial_pct": pd.to_numeric(w.get("Won%"), errors="coerce"),
        "n_aerials": won + lost,
        "beaten": pd.to_numeric(w.get("Tkld"), errors="coerce") / n90.clip(lower=0.5),
        "pass_pct": pd.to_numeric(w.get("Cmp%"), errors="coerce"),
        "passes_p90": pd.to_numeric(w.get("Att"), errors="coerce") / n90p.clip(lower=0.5),
        "prog_passes_p90": pd.to_numeric(w.get("PrgP"), errors="coerce") / n90p.clip(lower=0.5),
        "prog_dist_p90": pd.to_numeric(w.get("PrgDist"), errors="coerce") / n90p.clip(lower=0.5),
        "sca_p90": pd.to_numeric(w.get("SCA90"), errors="coerce"),
        "gca_p90": pd.to_numeric(w.get("GCA90"), errors="coerce"),
    })


def load() -> pd.DataFrame:
    """One row per player: measured quality, on natural units.

    Both seasons are pooled and the fuller one kept, because a rating is a
    description of a player and the bigger sample describes him better.
    """
    frames = []
    if (SRC / "2324_Defense.csv").exists():
        frames.append(_from_2324())
    if (SRC / "fbref_players_2425.csv").exists():
        frames.append(_from_2425())
    if not frames:
        return pd.DataFrame(columns=["player", *COLUMNS])
    d = pd.concat(frames, ignore_index=True)
    d = d[d["fbq_90s"].fillna(0) >= MIN_90S]
    d = d.sort_values("fbq_90s", ascending=False).drop_duplicates("player")

    d["duel_pct"] = _shrink(d["duel_pct"], d["n_challenges"], SHRINK_CHALLENGES)
    d["aerial_pct"] = _shrink(d["aerial_pct"], d["n_aerials"], SHRINK_AERIALS)
    d["beaten_rarely"] = (BEATEN_CEILING
                          - d["beaten"].clip(lower=0, upper=BEATEN_CEILING).fillna(
                              d["beaten"].median()))
    return d[["player", *COLUMNS]].reset_index(drop=True)


def by_key(keyfn) -> dict:
    """`load()` indexed by the caller's own name key, best sample winning."""
    d = load()
    out: dict[str, dict] = {}
    for r in d.sort_values("fbq_90s").itertuples(index=False):
        k = keyfn(r.player)
        if k:
            out[k] = {c: float(getattr(r, c)) if pd.notna(getattr(r, c)) else np.nan
                      for c in COLUMNS}
    return out


# ── goalkeepers ───────────────────────────────────────────────────────────────
# The deployed keeper score comes from StatsBomb save%/goals-conceded/clean
# sheets over whatever StatsBomb released, which for a modern keeper is thin and
# old: Yann Sommer scores 44.3 there while saving 76.4% of his shots in FBref's
# 2024/25, and Courtois 58.5 against 75.0%. FBref has both full seasons for
# essentially every big-five keeper.
#
# THE CAVEAT, KEPT IN THE OPEN: shot-stopping barely persists season to season
# (r +0.06 for PSxG+/- per 90, +0.17 for save%, measured on the 96 keepers with
# 8+ matches in both). A keeper rating is the least trustworthy number in the
# product whatever it is built from. Pooling the two seasons roughly doubles the
# sample, which is the only honest improvement available: weak on 60 matches
# beats weak on 10.
GK_MIN_90S = 5.0
GK_SHRINK_SOT = 60.0          # shots on target faced; ~60 is half weight
# Post-shot xG minus goals conceded is the metric built to remove shot quality,
# so it leads; raw save% still carries the bulk of the sample.
GK_WEIGHTS = {"psxg_net90": 0.55, "save_pct": 0.45}


def keepers() -> pd.DataFrame:
    """player -> a standardised shot-stopping score, pooled over both seasons.

    MEASURED AND NOT DEPLOYED (2026-09-07). Built, run, and rejected: the top of
    the list is Meza, Schwengber, Velho and Romero on 20-30 matches in weak
    leagues, and standardising within league and restricting to the big five
    only swaps them for Muric, Gregorio and Dahmen. Courtois, Oblak, Alisson and
    ter Stegen appear nowhere. Save% measures how easy your shots were, and the
    persistence number above says the same thing from the other direction. Kept
    because the evidence is worth re-reading, not because it is used.
    """
    frames = []
    f1, f2 = SRC / "2324_Goalkeeping.csv", SRC / "2324_Goalkeeping_adv.csv"
    if f1.exists() and f2.exists():
        a = pd.read_csv(f1).sort_values("minutes_90s", ascending=False).drop_duplicates(
            subset=["player", "club"])
        b = pd.read_csv(f2).sort_values("minutes_90s", ascending=False).drop_duplicates(
            subset=["player", "club"])
        keys = ["player", "club"]
        d = a[keys + ["minutes_90s", "save_pct", "shotsOT_a"]].merge(
            b[keys + ["psxg_net90"]], on=keys, how="left")
        frames.append(pd.DataFrame({
            "player": d["player"],
            "n90": pd.to_numeric(d["minutes_90s"], errors="coerce"),
            "sot": pd.to_numeric(d["shotsOT_a"], errors="coerce"),
            "save_pct": pd.to_numeric(d["save_pct"], errors="coerce"),
            "psxg_net90": pd.to_numeric(d["psxg_net90"], errors="coerce"),
        }))
    f3 = SRC / "fbref_players_2425.csv"
    if f3.exists():
        w = pd.read_csv(f3, low_memory=False)
        g = w[w["Pos"].astype(str).str.startswith("GK")]
        n = pd.to_numeric(g["90s"], errors="coerce")
        frames.append(pd.DataFrame({
            "player": g["Player"], "n90": n,
            "sot": pd.to_numeric(g.get("SoTA"), errors="coerce"),
            "save_pct": pd.to_numeric(g.get("Save%"), errors="coerce"),
            "psxg_net90": pd.to_numeric(g.get("PSxG+/-"), errors="coerce") / n.clip(lower=1),
        }))
    if not frames:
        return pd.DataFrame(columns=["player", "gk_z", "n90"])

    d = pd.concat(frames, ignore_index=True)
    d = d[d["n90"].fillna(0) >= GK_MIN_90S]
    # pool a keeper's two seasons rather than picking one: the sample is the
    # whole problem here
    g = d.groupby("player").apply(
        lambda x: pd.Series({
            "n90": x["n90"].sum(),
            "sot": x["sot"].sum(),
            "save_pct": np.average(x["save_pct"].fillna(x["save_pct"].mean()),
                                   weights=x["n90"].clip(lower=0.1)),
            "psxg_net90": np.average(x["psxg_net90"].fillna(0),
                                     weights=x["n90"].clip(lower=0.1)),
        }), include_groups=False).reset_index()

    z = np.zeros(len(g))
    for col, wgt in GK_WEIGHTS.items():
        v = g[col].astype(float)
        zz = (v - v.mean()) / (v.std() or 1.0)
        z += np.nan_to_num(zz.to_numpy()) * wgt
    cred = g["sot"].fillna(0) / (g["sot"].fillna(0) + GK_SHRINK_SOT)
    g["gk_z"] = np.clip(z * cred, -3, 3)
    return g[["player", "gk_z", "n90", "save_pct", "psxg_net90"]]
