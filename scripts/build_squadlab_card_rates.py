#!/usr/bin/env python3
"""What each SquadLab card DOES on the pitch, per 90 minutes.

The card overall says how good a player is; the live engine also needs to know
who shoots, who scores, who creates, who gets booked and who takes the
penalties — the things that decide WHICH player's name goes on an event, not
how many events a side has (that stays the team model's job). Market value
plays no part anywhere: only what the man did on the pitch.

Sources, by card kind, all on the same per-90 scale:

  ACTUAL  the player-props model's own state (Understat history + this
          season's ESPN rows), with its validated recipe: career rate shrunk to
          the position prior with K=900 minutes, blended with last-15 form for
          goals/shots; goals = 0.7·npxG + 0.3·npgoals, assists = 0.7·xA +
          0.3·assists. The penalty-taker share is the props model's too.
  PRIME   that SEASON's Understat rows (2014/15 onwards), else StatsBomb's
          season profile; a prime is measured in its own year.
  ICONO   the career: Understat when it covers the player, else StatsBomb.

A card with no rows (curated icons, a few primes) falls back to what its own
axes imply: per position, log(rate) regressed on the card's measured attack /
creation axis over well-sampled actual cards. That same axis prior is what
thin samples are shrunk toward, so a 3-match icon is not read off 3 matches.

Penalty CONVERSION is deliberately not per player: across alternating seasons
a taker's conversion correlates at r = −0.065 (66 takers, ≥6 kicks each half),
so it is not a skill this data can see. Everyone converts at the measured rate;
what differs is who is handed the ball.

Writes data/processed/squadlab_card_rates.csv (one row per card_id).

Run:
    python scripts/build_squadlab_card_rates.py
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mundialytics.identity.current_squads import NameIndex  # noqa: E402
from mundialytics.statistical_core.squadlab.cards import load_cards  # noqa: E402

OUT = ROOT / "data/processed/squadlab_card_rates.csv"
US_PM = ROOT / "data/external/advanced/understat/understat_player_match.csv"
US_SHOTS = ROOT / "data/external/advanced/understat/understat_shots.csv"
SB_SEASON = ROOT / "data/processed/player_profiles_by_season.csv"
SB_CAREER = ROOT / "data/processed/player_profiles_with_positions.csv"
SB_SHOTS = ROOT / "data/external/xg/statsbomb/statsbomb_xg_shots.csv"

K_MIN = 900.0          # the props model's credibility, in minutes
PEN_XG = (0.70, 0.80)  # Understat files penalties with situation=NaN and xG 0.74-0.76
# a StatsBomb appearance is ~82 minutes on average for the players these
# profiles keep (starters mostly); per-match rates are put on the per-90 scale
MIN_PER_APP = 82.0
GROUPS = {"Goalkeeper": "GK", "Defender": "DEF", "Midfielder": "MID", "Forward": "FWD"}


def fold(s: object) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


# ── understat, per player-season ───────────────────────────────────────────────
def understat_seasons() -> pd.DataFrame:
    pm = pd.read_csv(US_PM, usecols=["player", "season", "team", "minutes", "goals", "shots", "xg",
                                     "assists", "xa", "yellow_cards"])
    sh = pd.read_csv(US_SHOTS, usecols=["player", "season", "situation", "xg", "result"])
    pen = sh[sh["situation"].isna() & sh["xg"].between(*PEN_XG)].copy()
    pen["pg"] = (pen["result"] == "Goal").astype(int)
    pen = pen.groupby(["player", "season"]).agg(pen_att=("pg", "size"), pen_goal=("pg", "sum"))
    s = pm.groupby(["player", "season"]).sum(numeric_only=True).join(pen, how="left").fillna(0.0)
    s["npxg"] = (s["xg"] - 0.76 * s["pen_att"]).clip(lower=0)
    s["npgoals"] = (s["goals"] - s["pen_goal"]).clip(lower=0)
    return s.reset_index()


def per90(frame: pd.DataFrame, prior: dict[str, float], k: float = K_MIN) -> dict[str, float]:
    """Summed rows -> shrunk per-90 rates (the props recipe, without recency)."""
    mins = float(frame["minutes"].sum())
    out = {"minutes": mins}
    for key, num in (("g", 0.7 * frame["npxg"].sum() + 0.3 * frame["npgoals"].sum()),
                     ("a", 0.7 * frame["xa"].sum() + 0.3 * frame["assists"].sum()),
                     ("sh", frame["shots"].sum()), ("yc", frame["yellow_cards"].sum()),
                     ("pen", frame["pen_att"].sum())):
        raw = num / mins * 90.0 if mins > 0 else prior[key]
        cred = mins / (mins + k)
        out[key] = cred * raw + (1 - cred) * prior[key]
    return out


def statsbomb_rates(row, prior: dict[str, float], pens: tuple[float, float] = (0.0, 0.0)) -> dict[str, float]:
    """A StatsBomb profile row (per MATCH) on the per-90 scale, shrunk.

    `pens` = (attempts, goals) over the same matches, from the shot file: the
    profile's goals include penalties, and the engine draws those separately.
    """
    apps = float(row.get("matches", 0) or 0)
    mins = apps * MIN_PER_APP
    f = 90.0 / MIN_PER_APP
    pen_att, pen_goal = pens
    xg = row.get("npxg_per_match", np.nan)
    gl = row.get("goals_per_match", np.nan)
    if pd.notna(gl) and apps > 0:
        gl = max(float(gl) - pen_goal / apps, 0.0)
    g_raw = (0.7 * xg + 0.3 * gl) if pd.notna(xg) and pd.notna(gl) else gl
    vals = {"g": g_raw, "a": row.get("assists_per_match", np.nan),
            "sh": row.get("shots_per_match", np.nan), "yc": row.get("yellow_cards_per_match", np.nan)}
    cred = mins / (mins + K_MIN)
    out = {"minutes": mins,
           "pen": cred * (pen_att / mins * 90.0 if mins > 0 else 0.0) + (1 - cred) * prior["pen"]}
    cred = mins / (mins + K_MIN)
    for key, v in vals.items():
        v = float(v) * f if pd.notna(v) else prior[key]
        out[key] = cred * v + (1 - cred) * prior[key]
    return out


# ── props model state for the current cards ────────────────────────────────────
def props_state():
    from api import engine
    _, pp = engine.props_models()
    if pp is None:
        raise SystemExit("player props model unavailable")
    from mundialytics.enrichment.understat_team_aliases import to_foundation_name
    from mundialytics.props.player_props import K_RECENT, RECENT_W, STATS

    P = pp._players.copy()
    for c in STATS:
        prior = P["pgroup"].map(pp._pri[c]).fillna(pp._glob[c])
        raw = np.where(P["cmin"] > 0, P[f"c_{c}"] / P["cmin"].clip(lower=1e-9) * 90.0, prior)
        cred = P["cmin"] / (P["cmin"] + K_MIN)
        r_car = cred * raw + (1 - cred) * prior
        w = RECENT_W[c]
        if w > 0:
            rmin = P["rmin15"].fillna(0.0)
            raw_r = np.where(rmin > 0, P[f"rr_{c}"].fillna(0.0) / rmin.clip(lower=1e-9) * 90.0, r_car)
            cred_r = rmin / (rmin + K_RECENT)
            P[f"r_{c}"] = w * (cred_r * raw_r + (1 - cred_r) * r_car) + (1 - w) * r_car
        else:
            P[f"r_{c}"] = r_car
    t_pen = P["t_pen60"].fillna(0.0)
    # the props model's taker share: his cut of his club's last penalties,
    # shrunk towards nobody when the club has taken few
    P["taker"] = (t_pen / (t_pen + 4.0)) * (P["p_pen60"].fillna(0.0) / t_pen.clip(lower=1e-9))
    P["team_fd"] = P["team"].map(lambda t: to_foundation_name(str(t)))
    return P, pp


def main() -> None:
    cards = load_cards()
    P, pp = props_state()
    grp_prior = {g: {"g": 0.7 * pp._pri["npxg"].get(g, pp._glob["npxg"]) + 0.3 * pp._pri["npgoals"].get(g, pp._glob["npgoals"]),
                     "a": 0.7 * pp._pri["xa"].get(g, pp._glob["xa"]) + 0.3 * pp._pri["assists"].get(g, pp._glob["assists"]),
                     "sh": pp._pri["shots"].get(g, pp._glob["shots"]),
                     "yc": pp._pri["yellow_cards"].get(g, pp._glob["yellow_cards"]),
                     "pen": 0.0}
                 for g in ("GK", "DEF", "MID", "FWD")}

    # ACTUAL: the props player, same club first, then anywhere by name
    by_team: dict[str, NameIndex] = {}
    anywhere = NameIndex()
    for i, r in P.iterrows():
        rank = float(r["cmin"] or 0)
        by_team.setdefault(str(r["team_fd"]), NameIndex()).add(r["player"], i, rank)
        anywhere.add(r["player"], i, rank)

    us = understat_seasons()
    us["pk"] = us["player"].map(fold)
    us_idx = NameIndex()
    for name, g in us.groupby("player"):
        us_idx.add(name, name, float(g["minutes"].sum()))
    sb_season = pd.read_csv(SB_SEASON)
    sb_career = pd.read_csv(SB_CAREER)
    sb_idx = NameIndex()
    for i, r in sb_career.iterrows():
        sb_idx.add(r["player"], i, float(r.get("matches", 0) or 0))
    # StatsBomb penalties, shoot-outs excluded (they are filed from minute 120)
    sbs = pd.read_csv(SB_SHOTS, usecols=["player", "season", "minute", "shot_type", "outcome"])
    sbs = sbs[(sbs["shot_type"] == "Penalty") & (sbs["minute"] < 120)]
    sb_pens = (sbs.assign(goal=(sbs["outcome"] == "Goal").astype(int))
               .groupby(["player", "season"]).agg(att=("goal", "size"), goal=("goal", "sum"))
               .reset_index())

    rows = []
    for c in cards.itertuples(index=False):
        grp = GROUPS.get(c.position, "MID")
        prior = grp_prior[grp]
        rec, src = None, "prior"
        if c.kind == "actual":
            hit = by_team.get(str(c.club), NameIndex()).lookup(c.player) if str(c.club) in by_team else None
            hit = hit if hit is not None else anywhere.lookup(c.player)
            if hit is not None:
                r = P.loc[hit]
                rec = {"minutes": float(r["cmin"]),
                       "g": 0.7 * r["r_npxg"] + 0.3 * r["r_npgoals"],
                       "a": 0.7 * r["r_xa"] + 0.3 * r["r_assists"],
                       "sh": r["r_shots"], "yc": r["r_yellow_cards"], "taker": r["taker"]}
                src = "props"
        else:
            cands = []
            name = us_idx.lookup(c.player)
            if name is not None:
                rows_p = us[us["player"] == name]
                if c.kind == "prime":
                    code = _season_code(c.season_label)
                    rows_p = rows_p[rows_p["season"] == code] if code else rows_p.iloc[0:0]
                if len(rows_p) and rows_p["minutes"].sum() > 0:
                    cands.append((per90(rows_p, prior), "understat"))
            hit = sb_idx.lookup(c.player)
            if hit is not None:
                row = sb_career.loc[hit]
                pens = sb_pens.loc[sb_pens["player"] == row["player"]]
                if c.kind == "prime":
                    s = sb_season[(sb_season["player"] == row["player"])
                                  & (sb_season["season"].astype(str).map(_short) == str(c.season_label))]
                    pens = pens[pens["season"].astype(str).map(_short) == str(c.season_label)]
                    row = s.sort_values("matches").iloc[-1] if len(s) else None
                if row is not None:
                    cands.append((statsbomb_rates(row.to_dict(), prior,
                                                  (float(pens["att"].sum()), float(pens["goal"].sum()))),
                                  "statsbomb"))
            # a prime is its own season wherever it was measured; an icon is
            # read off whichever source saw more of his career
            if cands:
                rec, src = max(cands, key=lambda x: x[0]["minutes"])
        rows.append({"card_id": c.card_id, "kind": c.kind, "player": c.player, "position": c.position,
                     "raw_attack": c.raw_attack, "raw_creation": c.raw_creation,
                     "src": src, **(rec or {"minutes": 0.0})})

    out = pd.DataFrame(rows)
    out = _fill_from_axes(out)
    # the taker signal: the props share for current cards; for primes and icons
    # the penalty attempts per 90 they actually took, on the same 0..1 reading
    # (a club's penalties per 90 are ~0.12, so a sole taker runs ~0.12/90')
    out["taker"] = out.get("taker", pd.Series(np.nan, index=out.index))
    pen_share = (out["pen"] / 0.12).clip(0, 1) if "pen" in out else 0.0
    out["taker"] = out["taker"].fillna(pen_share).fillna(0.0).clip(0, 1)
    out.loc[out["position"] == "Goalkeeper", ["g", "a", "sh", "taker"]] = [0.0, 0.004, 0.01, 0.0]
    keep = ["card_id", "kind", "player", "position", "src", "minutes", "g", "a", "sh", "yc", "taker"]
    out = out[keep]
    for col in ("g", "a", "sh", "yc", "taker"):
        out[col] = out[col].astype(float).round(4)
    out["minutes"] = out["minutes"].astype(float).round(0)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT.name}: {len(out)} cards")
    print(out.groupby(["kind", "src"]).size().to_string())
    print(out.groupby("position")[["g", "a", "sh", "yc", "taker"]].mean().round(3).to_string())


def _season_code(label: object) -> int | None:
    """'15/16' -> 1516 (Understat's season code)."""
    s = str(label or "")
    if len(s) == 5 and s[2] == "/":
        try:
            return int(s[:2] + s[3:])
        except ValueError:
            return None
    return None


def _short(season: str) -> str:
    """'2015/2016' -> '15/16'."""
    s = str(season)
    return f"{s[2:4]}/{s[7:9]}" if len(s) == 9 else s


def _fill_from_axes(out: pd.DataFrame) -> pd.DataFrame:
    """Cards with no rows read their rates off their own axes.

    Per position, log(rate) ~ axis over well-sampled cards (≥1,800 minutes):
    attack drives goals and shots, creation drives assists; bookings are the
    position's mean, since no axis says who gets booked.
    """
    fit_on = out[(out["minutes"] >= 1800) & (out["position"] != "Goalkeeper")]
    for pos, idx in out.groupby("position").groups.items():
        sub = fit_on[fit_on["position"] == pos]
        miss = out.index.isin(idx) & out["g"].isna()
        if not miss.any():
            continue
        for rate, axis in (("g", "raw_attack"), ("sh", "raw_attack"), ("a", "raw_creation")):
            if len(sub) >= 15:
                b, a = np.polyfit(sub[axis], np.log(sub[rate].clip(lower=1e-3)), 1)
                out.loc[miss, rate] = np.exp(a + b * out.loc[miss, axis])
            else:
                out.loc[miss, rate] = fit_on[rate].median()
        out.loc[miss, "yc"] = sub["yc"].mean() if len(sub) else fit_on["yc"].mean()
        out.loc[miss, "pen"] = 0.0
    return out


if __name__ == "__main__":
    main()
