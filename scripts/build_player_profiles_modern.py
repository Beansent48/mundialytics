#!/usr/bin/env python3
"""Build MODERN per-season player profiles (2014/15-2025/26) by combining:
  - Understat season stats: goals, xg, np_xg, assists, xa, key_passes,
    xg_chain, xg_buildup, shots (the rich attacking/creation/buildup source).
  - Understat shot events -> finishing_per_shot (goals - xG per shot, PENALTIES
    INCLUDED per user 2026-07-02).
  - FBref 'misc' page (parsed offline from soccerdata's cached HTML):
    interceptions, tackles_won, crosses, fouls_drawn -- the ball-winning + wide
    signals Understat lacks.

Fills the modern gap where StatsBomb open data stops (e.g. Kanté's Chelsea
years). See [[project_data_state]]. Output feeds the unified role system.

Run with the project venv:
    .venv/Scripts/python.exe scripts/build_player_profiles_modern.py
"""
from __future__ import annotations

import glob
import io
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
US_SEASON = ROOT / "data/external/advanced/understat/understat_player_season.csv"
US_SHOTS = ROOT / "data/external/advanced/understat/understat_shots.csv"
FBREF_CACHE = Path.home() / "soccerdata/data/FBref"
OUT = ROOT / "data/processed/player_profiles_modern.csv"


def norm(name: str) -> str:
    """Accent-stripped, lowercased, apostrophe/space-normalised name key."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("'", "").replace("`", "").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def _fbref_misc() -> pd.DataFrame:
    """Parse all cached FBref misc season pages -> (pn, season, Int, TklW, Crs, Fld, 90s)."""
    rows = []
    for path in glob.glob(str(FBREF_CACHE / "players_*_misc.html")):
        parts = Path(path).stem.split("_")  # players, <league>, <season>, misc
        season = parts[-2]
        try:
            html = open(path, encoding="utf-8").read().replace("<!--", "").replace("-->", "")
            df = pd.read_html(io.StringIO(html), attrs={"id": "stats_misc"})[0]
        except Exception:
            continue
        df.columns = ["_".join(str(c) for c in col if "Unnamed" not in str(c)).strip("_")
                      for col in df.columns.to_flat_index()]
        df = df[df["Player"] != "Player"]  # drop repeated header rows
        keep = {"Player": "player", "90s": "nineties", "Performance_Int": "interceptions",
                "Performance_TklW": "tackles_won", "Performance_Crs": "crosses",
                "Performance_Fld": "fouls_drawn"}
        sub = df[[c for c in keep if c in df.columns]].rename(columns=keep)
        for c in ("nineties", "interceptions", "tackles_won", "crosses", "fouls_drawn"):
            if c in sub.columns:
                sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub["season"] = season
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    fb = pd.concat(rows, ignore_index=True)
    fb["pn"] = fb["player"].map(norm)
    return fb.groupby(["pn", "season"], as_index=False).agg(
        fb_nineties=("nineties", "sum"), interceptions=("interceptions", "sum"),
        tackles_won=("tackles_won", "sum"), crosses=("crosses", "sum"),
        fouls_drawn=("fouls_drawn", "sum"))


def main() -> None:
    us = pd.read_csv(US_SEASON)
    us["pn"] = us["player"].map(norm)
    us["season"] = us["season"].astype(str)
    agg = us.groupby(["pn", "season"], as_index=False).agg(
        player=("player", "first"), position=("position", "first"),
        matches=("matches", "sum"), minutes=("minutes", "sum"),
        goals=("goals", "sum"), xg=("xg", "sum"), np_goals=("np_goals", "sum"),
        np_xg=("np_xg", "sum"), assists=("assists", "sum"), xa=("xa", "sum"),
        shots=("shots", "sum"), key_passes=("key_passes", "sum"),
        xg_chain=("xg_chain", "sum"), xg_buildup=("xg_buildup", "sum"))

    # finishing from shots (penalties INCLUDED)
    sh = pd.read_csv(US_SHOTS, usecols=["season", "player", "xg", "result"])
    sh["season"] = sh["season"].astype(str)
    sh["pn"] = sh["player"].map(norm)
    sh["is_goal"] = sh["result"].astype(str) == "Goal"
    sh["xg"] = pd.to_numeric(sh["xg"], errors="coerce").fillna(0.0)
    fin = sh.groupby(["pn", "season"], as_index=False).agg(
        fin_goals=("is_goal", "sum"), fin_xg=("xg", "sum"), finishing_shots=("xg", "size"))
    fin["finishing_per_shot"] = (fin["fin_goals"] - fin["fin_xg"]) / fin["finishing_shots"].clip(lower=1)

    fb = _fbref_misc()

    m = agg.merge(fin[["pn", "season", "finishing_per_shot", "finishing_shots"]],
                  on=["pn", "season"], how="left")
    if not fb.empty:
        m = m.merge(fb, on=["pn", "season"], how="left")

    # per-90 rates (partial-season safe); p90 floored so short samples don't explode
    p90 = (m["minutes"] / 90.0).clip(lower=0.1)
    for col, src in [("goals_p90", "goals"), ("xg_p90", "xg"), ("np_xg_p90", "np_xg"),
                     ("assists_p90", "assists"), ("xa_p90", "xa"), ("shots_p90", "shots"),
                     ("key_passes_p90", "key_passes"), ("xg_chain_p90", "xg_chain"),
                     ("xg_buildup_p90", "xg_buildup")]:
        m[col] = m[src] / p90
    fb90 = m["fb_nineties"].fillna(0).clip(lower=0.1) if "fb_nineties" in m.columns else p90
    for col, src in [("interceptions_p90", "interceptions"), ("tackles_won_p90", "tackles_won"),
                     ("crosses_p90", "crosses"), ("fouls_drawn_p90", "fouls_drawn")]:
        if src in m.columns:
            m[col] = m[src] / fb90
        else:
            m[col] = np.nan

    m["finishing_per_shot"] = m["finishing_per_shot"].fillna(0.0)
    m["finishing_shots"] = m["finishing_shots"].fillna(0.0)
    m["source"] = "understat_fbref"
    m.to_csv(OUT, index=False)
    print(f"WROTE {OUT}: {len(m)} rows, {m['pn'].nunique()} players")
    fb_matched = int(m["interceptions_p90"].notna().sum()) if "interceptions_p90" in m.columns else 0
    print(f"FBref misc matched (int/tackles): {fb_matched}/{len(m)} season-rows")


if __name__ == "__main__":
    main()
