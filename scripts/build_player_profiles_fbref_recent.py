#!/usr/bin/env python3
"""Build a RECENT-SEASON player profiles CSV from FBref (via soccerdata),
covering 2017/18-2025/26 for the 5 Big5 leagues -- seasons StatsBomb's free
open-data doesn't reach (its Big5 coverage mostly stops well before 2020,
see [[project_player_rating_data]]).

IMPORTANT LIMITATIONS, confirmed 2026-07-01 by inspecting soccerdata's ACTUAL
returned columns directly (a first version of this script guessed column
names and silently produced garbage -- "matches" defaulted to 1 for every
row, so "goals_per_match" was actually raw season goal totals, and
xg_per_match/tklw_per_match were silently all zero):

  - soccerdata 1.9.0's read_player_season_stats(stat_type='shooting') does
    NOT include FBref's "Expected" (xG) section at all -- this was already
    known and recorded in [[project_data_state]] ("FBref xG via soccerdata:
    the shooting stat type doesn't include the Expected section") from an
    earlier session; this file should have checked that memory before
    claiming xG as a benefit. There is NO xg_per_match/npxg_per_match column
    in this output -- don't reintroduce a fake one.
  - Only 5 stat pages are exposed at season granularity: standard, shooting,
    misc, playing_time, keeper. No "Defensive Actions" or "Passing" page, so
    this file cannot populate duel_win_rate/dribbled_past_per_match/
    clearances_per_match/blocks_per_match/key_passes_per_match/
    pass_completion the way the StatsBomb-based pipeline does (see
    scripts/enrich_player_profiles_with_defense_creation.py). The closest
    thing available is tklw_per_match (tackles WON, a count not a rate) and
    interceptions_per_match.
  - ITA-Serie A failed entirely for all 4 stat types this run ("No objects
    to concatenate") -- something about that league/season combination
    doesn't resolve via soccerdata. Re-run just that league later; the rest
    (Premier League, La Liga, Bundesliga, Ligue 1) succeeded.

What this DOES provide that's genuinely useful: real per-season PLAYER-level
goals/assists/shots/SOT/fouls/cards for seasons StatsBomb doesn't cover at
all (2017/18-2025/26), so recent squads/players exist in the draftable pool
instead of being entirely absent. Their off_score will be fully populated;
def_score/creation_score will fall back to neutral via the existing
zero-credibility pattern (same as any player missing that data).

Run (slow -- ~20 FBref requests, ~2 minutes each due to soccerdata's
built-in rate limiting -- expect ~35-40 minutes total):
    python scripts/build_player_profiles_fbref_recent.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.identity.normalization import canonical_team_name

OUT_PATH = ROOT / "data/processed/player_profiles_fbref_recent.csv"

LEAGUES = ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]
LEAGUE_LABELS = {
    "ENG-Premier League": "Premier League", "ESP-La Liga": "LaLiga",
    "GER-Bundesliga": "Bundesliga", "ITA-Serie A": "Serie A", "FRA-Ligue 1": "Ligue 1",
}
SEASONS = ["1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]

# Real flattened column names, confirmed by direct inspection 2026-07-01 --
# do NOT guess these again, soccerdata's naming doesn't match FBref's own
# on-site labels 1:1 (e.g. "Sh" on-site is "Standard_Sh" here).
STANDARD_COLS = {
    "matches": "Playing Time_MP", "goals": "Performance_Gls", "assists": "Performance_Ast",
    "position_raw": "pos",
}
# FBref position codes (confirmed by inspection): GK/DF/MF/FW, or a
# comma-joined pair (e.g. "MF,FW") when a player features in two -- FBref
# lists the primary position first, so take the substring before any comma.
FBREF_POSITION_MAP = {"GK": "Goalkeeper", "DF": "Defender", "MF": "Midfielder", "FW": "Forward"}
SHOOTING_COLS = {"shots": "Standard_Sh", "sot": "Standard_SoT"}
MISC_COLS = {
    "fouls": "Performance_Fls", "yellow_cards": "Performance_CrdY",
    "tklw": "Performance_TklW", "interceptions": "Performance_Int",
}


def _season_label(code: str) -> str:
    return f"20{code[:2]}-20{code[2:]}"


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = ["_".join(c for c in col if c).strip() for col in df.columns.to_flat_index()]
    return df.reset_index()


def _fetch_league(league: str) -> pd.DataFrame | None:
    from soccerdata import FBref

    fb = FBref(leagues=league, seasons=SEASONS)
    frames = {}
    for stat in ("standard", "shooting", "misc"):
        t0 = time.time()
        try:
            frames[stat] = _flatten(fb.read_player_season_stats(stat_type=stat))
            print(f"  {stat}: {frames[stat].shape} ({time.time()-t0:.1f}s)")
        except Exception as exc:
            print(f"  {stat}: FAILED ({exc})")
            return None

    keys = ["league", "season", "team", "player"]
    standard = frames["standard"][keys + list(STANDARD_COLS.values())]
    shooting = frames["shooting"][keys + list(SHOOTING_COLS.values())]
    misc = frames["misc"][keys + list(MISC_COLS.values())]

    merged = standard.merge(shooting, on=keys, how="left").merge(misc, on=keys, how="left")
    merged["competition"] = LEAGUE_LABELS[league]
    return merged


def main() -> None:
    all_frames = []
    for league in LEAGUES:
        print(f"=== {league} ===")
        merged = _fetch_league(league)
        if merged is not None:
            all_frames.append(merged)

    if not all_frames:
        print("No data fetched.")
        return

    combined = pd.concat(all_frames, ignore_index=True, sort=False)
    print(f"\nCombined raw shape: {combined.shape}")

    out = pd.DataFrame()
    out["player"] = combined["player"]
    out["team"] = combined["team"].map(canonical_team_name)
    out["competition"] = combined["competition"]
    out["season"] = combined["season"].map(_season_label)
    primary_pos = combined[STANDARD_COLS["position_raw"]].astype(str).str.split(",").str[0]
    out["position"] = primary_pos.map(FBREF_POSITION_MAP).fillna("Unknown")

    matches = pd.to_numeric(combined[STANDARD_COLS["matches"]], errors="coerce").fillna(0.0)
    out["matches"] = matches.round().astype(int).clip(lower=1)
    m = out["matches"]

    for name, col in STANDARD_COLS.items():
        if name in ("matches", "position_raw"):
            continue
        out[f"{name}_per_match"] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0) / m
    for name, col in SHOOTING_COLS.items():
        out[f"{name}_per_match"] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0) / m
    for name, col in MISC_COLS.items():
        out[f"{name}_per_match"] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0) / m

    out = out[out["matches"] >= 1].reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(out)} (player, team, competition, season) rows -> {OUT_PATH}")
    print(f"Unique players: {out['player'].nunique()}, seasons: {sorted(out['season'].unique())}")
    print(f"Competitions: {sorted(out['competition'].unique())}")

    messi = out[out["player"].str.contains("Messi", case=False, na=False)]
    print("\nSanity check (Messi should show real per-match rates, not season totals):")
    print(messi[["player", "team", "season", "matches", "goals_per_match", "assists_per_match"]].to_string(index=False))


if __name__ == "__main__":
    main()
