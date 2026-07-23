from __future__ import annotations

"""ONE-COMMAND season update — keeps the whole system fresh for 2026/27+.

    .venv/Scripts/python.exe scripts/update_season.py            # weekly in-season
    .venv/Scripts/python.exe scripts/update_season.py --full     # + regenerate the
                                                                 # Resultados walk-forward cache

Steps (each tolerant — a failed download never corrupts existing data; all
Understat writes are append + dedupe):
  1. football-data.co.uk CSVs for the current (and previous) season
  2. Understat shots + player-match stats for the current season
  3. rebuild the Understat xG match aggregation (canonical + team-match)
  4. rebuild the modeling foundation from the raw CSVs
  5. re-attach the xG columns to the foundation (the builder predates xG)
  6. refresh canonical_matches_with_xg (walk-forward context joins)
  7. prune the fitted-props cache (the app refits + recaches on next load)
  8. [--full] regenerate the deployed-chain walk-forward cache (Resultados page)

The engines and props models need no manual retrain: they fit from the
foundation at app load, and the props joblib cache key includes the data's
max date, so it self-invalidates.
"""

import argparse
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
UNDERSTAT_LEAGUES = ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga",
                     "ITA-Serie A", "FRA-Ligue 1"]
SHOTS_CSV = ROOT / "data/external/advanced/understat/understat_shots.csv"
PLAYER_CSV = ROOT / "data/external/advanced/understat/understat_player_match.csv"
FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
TEAM_MATCH = ROOT / "data/processed/understat_team_match_xg.csv"


def season_codes(today: date) -> tuple[str, str]:
    """('2627', '2526') style current + previous season codes (July = new season)."""
    y = today.year % 100
    if today.month >= 7:
        return f"{y:02d}{(y + 1) % 100:02d}", f"{(y - 1) % 100:02d}{y:02d}"
    return f"{(y - 1) % 100:02d}{y:02d}", f"{(y - 2) % 100:02d}{(y - 1) % 100:02d}"


def run_step(name: str, cmd: list[str]) -> bool:
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    ok = r.returncode == 0
    print(f"    {'OK' if ok else 'FAILED'} ({time.time()-t0:.0f}s)", flush=True)
    return ok


def update_understat(season: str) -> None:
    """Append + dedupe shots and player-match stats for one season code."""
    from soccerdata import Understat
    for out, reader, dedupe in [
        (SHOTS_CSV, "read_shot_events", ["shot_id"]),
        (PLAYER_CSV, "read_player_match_stats", ["game_id", "player_id"]),
    ]:
        existing = pd.read_csv(out) if out.exists() else pd.DataFrame()
        frames = []
        for lg in UNDERSTAT_LEAGUES:
            try:
                u = Understat(leagues=lg, seasons=season)
                df = getattr(u, reader)().reset_index()
                frames.append(df)
                print(f"    OK   {out.name} {lg} {season}: {len(df)} rows", flush=True)
            except Exception as exc:
                print(f"    FAIL {out.name} {lg} {season}: {str(exc)[:120]}", flush=True)
        if not frames:
            continue
        combined = pd.concat([existing, *frames], ignore_index=True)
        keys = [k for k in dedupe if k in combined.columns]
        if keys:
            combined = combined.drop_duplicates(subset=keys, keep="first")
        combined.to_csv(out, index=False)
        print(f"    {out.name}: {len(existing)} -> {len(combined)} rows", flush=True)


def augment_foundation_with_xg() -> None:
    """Re-attach the 8 xG columns (the foundation builder predates xG)."""
    found = pd.read_csv(FOUND, low_memory=False)
    xg_cols = ["home_xg", "away_xg", "home_npxg", "away_npxg",
               "home_xg_op", "away_xg_op", "home_xg_sp", "away_xg_sp"]
    found = found.drop(columns=[c for c in xg_cols if c in found.columns])
    tm = pd.read_csv(TEAM_MATCH)[["date", "home_team_fd", "away_team_fd"] + xg_cols]
    tm = tm.rename(columns={"home_team_fd": "home_team", "away_team_fd": "away_team"})
    tm["date"] = pd.to_datetime(tm["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    tm = tm.drop_duplicates(subset=["date", "home_team", "away_team"])
    merged = found.merge(tm, on=["date", "home_team", "away_team"], how="left")
    merged.to_csv(FOUND, index=False)
    cov = merged["home_xg"].notna().mean()
    print(f"    foundation: {len(merged)} matches, xG coverage {cov:.1%}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=None, help="Season code like 2627 (auto from date)")
    ap.add_argument("--skip-understat", action="store_true", help="Skip the network-heavy Understat pulls")
    ap.add_argument("--full", action="store_true",
                    help="Also regenerate the deployed walk-forward cache (Resultados page, ~15 min)")
    args = ap.parse_args()

    cur, prev = season_codes(date.today())
    cur = args.season or cur
    print(f"Season update: current={cur} (also refreshing {prev})", flush=True)

    run_step("1/8 football-data CSVs",
             [PY, "scripts/download_football_data_stats.py", "--seasons", cur, prev,
              "--mode", "csv", "--leagues", "E0", "SP1", "I1", "D1", "F1"])

    if not args.skip_understat:
        print("\n=== 2/8 Understat (shots + player-match) ===", flush=True)
        sys.path.insert(0, str(ROOT / "src"))
        update_understat(cur)
    else:
        print("\n=== 2/8 Understat SKIPPED ===", flush=True)

    run_step("3/8 Understat xG aggregation", [PY, "scripts/build_understat_xg_matches.py"])
    run_step("4/8 foundation rebuild", [PY, "scripts/build_foundation_big5_historical.py"])

    print("\n=== 5/8 foundation xG augment ===", flush=True)
    augment_foundation_with_xg()

    run_step("6/8 canonical_matches_with_xg",
             [PY, "scripts/enrich_matches_with_xg.py",
              "--matches", "data/processed/foundation_big5_multi_season.csv",
              "--xg", "data/external/xg/understat/understat_xg_matches.csv",
              "--provider", "understat",
              "--out-dir", "data/processed/enriched/understat_xg",
              "--allow-missing-xg"])

    print("\n=== 7/8 prune fitted-props cache ===", flush=True)
    n = 0
    for f in (ROOT / "data/processed/cache").glob("props_models_*.joblib"):
        f.unlink(missing_ok=True)
        n += 1
    print(f"    removed {n} cache file(s); the app refits + recaches on next load", flush=True)

    if args.full:
        run_step("8/8 deployed walk-forward cache (Resultados)",
                 [PY, "scripts/generate_deployed_walkforward.py"])
    else:
        print("\n=== 8/8 walk-forward cache SKIPPED (use --full) ===", flush=True)

    # ── summary ────────────────────────────────────────────────────────────────
    print("\n===== SUMMARY =====", flush=True)
    for label, p, datecol in [
        ("foundation", FOUND, "date"),
        ("team xG", TEAM_MATCH, "date"),
        ("player-match", PLAYER_CSV, None),
        ("shots", SHOTS_CSV, None),
    ]:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            last = f" | last: {pd.to_datetime(df[datecol], errors='coerce').max():%Y-%m-%d}" if datecol else ""
            print(f"  {label:14s} {len(df):>8,} rows{last}", flush=True)


if __name__ == "__main__":
    main()
