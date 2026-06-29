#!/usr/bin/env python3
"""Convert normalized Creativesdev fixtures into Mundialytics matchday fixture input.

Input: outputs/creativesdev_probe_current/fixtures_by_date_fixtures_normalized.csv
Output: data/input/generated/today_fixtures_from_creativesdev.csv or custom path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd


def _resolve(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def _date_from_iso(value: object) -> str:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return ""
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc).date().isoformat()
    except Exception:
        return str(value)[:10]


def _year_from_date(value: str) -> str:
    return str(value)[:4] if value else ""


def _status_is_valid(row: pd.Series, include_finished: bool, include_live: bool) -> bool:
    status = str(row.get("status", "") or "").lower()
    short = str(row.get("status_short", "") or "").lower()
    if any(x in status or x in short for x in ["cancel", "postpon", "abandon", "award"]):
        return False
    if any(x in status or x in short for x in ["full", "ft", "finished"]):
        return include_finished
    if any(x in status or x in short for x in ["live", "1h", "2h", "started"]):
        return include_live
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Mundialytics fixtures input from Creativesdev normalized fixtures.")
    parser.add_argument("--fixtures", default="outputs/creativesdev_probe_current/fixtures_by_date_fixtures_normalized.csv")
    parser.add_argument("--out", default="data/input/generated/today_fixtures_from_creativesdev.csv")
    parser.add_argument("--date", default=None, help="Optional UTC date filter, YYYY-MM-DD")
    parser.add_argument("--team", action="append", default=[], help="Optional team filter. Can be repeated, e.g. --team Portugal --team Ghana")
    parser.add_argument("--competition", default=None, help="Optional contains filter over competition/tournament_stage/provider_league_id")
    parser.add_argument("--include-finished", action="store_true")
    parser.add_argument("--include-live", action="store_true")
    parser.add_argument("--team-type", default="club", choices=["club", "national_team"])
    parser.add_argument("--competition-context", default="unknown")
    parser.add_argument("--gender", default="men")
    args = parser.parse_args(argv)

    src = _resolve(args.fixtures)
    out = _resolve(args.out)
    if src is None or not src.exists():
        raise FileNotFoundError(src)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    if df.empty:
        raise RuntimeError(f"No fixture rows in {src}")
    df["date"] = df.get("kickoff_utc", "").map(_date_from_iso)
    df = df[df.apply(lambda r: _status_is_valid(r, args.include_finished, args.include_live), axis=1)].copy()

    if args.date:
        df = df[df["date"].astype(str).eq(args.date)].copy()

    teams = [t.strip().lower() for t in args.team if t and t.strip()]
    if teams:
        df = df[
            df["home_team"].astype(str).str.lower().isin(teams)
            | df["away_team"].astype(str).str.lower().isin(teams)
            | df["home_team_canonical"].astype(str).str.lower().isin(teams)
            | df["away_team_canonical"].astype(str).str.lower().isin(teams)
        ].copy()

    if args.competition:
        needle = args.competition.lower()
        comp_blob = (
            df.get("competition", "").astype(str) + " "
            + df.get("tournament_stage", "").astype(str) + " "
            + df.get("provider_league_id", "").astype(str)
        ).str.lower()
        df = df[comp_blob.str.contains(needle, na=False)].copy()

    out_df = pd.DataFrame({
        "match_id": df["provider_match_id"].astype(str),
        "date": df["date"],
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "neutral": 0,
        "competition": df.get("competition", "").replace("", pd.NA).fillna(df.get("provider_league_id", "")),
        "season": df["date"].map(_year_from_date),
        "stage": df.get("tournament_stage", ""),
        "group": "",
        "team_scope": "club" if args.team_type == "club" else "national",
        "team_type": args.team_type,
        "competition_context": args.competition_context,
        "gender": args.gender,
        "provider": df.get("provider", "creativesdev_live_football"),
        "provider_fixture_id": df["provider_fixture_id"].astype(str),
        "kickoff_utc": df.get("kickoff_utc", ""),
        "status": df.get("status", ""),
    })
    out_df.to_csv(out, index=False)
    print(f"Wrote {len(out_df)} fixtures -> {out}")
    if len(out_df):
        print(out_df[["match_id", "date", "home_team", "away_team", "competition", "kickoff_utc", "status"]].head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
