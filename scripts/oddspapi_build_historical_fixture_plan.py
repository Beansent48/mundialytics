#!/usr/bin/env python3
"""Build a conservative OddsPapi historical fixture-discovery plan from internal matches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.betting.historical_odds_backfill import normalize_internal_matches, build_fixture_request_windows, date_column_diagnostics, summarize_fixture_plan


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build historical OddsPapi fixture request windows from internal historical matches.")
    parser.add_argument("--matches", required=True, help="Internal historical matches CSV with match_id/date/home_team/away_team.")
    parser.add_argument("--out-dir", default="outputs/oddspapi_historical_fixture_plan_current")
    parser.add_argument("--min-date", default="2026-01-01", help="OddsPapi historical odds availability starts at 2026-01-01 per docs; keep this default unless verified.")
    parser.add_argument("--chunk-hours", type=int, default=216, help="Requested raw chunk size. Effective window is capped below the OddsPapi sportId+from/to 10-day limit.")
    parser.add_argument("--pad-hours", type=int, default=6, help="Extra hours added before/after each raw chunk for safer provider matching.")
    parser.add_argument("--api-max-window-hours", type=float, default=239.0, help="Safety cap for full API span from/to. Keep below 240h because OddsPapi docs require under 10 days.")
    parser.add_argument("--no-clamp-to-internal-range", action="store_true", help="Allow padding before the first internal match and after the last one. Default keeps API windows synced to the prepared internal date range.")
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args(argv)

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(_resolve(args.matches), low_memory=False)
    (out_dir / "input_column_diagnostics.json").write_text(json.dumps(date_column_diagnostics(raw), indent=2, ensure_ascii=False), encoding="utf-8")
    matches = normalize_internal_matches(raw, min_date=args.min_date)
    windows = build_fixture_request_windows(
        matches,
        chunk_hours=args.chunk_hours,
        pad_hours=args.pad_hours,
        max_windows=args.max_windows,
        api_max_window_hours=args.api_max_window_hours,
        clamp_to_internal_range=not args.no_clamp_to_internal_range,
    )
    matches.to_csv(out_dir / "internal_matches_prepared.csv", index=False)
    windows.to_csv(out_dir / "fixture_request_windows.csv", index=False)
    plan_diag = summarize_fixture_plan(matches, windows)
    summary = {
        "version": "v0.46.4_oddspapi_historical_fixture_plan_safe_windows",
        "input_rows": int(len(raw)),
        "prepared_matches": int(len(matches)),
        "request_windows": int(len(windows)),
        "min_date": args.min_date,
        "requested_chunk_hours": args.chunk_hours,
        "pad_hours": args.pad_hours,
        "api_max_window_hours": args.api_max_window_hours,
        "bookmaker_strategy": "bet365_only_recommended",
        "clamp_to_internal_range": not args.no_clamp_to_internal_range,
        "sync_policy": "internal_matches_minmax_after_min_date; api windows are generated from prepared internal match dates; default clamps first/last API window to the prepared internal range",
        "diagnostics": plan_diag,
        "outputs": ["input_column_diagnostics.json", "internal_matches_prepared.csv", "fixture_request_windows.csv", "fixture_plan_summary.json"],
    }
    (out_dir / "fixture_plan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
