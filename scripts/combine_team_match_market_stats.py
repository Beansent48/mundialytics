#!/usr/bin/env python3
"""Combine multiple team_match_market_stats CSVs into one processed table.

Use this when you have Football-Data CSV stats plus provider/API stats. Later files win on
same match_id/team/opponent/is_home, so put more precise/provider-rich sources later.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.data.extra_match_stats import clean_team_match_market_stats


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Combine team-match market stat CSV files")
    ap.add_argument("--csv", action="append", required=True, help="Processed stats CSV. Can be repeated; later files win on duplicates.")
    ap.add_argument("--out", default="data/processed/team_match_market_stats_combined.csv")
    args = ap.parse_args(argv)

    frames = []
    input_files = []
    for item in args.csv:
        p = _resolve(item)
        input_files.append(str(p))
        if p.exists():
            try:
                frames.append(pd.read_csv(p))
            except Exception:
                pass
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        df = pd.concat(frames, ignore_index=True, sort=False)
        df = clean_team_match_market_stats(df)
    else:
        df = clean_team_match_market_stats(pd.DataFrame())
    df.to_csv(out_path, index=False)
    summary = {
        "version": "v0.37.1_combine_team_match_market_stats",
        "input_files": input_files,
        "rows": int(len(df)),
        "matches": int(df["match_id"].nunique()) if not df.empty else 0,
        "corners_rows": int(df["corners_for"].notna().sum()) if "corners_for" in df.columns else 0,
        "saves_rows": int(df["saves_for"].notna().sum()) if "saves_for" in df.columns else 0,
        "output": str(out_path),
    }
    (out_path.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("MUNDIALYTICS COMBINE TEAM MATCH MARKET STATS")
    print(f"Input files: {len(input_files)}")
    print(f"Rows: {len(df)} | matches: {summary['matches']}")
    print(f"Corners rows: {summary['corners_rows']} | Saves rows: {summary['saves_rows']}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
