#!/usr/bin/env python3
"""Import Football-Data.co.uk CSVs into Mundialytics team-match market stats.

Football-Data columns supported:
- FTHG/FTAG: goals
- HS/AS: shots
- HST/AST: shots on target
- HC/AC: corners
- HF/AF: fouls
- HY/AY: yellow cards
- HR/AR: red cards

This is the easiest free historical source for corners in many club leagues.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.extra_match_stats import parse_football_data_csvs


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def _discover(paths: list[str], dirs: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in paths or []:
        p = _resolve(item)
        if p.exists():
            out.append(p)
    for d in dirs or []:
        dp = _resolve(d)
        if dp.exists():
            out.extend(sorted(dp.rglob("*.csv")))
    return list(dict.fromkeys(out))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Import Football-Data.co.uk match stats into team_match_market_stats.csv")
    ap.add_argument("--csv", action="append", default=[], help="One Football-Data CSV. Can be repeated.")
    ap.add_argument("--csv-dir", action="append", default=[], help="Directory containing Football-Data CSV files. Can be repeated.")
    ap.add_argument("--out", default="data/processed/team_match_market_stats.csv")
    ap.add_argument(
        "--derive-saves-from-sot",
        action="store_true",
        help=(
            "Explicitly estimate goalkeeper saves from opponent shots on target minus goals conceded. "
            "This is useful for historical research when Football-Data lacks saves, but it is lower quality "
            "than real provider saves and is flagged as derived."
        ),
    )
    args = ap.parse_args(argv)

    files = _discover(args.csv, args.csv_dir)
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = parse_football_data_csvs(files, derive_saves_from_sot=args.derive_saves_from_sot)
    df.to_csv(out_path, index=False)
    summary = {
        "version": "v0.37.2_extra_match_stats_ingestion",
        "source": "football-data.co.uk",
        "input_files": [str(p) for p in files],
        "rows": int(len(df)),
        "matches": int(df["match_id"].nunique()) if not df.empty else 0,
        "output": str(out_path),
        "corners_rows": int(df["corners_for"].notna().sum()) if "corners_for" in df.columns else 0,
        "saves_rows": int(df["saves_for"].notna().sum()) if "saves_for" in df.columns else 0,
        "derived_saves_from_sot": bool(args.derive_saves_from_sot),
        "derived_saves_rows": int((df.get("saves_data_quality_flag", pd.Series(dtype=str)).astype(str).eq("derived_saves_from_sot_minus_goals")).sum()) if not df.empty else 0,
        "note": "Football-Data usually has corners but not real goalkeeper saves. Use --derive-saves-from-sot for an auditable lower-quality saves approximation.",
    }
    (out_path.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("MUNDIALYTICS FOOTBALL-DATA STATS IMPORT")
    print(f"Input files: {len(files)}")
    print(f"Rows: {len(df)} | matches: {df['match_id'].nunique() if not df.empty else 0}")
    print(f"Output: {out_path}")
    print(f"Corners rows: {summary['corners_rows']} | Saves rows: {summary['saves_rows']} | Derived saves rows: {summary.get('derived_saves_rows', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
