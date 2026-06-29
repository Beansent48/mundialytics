#!/usr/bin/env python3
"""Import saved provider fixture-stat JSON files.

Currently supports API-Football-style response payloads for fixture statistics. This is where
corners and goalkeeper saves can enter the pipeline from a paid/free API without scraping.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.extra_match_stats import parse_api_football_fixture_stats_json


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
            out.extend(sorted(dp.rglob("*.json")))
    return list(dict.fromkeys(out))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Import provider fixture stats JSON into team_match_market_stats.csv")
    ap.add_argument("--json", action="append", default=[], help="One provider stats JSON. Can be repeated.")
    ap.add_argument("--json-dir", action="append", default=[], help="Directory with provider stats JSON files. Can be repeated.")
    ap.add_argument("--out", default="data/processed/provider_team_match_market_stats.csv")
    args = ap.parse_args(argv)

    files = _discover(args.json, args.json_dir)
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = parse_api_football_fixture_stats_json(files)
    df.to_csv(out_path, index=False)
    summary = {
        "version": "v0.37_provider_fixture_stats_ingestion",
        "source": "provider_fixture_stats",
        "input_files": [str(p) for p in files],
        "rows": int(len(df)),
        "matches": int(df["match_id"].nunique()) if not df.empty else 0,
        "output": str(out_path),
        "corners_rows": int(df["corners_for"].notna().sum()) if "corners_for" in df.columns else 0,
        "saves_rows": int(df["saves_for"].notna().sum()) if "saves_for" in df.columns else 0,
    }
    (out_path.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("MUNDIALYTICS PROVIDER FIXTURE STATS IMPORT")
    print(f"Input files: {len(files)}")
    print(f"Rows: {len(df)} | matches: {df['match_id'].nunique() if not df.empty else 0}")
    print(f"Output: {out_path}")
    print(f"Corners rows: {summary['corners_rows']} | Saves rows: {summary['saves_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
