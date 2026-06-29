#!/usr/bin/env python3
"""Convenience wrapper to download all StatsBomb Open Data event JSONs.

This is equivalent to:
  python scripts/download_statsbomb_open_data_events.py --all-competitions --skip-existing
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download all StatsBomb Open Data event JSONs")
    ap.add_argument("--out-dir", default="data/raw/statsbomb/events")
    ap.add_argument("--sleep", default="0.10")
    ap.add_argument("--max-matches", default="0", help="Optional global cap for testing. 0 means no cap.")
    ap.add_argument("--min-season-year", default="0", help="Optional lower bound for season start year. 0 means no date filter.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "download_statsbomb_open_data_events.py"),
        "--all-competitions",
        "--skip-existing",
        "--out-dir",
        args.out_dir,
        "--sleep",
        str(args.sleep),
    ]
    if str(args.max_matches) != "0":
        cmd += ["--max-matches", str(args.max_matches)]
    if str(args.min_season_year) != "0":
        cmd += ["--min-season-year", str(args.min_season_year)]
    if args.dry_run:
        cmd.append("--dry-run")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
