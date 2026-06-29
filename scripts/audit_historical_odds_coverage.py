#!/usr/bin/env python3
"""Audit join coverage between model_market_lines.csv and mapped historical odds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.betting.odds_readiness import audit_historical_odds_coverage


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check whether historical odds map cleanly onto model market lines.")
    parser.add_argument("--model-lines", required=True)
    parser.add_argument("--historical-odds", required=True)
    parser.add_argument("--out-dir", default="outputs/historical_odds_coverage_current")
    args = parser.parse_args(argv)

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_lines = pd.read_csv(_resolve(args.model_lines), low_memory=False)
    historical_odds = pd.read_csv(_resolve(args.historical_odds), low_memory=False)
    by_market, summary = audit_historical_odds_coverage(model_lines, historical_odds)
    by_market.to_csv(out_dir / "historical_odds_coverage_by_market.csv", index=False)
    (out_dir / "historical_odds_coverage_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("MUNDIALYTICS HISTORICAL ODDS COVERAGE v0.41")
    print(json.dumps(summary, indent=2, default=str))
    print("Outputs:")
    print(f"- {out_dir / 'historical_odds_coverage_by_market.csv'}")
    print(f"- {out_dir / 'historical_odds_coverage_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
