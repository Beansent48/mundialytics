#!/usr/bin/env python3
"""Audit OddsPapi historical backfill coverage before training/backtesting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.betting.historical_odds_backfill import audit_backfill_coverage


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def _read_optional(path_text: str | None) -> pd.DataFrame | None:
    if not path_text:
        return None
    p = _resolve(path_text)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit historical odds backfill coverage.")
    parser.add_argument("--fixture-mapping", required=True)
    parser.add_argument("--historical-odds-ticks", required=True)
    parser.add_argument("--snapshot-odds", default=None)
    parser.add_argument("--out-dir", default="outputs/oddspapi_backfill_coverage_current")
    args = parser.parse_args(argv)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(_resolve(args.fixture_mapping), low_memory=False)
    ticks = pd.read_csv(_resolve(args.historical_odds_ticks), low_memory=False)
    snapshots = _read_optional(args.snapshot_odds)
    summary = audit_backfill_coverage(mapping, ticks, snapshots)
    summary["version"] = "v0.46_oddspapi_backfill_coverage"
    (out_dir / "backfill_coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Detail tables.
    if not ticks.empty and "market_key" in ticks.columns:
        ticks.groupby(["market_key", "scope", "side"], dropna=False).size().reset_index(name="rows").sort_values("rows", ascending=False).to_csv(out_dir / "market_side_tick_coverage.csv", index=False)
    if snapshots is not None and not snapshots.empty and "snapshot_label" in snapshots.columns:
        snapshots.groupby(["snapshot_label", "market_key", "scope", "side"], dropna=False).size().reset_index(name="rows").sort_values(["snapshot_label", "rows"], ascending=[True, False]).to_csv(out_dir / "snapshot_market_side_coverage.csv", index=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
