#!/usr/bin/env python3
"""Build leakage-safe pre-match snapshots from historical odds ticks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.betting.historical_odds_backfill import build_snapshot_rows, SNAPSHOT_OFFSETS_SECONDS


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def _parse_snapshots(text: str) -> dict[str, int]:
    if not text:
        return SNAPSHOT_OFFSETS_SECONDS
    out = {}
    aliases = dict(SNAPSHOT_OFFSETS_SECONDS)
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if token in aliases:
            out[token] = aliases[token]
            continue
        if ":" in token:
            label, seconds = token.split(":", 1)
            out[label.strip()] = int(seconds.strip())
            continue
        raise ValueError(f"Unknown snapshot token: {token}. Use t24h,t6h,t1h,t10m,closing or label:seconds.")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build pre-match snapshot odds from historical odds ticks.")
    parser.add_argument("--historical-odds-ticks", required=True)
    parser.add_argument("--fixture-mapping", required=True)
    parser.add_argument("--out-dir", default="outputs/oddspapi_snapshot_odds_current")
    parser.add_argument("--snapshots", default="t24h,t6h,t1h,t10m,closing")
    parser.add_argument("--allow-closing-at-kickoff", action="store_true")
    args = parser.parse_args(argv)

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ticks = pd.read_csv(_resolve(args.historical_odds_ticks), low_memory=False)
    mapping = pd.read_csv(_resolve(args.fixture_mapping), low_memory=False)
    snapshots = _parse_snapshots(args.snapshots)
    out = build_snapshot_rows(ticks, mapping, snapshot_offsets=snapshots, allow_closing_at_kickoff=args.allow_closing_at_kickoff)
    out.to_csv(out_dir / "historical_odds_snapshots.csv", index=False)
    # A backward-compatible file for existing value scripts: default to t1h if available, otherwise closing.
    if not out.empty:
        preferred = out[out["snapshot_label"].eq("t1h")].copy()
        if preferred.empty:
            preferred = out[out["snapshot_label"].eq("closing")].copy()
        preferred.to_csv(out_dir / "historical_odds_input.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "historical_odds_input.csv", index=False)
    summary = {
        "version": "v0.46_oddspapi_snapshot_odds",
        "input_tick_rows": int(len(ticks)),
        "snapshot_rows": int(len(out)),
        "snapshot_labels": sorted(out["snapshot_label"].dropna().unique().tolist()) if not out.empty and "snapshot_label" in out.columns else [],
        "snapshots_requested": snapshots,
        "leakage_policy": "Each snapshot uses the latest available odds before kickoff minus the requested offset. Do not train on all ticks directly.",
        "outputs": ["historical_odds_snapshots.csv", "historical_odds_input.csv"],
    }
    (out_dir / "snapshot_odds_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
