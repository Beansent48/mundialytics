#!/usr/bin/env python3
"""Create a combined provider readiness report for OddsPapi + Creativesdev."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.providers.api_config import discover_config_path, provider_runtime


def _resolve(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def provider_score(provider: str, endpoints: dict) -> dict:
    keys = set(endpoints or {})
    if provider == "oddspapi":
        return {
            "role": "odds_provider",
            "training_use": "historical/pre-kickoff odds features + value/EV backtest",
            "must_have": ["sports", "bookmakers", "markets", "fixtures", "fixture_odds", "fixture_historical_odds"],
            "configured_keys": sorted(keys),
            "decision": "use_for_odds_pilot",
        }
    return {
        "role": "football_data_provider",
        "training_use": "fixtures, lineups, events, team/player stats; only use odds if endpoint quality is proven",
        "must_have": ["fixtures_by_date", "fixture_lineups", "fixture_statistics", "fixture_events"],
        "configured_keys": sorted(keys),
        "decision": "probe_before_training",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit external API provider config and readiness.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", default="outputs/provider_rapidapi_audit_current")
    args = parser.parse_args(argv)
    cfg_path = discover_config_path(_resolve(args.config))
    if cfg_path is None:
        raise SystemExit("No provider config found.")
    out_dir = _resolve(args.out_dir) or (ROOT / "outputs/provider_rapidapi_audit_current")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ["oddspapi", "creativesdev_live_football"]:
        rt = provider_runtime(name, cfg_path, required=False)
        rows.append({
            "provider": name,
            "mode": rt.mode,
            "base_url": rt.base_url,
            "host": rt.host,
            "monthly_budget": rt.monthly_budget,
            "max_calls_per_run": rt.max_calls_per_run,
            **provider_score(name, rt.endpoints),
        })
    report = {
        "version": "v0.44_multi_provider_rapidapi_config",
        "config_path": str(cfg_path),
        "providers": rows,
        "strict_rules": [
            "Do not train betting ROI models with post-kickoff or post-match odds.",
            "OddsPapi is the odds source of truth until Creativesdev odds coverage is proven.",
            "Creativesdev data must pass fixture/lineup/stat coverage audit before being merged into model features.",
            "Never overwrite local config files in update ZIPs.",
        ],
    }
    (out_dir / "provider_rapidapi_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
