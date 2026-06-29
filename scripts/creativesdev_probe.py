#!/usr/bin/env python3
"""Safely probe Creativesdev Free API Live Football Data via RapidAPI.

Because RapidAPI playground endpoints can change and are not reliably exposed in search,
this script reads endpoint paths from the external config. Use --dry-run first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.data.adapters.creativesdev import (
    CreativesDevClient,
    normalize_events,
    normalize_fixtures,
    normalize_lineups,
    normalize_match_statistics,
)
from mundialytics.providers.api_config import endpoint_spec, provider_runtime, render_template


def _resolve(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def _parse_vars(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    out = {}
    for part in text.split(","):
        if not part.strip():
            continue
        k, _, v = part.partition("=")
        out[k.strip()] = v.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe a configured Creativesdev RapidAPI endpoint with hard call limits.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--endpoint-key", required=True, help="Key under providers.creativesdev_live_football.endpoints, e.g. fixtures_by_date")
    parser.add_argument("--vars", default=None, help="Template variables, e.g. date=2026-06-23,fixture_id=123")
    parser.add_argument("--out-dir", default="outputs/creativesdev_probe_current")
    parser.add_argument("--max-api-calls", type=int, default=None)
    parser.add_argument("--monthly-budget", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config_path = _resolve(args.config)
    variables = _parse_vars(args.vars)
    out_dir = _resolve(args.out_dir) or (ROOT / "outputs/creativesdev_probe_current")
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    runtime = provider_runtime("creativesdev_live_football", config_path, required=True)
    spec = endpoint_spec(runtime, args.endpoint_key)
    path = render_template(spec.get("path"), variables)
    params = render_template(spec.get("params", {}), variables)
    plan = {
        "provider": "creativesdev_live_football",
        "endpoint_key": args.endpoint_key,
        "base_url": runtime.base_url,
        "host": runtime.host,
        "path": path,
        "params": params,
        "max_api_calls": args.max_api_calls or runtime.max_calls_per_run,
        "monthly_budget": args.monthly_budget or runtime.monthly_budget,
    }
    if args.dry_run:
        (out_dir / "probe_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(json.dumps(plan, indent=2))
        return 0

    client = CreativesDevClient.from_config(provider_config=config_path, max_calls=args.max_api_calls, monthly_budget=args.monthly_budget)
    payload = client.get(str(path), params if isinstance(params, dict) else {}, force=args.force, unwrap=False)
    (raw_dir / f"{args.endpoint_key}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    fixtures = normalize_fixtures(payload)
    lineups = normalize_lineups(payload)
    stats = normalize_match_statistics(payload)
    events = normalize_events(payload)
    fixtures.to_csv(out_dir / f"{args.endpoint_key}_fixtures_normalized.csv", index=False)
    lineups.to_csv(out_dir / f"{args.endpoint_key}_lineups_normalized.csv", index=False)
    stats.to_csv(out_dir / f"{args.endpoint_key}_stats_normalized.csv", index=False)
    events.to_csv(out_dir / f"{args.endpoint_key}_events_normalized.csv", index=False)

    summary = {
        "version": "v0.45_creativesdev_probe",
        "endpoint_key": args.endpoint_key,
        "calls_made": client.calls_made,
        "cache_hits": client.cache_hits,
        "fixtures_rows": int(len(fixtures)),
        "lineups_rows": int(len(lineups)),
        "stats_rows": int(len(stats)),
        "events_rows": int(len(events)),
        "outputs": [
            f"{args.endpoint_key}_fixtures_normalized.csv",
            f"{args.endpoint_key}_lineups_normalized.csv",
            f"{args.endpoint_key}_stats_normalized.csv",
            f"{args.endpoint_key}_events_normalized.csv",
        ],
    }
    (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
