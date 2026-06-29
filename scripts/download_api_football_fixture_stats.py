#!/usr/bin/env python3
"""Download API-Football fixture statistics JSONs for corners and goalkeeper saves.

Requires an API-Football/API-Sports key. Set it in an environment variable:
  $env:API_FOOTBALL_KEY="your_key"

Examples:
  python scripts/download_api_football_fixture_stats.py --league 39 --season 2024 --max-fixtures 20
  python scripts/download_api_football_fixture_stats.py --fixture-id 1035037 --fixture-id 1035038

Saved files are compatible with scripts/import_provider_fixture_stats.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://v3.football.api-sports.io"


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def _api_get(session: requests.Session, endpoint: str, params: dict, api_key: str, timeout: int = 40) -> dict:
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    headers = {"x-apisports-key": api_key, "User-Agent": "Mundialytics/0.37.1"}
    resp = session.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download API-Football fixture statistics JSONs")
    ap.add_argument("--api-key", default=os.environ.get("API_FOOTBALL_KEY") or os.environ.get("APISPORTS_KEY") or "")
    ap.add_argument("--league", action="append", default=[], help="API-Football league id, e.g. 39 Premier League. Can be repeated.")
    ap.add_argument("--season", action="append", default=[], help="Season year, e.g. 2024. Can be repeated.")
    ap.add_argument("--fixture-id", action="append", default=[], help="Specific fixture id. Can be repeated.")
    ap.add_argument("--out-dir", default="data/raw/provider_fixture_stats/api_football")
    ap.add_argument("--max-fixtures", type=int, default=0, help="Limit fixture count per league-season, useful for testing")
    ap.add_argument("--sleep", type=float, default=0.35, help="Pause between API requests")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.api_key and not args.dry_run:
        print("Falta API key. Usa: $env:API_FOOTBALL_KEY='tu_key' o --api-key tu_key")
        return 2

    session = requests.Session()
    fixture_items: list[dict] = []
    results: list[dict] = []

    # Direct fixtures supplied by user.
    for fid in args.fixture_id:
        fixture_items.append({"fixture": {"id": int(fid) if str(fid).isdigit() else fid}, "league": {}, "teams": {}})

    # Discover fixtures from league-season pairs.
    for league in args.league:
        for season in args.season:
            if args.dry_run:
                results.append({"kind": "fixtures", "league": league, "season": season, "status": "dry_run", "endpoint": "/fixtures"})
                continue
            try:
                payload = _api_get(session, "fixtures", {"league": league, "season": season}, args.api_key)
                rows = payload.get("response") or []
                if args.max_fixtures and len(rows) > args.max_fixtures:
                    rows = rows[: args.max_fixtures]
                fixture_items.extend(rows)
                results.append({"kind": "fixtures", "league": league, "season": season, "status": "ok", "rows": len(rows)})
            except Exception as e:
                results.append({"kind": "fixtures", "league": league, "season": season, "status": "error", "error": repr(e)})
            time.sleep(args.sleep)

    # De-duplicate fixtures.
    seen = set()
    deduped = []
    for item in fixture_items:
        fid = str((item.get("fixture") or {}).get("id") or "")
        if not fid or fid in seen:
            continue
        seen.add(fid)
        deduped.append(item)

    if args.dry_run:
        print("MUNDIALYTICS API-FOOTBALL STATS DOWNLOADER DRY RUN")
        print(f"Fixture ids supplied: {len(args.fixture_id)}")
        print(f"League-season pairs: {len(args.league) * len(args.season)}")
        return 0

    written = 0
    for item in deduped:
        fixture = item.get("fixture") or {}
        fid = str(fixture.get("id") or "")
        if not fid:
            continue
        try:
            stats_payload = _api_get(session, "fixtures/statistics", {"fixture": fid}, args.api_key)
            merged = {
                "fixture": fixture,
                "league": item.get("league") or {},
                "teams": item.get("teams") or {},
                "goals": item.get("goals") or {},
                "score": item.get("score") or {},
                "response": stats_payload.get("response") or [],
                "raw_statistics_payload": stats_payload,
            }
            league_id = str((item.get("league") or {}).get("id") or "unknown_league")
            season = str((item.get("league") or {}).get("season") or "unknown_season")
            target_dir = out_dir / league_id / season
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"fixture_{fid}_statistics.json"
            target.write_text(json.dumps(merged, indent=2), encoding="utf-8")
            written += 1
            results.append({"kind": "statistics", "fixture_id": fid, "status": "ok", "path": str(target), "teams": len(merged["response"])})
        except Exception as e:
            results.append({"kind": "statistics", "fixture_id": fid, "status": "error", "error": repr(e)})
        time.sleep(args.sleep)

    summary = {
        "version": "v0.37.1_api_football_stats_downloader",
        "out_dir": str(out_dir),
        "fixtures_discovered": len(deduped),
        "json_files_written": written,
        "results": results,
    }
    (out_dir / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("MUNDIALYTICS API-FOOTBALL STATS DOWNLOADER")
    print(f"Fixtures discovered: {len(deduped)}")
    print(f"JSON files written: {written}")
    print(f"Output dir: {out_dir}")
    print(f"Summary: {out_dir / 'download_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
