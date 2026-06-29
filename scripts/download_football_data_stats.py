#!/usr/bin/env python3
"""Download Football-Data.co.uk CSV stats/odds files for extra market training.

Why this exists:
- import_football_data_stats.py expects local CSVs.
- Football-Data publishes season CSV files and season ZIPs.
- The CSVs include common bookmaker-relevant targets such as HC/AC corners, HS/AS shots,
  HST/AST shots on target, HY/AY cards, etc. They generally do NOT include goalkeeper saves.

Examples:
  python scripts/download_football_data_stats.py --seasons 2526 2425 2324 --mode zip
  python scripts/download_football_data_stats.py --seasons 2425 --mode csv --leagues E0 SP1 I1 D1 F1
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]

BASE_URL = "https://www.football-data.co.uk/mmz4281"
DEFAULT_SEASONS = ["2526", "2425", "2324", "2223", "2122"]
DEFAULT_LEAGUES = [
    # England
    "E0", "E1", "E2", "E3",
    # Scotland
    "SC0", "SC1", "SC2", "SC3",
    # Germany, Italy, Spain, France
    "D1", "D2", "I1", "I2", "SP1", "SP2", "F1", "F2",
    # Other common European leagues
    "N1", "B1", "P1", "T1", "G1",
]


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def _get(session: requests.Session, url: str, timeout: int = 30) -> requests.Response:
    resp = session.get(url, timeout=timeout, headers={"User-Agent": "Mundialytics/0.37.1"})
    resp.raise_for_status()
    return resp


def _download_csv(session: requests.Session, season: str, league: str, out_dir: Path, dry_run: bool = False) -> dict:
    url = f"{BASE_URL}/{season}/{league}.csv"
    out_path = out_dir / f"{season}_{league}.csv"
    item = {"season": season, "league": league, "url": url, "path": str(out_path), "status": "pending"}
    if dry_run:
        item["status"] = "dry_run"
        return item
    try:
        resp = _get(session, url)
        text = resp.text
        # Football-Data sometimes returns tiny HTML error pages for absent leagues.
        if "<html" in text[:200].lower() or len(text.strip()) < 20 or "," not in text[:500]:
            item["status"] = "skipped_non_csv"
            item["bytes"] = len(resp.content)
            return item
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.content)
        item["status"] = "ok"
        item["bytes"] = len(resp.content)
    except Exception as e:
        item["status"] = "error"
        item["error"] = repr(e)
    return item


def _download_zip(session: requests.Session, season: str, out_dir: Path, dry_run: bool = False) -> dict:
    url = f"{BASE_URL}/{season}/data.zip"
    season_dir = out_dir / season
    item = {"season": season, "url": url, "path": str(season_dir), "status": "pending", "files": []}
    if dry_run:
        item["status"] = "dry_run"
        return item
    try:
        resp = _get(session, url, timeout=60)
        season_dir.mkdir(parents=True, exist_ok=True)
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        for name in csv_names:
            # Flatten into season directory and prefix season to avoid overwrites when later copied.
            target = season_dir / f"{season}_{Path(name).name}"
            target.write_bytes(zf.read(name))
            item["files"].append(str(target))
        item["status"] = "ok"
        item["bytes"] = len(resp.content)
        item["n_csv"] = len(csv_names)
    except Exception as e:
        item["status"] = "error"
        item["error"] = repr(e)
    return item


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download Football-Data.co.uk CSV files into data/raw/football_data")
    ap.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS, help="Season codes like 2526, 2425, 2324")
    ap.add_argument("--mode", choices=["zip", "csv"], default="zip", help="zip downloads data.zip per season; csv downloads league CSVs one by one")
    ap.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES, help="League codes used only in --mode csv, e.g. E0 SP1 I1 D1 F1")
    ap.add_argument("--out-dir", default="data/raw/football_data", help="Output folder inside the project")
    ap.add_argument("--sleep", type=float, default=0.2, help="Pause between downloads")
    ap.add_argument("--dry-run", action="store_true", help="Print planned URLs without downloading")
    args = ap.parse_args(argv)

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    results: list[dict] = []

    if args.mode == "zip":
        for season in args.seasons:
            results.append(_download_zip(session, season, out_dir, args.dry_run))
            time.sleep(args.sleep)
    else:
        for season in args.seasons:
            for league in args.leagues:
                results.append(_download_csv(session, season, league, out_dir, args.dry_run))
                time.sleep(args.sleep)

    ok = sum(1 for r in results if r.get("status") == "ok")
    n_csv = 0
    for r in results:
        if "n_csv" in r:
            n_csv += int(r.get("n_csv") or 0)
        elif r.get("status") == "ok":
            n_csv += 1
    summary = {
        "version": "v0.37.1_football_data_downloader",
        "source": "football-data.co.uk",
        "mode": args.mode,
        "seasons": args.seasons,
        "leagues": args.leagues if args.mode == "csv" else "all_available_in_season_zip",
        "out_dir": str(out_dir),
        "ok_downloads": ok,
        "csv_files_written": n_csv,
        "results": results,
    }
    (out_dir / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("MUNDIALYTICS FOOTBALL-DATA DOWNLOADER")
    print(f"Mode: {args.mode}")
    print(f"Seasons: {', '.join(args.seasons)}")
    print(f"Output dir: {out_dir}")
    print(f"OK downloads: {ok} | CSV files written: {n_csv}")
    print(f"Summary: {out_dir / 'download_summary.json'}")
    if args.dry_run:
        for r in results[:20]:
            print(r["url"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
