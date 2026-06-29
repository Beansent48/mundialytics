#!/usr/bin/env python3
"""Download StatsBomb Open Data raw event JSONs.

Examples:
  python scripts/download_statsbomb_open_data_events.py --list-competitions
  python scripts/download_statsbomb_open_data_events.py --all-competitions --out-dir data/raw/statsbomb/events
  python scripts/download_statsbomb_open_data_events.py --competition-id 43 --season-id 106 --max-matches 20

The script writes event files compatible with:
  scripts/build_statsbomb_raw_extra_stats.py
  scripts/build_statsbomb_raw_goalkeeper_stats.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def _get_json(session: requests.Session, url: str, timeout: int = 45) -> Any:
    resp = session.get(url, timeout=timeout, headers={"User-Agent": "Mundialytics/0.38.1"})
    resp.raise_for_status()
    return resp.json()


def _team_name(obj: Any) -> str:
    if isinstance(obj, dict):
        return str(obj.get("home_team_name") or obj.get("away_team_name") or obj.get("name") or obj.get("id") or "")
    return str(obj or "")


def _as_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _competition_row(c: dict[str, Any]) -> dict[str, Any]:
    return {
        "competition_id": c.get("competition_id"),
        "season_id": c.get("season_id"),
        "competition_name": c.get("competition_name"),
        "season_name": c.get("season_name"),
        "country_name": c.get("country_name"),
    }


def _passes_name_filters(row: dict[str, Any], include_terms: list[str], exclude_terms: list[str]) -> bool:
    haystack = " ".join(str(row.get(k) or "") for k in ["competition_name", "season_name", "country_name"]).lower()
    if include_terms and not any(t.lower() in haystack for t in include_terms):
        return False
    if exclude_terms and any(t.lower() in haystack for t in exclude_terms):
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download StatsBomb Open Data raw event files")
    ap.add_argument("--competition-id", action="append", default=[], help="StatsBomb competition_id. Can repeat.")
    ap.add_argument("--season-id", action="append", default=[], help="StatsBomb season_id. Can repeat.")
    ap.add_argument("--match-id", action="append", default=[], help="Specific match_id. Can repeat; skips competition discovery.")
    ap.add_argument("--out-dir", default="data/raw/statsbomb/events")
    ap.add_argument("--list-competitions", action="store_true")
    ap.add_argument("--all-competitions", action="store_true", help="Download every competition/season pair listed by StatsBomb Open Data.")
    ap.add_argument("--include-name", action="append", default=[], help="Optional text filter over competition/season/country names. Can repeat.")
    ap.add_argument("--exclude-name", action="append", default=[], help="Optional text exclusion filter over competition/season/country names. Can repeat.")
    ap.add_argument("--min-season-year", type=int, default=0, help="Optional lower bound from the first 4-digit year in season_name, e.g. 2010.")
    ap.add_argument("--max-competition-seasons", type=int, default=0, help="Optional cap on competition-season rows discovered.")
    ap.add_argument("--max-matches", type=int, default=0, help="Global max event files to download after discovery/deduplication.")
    ap.add_argument("--max-matches-per-season", type=int, default=0, help="Cap matches per competition-season row before global dedupe.")
    ap.add_argument("--skip-existing", action="store_true", help="Do not re-download event JSONs that already exist.")
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    results: list[dict[str, Any]] = []

    competitions_url = f"{BASE}/competitions.json"
    try:
        competitions = _get_json(session, competitions_url)
    except Exception as e:
        print(f"No he podido descargar competitions.json: {e!r}")
        return 2

    comp_rows = [_competition_row(c) for c in competitions]
    comp_rows = [r for r in comp_rows if _passes_name_filters(r, args.include_name, args.exclude_name)]
    if args.min_season_year:
        filtered = []
        for r in comp_rows:
            season = str(r.get("season_name") or "")
            year = None
            for token in season.replace("/", " ").split():
                if len(token) == 4 and token.isdigit():
                    year = int(token)
                    break
            if year is None or year >= args.min_season_year:
                filtered.append(r)
        comp_rows = filtered

    if args.list_competitions:
        print("StatsBomb Open Data competitions/seasons:")
        for r in comp_rows[:500]:
            print(f"competition_id={r['competition_id']} | season_id={r['season_id']} | {r['competition_name']} | {r['season_name']} | {r['country_name']}")
        print(f"Total rows: {len(comp_rows)}")
        return 0

    match_items: list[dict[str, Any]] = []
    discovered_competition_seasons: list[dict[str, Any]] = []

    if args.match_id:
        for mid in args.match_id:
            match_items.append({"match_id": str(mid)})
    else:
        if args.all_competitions:
            selected_pairs = comp_rows
        else:
            if not args.competition_id or not args.season_id:
                print("Indica --competition-id y --season-id, usa --all-competitions, usa --list-competitions, o pasa --match-id.")
                return 1
            selected_pairs = []
            # Preserve old behavior: all combinations of user-supplied competition/season ids.
            for comp_id in args.competition_id:
                for season_id in args.season_id:
                    selected_pairs.append({"competition_id": comp_id, "season_id": season_id, "competition_name": None, "season_name": None, "country_name": None})

        if args.max_competition_seasons:
            selected_pairs = selected_pairs[: args.max_competition_seasons]

        for row in selected_pairs:
            comp_id = row.get("competition_id")
            season_id = row.get("season_id")
            url = f"{BASE}/matches/{comp_id}/{season_id}.json"
            try:
                matches = _get_json(session, url)
                rows_before_cap = len(matches)
                if args.max_matches_per_season:
                    matches = matches[: args.max_matches_per_season]
                for m in matches:
                    m = dict(m)
                    m["competition_id"] = comp_id
                    m["season_id"] = season_id
                    if row.get("competition_name"):
                        m["competition_name"] = row.get("competition_name")
                    if row.get("season_name"):
                        m["season_name"] = row.get("season_name")
                    match_items.append(m)
                discovered_competition_seasons.append(dict(row, matches=len(matches), matches_available=rows_before_cap))
                results.append({"kind": "matches", "competition_id": comp_id, "season_id": season_id, "status": "ok", "rows": len(matches), "rows_available": rows_before_cap})
            except Exception as e:
                results.append({"kind": "matches", "competition_id": comp_id, "season_id": season_id, "status": "error", "error": repr(e)})
            time.sleep(args.sleep)

    seen = set()
    deduped = []
    for m in match_items:
        mid = str(m.get("match_id") or "")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        deduped.append(m)

    if args.max_matches:
        deduped = deduped[: args.max_matches]

    if args.dry_run:
        print("MUNDIALYTICS STATSBOMB OPEN DATA DOWNLOADER DRY RUN")
        print(f"Competition-season rows selected: {len(discovered_competition_seasons)}")
        print(f"Matches discovered: {len(deduped)}")
        for m in deduped[:30]:
            print(m.get("match_id"), m.get("match_date"), _team_name(m.get("home_team")), "vs", _team_name(m.get("away_team")))
        dry_summary = {
            "version": "v0.38.1_statsbomb_open_data_downloader",
            "mode": "dry_run",
            "all_competitions": bool(args.all_competitions),
            "competition_seasons_selected": discovered_competition_seasons,
            "matches_discovered": len(deduped),
        }
        (out_dir / "download_dry_run_summary.json").write_text(json.dumps(dry_summary, indent=2), encoding="utf-8")
        print(f"Dry summary: {out_dir / 'download_dry_run_summary.json'}")
        return 0

    written = 0
    skipped_existing = 0
    for m in deduped:
        mid = str(m.get("match_id"))
        event_path = out_dir / f"{mid}.json"
        if args.skip_existing and event_path.exists():
            skipped_existing += 1
            results.append({"kind": "events", "match_id": mid, "status": "skipped_existing"})
            continue
        url = f"{BASE}/events/{mid}.json"
        try:
            events = _get_json(session, url)
            event_path.write_text(json.dumps(events), encoding="utf-8")
            meta = {
                "match_id": mid,
                "match_date": m.get("match_date"),
                "competition_id": m.get("competition_id"),
                "season_id": m.get("season_id"),
                "competition_name": _team_name(m.get("competition")) or m.get("competition_name"),
                "season_name": _team_name(m.get("season")) or m.get("season_name"),
                "home_team": _team_name(m.get("home_team")),
                "away_team": _team_name(m.get("away_team")),
            }
            (out_dir / f"{mid}.metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            written += 1
            results.append({"kind": "events", "match_id": mid, "status": "ok", "events": len(events) if isinstance(events, list) else None})
        except Exception as e:
            results.append({"kind": "events", "match_id": mid, "status": "error", "error": repr(e)})
        time.sleep(args.sleep)

    summary = {
        "version": "v0.38.1_statsbomb_open_data_downloader",
        "out_dir": str(out_dir),
        "all_competitions": bool(args.all_competitions),
        "competition_seasons_selected": discovered_competition_seasons,
        "competition_season_rows_selected": len(discovered_competition_seasons),
        "matches_discovered": len(deduped),
        "event_files_written": written,
        "event_files_skipped_existing": skipped_existing,
        "results": results,
    }
    (out_dir / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("MUNDIALYTICS STATSBOMB OPEN DATA DOWNLOADER")
    print(f"All competitions: {bool(args.all_competitions)}")
    print(f"Competition-season rows selected: {len(discovered_competition_seasons)}")
    print(f"Matches discovered: {len(deduped)}")
    print(f"Event files written: {written} | skipped existing: {skipped_existing}")
    print(f"Output dir: {out_dir}")
    print(f"Summary: {out_dir / 'download_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
