from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Best-effort FBref advanced downloader using the optional soccerdata package. "
            "It writes raw provider CSVs; normalize them with scripts/import_advanced_csv.py. "
            "If soccerdata is not installed or FBref blocks access, the script reports blocked instead of failing silently."
        )
    )
    parser.add_argument("--league", nargs="+", required=True, help='soccerdata league IDs, e.g. "ENG-Premier League"')
    parser.add_argument("--season", nargs="+", required=True, help="Seasons accepted by soccerdata, e.g. 2021 2022 2023")
    parser.add_argument("--stat-type", nargs="+", default=["schedule", "shooting", "keeper", "misc"], help="FBref stat types")
    parser.add_argument("--out-dir", default="data/external/advanced/fbref")
    parser.add_argument("--include-player-match-stats", action="store_true", help="Also download FBref player match stats for the requested stat types.")
    parser.add_argument("--include-lineups", action="store_true", help="Also download FBref lineups when soccerdata supports it.")
    parser.add_argument("--include-events", action="store_true", help="Also download FBref events when soccerdata supports it.")
    args = parser.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "fbref_download_report.json"
    files: list[str] = []
    failures: list[dict[str, str]] = []

    try:
        import soccerdata as sd  # type: ignore
    except Exception as exc:
        report = {
            "version": "v0.50.0_advanced_football_data_layer",
            "provider": "soccerdata_fbref",
            "status": "blocked",
            "reason": "missing_optional_dependency_soccerdata",
            "error": str(exc),
            "hint": "Install soccerdata locally if you want direct FBref scraping, or export FBref/worldfootballR CSVs and use import_advanced_csv.py.",
            "raw_data_changed": False,
            "outputs": {"report": str(report_path)},
        }
        _write_json(report, report_path)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    try:
        seasons = [int(s) if str(s).isdigit() else s for s in args.season]
        fbref = sd.FBref(leagues=args.league, seasons=seasons)
        for stat_type in args.stat_type:
            try:
                df = fbref.read_team_match_stats(stat_type=stat_type)
                path = out_dir / f"fbref_team_match_stats_{stat_type}.csv"
                df.reset_index().to_csv(path, index=False)
                files.append(str(path))
            except Exception as exc:
                failures.append({"stat_type": stat_type, "scope": "team_match_stats", "error": str(exc)})

            if args.include_player_match_stats:
                try:
                    df = fbref.read_player_match_stats(stat_type=stat_type)
                    path = out_dir / f"fbref_player_match_stats_{stat_type}.csv"
                    df.reset_index().to_csv(path, index=False)
                    files.append(str(path))
                except Exception as exc:
                    failures.append({"stat_type": stat_type, "scope": "player_match_stats", "error": str(exc)})

        if args.include_lineups:
            try:
                df = fbref.read_lineup()
                path = out_dir / "fbref_lineups.csv"
                df.reset_index().to_csv(path, index=False)
                files.append(str(path))
            except Exception as exc:
                failures.append({"scope": "lineups", "error": str(exc)})

        if args.include_events:
            try:
                df = fbref.read_events()
                path = out_dir / "fbref_events.csv"
                df.reset_index().to_csv(path, index=False)
                files.append(str(path))
            except Exception as exc:
                failures.append({"scope": "events", "error": str(exc)})
    except Exception as exc:
        failures.append({"stage": "fbref_init_or_download", "error": str(exc)})

    status = "ok" if files else "blocked"
    report = {
        "version": "v0.50.0_advanced_football_data_layer",
        "provider": "soccerdata_fbref",
        "status": status,
        "leagues": args.league,
        "seasons": args.season,
        "stat_types": args.stat_type,
        "include_player_match_stats": args.include_player_match_stats,
        "include_lineups": args.include_lineups,
        "include_events": args.include_events,
        "files": files,
        "failures": failures,
        "failure_count": len(failures),
        "terms_note": "Respect FBref/soccerdata caching, robots, rate limits and provider terms. Prefer local cache and do not scrape aggressively.",
        "raw_data_changed": False,
        "outputs": {"report": str(report_path)},
    }
    _write_json(report, report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
