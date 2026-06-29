from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.loaders import load_matches
from mundialytics.data_quality.team_registry import load_team_registry
from mundialytics.enrichment.clubelo import download_clubelo_daily_ratings, download_clubelo_team_histories


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download cached ClubElo data. Default mode downloads one full team history per club, "
            "which is much faster than the legacy one-snapshot-per-match-date mode."
        )
    )
    parser.add_argument("--matches", required=True, help="Canonical/enriched matches CSV with date, home_team and away_team columns.")
    parser.add_argument("--registry", default=None, help="Optional team_registry.csv with reviewed clubelo_name aliases.")
    parser.add_argument("--out-dir", default="data/external/clubelo")
    parser.add_argument("--mode", choices=["team-history", "daily-snapshot"], default="team-history")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached files exist.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    args = parser.parse_args()

    matches = load_matches(_resolve(args.matches))
    out_dir = _resolve(args.out_dir)

    if args.mode == "team-history":
        registry = load_team_registry(str(_resolve(args.registry))) if args.registry else None
        outputs = download_clubelo_team_histories(
            matches,
            out_dir,
            registry=registry,
            force=args.force,
            timeout=args.timeout,
            sleep_seconds=args.sleep_seconds,
        )
    else:
        outputs = download_clubelo_daily_ratings(
            matches["date"],
            out_dir,
            force=args.force,
            timeout=args.timeout,
            sleep_seconds=args.sleep_seconds,
        )

    report_path = out_dir / "clubelo_download_report.json"
    report = dict(outputs.report)
    report["inputs"] = {"matches": str(_resolve(args.matches)), "registry": str(_resolve(args.registry)) if args.registry else None}
    report["outputs"] = {"report": str(report_path)}
    _write_json(report, report_path)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
