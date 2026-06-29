from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.enrichment.advanced import import_advanced_provider_csv


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize a provider/manual advanced match-stat CSV to Mundialytics' canonical advanced match contract. "
            "Use this for FBref/soccerdata/worldfootballR exports, Kaggle exports, RapidAPI exports or any future provider."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--provider", default="provider_csv")
    parser.add_argument("--out-dir", default="data/external/advanced/provider_csv")
    args = parser.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = import_advanced_provider_csv(_resolve(args.input), provider=args.provider)

    match_path = out_dir / f"{args.provider}_advanced_match_stats.csv"
    player_path = out_dir / f"{args.provider}_player_match_stats.csv"
    shot_path = out_dir / f"{args.provider}_shot_events.csv"
    lineup_path = out_dir / f"{args.provider}_lineups.csv"
    report_path = out_dir / f"{args.provider}_advanced_import_report.json"

    outputs.advanced_matches.to_csv(match_path, index=False)
    outputs.player_matches.to_csv(player_path, index=False)
    outputs.shot_events.to_csv(shot_path, index=False)
    outputs.lineups.to_csv(lineup_path, index=False)

    report = dict(outputs.report)
    report["outputs"] = {
        "report": str(report_path),
        "advanced_match_stats": str(match_path),
        "player_match_stats": str(player_path),
        "shot_events": str(shot_path),
        "lineups": str(lineup_path),
    }
    _write_json(report, report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
