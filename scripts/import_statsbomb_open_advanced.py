from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.enrichment.advanced import import_statsbomb_open_advanced


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Import StatsBomb Open Data into advanced match, player and shot-event contracts. "
            "This is free/offline but coverage is partial."
        )
    )
    parser.add_argument("--data-dir", default="data/raw/statsbomb/open-data/data")
    parser.add_argument("--out-dir", default="data/external/advanced/statsbomb")
    parser.add_argument("--competition-id", nargs="*", default=None)
    parser.add_argument("--season-id", nargs="*", default=None)
    parser.add_argument("--max-matches", type=int, default=None)
    args = parser.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = import_statsbomb_open_advanced(
        _resolve(args.data_dir),
        competition_ids=args.competition_id,
        season_ids=args.season_id,
        max_matches=args.max_matches,
    )

    match_path = out_dir / "statsbomb_advanced_match_stats.csv"
    player_path = out_dir / "statsbomb_player_match_stats.csv"
    shot_path = out_dir / "statsbomb_shot_events.csv"
    lineup_path = out_dir / "statsbomb_lineups.csv"
    report_path = out_dir / "statsbomb_advanced_import_report.json"

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
