from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.features.team_match_stats import build_team_match_stats_from_matches, build_team_match_stats_from_player_events, add_match_totals


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    p = argparse.ArgumentParser(description="Build team_match_stats.csv from player events or wide matches.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--player-events")
    src.add_argument("--matches")
    p.add_argument("--out", default="outputs/team_match_stats.csv")
    p.add_argument("--report-out", default="outputs/team_match_stats_report.json")
    args = p.parse_args()

    if args.player_events:
        df = pd.read_csv(_resolve(args.player_events))
        out = build_team_match_stats_from_player_events(df)
        source = args.player_events
        source_kind = "player_events"
    else:
        df = pd.read_csv(_resolve(args.matches))
        out = build_team_match_stats_from_matches(df)
        source = args.matches
        source_kind = "matches"
    out = add_match_totals(out)
    out_path = _resolve(args.out)
    assert out_path is not None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    report = {
        "status": "TEAM_MATCH_STATS_BUILT",
        "source": source,
        "source_kind": source_kind,
        "out": str(out_path),
        "rows": int(len(out)),
        "matches": int(out["match_id"].nunique()) if "match_id" in out.columns else 0,
        "columns": out.columns.tolist(),
        "corners_available": "corners_for" in out.columns and out["corners_for"].notna().any(),
        "warning": None if ("corners_for" in out.columns and out["corners_for"].notna().any()) else "corners_not_available_or_all_missing; do not create corner markets from this dataset",
    }
    report_path = _resolve(args.report_out)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
