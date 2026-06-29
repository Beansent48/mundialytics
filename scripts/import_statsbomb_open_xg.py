from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.enrichment.statsbomb_open import import_statsbomb_open_xg


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Import free official StatsBomb Open Data shot xG from a local open-data checkout. "
            "This is the preferred free xG path when Understat direct scraping is blocked. "
            "Coverage is partial and depends on the matches present under data/raw/statsbomb/open-data/data."
        )
    )
    parser.add_argument("--data-dir", default="data/raw/statsbomb/open-data/data")
    parser.add_argument("--out-dir", default="data/external/xg/statsbomb")
    parser.add_argument("--competition-id", nargs="*", default=None, help="Optional StatsBomb competition IDs to include.")
    parser.add_argument("--season-id", nargs="*", default=None, help="Optional StatsBomb season IDs to include.")
    parser.add_argument("--max-matches", type=int, default=None, help="Optional smoke-test limit.")
    args = parser.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = import_statsbomb_open_xg(
        _resolve(args.data_dir),
        competition_ids=args.competition_id,
        season_ids=args.season_id,
        max_matches=args.max_matches,
    )

    xg_matches_path = out_dir / "statsbomb_xg_matches.csv"
    xg_shots_path = out_dir / "statsbomb_xg_shots.csv"
    report_path = out_dir / "statsbomb_xg_import_report.json"

    outputs.xg_matches.to_csv(xg_matches_path, index=False)
    outputs.xg_shots.to_csv(xg_shots_path, index=False)

    report = dict(outputs.report)
    report["outputs"] = {
        "report": str(report_path),
        "canonical_xg_matches": str(xg_matches_path),
        "canonical_xg_shots": str(xg_shots_path),
    }
    _write_json(report, report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
