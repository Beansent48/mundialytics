from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.loaders import load_matches
from mundialytics.data_quality.team_registry import build_team_registry


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an editable team registry for provider alias matching "
            "(Football-Data, ClubElo, Understat, StatsBomb)."
        )
    )
    parser.add_argument("--matches", nargs="+", required=True, help="Canonical match CSVs to scan.")
    parser.add_argument("--out-dir", default="data/processed/entities")
    parser.add_argument("--dataset-name", default="team_registry")
    args = parser.parse_args()

    frames = [load_matches(_resolve(path)) for path in args.matches]
    matches = pd.concat(frames, ignore_index=True, sort=False)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = build_team_registry(matches, dataset_name=args.dataset_name)

    registry_path = out_dir / "team_registry.csv"
    report_path = out_dir / "team_registry_report.json"
    outputs.registry.to_csv(registry_path, index=False)
    summary = dict(outputs.summary)
    summary["inputs"] = {"matches": [str(_resolve(p)) for p in args.matches]}
    summary["outputs"] = {"team_registry": str(registry_path), "team_registry_report": str(report_path)}
    _write_json(summary, report_path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
