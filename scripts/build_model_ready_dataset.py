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
from mundialytics.data_quality.model_ready_snapshots import build_model_ready_match_snapshots


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe model-ready match snapshots from canonical_matches.csv. "
            "This prepares the hybrid Big-5 data contract; it does not train a model."
        )
    )
    parser.add_argument("--matches", required=True, help="Canonical matches CSV produced by the data-foundation step.")
    parser.add_argument("--out-dir", default="outputs/model_ready_match_snapshots")
    parser.add_argument("--dataset-name", default="model_ready_match_snapshots")
    args = parser.parse_args()

    matches_path = _resolve(args.matches)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matches = load_matches(matches_path)
    outputs = build_model_ready_match_snapshots(matches, dataset_name=args.dataset_name)

    snapshots_path = out_dir / "model_ready_match_snapshots.csv"
    contract_path = out_dir / "model_ready_feature_contract.csv"
    report_path = out_dir / "model_ready_snapshot_report.json"

    outputs.snapshots.to_csv(snapshots_path, index=False)
    outputs.feature_contract.to_csv(contract_path, index=False)

    summary = dict(outputs.summary)
    summary["inputs"] = {"matches": str(matches_path)}
    summary["outputs"] = {
        "model_ready_match_snapshots": str(snapshots_path),
        "model_ready_feature_contract": str(contract_path),
        "model_ready_snapshot_report": str(report_path),
    }
    _write_json(summary, report_path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
