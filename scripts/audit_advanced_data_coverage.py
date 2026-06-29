from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.enrichment.advanced import audit_advanced_data_coverage


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit advanced data coverage by feature group.")
    parser.add_argument("--matches", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-name", default="advanced_data_coverage")
    args = parser.parse_args()

    matches = pd.read_csv(_resolve(args.matches))
    outputs = audit_advanced_data_coverage(matches, dataset_name=args.dataset_name)

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "advanced_data_coverage_report.json"
    coverage_path = out_dir / "advanced_feature_coverage.csv"
    outputs["coverage"].to_csv(coverage_path, index=False)
    report = dict(outputs["summary"])
    report["outputs"] = {
        "report": str(report_path),
        "feature_coverage": str(coverage_path),
    }
    _write_json(report, report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
