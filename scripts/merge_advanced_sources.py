from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.enrichment.advanced import merge_advanced_match_sources


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Use --source provider=path/to/file.csv")
    provider, path = value.split("=", 1)
    return provider, _resolve(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple advanced match-stat providers into one canonical priority-selected file.")
    parser.add_argument("--source", nargs="+", required=True, help="provider=csv pairs")
    parser.add_argument("--provider-priority", nargs="*", default=None)
    parser.add_argument("--out-dir", default="data/external/advanced/canonical")
    args = parser.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = []
    missing = []
    for item in args.source:
        provider, path = _parse_source(item)
        if path.exists():
            sources.append((provider, pd.read_csv(path)))
        else:
            missing.append({"provider": provider, "path": str(path), "error": "file_not_found"})

    outputs = merge_advanced_match_sources(sources, provider_priority=args.provider_priority)
    match_path = out_dir / "canonical_advanced_match_stats.csv"
    report_csv = out_dir / "advanced_provider_priority_report.csv"
    report_json = out_dir / "advanced_provider_priority_report.json"

    outputs.canonical_advanced_matches.to_csv(match_path, index=False)
    outputs.provider_priority_report.to_csv(report_csv, index=False)
    summary = dict(outputs.summary)
    summary["missing_sources"] = missing
    summary["outputs"] = {
        "canonical_advanced_match_stats": str(match_path),
        "provider_priority_report_csv": str(report_csv),
        "provider_priority_report_json": str(report_json),
    }
    _write_json(summary, report_json)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
