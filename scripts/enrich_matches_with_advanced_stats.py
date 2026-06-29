from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.enrichment.advanced import enrich_matches_with_advanced_stats


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach canonical advanced match stats to canonical matches using date/team/provider aliases."
    )
    parser.add_argument("--matches", required=True)
    parser.add_argument("--advanced", required=True)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--manual-aliases", default=None, help="Optional CSV with football_data_name,provider_name,canonical_name columns.")
    parser.add_argument("--provider-alias-column", default="football_data_name")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-name", default="advanced_enriched_matches")
    args = parser.parse_args()

    matches = pd.read_csv(_resolve(args.matches))
    advanced = pd.read_csv(_resolve(args.advanced))
    registry = pd.read_csv(_resolve(args.registry)) if args.registry else None
    manual_aliases = pd.read_csv(_resolve(args.manual_aliases)) if args.manual_aliases else None

    outputs = enrich_matches_with_advanced_stats(
        matches,
        advanced,
        registry=registry,
        manual_aliases=manual_aliases,
        provider_alias_column=args.provider_alias_column,
        dataset_name=args.dataset_name,
    )

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = out_dir / "canonical_matches_with_advanced_stats.csv"
    canonical_path = out_dir / "canonical_advanced_match_stats.csv"
    join_path = out_dir / "advanced_join_report.csv"
    report_path = out_dir / "advanced_enrichment_report.json"

    outputs.enriched_matches.to_csv(enriched_path, index=False)
    outputs.canonical_advanced_matches.to_csv(canonical_path, index=False)
    outputs.join_report.to_csv(join_path, index=False)
    report = dict(outputs.summary)
    report["outputs"] = {
        "canonical_matches_with_advanced_stats": str(enriched_path),
        "canonical_advanced_match_stats": str(canonical_path),
        "advanced_join_report": str(join_path),
        "advanced_enrichment_report": str(report_path),
    }
    _write_json(report, report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
