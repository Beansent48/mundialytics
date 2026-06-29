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
from mundialytics.enrichment.clubelo import enrich_matches_with_clubelo


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach cached ClubElo team histories or daily ratings to canonical match rows as pre-match external strength features."
    )
    parser.add_argument("--matches", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--clubelo-dir", default="data/external/clubelo")
    parser.add_argument("--out-dir", default="data/processed/enriched/clubelo")
    parser.add_argument("--dataset-name", default="clubelo_enriched_matches")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.94)
    parser.add_argument("--source-mode", choices=["auto", "team-history", "daily-snapshot"], default="auto")
    args = parser.parse_args()

    matches = load_matches(_resolve(args.matches))
    registry = load_team_registry(str(_resolve(args.registry)))
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = enrich_matches_with_clubelo(
        matches,
        registry,
        _resolve(args.clubelo_dir),
        dataset_name=args.dataset_name,
        fuzzy_threshold=args.fuzzy_threshold,
        source_mode=args.source_mode,
    )

    enriched_path = out_dir / "canonical_matches_with_clubelo.csv"
    features_path = out_dir / "clubelo_match_features.csv"
    join_report_path = out_dir / "clubelo_join_report.csv"
    report_path = out_dir / "clubelo_enrichment_report.json"

    outputs.enriched_matches.to_csv(enriched_path, index=False)
    outputs.match_features.to_csv(features_path, index=False)
    outputs.join_report.to_csv(join_report_path, index=False)

    summary = dict(outputs.summary)
    summary["inputs"] = {
        "matches": str(_resolve(args.matches)),
        "registry": str(_resolve(args.registry)),
        "clubelo_dir": str(_resolve(args.clubelo_dir)),
    }
    summary["outputs"] = {
        "canonical_matches_with_clubelo": str(enriched_path),
        "clubelo_match_features": str(features_path),
        "clubelo_join_report": str(join_report_path),
        "clubelo_enrichment_report": str(report_path),
    }
    _write_json(summary, report_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
