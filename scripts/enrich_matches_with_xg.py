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
from mundialytics.data_quality.team_registry import load_team_registry
from mundialytics.enrichment.xg import CANONICAL_XG_COLUMNS, enrich_matches_with_xg


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Attach match-level xG to canonical matches from a provider/manual CSV. "
            "The enriched match rows may contain post-match xG, but only prior rolling xG features should be used for modelling. "
            "If --xg is missing and --allow-missing-xg is set, the pipeline writes an enriched file with xG coverage 0 instead of crashing."
        )
    )
    parser.add_argument("--matches", required=True)
    parser.add_argument("--xg", required=True, help="Provider/manual xG CSV. Understat downloader writes understat_xg_matches.csv.")
    parser.add_argument("--registry", default=None, help="Optional team_registry.csv for provider aliases.")
    parser.add_argument("--provider", default="unknown")
    parser.add_argument("--provider-alias-column", default="understat_name")
    parser.add_argument("--out-dir", default="data/processed/enriched/xg")
    parser.add_argument("--dataset-name", default="xg_enriched_matches")
    parser.add_argument(
        "--allow-missing-xg",
        action="store_true",
        help="Continue with an empty xG dataframe if --xg does not exist. Useful for batch jobs where xG is optional.",
    )
    args = parser.parse_args()

    matches = load_matches(_resolve(args.matches))
    xg_path = _resolve(args.xg)
    if xg_path.exists():
        xg = pd.read_csv(xg_path)
    elif args.allow_missing_xg:
        xg = pd.DataFrame(columns=CANONICAL_XG_COLUMNS)
    else:
        raise FileNotFoundError(
            f"xG file not found: {xg_path}. Run scripts/download_understat_xg.py successfully, "
            "or import a provider CSV with scripts/import_xg_csv.py, or pass --allow-missing-xg."
        )
    registry = load_team_registry(str(_resolve(args.registry))) if args.registry else None

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = enrich_matches_with_xg(
        matches,
        xg,
        registry=registry,
        provider=args.provider,
        provider_alias_column=args.provider_alias_column,
        dataset_name=args.dataset_name,
    )

    enriched_path = out_dir / "canonical_matches_with_xg.csv"
    canonical_xg_path = out_dir / "external_xg_matches_canonical.csv"
    join_report_path = out_dir / "xg_join_report.csv"
    report_path = out_dir / "xg_enrichment_report.json"

    outputs.enriched_matches.to_csv(enriched_path, index=False)
    outputs.canonical_xg.to_csv(canonical_xg_path, index=False)
    outputs.join_report.to_csv(join_report_path, index=False)

    summary = dict(outputs.summary)
    summary["inputs"] = {
        "matches": str(_resolve(args.matches)),
        "xg": str(xg_path),
        "xg_exists": bool(xg_path.exists()),
        "allow_missing_xg": bool(args.allow_missing_xg),
        "registry": str(_resolve(args.registry)) if args.registry else None,
    }
    if not xg_path.exists():
        summary["status"] = "warning"
        summary["warning"] = "xg_file_missing_empty_xg_used"
    summary["outputs"] = {
        "canonical_matches_with_xg": str(enriched_path),
        "external_xg_matches_canonical": str(canonical_xg_path),
        "xg_join_report": str(join_report_path),
        "xg_enrichment_report": str(report_path),
    }
    _write_json(summary, report_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
