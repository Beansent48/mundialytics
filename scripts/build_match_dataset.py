from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.adapters import football_data_uk_to_matches, international_results_to_matches, openfootball_json_to_matches
from mundialytics.data.loaders import load_matches
from mundialytics.data_quality.match_dataset_foundation import prepare_match_dataset


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        resolved_pattern = str(_resolve(pattern))
        matches = sorted(glob.glob(resolved_pattern))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            p = _resolve(pattern)
            if p.exists():
                paths.append(p)
    unique: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_one(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    if args.source == "canonical":
        df = load_matches(path)
    elif args.source == "football-data-uk":
        df = football_data_uk_to_matches(path, season=args.season)
    elif args.source == "international-results":
        df = international_results_to_matches(path)
    elif args.source == "openfootball":
        df = openfootball_json_to_matches(
            path,
            competition=args.competition or "openfootball",
            season=args.season or "unknown",
            team_scope=args.team_scope,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported source: {args.source}")
    df["source_file"] = str(path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a cleaned, profiled canonical match dataset before model validation. "
            "This is a data-foundation step, not a model-training step."
        )
    )
    parser.add_argument("--source", choices=["canonical", "football-data-uk", "international-results", "openfootball"], required=True)
    parser.add_argument("--inputs", nargs="+", required=True, help="Input files or glob patterns. Quote globs in PowerShell.")
    parser.add_argument("--out-dir", default="outputs/match_dataset_foundation")
    parser.add_argument("--dataset-name", default="match_dataset_foundation")
    parser.add_argument("--season", default=None, help="Optional season override. Usually leave empty for multi-season Football-Data inputs.")
    parser.add_argument("--competition", default=None, help="OpenFootball competition name when needed.")
    parser.add_argument("--team-scope", choices=["club", "national"], default="club")
    parser.add_argument("--drop-incomplete-goals", action="store_true", help="Drop rows missing full-time goals. Use for training datasets.")
    parser.add_argument("--skip-bad-files", action="store_true", help="Continue if one input file fails conversion; failures are reported.")
    args = parser.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_paths = _expand_inputs(args.inputs)
    if not input_paths:
        raise FileNotFoundError(f"No input files matched: {args.inputs}")

    frames: list[pd.DataFrame] = []
    file_errors: list[dict[str, str]] = []
    for path in input_paths:
        try:
            frames.append(_load_one(path, args))
        except Exception as exc:
            if not args.skip_bad_files:
                raise
            file_errors.append({"path": str(path), "error": str(exc)})

    if not frames:
        raise RuntimeError("No input files could be converted into canonical matches.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(out_dir / "canonical_matches_raw_combined.csv", index=False)

    outputs = prepare_match_dataset(
        combined,
        dataset_name=args.dataset_name,
        drop_incomplete_goals=args.drop_incomplete_goals,
    )

    outputs.cleaned_matches.to_csv(out_dir / "canonical_matches.csv", index=False)
    outputs.feature_coverage.to_csv(out_dir / "match_dataset_feature_coverage.csv", index=False)
    outputs.quality_by_competition_season.to_csv(out_dir / "match_dataset_quality_by_competition_season.csv", index=False)
    outputs.anomalies.to_csv(out_dir / "match_dataset_anomalies.csv", index=False)
    outputs.dropped_rows.to_csv(out_dir / "match_dataset_dropped_rows.csv", index=False)

    summary = dict(outputs.summary)
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary["input_files"] = [str(p) for p in input_paths]
    summary["input_file_count"] = len(input_paths)
    summary["file_errors"] = file_errors
    summary["outputs"] = {
        "canonical_matches": str(out_dir / "canonical_matches.csv"),
        "raw_combined": str(out_dir / "canonical_matches_raw_combined.csv"),
        "feature_coverage": str(out_dir / "match_dataset_feature_coverage.csv"),
        "quality_by_competition_season": str(out_dir / "match_dataset_quality_by_competition_season.csv"),
        "anomalies": str(out_dir / "match_dataset_anomalies.csv"),
        "dropped_rows": str(out_dir / "match_dataset_dropped_rows.csv"),
    }
    _write_json(summary, out_dir / "match_dataset_foundation_report.json")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
