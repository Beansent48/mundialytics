from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.enrichment.understat import (
    UNDERSTAT_LEAGUES,
    download_understat_xg,
    normalize_provider_xg_csv,
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_league_season(values: list[str] | None) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for value in values or []:
        if ":" not in value:
            raise ValueError("Use --league-season values like EPL:2021 or LaLiga:2024")
        league, season = value.split(":", 1)
        pairs.append((league, int(season)))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Optional research downloader/importer for match-level xG. "
            "Direct Understat scraping is best-effort; provider/manual CSV import is supported."
        )
    )
    parser.add_argument(
        "--league-season",
        nargs="+",
        help="Pairs like EPL:2021 LaLiga:2024 Serie_A:2023 Bundesliga:2022 Ligue_1:2025",
    )
    parser.add_argument(
        "--input-csv",
        help=(
            "Already downloaded provider/manual xG CSV. When set, the script normalizes this CSV "
            "to the canonical understat_xg_matches.csv contract instead of scraping."
        ),
    )
    parser.add_argument("--provider", default="understat", help="Provider label for --input-csv mode.")
    parser.add_argument("--out-dir", default="data/external/xg/understat")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_csv:
        outputs = normalize_provider_xg_csv(
            _resolve(args.input_csv),
            out_dir,
            provider=args.provider,
            output_filename="understat_xg_matches.csv",
        )
    else:
        pairs = _parse_league_season(args.league_season)
        if not pairs:
            raise ValueError("Provide --league-season values or --input-csv.")
        outputs = download_understat_xg(
            pairs,
            out_dir,
            timeout=args.timeout,
            sleep_seconds=args.sleep_seconds,
        )

    report_path = out_dir / "understat_xg_download_report.json"
    report = dict(outputs.report)
    report["known_league_aliases"] = sorted(UNDERSTAT_LEAGUES.keys())
    report["outputs"] = {
        "report": str(report_path),
        "canonical_xg_matches": str(out_dir / "understat_xg_matches.csv"),
    }
    _write_json(report, report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
