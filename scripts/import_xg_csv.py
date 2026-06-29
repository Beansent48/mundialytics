from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.enrichment.understat import normalize_provider_xg_csv


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize an already downloaded xG provider/manual CSV to Mundialytics' canonical "
            "external xG contract. This is the supported fallback when direct Understat scraping is blocked."
        )
    )
    parser.add_argument("--input", required=True, help="Provider/manual CSV containing match-level xG.")
    parser.add_argument("--provider", default="provider_csv", help="Provider label, e.g. understat_export, thestatsapi, apify.")
    parser.add_argument("--out-dir", default="data/external/xg/understat")
    args = parser.parse_args()

    out_dir = _resolve(args.out_dir)
    outputs = normalize_provider_xg_csv(
        _resolve(args.input),
        out_dir,
        provider=args.provider,
        output_filename="understat_xg_matches.csv",
    )
    report_path = out_dir / "understat_xg_download_report.json"
    report = dict(outputs.report)
    report["outputs"] = {
        "report": str(report_path),
        "canonical_xg_matches": str(out_dir / "understat_xg_matches.csv"),
    }
    _write_json(report, report_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
