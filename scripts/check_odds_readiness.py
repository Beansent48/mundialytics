#!/usr/bin/env python3
"""Create a provider-shopping/readiness report before buying or wiring an odds API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from mundialytics.betting.odds_readiness import (
    file_readiness_report,
    market_requirements_frame,
    summarize_odds_template,
)


def _resolve(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit whether Mundialytics is ready to request historical odds from a provider.")
    parser.add_argument("--odds-template", default="outputs/odds_ready_current/odds_needed_template.csv")
    parser.add_argument("--line-signals", default="outputs/event_line_backtest_current_v0391/settled_event_line_signals.csv")
    parser.add_argument("--decision-matrix", default="outputs/market_distribution_lab_current_v0391_clean/market_side_decision_matrix.csv")
    parser.add_argument("--out-dir", default="outputs/odds_readiness_current")
    args = parser.parse_args(argv)

    out_dir = _resolve(args.out_dir)
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    line_signals = _resolve(args.line_signals)
    decision_matrix = _resolve(args.decision_matrix)
    odds_template = _resolve(args.odds_template)

    files = file_readiness_report({
        "line_signals": line_signals,
        "decision_matrix": decision_matrix,
        "odds_template": odds_template,
    })

    if odds_template and odds_template.exists() and odds_template.stat().st_size > 0:
        template_df = pd.read_csv(odds_template, low_memory=False)
        shopping_list, template_summary = summarize_odds_template(template_df)
    else:
        shopping_list, template_summary = summarize_odds_template(pd.DataFrame())

    req = market_requirements_frame()
    req.to_csv(out_dir / "provider_market_requirements.csv", index=False)
    shopping_list.to_csv(out_dir / "odds_market_coverage_shopping_list.csv", index=False)
    report = {
        "version": "v0.41_odds_readiness_audit",
        "files": files,
        "odds_template_summary": template_summary,
        "recommendation": "Generate odds_needed_template.csv first, then use this shopping list to compare providers. Do not implement an API until coverage is confirmed.",
    }
    (out_dir / "odds_readiness_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("MUNDIALYTICS ODDS READINESS v0.41")
    print(json.dumps(report["files"], indent=2, default=str))
    print("Outputs:")
    print(f"- {out_dir / 'provider_market_requirements.csv'}")
    print(f"- {out_dir / 'odds_market_coverage_shopping_list.csv'}")
    print(f"- {out_dir / 'odds_readiness_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
