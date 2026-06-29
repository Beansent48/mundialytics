from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.features.team_match_stats import TEAM_PROP_MARKETS, poisson_over_probability, predict_team_props_simple


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    p = argparse.ArgumentParser(description="Predict team and match-total event props for current fixtures.")
    p.add_argument("--team-match-stats", required=True)
    p.add_argument("--fixtures", required=True)
    p.add_argument("--out", default="outputs/team_props_predictions.csv")
    p.add_argument("--report-out", default="outputs/team_props_prediction_report.json")
    p.add_argument("--markets", nargs="+", default=TEAM_PROP_MARKETS)
    p.add_argument("--recent-window", type=int, default=5)
    p.add_argument("--line", type=float, default=None, help="Optional line for P(over line) on each team market.")
    args = p.parse_args()
    hist = pd.read_csv(_resolve(args.team_match_stats))
    fixtures = pd.read_csv(_resolve(args.fixtures))
    out = predict_team_props_simple(hist, fixtures, markets=args.markets)
    if args.line is not None:
        for market in args.markets:
            exp_col = f"expected_{market}"
            if exp_col in out.columns:
                out[f"p_{market}_over_{args.line}"] = out[exp_col].map(lambda x: poisson_over_probability(x, args.line))
    out_path = _resolve(args.out)
    assert out_path is not None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    report = {
        "status": "TEAM_PROPS_PREDICTED",
        "team_match_stats": str(_resolve(args.team_match_stats)),
        "fixtures": str(_resolve(args.fixtures)),
        "out": str(out_path),
        "rows": int(len(out)),
        "markets": args.markets,
        "unavailable_markets": [m for m in args.markets if f"expected_{m}" not in out.columns or out[f"expected_{m}"].isna().all()],
        "note": "MVP simple recent-rate model; use as auditable baseline before calibrated Negative Binomial.",
    }
    report_path = _resolve(args.report_out)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
