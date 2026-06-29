from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.evaluation.readiness import ReadinessThresholds, evaluate_readiness


def _read_json(path: str | None):
    if not path:
        return None
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description="Cautious readiness gate for datasets/backtests. Keeps the bot honest.")
    p.add_argument("--data-report", default=None, help="JSON from diagnose_dataset.py")
    p.add_argument("--backtest-summary", default=None, help="JSON from backtest_from_csv.py")
    p.add_argument("--out", default="outputs/quality_gate.json")
    p.add_argument("--min-matches", type=int, default=200)
    p.add_argument("--min-backtest-predictions", type=int, default=100)
    args = p.parse_args()

    result = evaluate_readiness(
        _read_json(args.data_report),
        _read_json(args.backtest_summary),
        ReadinessThresholds(min_matches=args.min_matches, min_backtest_predictions=args.min_backtest_predictions),
    )
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
