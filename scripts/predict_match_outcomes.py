from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from mundialytics.statistical_core.match_model import MatchOutcomeModel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--historical-events", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fixtures = pd.read_csv(args.fixtures)
    hist = pd.read_csv(args.historical_events) if args.historical_events else pd.DataFrame()
    pred, score = MatchOutcomeModel().fit(hist).predict_fixtures(fixtures)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out, index=False)
    score.to_csv(str(Path(args.out).with_name(Path(args.out).stem + "_scorelines.csv")), index=False)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
