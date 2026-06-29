from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from mundialytics.statistical_core.match_model import _team_match_goal_frame


def main() -> int:
    ap = argparse.ArgumentParser(description="Build simple team-match goal features from historical player/event rows.")
    ap.add_argument("--historical-events", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    hist = pd.read_csv(args.historical_events)
    out = _team_match_goal_frame(hist)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
