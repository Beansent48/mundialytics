from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from mundialytics.statistical_core.team_stats_model import TeamStatsModel


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit v0.20 team stat profiles and save metadata JSON.")
    ap.add_argument("--historical-events", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    hist = pd.read_csv(args.historical_events)
    model = TeamStatsModel().fit(hist)
    data = {"audit": model.audit, "global_means": model.global_means, "availability": model.availability, "profiles": {k: v.__dict__ for k, v in model.profiles.items()}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved team stats profiles to {args.out}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
