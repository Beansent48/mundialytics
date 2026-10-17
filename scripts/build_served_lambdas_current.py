from __future__ import annotations

"""Walk-forward lambdas for the SEASON IN PROGRESS -> data/processed/served_lambdas_current.csv

generate_deployed_walkforward.py does this for past seasons so the Resultados
page can show served probabilities. The market-level correction
(features/market_level.py) needs the same thing for the current season: the
lambda the model serving that week produced, for every match already played.

Without it the level is measured off the pre-kickoff log alone, which carries
lambda_home/lambda_away only since PR #25 -- 47 matches on 2026-10-17, about 9
per league, which at K=70 leaves the correction at 11% of its strength. With it,
the whole season counts from the first refresh.

Slow (~25 min: one engine fit per week), so it is not a daily step. Re-run it
when a few rounds have accumulated; between runs the pre-kickoff log tops the
level up by itself.

    python scripts/build_served_lambdas_current.py [SEASON]
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/served_lambdas_current.csv"


def main() -> int:
    if len(sys.argv) > 1:
        season = sys.argv[1]
    else:
        now = pd.Timestamp.now()
        y = now.year if now.month >= 7 else now.year - 1
        season = f"{y}-{y + 1}"
    tmp = OUT.with_suffix(".tmp.csv")
    cmd = [sys.executable, str(ROOT / "scripts/goals_level/wf_lambdas.py"),
           season, str(tmp), "--src", "foundation"]
    print(f"walk-forward {season} -> {OUT.relative_to(ROOT)} (one fit per week, slow)",
          flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0 or not tmp.exists():
        print("failed; the deployed file is left alone", flush=True)
        return r.returncode or 1
    d = pd.read_csv(tmp)
    d["fecha"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    d[["fecha", "season", "comp", "home", "away", "lh", "la"]].to_csv(OUT, index=False)
    tmp.unlink(missing_ok=True)
    print(f"WROTE {OUT.relative_to(ROOT)} ({len(d)} matches)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
