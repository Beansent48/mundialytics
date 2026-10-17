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

This is a ONE-OFF backfill, not recurring maintenance. The pre-kickoff log has
carried the lambdas since PR #25 and grows every day, so the only gap is the
part of the season played before that. Once the backfill reaches the log's first
row the two together cover everything, and the level strengthens on its own.

It is slow (~25 min: one engine fit per week), so it checks first and exits
without doing anything when the gap is already closed. --force overrides.

    python scripts/build_served_lambdas_current.py [SEASON] [--force]
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/served_lambdas_current.csv"


PRED_LOG = ROOT / "data/processed/logs/predictions_log.csv"


def gap_is_closed(season: str) -> bool:
    """True when the cache already reaches the first date the log has lambdas for."""
    if not OUT.exists() or not PRED_LOG.exists():
        return False
    try:
        c = pd.read_csv(OUT)
        lg = pd.read_csv(PRED_LOG, low_memory=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return False
    lg = lg[(lg.get("mercado") == "1X2") & lg.get("lambda_home").notna()]
    if c.empty or lg.empty or "fecha" not in c.columns:
        return False
    if "season" in c.columns and not (c["season"].astype(str) == season).any():
        return False          # the cache is from a previous season
    c = c[c["season"].astype(str) == season] if "season" in c.columns else c
    # the log's earliest lambda must fall inside, or right after, what the cache covers
    return str(c["fecha"].max()) >= str(lg["fecha"].min())


def main() -> int:
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        season = args[0]
    else:
        now = pd.Timestamp.now()
        y = now.year if now.month >= 7 else now.year - 1
        season = f"{y}-{y + 1}"
    if not force and gap_is_closed(season):
        print(f"{season}: the backfill already reaches the log's first lambda; "
              f"nothing to do (--force to rebuild anyway)", flush=True)
        return 0
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
