"""
Backfill competition forecast bundles for the visual layer.

Pre-generates full-season multi-snapshot bundles (statistical_core/competition/
forecast_cache) for the Big-5 leagues so the "Pronóstico de liga" page has instant
content everywhere. Each bundle lets the matchday slider scrub the whole season
from cache with no compute on page load.

This is the once-per-matchday background job in disguise: run it whenever new data
lands. It's incremental — existing immutable snapshots are reused, only missing
matchdays are computed.

Usage:
    python scripts/backfill_competition_forecasts.py                # 2 latest seasons, all Big5
    python scripts/backfill_competition_forecasts.py --seasons 3
    python scripts/backfill_competition_forecasts.py --leagues LaLiga "Premier League"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.statistical_core.competition import forecast_cache as fc  # noqa: E402

FOUNDATION = ROOT / "data/processed/foundation_big5_multi_season.csv"
CACHE_DIR = ROOT / "data/processed/competition_cache"
BIG5 = ["LaLiga", "Premier League", "Serie A", "Bundesliga", "Ligue 1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="+", default=BIG5)
    ap.add_argument("--seasons", type=int, default=2, help="how many recent seasons per league")
    ap.add_argument("--step", type=int, default=5, help="matchday grid step")
    ap.add_argument("--sims", type=int, default=10_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    found = pd.read_csv(FOUNDATION, low_memory=False)
    t0 = time.time()
    done = 0
    for comp in args.leagues:
        seasons = sorted(found.loc[found["competition"] == comp, "season"].unique(), reverse=True)[: args.seasons]
        for season in seasons:
            t = time.time()
            bundle = fc.get_or_build(comp, season, found, timeline_step=args.step,
                                     n_sims=args.sims, cache_dir=CACHE_DIR, force=args.force)
            mds = fc.available_matchdays(bundle)
            top = bundle["snapshots"][str(mds[-1])]["forecast"]["team_probs"][0]
            print(f"[{done+1:2d}] {comp:16s} {season}: {len(mds)} snapshots "
                  f"(mj {mds[0]}-{mds[-1]}) · fav {top['team']} {top['p_champion']*100:.0f}% "
                  f"· {time.time()-t:.0f}s", flush=True)
            done += 1
    print(f"\nDone: {done} bundles in {(time.time()-t0)/60:.1f} min -> {CACHE_DIR}")


if __name__ == "__main__":
    main()
