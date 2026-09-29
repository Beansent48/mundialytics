from __future__ import annotations

"""Build data/processed/squad_values.csv (squad market value per club, dated).

    python scripts/build_squad_values.py              # today's snapshot from ESPN rosters
    python scripts/build_squad_values.py --history    # + the weekly history 2019-> (minutes)
    python scripts/build_squad_values.py --download   # fetch the Kaggle dump first (~250 MB)

The daily refresh runs the first form after the current squads are rebuilt; the history is
built once (it only depends on the frozen dump). See src/mundialytics/features/squad_value.py.
"""

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.features import squad_value as sv  # noqa: E402

KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/davidcariboo/player-scores"
NEEDED = ["games.csv", "player_valuations.csv", "players.csv", "transfers.csv"]


def download() -> None:
    sv.TM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {KAGGLE} ...", flush=True)
    with urllib.request.urlopen(KAGGLE, timeout=900) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    for name in NEEDED:
        z.extract(name, sv.TM_DIR)
    print("  extracted", NEEDED, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="store_true", help="rebuild the club map and weekly history")
    ap.add_argument("--download", action="store_true", help="download the Kaggle dump first")
    args = ap.parse_args()
    if args.download:
        download()
    missing = [n for n in NEEDED if not (sv.TM_DIR / n).exists()]
    if missing:
        print(f"Transfermarkt dump missing {missing} in {sv.TM_DIR}; run with --download", flush=True)
        return 2

    if args.history or not sv.CLUB_MAP_PATH.exists():
        club_map = sv.build_club_map()
        club_map.to_csv(sv.CLUB_MAP_PATH, index=False)
        print(f"club map: {len(club_map)} Transfermarkt clubs -> canonical teams", flush=True)
    club_map = pd.read_csv(sv.CLUB_MAP_PATH)

    existing = sv.load_squad_values()
    if args.history:
        hist = sv.build_history(club_map=club_map)
        print(f"history: {hist['snap'].nunique()} weekly snapshots "
              f"{hist['snap'].min():%Y-%m-%d}..{hist['snap'].max():%Y-%m-%d}, "
              f"known share {hist['known_share'].mean():.1%}", flush=True)
        existing = sv.upsert_snapshots(existing, hist)

    if not sv.ROSTERS_PATH.exists():
        print(f"no rosters at {sv.ROSTERS_PATH}; current snapshot skipped", flush=True)
    else:
        rosters = pd.read_csv(sv.ROSTERS_PATH)
        cache = pd.read_csv(sv.MATCH_CACHE_PATH) if sv.MATCH_CACHE_PATH.exists() else None
        teams, matches = sv.build_current(rosters, club_map=club_map, cache=cache)
        matches.to_csv(sv.MATCH_CACHE_PATH, index=False)
        existing = sv.upsert_snapshots(existing, teams)
        low = teams[teams["known_share"] < 0.6].sort_values("known_share")
        print(f"current: {len(teams)} teams, {matches['player_id'].notna().mean():.1%} of "
              f"{len(matches)} roster players valued"
              + (f"; low coverage: {', '.join(f'{t} {s:.0%}' for t, s in zip(low.team, low.known_share))}"
                 if len(low) else ""), flush=True)

    existing.to_csv(sv.SQUAD_VALUES_PATH, index=False)
    print(f"WROTE {sv.SQUAD_VALUES_PATH} ({len(existing)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
