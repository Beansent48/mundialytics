from __future__ import annotations

"""Build data/processed/squad_values.csv (squad market value per club, dated).

    python scripts/build_squad_values.py              # current snapshot (live Transfermarkt, weekly)
    python scripts/build_squad_values.py --history    # + the weekly history 2019-> (minutes)
    python scripts/build_squad_values.py --download   # fetch the Kaggle dump first (~250 MB)
    python scripts/build_squad_values.py --force-live # refetch Transfermarkt even if fresh
    python scripts/build_squad_values.py --offline    # never touch Transfermarkt

Current snapshot, in order of preference:
  1. transfermarkt_live -- every Big Five squad and value read from the site, refreshed
     when the last live snapshot is LIVE_MAX_AGE_DAYS old (fresh values are worth -0.0003
     RPS over June-frozen ones); a fresh one is left alone;
  2. espn_x_dump -- only when the site fails: today's ESPN rosters, each player carrying
     his latest value from the frozen Kaggle dump.
The daily refresh runs this after the current squads are rebuilt. The history only
depends on the dump, so it is built once. See src/mundialytics/features/squad_value.py.
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
LIVE_MAX_AGE_DAYS = 7
TM_CACHE = sv.TM_DIR / "live_cache"


def download() -> None:
    sv.TM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {KAGGLE} ...", flush=True)
    with urllib.request.urlopen(KAGGLE, timeout=900) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    for name in NEEDED:
        z.extract(name, sv.TM_DIR)
    print("  extracted", NEEDED, flush=True)


def live_snapshot(club_map: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    from mundialytics.providers.transfermarkt import fetch_live_squads

    tm = fetch_live_squads(TM_CACHE)
    teams, unmapped = sv.build_live(tm, club_map, current)
    missing = sorted(set(current["team"]) - set(teams["team"]))
    print(f"live: {len(teams)} teams, {tm['value_eur'].notna().mean():.1%} of {len(tm)} players valued"
          + (f"; unmapped Transfermarkt clubs: {unmapped}" if unmapped else "")
          + (f"; current teams without a live value: {missing}" if missing else ""), flush=True)
    if len(teams) < 0.9 * current["team"].nunique():
        raise RuntimeError(f"only {len(teams)} of {current['team'].nunique()} teams mapped")
    return teams.assign(source="transfermarkt_live")


def espn_x_dump_snapshot(club_map: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    cache = pd.read_csv(sv.MATCH_CACHE_PATH) if sv.MATCH_CACHE_PATH.exists() else None
    teams, matches = sv.build_current(rosters, club_map=club_map, cache=cache)
    matches.to_csv(sv.MATCH_CACHE_PATH, index=False)
    print(f"espn x dump: {len(teams)} teams, {matches['player_id'].notna().mean():.1%} of "
          f"{len(matches)} roster players valued", flush=True)
    return teams.assign(source="espn_x_dump")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="store_true", help="rebuild the club map and weekly history")
    ap.add_argument("--download", action="store_true", help="download the Kaggle dump first")
    ap.add_argument("--force-live", action="store_true", help="refetch Transfermarkt even if fresh")
    ap.add_argument("--offline", action="store_true", help="never fetch Transfermarkt")
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
        hist = sv.build_history(club_map=club_map).assign(source="history")
        print(f"history: {hist['snap'].nunique()} weekly snapshots "
              f"{hist['snap'].min():%Y-%m-%d}..{hist['snap'].max():%Y-%m-%d}, "
              f"known share {hist['known_share'].mean():.1%}", flush=True)
        existing = sv.upsert_snapshots(existing, hist)

    if not sv.ROSTERS_PATH.exists():
        print(f"no rosters at {sv.ROSTERS_PATH}; current snapshot skipped", flush=True)
    else:
        rosters = pd.read_csv(sv.ROSTERS_PATH)
        current = rosters.assign(team=rosters["team"].map(sv.canonical_name))[["competition", "team"]].drop_duplicates()
        last_live = None
        if existing is not None and "source" in existing.columns:
            live_rows = existing[existing["source"] == "transfermarkt_live"]
            last_live = live_rows["snap"].max() if len(live_rows) else None
        age = (pd.Timestamp.now().normalize() - last_live).days if last_live is not None else None
        snap = None
        if args.offline:
            print("offline: Transfermarkt skipped", flush=True)
        elif age is not None and age < LIVE_MAX_AGE_DAYS and not args.force_live:
            print(f"live snapshot from {last_live:%Y-%m-%d} ({age} d old) is fresh; kept", flush=True)
        else:
            try:
                snap = live_snapshot(club_map, current)
            except Exception as exc:  # the site changed or refused: fall back, never retry harder
                print(f"live Transfermarkt failed ({type(exc).__name__}: {exc}); falling back", flush=True)
        # the fallback only runs when there is no live snapshot to keep: a newer
        # espn_x_dump row would otherwise shadow a fresh live one in squad_values_asof
        if snap is None and (args.offline or age is None or age >= LIVE_MAX_AGE_DAYS or args.force_live):
            snap = espn_x_dump_snapshot(club_map, rosters)
        if snap is not None:
            existing = sv.upsert_snapshots(existing, snap)
            low = snap[snap["known_share"] < 0.6].sort_values("known_share")
            if len(low):
                print("low coverage: " + ", ".join(f"{t} {s:.0%}" for t, s in zip(low.team, low.known_share)),
                      flush=True)

    existing.to_csv(sv.SQUAD_VALUES_PATH, index=False)
    print(f"WROTE {sv.SQUAD_VALUES_PATH} ({len(existing)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
