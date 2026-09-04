from __future__ import annotations

"""Which European fixtures can be priced, and which clubs block the rest.

The European layer prices a match from the two sides' ClubElo ratings, so a
fixture is unusable when either club has no downloaded ClubElo history. This
reports the coverage and names the blocking clubs, so the gap is a list of
downloads rather than a vague percentage.

Run it after any ClubElo download to see the coverage move.

    python scripts/audit_european_elo_coverage.py
"""

import collections
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.statistical_core.competition.european import make_resolver  # noqa: E402

UEFA = ROOT / "data/external/uefa"
ELO_DIR = ROOT / "data/external/clubelo/teams"
SNAPSHOT_DIR = ROOT / "data/external/clubelo/daily"


def club_names(pattern: str) -> list[str]:
    out = []
    for f in glob.glob(pattern):
        try:
            out.append(str(pd.read_csv(f, usecols=["Club"], nrows=1)["Club"].iloc[0]))
        except Exception:
            continue
    return out


def main() -> None:
    hist = sorted(club_names(str(ELO_DIR / "*.csv")))
    resolve = make_resolver(hist)

    # the daily snapshot lists far more clubs than we hold histories for; a club
    # that appears there is a download away, which is worth separating out
    snaps = sorted(glob.glob(str(SNAPSHOT_DIR / "*.csv")))
    snap_names: list[str] = []
    if snaps:
        snap_names = pd.read_csv(snaps[-1])["Club"].dropna().astype(str).tolist()
    resolve_snap = make_resolver(snap_names) if snap_names else (lambda _: None)

    total = 0
    missing: collections.Counter[str] = collections.Counter()
    for f in glob.glob(str(UEFA / "raw_*.csv")):
        d = pd.read_csv(f)
        for col in ("Home Team", "Away Team"):
            if col not in d.columns:
                continue
            for name in d[col].dropna().astype(str):
                total += 1
                if not resolve(name):
                    missing[name] += 1

    ok = total - sum(missing.values())
    print(f"club appearances in UEFA fixtures : {total}")
    print(f"  priceable (ClubElo history held): {ok}  ({ok / max(total, 1):.1%})")
    print(f"  blocked                         : {sum(missing.values())}"
          f"  across {len(missing)} clubs")

    in_snapshot = [n for n in missing if resolve_snap(n)]
    print(f"\n{len(in_snapshot)} of those clubs appear in the latest daily snapshot, so they")
    print("are a download away; the rest are not in the snapshot either.\n")
    for name, n in missing.most_common():
        tag = "  [in snapshot]" if name in in_snapshot else ""
        print(f"  {n:4d}  {name}{tag}")


if __name__ == "__main__":
    main()
