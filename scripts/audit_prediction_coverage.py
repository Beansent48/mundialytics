from __future__ import annotations

"""Which played matches have no prediction logged before kick-off?

The logger's own coverage check (log_upcoming_round.py) looks forward: every
fixture in its window must end up logged. This looks back, at what was actually
played, and so also catches the days the logger never ran at all -- the machine
off for a week, the task disabled, a crash before the check. A match missed
here is a matchday the model can never be tested on.

    python scripts/audit_prediction_coverage.py            # last 7 days, exit 3 if any missed
    python scripts/audit_prediction_coverage.py --season   # whole season, report only

Domestic calendars come from ESPN, European ones from the same source the
logger uses, so both sides see the same fixtures.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mundialytics.serving.logged_prediction import find_logged  # noqa: E402

OUT = ROOT / "data/processed/logs/coverage_audit.json"
# Misses that are explained and whose cause is fixed. Listed with a reason so a
# known hole does not fail every run for a week -- and a NEW hole always does.
KNOWN_GAPS = ROOT / "data/curated/coverage_known_gaps.csv"
EXIT_MISSED = 3


def played_fixtures(year: int) -> pd.DataFrame:
    """Every played fixture of the season: competition, date, home, away."""
    from log_upcoming_round import fetch_fixtures

    from mundialytics.statistical_core.competition.european import (
        FD_SLUG, fetch_season_fixtures)

    frames = []
    dom = fetch_fixtures(year)
    if len(dom):
        d = dom[dom["played"]]
        frames.append(pd.DataFrame({"competition": d["competition"], "date": d["date"],
                                    "home": d["home"], "away": d["away"]}))
    for comp in FD_SLUG:
        fx = fetch_season_fixtures(ROOT, comp, year)
        if fx is None or fx.empty:
            continue
        played = fx["Result"].astype(str).str.contains(r"\d+\s*-\s*\d+", regex=True)
        e = fx[played]
        frames.append(pd.DataFrame({
            "competition": comp,
            "date": pd.to_datetime(e["Date"], dayfirst=True, errors="coerce"),
            "home": e["Home Team"].astype(str).str.lower(),
            "away": e["Away Team"].astype(str).str.lower(),
        }))
    if not frames:
        return pd.DataFrame(columns=["competition", "date", "home", "away"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    return out.dropna(subset=["date"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="look back this many days")
    ap.add_argument("--season", action="store_true", help="whole season, report only")
    ap.add_argument("--year", type=int, default=None)
    args = ap.parse_args()

    now = pd.Timestamp.now()
    year = args.year or (now.year if now.month >= 7 else now.year - 1)
    fx = played_fixtures(year)
    if not args.season:
        fx = fx[fx["date"] >= now.normalize() - pd.Timedelta(days=args.days)]

    fx = fx.assign(logged=[find_logged(r.home, r.away, r.date) is not None
                           for r in fx.itertuples(index=False)])
    missed = fx[~fx["logged"]].sort_values("date")
    known = set()
    if KNOWN_GAPS.exists():
        k = pd.read_csv(KNOWN_GAPS)
        known = {(str(d)[:10], h, a) for d, h, a in zip(k["date"], k["home"], k["away"])}
    is_known = [(f"{r.date:%Y-%m-%d}", r.home, r.away) in known
                for r in missed.itertuples(index=False)]
    unexplained = missed[[not x for x in is_known]]
    scope = "temporada" if args.season else f"ultimos {args.days} dias"
    print(f"{scope}: {int(fx['logged'].sum())}/{len(fx)} partidos jugados con prediccion previa")
    for comp, g in fx.groupby("competition"):
        print(f"  {comp:18s} {int(g['logged'].sum()):4d}/{len(g)}")
    if len(missed):
        print(f"\nSIN PREDICCION PREVIA ({len(missed)}):")
        for r, kn in zip(missed.head(40).itertuples(index=False), is_known):
            tag = "  (conocido)" if kn else ""
            print(f"  {r.date:%Y-%m-%d} {r.competition:16s} {r.home} vs {r.away}{tag}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "checked_at": now.isoformat(timespec="seconds"),
        "scope": scope,
        "played": int(len(fx)),
        "logged": int(fx["logged"].sum()),
        "missed": [f"{r.date:%Y-%m-%d} {r.competition}: {r.home} vs {r.away}"
                   for r in missed.itertuples(index=False)],
        "unexplained": [f"{r.date:%Y-%m-%d} {r.competition}: {r.home} vs {r.away}"
                        for r in unexplained.itertuples(index=False)],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return EXIT_MISSED if (len(unexplained) and not args.season) else 0


if __name__ == "__main__":
    raise SystemExit(main())
