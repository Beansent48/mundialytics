from __future__ import annotations

"""Referee per match for the big five, from ESPN's public match summary.

football-data carries a Referee column for the Premier League only, which is
why the referee feature in props/team_props.py has been EPL-only. ESPN's
`summary.gameInfo.officials` names the referee for all five leagues — measured
on 2026-09-29: every sampled match from calendar 2022 on, none before (2019-21
come back empty), so history starts with season 2021/22.

One summary request per match, cached by ESPN event id in the output CSV, so
a re-run only asks for matches it has not seen (the daily cost is one round).

    python scripts/fetch_espn_referees.py                # 2021/22 -> today
    python scripts/fetch_espn_referees.py --from-year 2026
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.identity.normalization import canonical_team_name  # noqa: E402
from mundialytics.providers.espn_fixtures import (  # noqa: E402
    ESPN_LEAGUE_CODES, _load_aliases, espn_json, fetch_scoreboard_events)
from mundialytics.utils import atomic_to_csv  # noqa: E402

OUT = ROOT / "data/processed/espn_referees.csv"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/summary?event={eid}"
COLUMNS = ["event_id", "competition", "date", "home_team", "away_team",
           "home_espn", "away_espn", "referee"]


def _referee(code: str, eid: str) -> str | None:
    """The match referee's name, '' when ESPN lists none, None when the call fails."""
    try:
        s = espn_json(SUMMARY.format(code=code, eid=eid), timeout=30, tries=3)
    except Exception:
        return None
    for o in (s.get("gameInfo", {}) or {}).get("officials", []) or []:
        pos = str((o.get("position") or {}).get("name", "")).lower()
        if pos == "referee" or o.get("order") == 1:
            return str(o.get("fullName") or o.get("displayName") or "").strip()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Referee per big-five match from ESPN summaries.")
    ap.add_argument("--from-year", type=int, default=2021,
                    help="first season start year (2021 = 2021/22)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    old = pd.read_csv(OUT, dtype={"event_id": str}) if OUT.exists() else pd.DataFrame(columns=COLUMNS)
    # a name counts as done; '' (none listed) is asked again only for a week --
    # ESPN may fill a recent match late, but early 2021/22 simply has none
    named = old["referee"].fillna("").astype(str).str.len() > 0
    settled = pd.to_datetime(old["date"], errors="coerce") < pd.Timestamp(date.today()) - pd.Timedelta(days=7)
    done = set(old.loc[named | settled, "event_id"].astype(str))
    alias = _load_aliases(ROOT)

    def canon(name: str) -> str:
        return alias.get(name, canonical_team_name(name))

    todo = []
    for comp, code in ESPN_LEAGUE_CODES.items():
        events = fetch_scoreboard_events(code, date(args.from_year, 7, 1), date.today(),
                                         timeout=45, tries=3)
        for ev in events:
            c = (ev.get("competitions") or [{}])[0]
            if not c.get("status", {}).get("type", {}).get("completed"):
                continue
            eid = str(ev.get("id"))
            if eid in done:
                continue
            teams = {x.get("homeAway"): x["team"]["displayName"] for x in c.get("competitors", [])}
            if "home" not in teams or "away" not in teams:
                continue
            todo.append({"event_id": eid, "competition": comp, "code": code,
                         "date": str(ev.get("date", ""))[:10],
                         "home_espn": teams["home"], "away_espn": teams["away"]})
        print(f"{comp}: {len(events)} events, {sum(t['code'] == code for t in todo)} to fetch", flush=True)

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (t, ref) in enumerate(zip(todo, ex.map(lambda t: _referee(t["code"], t["event_id"]), todo))):
            if ref is None:
                continue          # failed call: leave it for the next run
            rows.append({**{k: t[k] for k in ("event_id", "competition", "date", "home_espn", "away_espn")},
                         "home_team": canon(t["home_espn"]), "away_team": canon(t["away_espn"]),
                         "referee": ref})
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)

    new = pd.DataFrame(rows, columns=COLUMNS)
    out = (pd.concat([old[~old["event_id"].astype(str).isin(set(new["event_id"]))], new],
                     ignore_index=True)
           .sort_values(["date", "competition", "event_id"]))
    atomic_to_csv(out[COLUMNS], OUT)
    named = out["referee"].fillna("").astype(str).str.len() > 0
    print(f"wrote {len(out)} matches ({named.mean():.1%} with a referee) -> {OUT.relative_to(ROOT)}")
    print(out[named].groupby("competition")["date"].agg(["min", "max", "size"]).to_string())


if __name__ == "__main__":
    main()
