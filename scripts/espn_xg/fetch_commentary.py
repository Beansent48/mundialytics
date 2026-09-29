"""Download ESPN match commentary (Opta-style text, one line per shot) for a season.

Why: Sofascore, the live xG source, answers 403 since 2026-09-26, and Understat and FBref
were already gone. ESPN's summary payload carries the full match commentary, and every
attempt is described in the Opta standard: "Attempt saved. X (Team) left footed shot from
the left side of the six yard box is saved ... Assisted by Y with a cross." Body part,
zone and situation are enough for a text-based xG model (scripts/espn_xg/text_xg.py).

Writes one JSON line per match to data/external/advanced/espn/commentary/<season>.jsonl
(event_id, league, date, teams, score, commentary [time, text]); resumes where it left off.

    python scripts/espn_xg/fetch_commentary.py 2025          # 2025/26, Big Five
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.providers.espn_fixtures import espn_json, fetch_scoreboard_events  # noqa: E402

LEAGUES = {"eng.1": "Premier League", "esp.1": "LaLiga", "ger.1": "Bundesliga",
           "ita.1": "Serie A", "fra.1": "Ligue 1"}
BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}"
OUT_DIR = ROOT / "data/external/advanced/espn/commentary"
PAUSE_S = 0.5


def main() -> int:
    fetch_season(int(sys.argv[1]))
    return 0


def fetch_season(year: int, log=print) -> int:
    """Fetch every completed Big Five match of season year/year+1 not yet on disk; returns
    how many were added."""
    out = OUT_DIR / f"{year}-{year + 1}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["event_id"] for line in out.read_text(encoding="utf-8").splitlines() if line}
    start, end = pd.Timestamp(f"{year}-07-15").date(), pd.Timestamp(f"{year + 1}-06-15").date()
    n = 0
    with out.open("a", encoding="utf-8") as fh:
        for code, comp in LEAGUES.items():
            evs = [e for e in fetch_scoreboard_events(code, start, end, timeout=40, tries=3)
                   if e["competitions"][0].get("status", {}).get("type", {}).get("completed")]
            todo = [e for e in evs if str(e["id"]) not in done]
            log(f"{comp}: {len(evs)} completed, {len(todo)} to fetch")
            for e in todo:
                try:
                    s = espn_json(f"{BASE.format(code=code)}/summary?event={e['id']}", timeout=40, tries=3)
                except Exception as exc:  # one bad match must not stop the season
                    log(f"  {e['id']}: {exc}")
                    continue
                comps = e["competitions"][0]["competitors"]
                side = {c["homeAway"]: c for c in comps}
                box = {t["team"]["displayName"]: {x["name"]: x.get("displayValue") for x in t.get("statistics", [])}
                       for t in (s.get("boxscore") or {}).get("teams", [])}
                fh.write(json.dumps({
                    "event_id": str(e["id"]), "league": code, "competition": comp, "date": e["date"][:10],
                    "home": side["home"]["team"]["displayName"], "away": side["away"]["team"]["displayName"],
                    "home_goals": int(side["home"].get("score") or 0), "away_goals": int(side["away"].get("score") or 0),
                    "box": box,
                    "commentary": [{"t": (c.get("time") or {}).get("displayValue"), "text": c.get("text")}
                                   for c in s.get("commentary") or []],
                }, ensure_ascii=False) + "\n")
                n += 1
                if n % 200 == 0:
                    log(f"  {n} fetched")
                time.sleep(PAUSE_S)
    log(f"DONE {n} new matches -> {out}")
    return n


if __name__ == "__main__":
    sys.exit(main())
