from __future__ import annotations

"""Per-player match stats for the CURRENT season, from ESPN's public JSON.

WHY THIS EXISTS. Player markets are logged pre-kickoff for all five leagues, and
the only thing able to settle them was FBref -- a scraper that rate-limits, whose
season labels are wrong (asking for "2627" returns 1926-27), and whose five-league
fetch stalled without writing a byte. Only the Premier League was ever downloaded,
leaving 79% of the logged player predictions unsettleable: the same failure that
made booking points worthless for months.

WHY THIS ONE IS STURDIER.
  * JSON API, not a page scrape: no key, no rate-limit dance.
  * Matches are asked for BY DATE, so there is no season label to be wrong about.
  * It carries the full ROSTER, not just the players who did something. That
    distinction decides whether the track record is honest: ESPN's event list
    names only scorers and booked players, so settling from events alone would
    score a striker's market only when he scored, and the measured hit rate
    would be near 100% by construction. The roster says who actually played, so
    "played and did not score" settles as a miss, exactly as it should.
  * Own goals are separated from real ones (ESPN credits an own goal to the
    BENEFITING team, so its scorer belongs to the other side and must never
    settle his own scorer market as a hit).

One call per league to list the season's matches, then one per match, skipping
matches already stored. Cross-checked on the first build: all 97 played matches
agreed with football-data's scorelines, and all 100 reconstructed from their own
scorers.

Run: .venv/Scripts/python.exe scripts/fetch_espn_match_events.py
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DIR = ROOT / "data/external/advanced/espn"
OUT_P = DIR / "espn_player_match_current.csv"
OUT_M = DIR / "espn_matches_current.csv"
OUT_E = DIR / "espn_player_events_current.csv"
ALIASES = ROOT / "data/curated/fixture_team_aliases.csv"
BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}"

LEAGUES = {
    "eng.1": "Premier League",
    "esp.1": "LaLiga",
    "ger.1": "Bundesliga",
    "ita.1": "Serie A",
    "fra.1": "Ligue 1",
}

# ESPN stat name -> our column
STATS = {
    "totalGoals": "goals", "totalShots": "shots", "shotsOnTarget": "sot",
    "goalAssists": "assists", "yellowCards": "yellow_cards",
    "redCards": "red_cards", "ownGoals": "own_goals",
    "appearances": "appearances", "subIns": "sub_ins",
}


def _get(url: str, tries: int = 3) -> dict:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=40) as fh:
                return json.load(fh)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    return {}


def _aliases() -> dict[str, str]:
    if not ALIASES.exists():
        return {}
    df = pd.read_csv(ALIASES)
    return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))


def list_matches(code: str, comp: str, lo: str, hi: str,
                 canon) -> tuple[list[dict], list[dict]]:
    """Completed matches for one league, from the scoreboard (one request).

    The timeline comes from here too. It is tempting to read it from the richer
    per-match summary instead, but that endpoint's `keyEvents` carries no
    `athletesInvolved` at all -- an extraction written against it silently
    yields nothing. The scoreboard's `details` is the only place the minute, the
    scorer and the penalty flag appear together, and it costs no extra request.
    """
    data = _get(f"{BASE.format(code=code)}/scoreboard?dates={lo}-{hi}&limit=1000")
    games, events = [], []
    for ev in data.get("events", []):
        c = ev["competitions"][0]
        if not c.get("status", {}).get("type", {}).get("completed"):
            continue
        home = next((x for x in c.get("competitors", []) if x.get("homeAway") == "home"), None)
        away = next((x for x in c.get("competitors", []) if x.get("homeAway") == "away"), None)
        if home is None or away is None:
            continue
        eid, date = str(ev["id"]), str(ev.get("date", ""))[:10]
        games.append({
            "event_id": eid, "league": code, "competition": comp, "date": date,
            "home_team": canon(home["team"]["displayName"]),
            "away_team": canon(away["team"]["displayName"]),
            "home_goals": int(home.get("score") or 0),
            "away_goals": int(away.get("score") or 0),
        })

        sides = {str(x["team"]["id"]): canon(x["team"]["displayName"])
                 for x in c.get("competitors", [])}
        for d in c.get("details", []) or []:
            ath = d.get("athletesInvolved") or []
            if not ath or not ath[0].get("displayName"):
                continue
            tid = str((d.get("team") or {}).get("id", ""))
            credited = sides.get(tid, "")
            own = bool(d.get("ownGoal"))
            # an own goal is credited to the other team, so its scorer's own
            # side is the one that did NOT get the goal
            team = credited
            if own:
                other = [v for k, v in sides.items() if k != tid]
                team = other[0] if other else credited
            events.append({
                "event_id": eid, "competition": comp, "date": date, "team": team,
                "player": ath[0]["displayName"],
                "minute": (d.get("clock") or {}).get("displayValue", ""),
                "type": (d.get("type") or {}).get("text", ""),
                "scoring": bool(d.get("scoringPlay")),
                "penalty": bool(d.get("penaltyKick")),
                "own_goal": own, "shootout": bool(d.get("shootout")),
            })
    return games, events


def match_players(code: str, game: dict, canon) -> list[dict]:
    """Full roster for one match (one request)."""
    s = _get(f"{BASE.format(code=code)}/summary?event={game['event_id']}")
    players = []
    for side in s.get("rosters", []) or []:
        team = canon(side.get("team", {}).get("displayName", ""))
        for p in side.get("roster", []) or []:
            who = (p.get("athlete") or {}).get("displayName")
            if not who:
                continue
            raw = {x.get("name"): x.get("value") for x in (p.get("stats") or [])}
            if not raw:
                continue
            row = {"event_id": game["event_id"], "competition": game["competition"],
                   "date": game["date"], "team": team, "player": who,
                   # ESPN's per-match position ("G", "CD-L", "AM", "F"...). It is
                   # the only positional signal in the current-season data, and
                   # the current-squad table needs it to fill an XI by role.
                   "position": ((p.get("position") or {}).get("abbreviation") or ""),
                   "starter": bool(p.get("starter")),
                   "subbed_in": bool(p.get("subbedIn"))}
            for k, col in STATS.items():
                v = raw.get(k)
                row[col] = pd.to_numeric(v, errors="coerce") if v is not None else 0
            players.append(row)
    return players


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="lo", default=None, help="AAAAMMDD")
    ap.add_argument("--to", dest="hi", default=None, help="AAAAMMDD")
    ap.add_argument("--rebuild", action="store_true", help="ignora lo ya descargado")
    args = ap.parse_args()

    from mundialytics.identity.normalization import canonical_team_name
    alias = _aliases()

    def canon(name: str) -> str:
        return alias.get(name, canonical_team_name(name))

    now = pd.Timestamp.now()
    y = now.year if now.month >= 7 else now.year - 1
    lo, hi = args.lo or f"{y}0701", args.hi or now.strftime("%Y%m%d")
    season = f"{y}-{y + 1}"
    print(f"ESPN | temporada {season} | {lo} a {hi}")

    reuse = OUT_P.exists() and not args.rebuild
    old_p = pd.read_csv(OUT_P, low_memory=False) if reuse else pd.DataFrame()
    old_e = (pd.read_csv(OUT_E, low_memory=False)
             if (OUT_E.exists() and not args.rebuild) else pd.DataFrame())
    have = set(old_p["event_id"].astype(str)) if "event_id" in old_p.columns else set()

    games, players, events = [], [], []
    for code, comp in LEAGUES.items():
        try:
            g, ev = list_matches(code, comp, lo, hi, canon)
        except Exception as exc:
            print(f"  {comp:15s} FALLO listado: {type(exc).__name__} {str(exc)[:60]}")
            continue
        games += g
        events += ev
        todo = [x for x in g if x["event_id"] not in have]
        n_ok = 0
        for x in todo:
            try:
                pr = match_players(code, x, canon)
            except Exception:
                continue
            if pr:
                players += pr
                n_ok += 1
            time.sleep(0.25)
        print(f"  {comp:15s} {len(g):3d} partidos "
              f"({len(g) - len(todo)} en cache, {n_ok} descargados)")

    if not games:
        print("Nada descargado.")
        raise SystemExit(1)

    DIR.mkdir(parents=True, exist_ok=True)
    gdf = pd.DataFrame(games).drop_duplicates("event_id")
    gdf.insert(1, "season", season)
    gdf.to_csv(OUT_M, index=False)

    pdf = pd.DataFrame(players)
    if len(old_p):
        pdf = pd.concat([old_p, pdf], ignore_index=True)
    if pdf.empty:
        print("Sin filas de jugador.")
        raise SystemExit(1)
    pdf = pdf.drop_duplicates(subset=["event_id", "team", "player"], keep="last")
    if "season" not in pdf.columns:
        pdf.insert(1, "season", season)
    pdf.to_csv(OUT_P, index=False)

    edf = pd.DataFrame(events)
    if len(old_e) and "event_id" in old_e.columns:
        edf = pd.concat([old_e, edf], ignore_index=True).drop_duplicates()
    if len(edf):
        edf.to_csv(OUT_E, index=False)

    app = pd.to_numeric(pdf.get("appearances"), errors="coerce").fillna(0)
    print(f"\nESCRITO {OUT_P.name}: {len(pdf):,} filas jugador-partido "
          f"({int((app > 0).sum()):,} con minutos) - {pdf['player'].nunique():,} jugadores")
    print(f"        {OUT_M.name}: {len(gdf):,} partidos")
    for c in ("goals", "shots", "assists", "yellow_cards", "red_cards"):
        if c in pdf.columns:
            tot = int(pd.to_numeric(pdf[c], errors="coerce").fillna(0).sum())
            print(f"          {c:13s} {tot}")


if __name__ == "__main__":
    main()
