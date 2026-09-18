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
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.providers.espn_fixtures import espn_json  # noqa: E402

DIR = ROOT / "data/external/advanced/espn"
OUT_P = DIR / "espn_player_match_current.csv"
OUT_M = DIR / "espn_matches_current.csv"
OUT_E = DIR / "espn_player_events_current.csv"
OUT_T = DIR / "espn_team_match_current.csv"
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

# ESPN team boxscore stat name -> our column. Team-level totals (possession,
# corners, fouls) that never appear per player; read from the same summary call.
TEAM_STATS = {
    "possessionPct": "possession", "totalShots": "shots",
    "shotsOnTarget": "sot", "wonCorners": "corners",
    "foulsCommitted": "fouls", "yellowCards": "yellows", "redCards": "reds",
}


def _event_code(kind: str, scoring: bool) -> str | None:
    """ESPN keyEvent `type.type` -> our normalised code, or None to skip.

    Only the events a timeline shows survive: goals (however scored), own goals,
    scored penalties and cards. Kickoffs, delays, substitutions, VAR checks and
    missed penalties fall through as None. A second yellow (`yellow-red-card`) is
    a sending-off, so it settles as a red.
    """
    k = (kind or "").lower()
    if k in ("red-card", "yellow-red-card"):
        return "red"
    if k == "yellow-card":
        return "yellow"
    if not scoring:  # non-scoring, non-card -> not a timeline event
        return None
    if k == "own-goal":
        return "own_goal"
    if k.startswith("penalty"):
        return "penalty"
    return "goal"


def _minute_num(disp: str) -> float:
    """"45'+2'" -> 45.02, "90'" -> 90.0 -- a scalar the timeline can sort on.

    Stoppage time is folded into the hundredths so it orders after its own base
    minute but before the next (45'+2' < 46'); base minutes never reach 100.
    """
    s = str(disp or "").replace("'", "").strip()
    if not s:
        return -1.0
    if "+" in s:
        base, _, extra = s.partition("+")
        try:
            return float(int(base or 0)) + int(extra or 0) / 100.0
        except ValueError:
            return -1.0
    try:
        return float(int(s))
    except ValueError:
        return -1.0


def _get(url: str, tries: int = 3) -> dict:
    """Thin alias for the shared caller: the User-Agent rule lives in one place."""
    return espn_json(url, timeout=40, tries=tries)


def _aliases() -> dict[str, str]:
    if not ALIASES.exists():
        return {}
    df = pd.read_csv(ALIASES)
    return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))


def list_matches(code: str, comp: str, lo: str, hi: str, canon) -> list[dict]:
    """Completed matches for one league between two AAAAMMDD dates.

    One scoreboard request per calendar year the window touches: ESPN has
    rejected explicit date ranges since 2026-09-16 (see
    providers/espn_fixtures.fetch_scoreboard_events).
    """
    from mundialytics.providers.espn_fixtures import fetch_scoreboard_events

    start = pd.Timestamp(lo).date()
    end = pd.Timestamp(hi).date()
    games = []
    for ev in fetch_scoreboard_events(code, start, end, timeout=40, tries=3):
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
    return games


def roster_rows(s: dict, game: dict, canon) -> list[dict]:
    """Full roster for one match, from an already-fetched summary payload."""
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
                   "starter": bool(p.get("starter")),
                   "subbed_in": bool(p.get("subbedIn"))}
            for k, col in STATS.items():
                v = raw.get(k)
                row[col] = pd.to_numeric(v, errors="coerce") if v is not None else 0
            players.append(row)
    return players


def team_rows(s: dict, game: dict, canon) -> list[dict]:
    """Team-level totals for one match, from the summary's boxscore.

    Possession, corners and fouls are only ever team totals -- they never appear
    in the per-player roster -- so they come from `boxscore.teams[].statistics`,
    the same summary payload already fetched for the roster and timeline. Each
    entry is a flat {name, value}; some feeds nest them under a group's `stats`,
    so both shapes are handled.
    """
    rows = []
    for tm in (s.get("boxscore") or {}).get("teams", []) or []:
        team = canon((tm.get("team") or {}).get("displayName", ""))
        raw: dict = {}
        for st in tm.get("statistics", []) or []:
            if isinstance(st, dict) and "stats" in st:
                for x in st.get("stats") or []:
                    raw[x.get("name")] = x.get("value", x.get("displayValue"))
            else:
                raw[st.get("name")] = st.get("value", st.get("displayValue"))
        if not raw:
            continue
        row = {"event_id": game["event_id"], "competition": game["competition"],
               "date": game["date"], "team": team}
        for k, col in TEAM_STATS.items():
            v = raw.get(k)
            row[col] = pd.to_numeric(v, errors="coerce") if v is not None else pd.NA
        rows.append(row)
    return rows


def event_rows(s: dict, game: dict, canon) -> list[dict]:
    """Minute-by-minute timeline for one match, from the summary's `keyEvents`.

    The summary is the only endpoint that names the assister: its keyEvents carry
    `participants` (scorer at [0], assister at [1] for a goal) plus the minute,
    the scoring team and a normalised `type.type`. The scoreboard's `details`,
    used before, has none of that, so the timeline is read from here -- the same
    call that already fetches the roster, so it costs no extra request.

    An own goal is credited by ESPN to the team that BENEFITS, and it stays that
    way here so per-side goal counts still add up to the scoreline; the scorer
    (an opponent) is kept as `player` and flagged with `own_goal` so the reader
    can be told it was an own goal.
    """
    sides = {str((sd.get("team") or {}).get("id")):
             canon((sd.get("team") or {}).get("displayName", ""))
             for sd in (s.get("rosters") or [])}
    rows = []
    for k in s.get("keyEvents") or []:
        kind = (k.get("type") or {}).get("type", "")
        scoring = bool(k.get("scoringPlay"))
        code = _event_code(kind, scoring)
        if code is None:
            continue
        parts = k.get("participants") or []
        player = ((parts[0].get("athlete") or {}) if parts else {}).get("displayName")
        if not player:
            continue
        assist = ""
        if code in ("goal", "penalty") and len(parts) > 1:
            assist = ((parts[1].get("athlete") or {}).get("displayName") or "")
        tid = str((k.get("team") or {}).get("id", ""))
        disp = (k.get("clock") or {}).get("displayValue", "")
        rows.append({
            "event_id": game["event_id"], "competition": game["competition"],
            "date": game["date"], "team": sides.get(tid, ""),
            "player": player, "assist": assist,
            "minute": disp, "minute_num": _minute_num(disp),
            "period": int((k.get("period") or {}).get("number", 0) or 0),
            "type": (k.get("type") or {}).get("text", ""), "type_code": code,
            "scoring": scoring, "penalty": code == "penalty",
            "own_goal": code == "own_goal", "shootout": bool(k.get("shootout")),
        })
    return rows


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
    # The events file gained an `assist` column and a summary-based source; an old
    # file written before that has neither, so drop it and rebuild once -- every
    # match then gains its assist rather than half the season staying without one.
    if "assist" not in old_e.columns:
        old_e = pd.DataFrame()
    have_e = set(old_e["event_id"].astype(str)) if "event_id" in old_e.columns else set()
    old_t = (pd.read_csv(OUT_T, low_memory=False)
             if (OUT_T.exists() and not args.rebuild) else pd.DataFrame())
    # Team totals were added later; an older file predates the `possession` column,
    # so drop it and rebuild once so every match gains its team stats at once.
    if "possession" not in old_t.columns:
        old_t = pd.DataFrame()
    have_t = set(old_t["event_id"].astype(str)) if "event_id" in old_t.columns else set()

    games, players, events, teams = [], [], [], []
    for code, comp in LEAGUES.items():
        try:
            g = list_matches(code, comp, lo, hi, canon)
        except Exception as exc:
            print(f"  {comp:15s} FALLO listado: {type(exc).__name__} {str(exc)[:60]}")
            continue
        games += g
        # One summary request per match, fetched only when the roster OR the
        # timeline is still missing, and parsed for whichever is.
        todo = [x for x in g
                if x["event_id"] not in have
                or x["event_id"] not in have_e
                or x["event_id"] not in have_t]
        n_ok = 0
        for x in todo:
            try:
                s = _get(f"{BASE.format(code=code)}/summary?event={x['event_id']}")
            except Exception:
                continue
            if x["event_id"] not in have:
                pr = roster_rows(s, x, canon)
                if pr:
                    players += pr
                    n_ok += 1
            if x["event_id"] not in have_e:
                events += event_rows(s, x, canon)
            if x["event_id"] not in have_t:
                teams += team_rows(s, x, canon)
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
        edf = pd.concat([old_e, edf], ignore_index=True)
    if len(edf):
        # Dedupe on the event's identity, not the whole row: a stray NaN in one
        # copy once let byte-identical events survive `drop_duplicates()` twice.
        edf = edf.drop_duplicates(
            subset=["event_id", "team", "player", "minute", "type_code"],
            keep="last",
        ).sort_values(["date", "event_id", "minute_num"], kind="stable")
        edf.to_csv(OUT_E, index=False)

    tdf = pd.DataFrame(teams)
    if len(old_t) and "event_id" in old_t.columns:
        tdf = pd.concat([old_t, tdf], ignore_index=True)
    if len(tdf):
        tdf = tdf.drop_duplicates(subset=["event_id", "team"], keep="last") \
                 .sort_values(["date", "event_id"], kind="stable")
        tdf.to_csv(OUT_T, index=False)

    app = pd.to_numeric(pdf.get("appearances"), errors="coerce").fillna(0)
    print(f"\nESCRITO {OUT_P.name}: {len(pdf):,} filas jugador-partido "
          f"({int((app > 0).sum()):,} con minutos) - {pdf['player'].nunique():,} jugadores")
    print(f"        {OUT_M.name}: {len(gdf):,} partidos")
    if len(tdf):
        print(f"        {OUT_T.name}: {len(tdf):,} filas equipo-partido")
    for c in ("goals", "shots", "assists", "yellow_cards", "red_cards"):
        if c in pdf.columns:
            tot = int(pd.to_numeric(pdf[c], errors="coerce").fillna(0).sum())
            print(f"          {c:13s} {tot}")


if __name__ == "__main__":
    main()
