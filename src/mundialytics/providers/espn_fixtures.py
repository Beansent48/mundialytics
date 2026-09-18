"""League calendars from ESPN's public JSON.

WHY. Everything forward-looking in this project -- the live league forecast and
the pre-kickoff round logger, which is the only reason the track record is
evidence rather than a story -- read their fixtures from fixturedownload. On
2026-09-04 that host stopped answering for all five leagues at once, and both
went down with it: the forecast page reported "no matches remaining" for a
league three rounds old, and no upcoming round could be logged at all.

That was the fourth source to degrade in the same stretch (ClubElo's API 502ing
for three days, Understat not publishing the season, FBref stalling mid-fetch),
so the fixture calendar gets a second path rather than a retry.

ESPN answers the whole season in ONE request -- 380 fixtures with their status,
played and unplayed alike -- against fixturedownload's habit of listing 394
fixtures for a 380-match season because it lags on results. Team names go
through the same curated alias file, with the normaliser as fallback rather
than fixturedownload's plain dict lookup, which silently dropped every club it
had never been taught.

This is additive: callers try ESPN and keep fixturedownload behind it, so a bad
day at either host is survivable.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

ESPN_LEAGUE_CODES = {
    "Premier League": "eng.1",
    "LaLiga": "esp.1",
    "Bundesliga": "ger.1",
    "Serie A": "ita.1",
    "Ligue 1": "fra.1",
}
_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard"
_ALIASES = Path("data/curated/fixture_team_aliases.csv")

COLUMNS = ["match_id", "competition", "season", "date", "matchday",
           "home_team", "away_team", "completed", "home_goals", "away_goals"]


# ESPN's edge refuses browser-shaped User-Agents. A request that claims to be
# Chrome while carrying Python's TLS fingerprint reads as a bot and gets a hard
# 403: measured on 2026-09-18, three tries each, "Mozilla/5.0", a full Chrome
# string and a plain "Mundialytics/0.50" all failed on every endpoint, big five
# and UEFA alike, while sending no User-Agent at all succeeded every time. The
# header is therefore deliberately absent, and every ESPN caller in the repo
# goes through this one function so it cannot creep back in one script at a time.
def espn_json(url: str, timeout: int = 45, tries: int = 3) -> dict:
    """One ESPN JSON call, retried with a backoff. Raises if the last try fails."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                return json.load(fh)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return {}


def fetch_scoreboard_events(
    code: str,
    start: date,
    end: date,
    timeout: int = 45,
    tries: int = 1,
) -> list[dict]:
    """Every ESPN scoreboard event for one competition dated within [start, end].

    ESPN stopped accepting an explicit YYYYMMDD-YYYYMMDD range on 2026-09-16 (it
    answers 400). `dates=<year>` still works, but it means the CALENDAR year: for
    LaLiga `dates=2026` is January-December 2026, i.e. the second half of 2025/26
    plus the first half of 2026/27. A July-June season therefore takes both years
    and a date filter. Asking for the start year alone -- the first fix -- mixed
    the previous season into the calendar (September fixtures numbered round 28)
    and left out everything after 31 December.

    Raises if any request fails: half a season is worse than none, because a
    caller can fall back on an empty result but cannot tell a partial one apart.
    """
    lo, hi = start.isoformat(), end.isoformat()
    events: dict[str, dict] = {}
    for year in range(start.year, end.year + 1):
        url = f"{_BASE.format(code=code)}?dates={year}&limit=1000"
        data = espn_json(url, timeout=timeout, tries=tries)
        for ev in data.get("events", []) or []:
            if lo <= str(ev.get("date", ""))[:10] <= hi:
                events[str(ev.get("id"))] = ev
    return sorted(events.values(), key=lambda ev: str(ev.get("date", "")))


def _load_aliases(root: str | Path = ".") -> dict[str, str]:
    path = Path(root) / _ALIASES
    if not path.exists():
        return {}
    try:
        return dict(pd.read_csv(path).itertuples(index=False, name=None))
    except Exception:
        return {}


def fetch_season_fixtures(
    competition: str,
    season: str,
    alias: dict[str, str] | None = None,
    root: str | Path = ".",
    timeout: int = 45,
) -> pd.DataFrame:
    """Every fixture of one league-season, played and unplayed.

    Returns an empty frame (never raises) when the league is unknown or the host
    is unreachable, so a caller can fall through to its previous source.
    """
    from mundialytics.identity.normalization import canonical_team_name

    code = ESPN_LEAGUE_CODES.get(competition)
    if not code:
        return pd.DataFrame(columns=COLUMNS)
    if alias is None:
        alias = _load_aliases(root)

    def canon(name: str) -> str:
        return alias.get(name, canonical_team_name(name))

    y1 = int(str(season)[:4])
    try:
        events = fetch_scoreboard_events(code, date(y1, 7, 1), date(y1 + 1, 6, 30),
                                         timeout=timeout)
    except Exception:
        return pd.DataFrame(columns=COLUMNS)

    rows = []
    for ev in events:
        try:
            c = ev["competitions"][0]
        except (KeyError, IndexError):
            continue
        home = next((x for x in c.get("competitors", []) if x.get("homeAway") == "home"), None)
        away = next((x for x in c.get("competitors", []) if x.get("homeAway") == "away"), None)
        if home is None or away is None:
            continue
        done = bool(c.get("status", {}).get("type", {}).get("completed"))
        rows.append({
            "match_id": f"espn_{ev.get('id')}",
            "competition": competition,
            "season": season,
            "date": pd.to_datetime(str(ev.get("date", ""))[:10], errors="coerce"),
            "home_team": canon(home["team"]["displayName"]),
            "away_team": canon(away["team"]["displayName"]),
            "completed": done,
            # only meaningful once completed; left as NA otherwise so a caller
            # can never mistake a scheduled fixture for a 0-0
            "home_goals": float(home.get("score")) if done and home.get("score") is not None else pd.NA,
            "away_goals": float(away.get("score")) if done and away.get("score") is not None else pd.NA,
        })
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date")
    out["matchday"] = _matchdays(out)
    return out[COLUMNS].reset_index(drop=True)


def _matchdays(fx: pd.DataFrame) -> list[int]:
    """Round number per fixture, counted per team rather than per date.

    ESPN publishes no matchweek at all, and slicing the calendar into blocks of
    n/2 fixtures breaks the moment one is postponed -- every later round shifts.
    Counting each club's own fixtures survives that.

    The two clubs can disagree when one has a match in hand, and the LOWER count
    is the round the rest of the league is playing: taking the higher one lets a
    single brought-forward fixture drag its opponents a round ahead. Measured
    against the rounds already in the prediction log on 2026-09-04, the lower
    count agreed on all 88 upcoming fixtures and the higher one missed four.
    """
    seen: dict[str, int] = {}
    out = []
    for h, a in zip(fx["home_team"], fx["away_team"]):
        nh = seen[h] = seen.get(h, 0) + 1
        na = seen[a] = seen.get(a, 0) + 1
        out.append(min(nh, na))
    return out


def remaining_fixtures(
    competition: str,
    season: str,
    played: pd.DataFrame | None = None,
    alias: dict[str, str] | None = None,
    root: str | Path = ".",
) -> pd.DataFrame:
    """Fixtures still to play, with anything we already hold a result for removed.

    ESPN's own `completed` flag does most of the work, but our foundation can be
    ahead of it for a match finishing right now, so results we already hold win.
    """
    from mundialytics.identity.normalization import canonical_team_name

    fx = fetch_season_fixtures(competition, season, alias=alias, root=root)
    if fx.empty:
        return fx
    fx = fx[~fx["completed"].astype(bool)].copy()
    if played is not None and len(played):
        done = {(canonical_team_name(h), canonical_team_name(a))
                for h, a in zip(played["home_team"], played["away_team"])}
        keep = [(canonical_team_name(h), canonical_team_name(a)) not in done
                for h, a in zip(fx["home_team"], fx["away_team"])]
        fx = fx[keep]
    return fx.reset_index(drop=True)
