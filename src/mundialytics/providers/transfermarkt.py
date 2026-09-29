"""Live squads and market values from transfermarkt.com, for the squad-value input.

The Kaggle dump the history is built from stopped updating in June 2026. Weekly values
instead of June-frozen ones are worth another -0.0003 RPS in the backtest
(scripts/squad_value/README.md), so the current snapshot is read from the site.

Politeness: the site's robots.txt allows these pages. Requests identify this project
rather than pretend to be a browser, wait DELAY_S between calls, and each page is cached
for the day, so a re-run does not fetch again. About 100 pages a refresh (5 league pages +
one squad page per club), and the caller refreshes at most weekly. If the site starts
refusing, this fails and the caller falls back to ESPN rosters x the dump; it never tries
to get around a block.
"""
from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

BASE = "https://www.transfermarkt.com"
USER_AGENT = "mundialytics-research/0.55 (+https://github.com/Beansent48/mundialytics)"
DELAY_S = 3.0
LEAGUES = {
    "Premier League": ("premier-league", "GB1"),
    "LaLiga": ("laliga", "ES1"),
    "Bundesliga": ("bundesliga", "L1"),
    "Serie A": ("serie-a", "IT1"),
    "Ligue 1": ("ligue-1", "FR1"),
}
_UNITS = {"bn": 1e9, "m": 1e6, "k": 1e3}


def season_id(date=None) -> int:
    """Transfermarkt's season id is the starting year (2026 = 2026/27)."""
    d = pd.Timestamp(date) if date is not None else pd.Timestamp.now()
    return d.year if d.month >= 7 else d.year - 1


def parse_value(text: str | None) -> float | None:
    """'€30.00m' -> 30e6, '€500k' -> 5e5, '€1.43bn' -> 1.43e9, '-' -> None."""
    if not text:
        return None
    m = re.search(r"([\d.,]+)\s*(bn|m|k)\b", text.replace("\xa0", " "), flags=re.I)
    if not m:
        return None
    return float(m.group(1).replace(",", "")) * _UNITS[m.group(2).lower()]


def parse_league_clubs(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.items")
    out, seen = [], set()
    for row in table.select("tbody > tr") if table else []:
        a = row.select_one("td.hauptlink a[href*='/verein/']")
        if not a:
            continue
        m = re.search(r"^/([^/]+)/startseite/verein/(\d+)", a["href"])
        if m and int(m.group(2)) not in seen:
            seen.add(int(m.group(2)))
            out.append({"club_id": int(m.group(2)), "slug": m.group(1),
                        "club_name": a.get("title") or a.get_text(strip=True)})
    return out


def parse_squad(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.items")
    out, seen = [], set()
    for row in table.select("tbody > tr") if table else []:
        a = row.select_one("td.hauptlink a[href*='/profil/spieler/']")
        if not a:
            continue
        pid = int(re.search(r"/spieler/(\d+)", a["href"]).group(1))
        if pid in seen:
            continue
        seen.add(pid)
        v = row.select_one("td.rechts.hauptlink")
        out.append({"player_id": pid, "player": a.get_text(strip=True),
                    "value_eur": parse_value(v.get_text(strip=True) if v else None)})
    return out


def _fetch(url: str, cache_dir: Path, today: str) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "_", url.replace(BASE, ""))[:150]
    f = cache_dir / today / f"{key}.html"
    if f.exists():
        return f.read_text(encoding="utf-8")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", errors="replace")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(html, encoding="utf-8")
    time.sleep(DELAY_S)
    return html


def fetch_live_squads(cache_dir: Path, season: int | None = None, leagues: dict | None = None,
                      log=print) -> pd.DataFrame:
    """One row per (club, player): competition, club_id, club_name, player_id, player, value_eur."""
    season = season if season is not None else season_id()
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    rows = []
    for comp, (slug, code) in (leagues or LEAGUES).items():
        clubs = parse_league_clubs(_fetch(f"{BASE}/{slug}/startseite/wettbewerb/{code}/saison_id/{season}",
                                          cache_dir, today))
        if not clubs:
            raise RuntimeError(f"no clubs parsed for {comp}; page layout changed?")
        for c in clubs:
            squad = parse_squad(_fetch(f"{BASE}/{c['slug']}/kader/verein/{c['club_id']}/saison_id/{season}/plus/1",
                                       cache_dir, today))
            for p in squad:
                rows.append({"competition": comp, **c, **p})
        log(f"  transfermarkt {comp}: {len(clubs)} clubs")
    return pd.DataFrame(rows)
