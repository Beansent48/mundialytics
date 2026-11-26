#!/usr/bin/env python3
"""Fetch real betting odds from The Odds API (free tier, 500 req/day).

Logs O/U goals, corners and yellows odds per match into a CSV that the bot
pipeline and the backtest can consume.  Run daily before the bot JSON step.

Usage:
    python scripts/fetch_theoddsapi_odds.py              # today + tomorrow
    python scripts/fetch_theoddsapi_odds.py --days 3      # next 3 days
    python scripts/fetch_theoddsapi_odds.py --dry-run     # show quota cost only

Requires env var THE_ODDS_API_KEY (free key from https://the-odds-api.com).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "odds_api"
LOG_CSV = OUT_DIR / "odds_log.csv"
CACHE_DIR = OUT_DIR / "cache"

BASE = "https://api.the-odds-api.com/v4"

SPORT_KEYS = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
    "soccer_efl_champ",
    "soccer_spain_segunda_division",
    "soccer_italy_serie_b",
    "soccer_germany_bundesliga2",
    "soccer_france_ligue_two",
    "soccer_uefa_europa_league",
    "soccer_uefa_europa_conference_league",
]

# Leagues where corners/cards odds actually exist on The Odds API.
# Segunda, Serie B, Ligue 2 and Conference return 0 rows — skip them.
CORNERS_CARDS_SPORTS = {
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
    "soccer_efl_champ",
    "soccer_germany_bundesliga2",
    "soccer_uefa_europa_league",
}

BULK_MARKETS = "h2h,totals"
EVENT_MARKETS = "alternate_totals_corners,alternate_totals_cards"

CSV_COLS = [
    "fetch_ts", "sport", "event_id", "commence_time",
    "home_team", "away_team",
    "market", "bookmaker", "line", "over_price", "under_price", "draw_price",
]


def _api_key() -> str:
    key = os.environ.get("THE_ODDS_API_KEY", "").strip()
    if not key:
        sys.exit("Set THE_ODDS_API_KEY env var (free key from https://the-odds-api.com)")
    return key


def _get(url: str, params: dict, label: str = "") -> tuple[dict | list | None, dict]:
    """GET with quota tracking. Returns (json_body, headers)."""
    resp = requests.get(url, params=params, timeout=30)
    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    print(f"  [{label}] {resp.status_code}  quota: {used} used / {remaining} left")
    if resp.status_code == 401:
        sys.exit("Invalid API key")
    if resp.status_code == 429:
        print("  RATE LIMITED — stopping")
        return None, {}
    if resp.status_code == 422:
        print(f"  Unprocessable (market not available?): {resp.text[:200]}")
        return None, {}
    resp.raise_for_status()
    return resp.json(), dict(resp.headers)


def fetch_sport_events(api_key: str, sport: str,
                       from_dt: str, to_dt: str) -> list[dict]:
    """List upcoming events for a sport (free, no quota cost)."""
    url = f"{BASE}/sports/{sport}/events"
    params = {
        "apiKey": api_key,
        "commenceTimeFrom": from_dt,
        "commenceTimeTo": to_dt,
    }
    data, _ = _get(url, params, f"events {sport}")
    return data or []


def fetch_bulk_odds(api_key: str, sport: str,
                    from_dt: str, to_dt: str) -> list[dict]:
    """Fetch totals odds for all events in a sport (1 quota unit per region)."""
    url = f"{BASE}/sports/{sport}/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu,uk",
        "markets": BULK_MARKETS,
        "oddsFormat": "decimal",
        "commenceTimeFrom": from_dt,
        "commenceTimeTo": to_dt,
    }
    data, _ = _get(url, params, f"totals {sport}")
    return data or []


def fetch_event_odds(api_key: str, sport: str, event_id: str,
                     markets: str) -> dict | None:
    """Fetch corners/cards odds for a single event.

    Uses eu region only (Pinnacle is there) to halve quota cost.
    """
    url = f"{BASE}/sports/{sport}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": markets,
        "oddsFormat": "decimal",
    }
    data, _ = _get(url, params, f"event {event_id[:8]}")
    return data


def parse_odds_rows(events: list[dict], market_filter: str | None = None,
                    fetch_ts: str = "") -> list[dict]:
    """Extract rows from the API response."""
    rows = []
    for ev in events:
        base = {
            "fetch_ts": fetch_ts,
            "sport": ev.get("sport_key", ""),
            "event_id": ev.get("id", ""),
            "commence_time": ev.get("commence_time", ""),
            "home_team": ev.get("home_team", ""),
            "away_team": ev.get("away_team", ""),
        }
        for bk in ev.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                mkt_key = mkt.get("key", "")
                if market_filter and mkt_key != market_filter:
                    continue
                outcomes = mkt.get("outcomes", [])
                if mkt_key == "h2h":
                    home = next((o for o in outcomes if o.get("name") == ev.get("home_team")), None)
                    draw = next((o for o in outcomes if o.get("name") == "Draw"), None)
                    away = next((o for o in outcomes if o.get("name") == ev.get("away_team")), None)
                    if home and draw and away:
                        rows.append({
                            **base,
                            "market": "h2h",
                            "bookmaker": bk.get("key", ""),
                            "line": "1x2",
                            "over_price": home.get("price", ""),
                            "under_price": away.get("price", ""),
                            "draw_price": draw.get("price", ""),
                        })
                    continue
                over = next((o for o in outcomes if o.get("name") == "Over"), None)
                under = next((o for o in outcomes if o.get("name") == "Under"), None)
                if over and under:
                    rows.append({
                        **base,
                        "market": mkt_key,
                        "bookmaker": bk.get("key", ""),
                        "line": over.get("point", ""),
                        "over_price": over.get("price", ""),
                        "under_price": under.get("price", ""),
                    })
    return rows


def append_csv(rows: list[dict]):
    """Append rows to the odds log CSV."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exists = LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--corners-cards", action="store_true",
                        help="Also fetch corners/cards per event (costly)")
    args = parser.parse_args()

    api_key = _api_key()
    now = datetime.now(timezone.utc)
    from_dt = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    to_dt = (now + timedelta(days=args.days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    fetch_ts = from_dt

    print(f"Window: {from_dt[:16]} -> {to_dt[:16]}")
    print(f"Sports: {len(SPORT_KEYS)}")
    print()

    all_rows: list[dict] = []
    event_count = 0

    for sport in SPORT_KEYS:
        # Totals (O/U goals) — bulk, cheap
        events = fetch_bulk_odds(api_key, sport, from_dt, to_dt)
        if events:
            rows = parse_odds_rows(events, fetch_ts=fetch_ts)
            all_rows.extend(rows)
            event_count += len(events)
            print(f"    {sport}: {len(events)} events, {len(rows)} odds rows")

        if args.corners_cards and events and sport in CORNERS_CARDS_SPORTS:
            for ev in events:
                eid = ev.get("id", "")
                if not eid:
                    continue
                ev_data = fetch_event_odds(api_key, sport, eid, EVENT_MARKETS)
                if ev_data:
                    rows = parse_odds_rows([ev_data], fetch_ts=fetch_ts)
                    all_rows.extend(rows)
                    print(f"      +{len(rows)} corners/cards rows")

    print(f"\nTotal: {event_count} events, {len(all_rows)} odds rows")

    if args.dry_run:
        print("Dry run — not writing CSV")
        if all_rows:
            for r in all_rows[:5]:
                print(f"  {r['home_team']} vs {r['away_team']} | "
                      f"{r['market']} {r['line']} | "
                      f"O {r['over_price']} / U {r['under_price']} [{r['bookmaker']}]")
        return

    if all_rows:
        append_csv(all_rows)
        print(f"Appended to {LOG_CSV}")

        # Also save raw JSON for debugging
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_f = CACHE_DIR / f"fetch_{now.strftime('%Y%m%d_%H%M%S')}.json"
        cache_f.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        print(f"Raw cache: {cache_f}")
    else:
        print("No odds found")


if __name__ == "__main__":
    main()
