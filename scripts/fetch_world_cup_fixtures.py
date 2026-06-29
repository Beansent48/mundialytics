from __future__ import annotations

"""Convenience wrapper for FIFA World Cup 2026 API-Football fixtures.

Defaults to today's World Cup fixtures in US Eastern time. It deliberately
filters by API-Football league=1 and season=2026, and then post-filters kickoff
calendar date in the requested timezone so users do not accidentally get games
from yesterday/tomorrow because of timezone boundaries.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fetch_api_football_fixtures import main


def _has_date_selector(argv: list[str]) -> bool:
    return any(a in {"--today", "--tomorrow", "--date", "--from-date", "--to-date"} for a in argv)


if __name__ == "__main__":
    argv = sys.argv[1:]
    patched = [sys.argv[0]]
    if "--world-cup" not in argv:
        patched.append("--world-cup")
    if not _has_date_selector(argv):
        patched.append("--today")
    if "--timezone" not in argv:
        patched.extend(["--timezone", "America/New_York"])
    if "--out" not in argv:
        patched.extend(["--out", "outputs/api_football_world_cup_today_et.csv"])
    if "--raw-out" not in argv:
        patched.extend(["--raw-out", "outputs/api_football_world_cup_today_et.json"])
    if "--print-table" not in argv:
        patched.append("--print-table")
    patched.extend(argv)
    sys.argv = patched
    main()
