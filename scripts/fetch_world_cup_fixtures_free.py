from __future__ import annotations

"""Convenience wrapper for free/keyless World Cup fixtures."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fetch_today_fixtures import main


def _has_date_selector(argv: list[str]) -> bool:
    return any(a in {"--today", "--tomorrow", "--date"} for a in argv)


if __name__ == "__main__":
    argv = sys.argv[1:]
    patched = [sys.argv[0], "--competition", "world_cup"]
    if not _has_date_selector(argv):
        patched.append("--today")
    if "--timezone" not in argv:
        patched.extend(["--timezone", "America/New_York"])
    patched.extend(argv)
    sys.argv = patched
    main()
