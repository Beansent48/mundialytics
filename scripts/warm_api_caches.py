"""Pre-compute the API's slow answers right after a data refresh.

The league forecast (10,000 simulations of the rest of the season, ~35 s) and
the Golden Boot race (~30 s) are cached on disk, keyed on how many matches the
season has played, so every refresh that brings a result invalidates them. The
first visitor after it paid for the recomputation — and the web client aborts
an API call after 20 s, so for that visitor the league page rendered as a 404
and the awards page as an error.

This calls the same endpoint functions the web calls, in-process, so the cache
files, keys and shapes are exactly the ones the API reads back.

    python scripts/warm_api_caches.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from api import catalogue as cat
    from api.main import awards, league

    failures = 0
    for slug in cat.COMPETITIONS:
        for name, endpoint in (("liga", league), ("bota de oro", awards)):
            t0 = time.time()
            try:
                endpoint(slug)
                print(f"  {slug:15s} {name:12s} OK ({time.time() - t0:.0f}s)", flush=True)
            except Exception as exc:
                failures += 1
                print(f"  {slug:15s} {name:12s} FALLO {type(exc).__name__}: {str(exc)[:80]}",
                      flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
