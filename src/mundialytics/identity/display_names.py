from __future__ import annotations

"""Short display names for players.

Full legal names ([given names][paternal surname][maternal surname], very
common for Spanish/Latin players) break naive `name.split()[-1]` truncation:
it grabs the MATERNAL surname ("Messi Cuccittini" -> "Cuccittini"). This module
prefers a hand-curated CSV (data/identity/player_display_names.csv), and for
everyone else applies a smarter heuristic than last-word.
"""

import functools
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
_CSV = _ROOT / "data/identity/player_display_names.csv"

# common Spanish/Portuguese surname particles that belong WITH the next word
_PARTICLES = {"de", "del", "de la", "da", "dos", "van", "von", "van der",
              "di", "la", "le", "el", "al", "bin", "mac", "mc", "o'"}


@functools.lru_cache(maxsize=1)
def _curated() -> dict[str, str]:
    if not _CSV.exists():
        return {}
    df = pd.read_csv(_CSV)
    return dict(zip(df["player"].astype(str), df["display_name"].astype(str)))


def _heuristic(full: str) -> str:
    """Best-effort short name when the player isn't in the curated CSV.

    Last word is the surname for the vast majority of names (Haaland,
    Lewandowski, Fernandes...); the cases where it's WRONG — Hispanic
    [given][paternal][MATERNAL] like Messi Cuccittini — are exactly what the
    curated CSV covers. The one systematic fix here: glue a preceding particle
    ('de', 'van', 'dos'...) so "Kevin De Bruyne" -> "De Bruyne", not "Bruyne".
    """
    parts = full.split()
    if len(parts) == 1:
        return full
    last = parts[-1]
    prev = parts[-2].lower()
    if prev in _PARTICLES:
        return f"{parts[-2]} {last}"
    return last


def display_name(full: str | None) -> str:
    """Short, recognizable name. CSV first, smart heuristic otherwise."""
    if not full:
        return ""
    full = str(full).strip()
    hit = _curated().get(full)
    if hit:
        return hit
    return _heuristic(full)


def short(full: str | None, max_len: int | None = None) -> str:
    """display_name, optionally truncated (UI chips)."""
    s = display_name(full)
    return s[:max_len] if max_len else s
