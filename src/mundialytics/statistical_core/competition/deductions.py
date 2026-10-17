"""Administrative points deductions, which no match result can tell you about.

Finding 4 of docs/FOOTBALL_SEMANTICS_AUDIT_2026_10_17.md: `compute_standings`
derives points purely from played matches, so for a sanctioned club the live
table -- and every title, top-four and relegation probability built on it -- is
off by the whole deduction. Juventus -10 (2022/23), Everton -8 and Nottingham
Forest -4 (2023/24) all sit inside the backtest window.

The table lives in `data/reference/points_deductions.csv` and is **data, not
code**: it has to be maintained by hand, because a deduction is a decision by a
federation, not something any feed publishes in a results file.

Every seeded row is marked `verified=no`. They are written from memory and the
appeals were messy -- Juventus was docked 15 in January 2023, had it annulled in
April and was docked 10 in May -- so the numbers want checking against the
federations before anyone leans on them. An unverified row is still applied: a
deduction that is roughly right beats a table that pretends it never happened.
Set `verified=yes` once checked, and `strict=True` below to ignore the rest.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mundialytics.statistical_core.schemas import canonical_name

TABLE = Path(__file__).resolve().parents[4] / "data/reference/points_deductions.csv"


def load_deductions(season: str | None = None, competition: str | None = None, *,
                    strict: bool = False, path: str | Path | None = None) -> dict[str, int]:
    """{canonical team: points to add (negative)} for one season and competition.

    A missing file, an unreadable one or no matching row all give {} -- a league
    with no sanctions is the normal case, and this must never be the reason a
    table fails to build.
    """
    p = Path(path) if path else TABLE
    try:
        df = pd.read_csv(p)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return {}
    if df.empty or not {"team", "points"} <= set(df.columns):
        return {}
    if strict and "verified" in df.columns:
        df = df[df["verified"].astype(str).str.strip().str.lower().isin(("yes", "true", "1"))]
    if season is not None and "season" in df.columns:
        df = df[df["season"].astype(str) == str(season)]
    if competition is not None and "competition" in df.columns:
        want = canonical_name(competition).lower()
        df = df[df["competition"].astype(str).map(lambda c: canonical_name(c).lower()) == want]
    out: dict[str, int] = {}
    for r in df.itertuples(index=False):
        pts = pd.to_numeric(getattr(r, "points", None), errors="coerce")
        if pd.isna(pts):
            continue
        # several rows for one club in a season add up (Everton's -6 and -2)
        out[canonical_name(str(r.team))] = out.get(canonical_name(str(r.team)), 0) + int(pts)
    return out
