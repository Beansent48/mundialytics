"""Big-five clubs' European matches, as rows the walk-forward form can consume.

WHY. The engine's training foundation is 46,091 rows of football-data.co.uk and
nothing else: five domestic leagues. A club in Europe plays a median 8-12 extra
competitive matches a season (+32-41% on its 38 league games, 2024/25-2025/26),
and the xG-rate model's form windows (5/10/19 + EWMA halflife 5) never see any
of them. "The last five matches" of a Champions League side spans eight real
ones. This module is the evidence that is currently invisible.

Source: data/external/uefa/raw_<competition>_<year>.csv, the fixture files the
European layer already caches (Round/Date/Home Team/Away Team/Result). Names go
through the same alias table the text-xG builder uses, then canonical_team_name;
without the alias table only ~60% of the big-five clubs resolve.

There is NO xG for these matches: Understat covered the domestic leagues only,
and ESPN commentary has been fetched for the big five alone. `pseudo_xg` scales
the goals onto the xG level the engine's form is measured in (Understat runs
hotter than goals), per league-season, so the arms stay on one scale. That is
the experiment's main weakness and it is deliberate: a cheap first test of
whether the extra evidence helps at all, before paying for real European xG.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

from mundialytics.identity.normalization import canonical_team_name

UEFA_DIR = "data/external/uefa"
ALIASES = "data/curated/fixture_team_aliases.csv"
RESULT_RE = re.compile(r"\s*(\d+)\s*-\s*(\d+)")
FILE_RE = re.compile(r"raw_(.+?)_(\d{4})\.csv$")


def _canonicaliser(root: Path):
    alias: dict[str, str] = {}
    p = root / ALIASES
    if p.exists():
        a = pd.read_csv(p, encoding="utf-8")
        alias = {str(r.fixture_name).strip().lower(): str(r.canonical).strip()
                 for r in a.itertuples()}

    def cn(name) -> str:
        s = str(name).strip()
        return alias.get(s.lower()) or canonical_team_name(s)

    return cn


def load_european_matches(root: str | Path) -> pd.DataFrame:
    """Every played European match on disk: date, competition, teams, goals."""
    root = Path(root)
    cn = _canonicaliser(root)
    rows = []
    for p in sorted(glob.glob(str(root / UEFA_DIR / "raw_*.csv"))):
        m = FILE_RE.search(p)
        if m is None:
            continue
        d = pd.read_csv(p)
        if not {"Date", "Home Team", "Away Team", "Result"} <= set(d.columns):
            continue
        d["date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
        d = d[d["Result"].notna() & d["date"].notna()]
        for _, r in d.iterrows():
            sc = RESULT_RE.match(str(r["Result"]))
            if sc is None:          # "Postponed", a penalty-shootout note, etc.
                continue
            rows.append({"date": r["date"], "competition": m.group(1),
                         "home_team": cn(r["Home Team"]), "away_team": cn(r["Away Team"]),
                         "home_goals": int(sc.group(1)), "away_goals": int(sc.group(2))})
    if not rows:
        return pd.DataFrame(columns=["date", "competition", "home_team", "away_team",
                                     "home_goals", "away_goals"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def pseudo_xg(eu: pd.DataFrame, foundation: pd.DataFrame) -> pd.DataFrame:
    """Add home_xg/away_xg: goals put on the foundation's xG scale.

    One factor for the whole table, measured as sum(xG)/sum(goals) over the
    foundation rows that have xG. Per-season factors were within 1.5% of each
    other, so a single number avoids pretending to a precision we don't have.
    """
    f = foundation.dropna(subset=["home_xg", "away_xg"])
    goals = float(f["home_goals"].sum() + f["away_goals"].sum())
    xg = float(f["home_xg"].sum() + f["away_xg"].sum())
    k = (xg / goals) if goals > 0 else 1.0
    out = eu.copy()
    out["home_xg"] = out["home_goals"] * k
    out["away_xg"] = out["away_goals"] * k
    out.attrs["k"] = k
    return out


def big_five_rows(eu: pd.DataFrame, teams: set[str]) -> pd.DataFrame:
    """Only matches with at least one club the engine knows."""
    return eu[eu["home_team"].isin(teams) | eu["away_team"].isin(teams)].copy()
