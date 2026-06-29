from __future__ import annotations

from pathlib import Path

import pandas as pd

from mundialytics.data.schema import normalize_matches

_FD_MAP = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "Div": "competition",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_sot",
    "AST": "away_sot",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellow_cards",
    "AY": "away_yellow_cards",
}

_DIV_LABELS = {
    "E0": "Premier League",
    "E1": "Championship",
    "SP1": "LaLiga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
    "P1": "Primeira Liga",
}


def football_data_uk_to_matches(path: str | Path, season: str | None = None) -> pd.DataFrame:
    """Convert Football-Data.co.uk match CSV into the canonical schema.

    Works with the standard league CSV format: Date, HomeTeam, AwayTeam,
    FTHG, FTAG, and optional match-stat columns such as HS/HST/HC/HF/HY.
    """
    raw = pd.read_csv(path)
    missing = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"} - set(raw.columns)
    if missing:
        raise ValueError(f"Football-Data CSV missing columns: {sorted(missing)}")
    keep = [c for c in _FD_MAP if c in raw.columns]
    out = raw[keep].rename(columns=_FD_MAP).copy()
    # Football-Data.co.uk dates are day-first in common league CSVs
    # (e.g. 15/08/25). Parse here before schema normalization to avoid
    # month/day ambiguity in pandas/dateutil.
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")
    out["match_id"] = [f"fduk_{Path(path).stem}_{i:05d}" for i in range(len(out))]
    out["neutral"] = 0
    out["team_scope"] = "club"
    out["source"] = "football-data.co.uk"
    out["stage"] = "Regular Season"
    if "competition" in out.columns:
        out["competition"] = out["competition"].map(lambda x: _DIV_LABELS.get(str(x), str(x)))
    else:
        out["competition"] = "unknown_club_league"
    out["season"] = season or _infer_season_from_path(path)
    out = out.dropna(subset=["home_goals", "away_goals"]).copy()
    return normalize_matches(out)


def _infer_season_from_path(path: str | Path) -> str:
    text = str(path)
    # Football-data URLs often contain mmz4281/2526/E0.csv.
    import re

    m = re.search(r"/(\d{4})/[^/]+$", text.replace("\\", "/"))
    if m:
        yy = m.group(1)
        return f"20{yy[:2]}-20{yy[2:]}"
    return "unknown"


_ODDS_MAP_1X2 = {
    "B365": ("B365H", "B365D", "B365A"),
    "BW": ("BWH", "BWD", "BWA"),
    "IW": ("IWH", "IWD", "IWA"),
    "PS": ("PSH", "PSD", "PSA"),
    "WH": ("WHH", "WHD", "WHA"),
    "VC": ("VCH", "VCD", "VCA"),
    "Max": ("MaxH", "MaxD", "MaxA"),
    "Avg": ("AvgH", "AvgD", "AvgA"),
}


def football_data_uk_to_match_odds(path: str | Path) -> pd.DataFrame:
    """Extract historical 1X2 decimal odds from Football-Data.co.uk CSVs.

    Output schema is compatible with ``reports.match_value``. Match ids match
    ``football_data_uk_to_matches`` for the same CSV, so odds can be joined to
    backtest/prediction rows without relying on fuzzy team/date matching.
    """
    raw = pd.read_csv(path)
    required = {"Date", "HomeTeam", "AwayTeam"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Football-Data CSV missing odds context columns: {sorted(missing)}")
    rows = []
    for i, r in raw.iterrows():
        match_id = f"fduk_{Path(path).stem}_{i:05d}"
        for bookmaker, cols in _ODDS_MAP_1X2.items():
            if not all(c in raw.columns for c in cols):
                continue
            h, d, a = (r.get(c) for c in cols)
            for selection, odds in [("home", h), ("draw", d), ("away", a)]:
                if pd.isna(odds):
                    continue
                try:
                    odds = float(odds)
                except (TypeError, ValueError):
                    continue
                if odds <= 1:
                    continue
                rows.append({
                    "match_id": match_id,
                    "date": pd.to_datetime(r["Date"], dayfirst=True, errors="coerce"),
                    "home_team": r["HomeTeam"],
                    "away_team": r["AwayTeam"],
                    "bookmaker": bookmaker,
                    "market_type": "match_winner",
                    "selection": selection,
                    "odds": odds,
                    "source": "football-data.co.uk",
                })
    return pd.DataFrame(rows)
