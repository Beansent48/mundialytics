"""
Utility functions for PredictionEngine:
  - Data loading (clubs + international)
  - H2H history
  - Recent form
  - Bracket visualization helpers
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

# Competitions treated as "serious" for international model training
ELITE_COMPS = {
    "FIFA World Cup",
    "UEFA European Championship",
    "Copa América",
    "Africa Cup of Nations",
    "AFC Asian Cup",
    "CONCACAF Gold Cup",
    "UEFA Nations League",
    "CONMEBOL-UEFA Cup of Champions",
    "FIFA World Cup qualification",
    "UEFA Euro qualification",
}

# Map common competition names to display names
COMP_DISPLAY = {
    "FIFA World Cup": "FIFA World Cup",
    "UEFA European Championship": "UEFA Euro",
    "Copa América": "Copa América",
    "Africa Cup of Nations": "AFCON",
    "UEFA Nations League": "UEFA Nations League",
}


def load_clubs_data(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else ROOT / "data/processed/foundation_big5_multi_season.csv"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals"])
    df["team_scope"] = "club"
    return df


def load_international_data(
    path: str | Path | None = None,
    min_year: int = 2006,
    competitions: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load and filter international match results.

    Uses 2006+ by default so the dataset covers multiple World Cup cycles and
    avoids over-indexing on any single dominant team. Friendlies are excluded
    (too noisy); only competitive matches from ELITE_COMPS are kept.
    """
    p = Path(path) if path else ROOT / "data/processed/national_matches.csv"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals", "date"])
    df = df[df["date"].dt.year >= min_year].copy()

    comps = set(competitions) if competitions else ELITE_COMPS
    combined = df[df["competition"].isin(comps)].copy()
    combined["team_scope"] = "national"
    return combined.drop_duplicates(subset=["home_team", "away_team", "date"]).reset_index(drop=True)


def get_h2h(
    team1: str,
    team2: str,
    df: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Return last N head-to-head matches between two teams (either direction)."""
    mask = (
        ((df["home_team"] == team1) & (df["away_team"] == team2))
        | ((df["home_team"] == team2) & (df["away_team"] == team1))
    )
    h2h = df[mask].copy()
    if "date" in h2h.columns:
        h2h = h2h.sort_values("date", ascending=False)
    h2h = h2h.head(n).copy()

    def result_for(row: pd.Series, team: str) -> str:
        is_home = row["home_team"] == team
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        goals_for = hg if is_home else ag
        goals_against = ag if is_home else hg
        if goals_for > goals_against:
            return "W"
        if goals_for < goals_against:
            return "L"
        return "D"

    h2h["result_t1"] = h2h.apply(lambda r: result_for(r, team1), axis=1)
    h2h["score"] = h2h["home_goals"].astype(int).astype(str) + "-" + h2h["away_goals"].astype(int).astype(str)
    return h2h[["date", "home_team", "away_team", "score", "competition", "result_t1"]].reset_index(drop=True)


def get_form(
    team: str,
    df: pd.DataFrame,
    n: int = 5,
) -> list[dict]:
    """Return last N results for a team."""
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    recent = df[mask].copy()
    if "date" in recent.columns:
        recent = recent.sort_values("date", ascending=False)
    recent = recent.head(n)
    rows = []
    for _, r in recent.iterrows():
        is_home = r["home_team"] == team
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        gf = hg if is_home else ag
        ga = ag if is_home else hg
        opp = r["away_team"] if is_home else r["home_team"]
        res = "W" if gf > ga else ("D" if gf == ga else "L")
        rows.append({
            "date": str(r.get("date", ""))[:10],
            "opponent": opp,
            "score": f"{gf}-{ga}",
            "result": res,
            "home_away": "H" if is_home else "A",
            "competition": str(r.get("competition", "")),
        })
    return rows


def h2h_summary(team1: str, team2: str, df: pd.DataFrame) -> dict:
    """W/D/L totals for team1 vs team2 in all history."""
    h2h = get_h2h(team1, team2, df, n=1000)
    if h2h.empty:
        return {"w": 0, "d": 0, "l": 0, "total": 0}
    counts = h2h["result_t1"].value_counts()
    return {
        "w": int(counts.get("W", 0)),
        "d": int(counts.get("D", 0)),
        "l": int(counts.get("L", 0)),
        "total": len(h2h),
    }


def bracket_html(
    sim_df: pd.DataFrame,
    n_groups: int,
) -> str:
    """Generate a full HTML bracket showing table + odds bar per stage."""
    top = sim_df.head(24).copy()
    top["team"] = top["team"].str.title()

    def pct(val: float) -> str:
        return f"{val:.0%}"

    def color(val: float) -> str:
        if val >= 0.25: return "#16a34a"
        if val >= 0.10: return "#2563eb"
        if val >= 0.05: return "#9333ea"
        return "#6b7280"

    # Determine which round columns exist
    has_r16      = "p_r16"      in top.columns and top["p_r16"].sum() > 0
    has_quarters = "p_quarters" in top.columns and top["p_quarters"].sum() > 0
    has_semis    = "p_semis"    in top.columns and top["p_semis"].sum() > 0
    has_final    = "p_final"    in top.columns and top["p_final"].sum() > 0

    stage_cols = []
    if "p_advance_groups" in top.columns: stage_cols.append(("Grupos",   "p_advance_groups", "#10b981"))
    if has_r16:      stage_cols.append(("R16",       "p_r16",            "#3b82f6"))
    if has_quarters: stage_cols.append(("QF",        "p_quarters",       "#8b5cf6"))
    if has_semis:    stage_cols.append(("SF",        "p_semis",          "#f59e0b"))
    if has_final:    stage_cols.append(("Final",     "p_final",          "#ef4444"))
    stage_cols.append(("🏆 Campeón", "p_win", "#16a34a"))

    header_cells = "".join(
        f'<th style="padding:6px 8px;text-align:center;color:#6b7280;font-size:11px;'
        f'font-weight:500;border-bottom:0.5px solid #e5e7eb">{s[0]}</th>'
        for s in stage_cols
    )

    rows_html = ""
    for _, r in top.iterrows():
        p_win = float(r.get("p_win", 0))
        c = color(p_win)
        cells = f'<td style="padding:6px 10px;font-weight:500;color:{c}">{r["team"]}</td>'
        for _, col, bar_color in stage_cols:
            val = float(r.get(col, 0))
            bar = int(val * 120)
            cells += (
                f'<td style="padding:4px 8px">'
                f'<div style="display:flex;align-items:center;gap:6px">'
                f'<div style="width:{bar}px;max-width:120px;height:10px;background:{bar_color};'
                f'border-radius:2px;opacity:{max(0.3, val*2):.1f}"></div>'
                f'<span style="font-size:11px;color:#4b5563;min-width:34px">{pct(val)}</span>'
                f'</div></td>'
            )
        rows_html += f"<tr>{cells}</tr>"

    return (
        f'<table style="width:100%;border-collapse:collapse;font-size:12px">'
        f'<thead><tr>'
        f'<th style="padding:6px 10px;text-align:left;border-bottom:0.5px solid #e5e7eb">Equipo</th>'
        f'{header_cells}'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
    )

