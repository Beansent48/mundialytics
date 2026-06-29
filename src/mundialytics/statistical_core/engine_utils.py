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
    min_year: int = 2010,
    competitions: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load and filter international match results."""
    p = Path(path) if path else ROOT / "data/processed/national_matches.csv"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals", "date"])
    df = df[df["date"].dt.year >= min_year].copy()

    comps = set(competitions) if competitions else ELITE_COMPS
    # Allow friendlies after 2022 as signal for latest strength
    major = df[df["competition"].isin(comps)].copy()
    recent_friendly = df[
        (df["competition"] == "Friendly") & (df["date"].dt.year >= 2022)
    ].copy()
    combined = pd.concat([major, recent_friendly], ignore_index=True)
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
    """Generate an HTML bracket tree from simulation results.

    Args:
        sim_df : team_stats DataFrame with p_win, p_final, p_semis, p_quarters columns
        n_groups : number of groups (determines bracket depth)
    """
    top = sim_df.head(16).copy()
    top["team"] = top["team"].str.title()

    def pct(val: float) -> str:
        return f"{val:.0%}"

    def color(val: float) -> str:
        if val >= 0.25: return "#16a34a"
        if val >= 0.10: return "#2563eb"
        if val >= 0.05: return "#9333ea"
        return "#6b7280"

    # Build simple bracket table HTML
    rows_html = ""
    for _, r in top.iterrows():
        p_w = float(r.get("p_win", 0))
        p_f = float(r.get("p_final", 0))
        p_s = float(r.get("p_semis", 0))
        p_q = float(r.get("p_quarters", 0))
        p_g = float(r.get("p_advance_groups", 0))
        c = color(p_w)
        bar_w = int(p_w * 200)
        rows_html += f"""
        <tr>
          <td style="padding:6px 10px;font-weight:600;color:{c}">{r['team']}</td>
          <td style="padding:6px;text-align:center;color:#4b5563">{pct(p_g)}</td>
          <td style="padding:6px;text-align:center;color:#4b5563">{pct(p_q)}</td>
          <td style="padding:6px;text-align:center;color:#4b5563">{pct(p_s)}</td>
          <td style="padding:6px;text-align:center;color:#4b5563">{pct(p_f)}</td>
          <td style="padding:6px;text-align:center;font-weight:700;color:{c}">{pct(p_w)}</td>
          <td style="padding:6px 10px">
            <div style="width:{bar_w}px;height:14px;background:{c};border-radius:3px;max-width:200px"></div>
          </td>
        </tr>"""

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem">
      <thead>
        <tr style="border-bottom:2px solid #e5e7eb">
          <th style="padding:6px 10px;text-align:left">Equipo</th>
          <th style="padding:6px;text-align:center;color:#6b7280;font-size:0.8rem">Grupos</th>
          <th style="padding:6px;text-align:center;color:#6b7280;font-size:0.8rem">QF</th>
          <th style="padding:6px;text-align:center;color:#6b7280;font-size:0.8rem">SF</th>
          <th style="padding:6px;text-align:center;color:#6b7280;font-size:0.8rem">Final</th>
          <th style="padding:6px;text-align:center;color:#f59e0b;font-weight:700">🏆 Campeón</th>
          <th></th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>"""
