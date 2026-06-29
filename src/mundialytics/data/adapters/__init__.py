from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

def _season_from_path(path: str | Path) -> str:
    name = Path(path).stem
    prefix = name.split("_")[0]
    if len(prefix) == 4 and prefix.isdigit():
        return f"20{prefix[:2]}-20{prefix[2:]}"
    return "unknown"

_COMPETITION_MAP = {
    "E0": "Premier League",
    "SP1": "LaLiga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}

def football_data_uk_to_matches(path: str | Path, season: str | None = None) -> pd.DataFrame:
    raw = pd.read_csv(path)
    code = Path(path).stem.split("_")[-1]
    comp = _COMPETITION_MAP.get(code, code)
    out = pd.DataFrame()
    out["match_id"] = [f"fduk_{Path(path).stem}_{i:05d}" for i in range(len(raw))]
    out["date"] = pd.to_datetime(raw.get("Date"), dayfirst=True, errors="coerce")
    out["competition"] = comp
    out["season"] = season or _season_from_path(path)
    out["stage"] = "league"
    out["team_scope"] = "club"
    out["home_team"] = raw.get("HomeTeam")
    out["away_team"] = raw.get("AwayTeam")
    out["home_goals"] = pd.to_numeric(raw.get("FTHG"), errors="coerce")
    out["away_goals"] = pd.to_numeric(raw.get("FTAG"), errors="coerce")
    mappings = {
        "home_shots": "HS", "away_shots": "AS",
        "home_sot": "HST", "away_sot": "AST",
        "home_corners": "HC", "away_corners": "AC",
        "home_fouls": "HF", "away_fouls": "AF",
        "home_yellow_cards": "HY", "away_yellow_cards": "AY",
        "home_red_cards": "HR", "away_red_cards": "AR",
    }
    for out_col, raw_col in mappings.items():
        if raw_col in raw.columns:
            out[out_col] = pd.to_numeric(raw[raw_col], errors="coerce")
    out["neutral"] = 0
    for col in ["home_team", "away_team"]:
        out[col] = out[col].astype(str).str.strip().str.lower()
    return out

def international_results_to_matches(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    out = pd.DataFrame()
    out["match_id"] = raw.get("match_id", pd.Series([f"intl_{i:05d}" for i in range(len(raw))]))
    out["date"] = pd.to_datetime(raw.get("date"), errors="coerce")
    out["competition"] = raw.get("tournament", raw.get("competition", "international"))
    out["season"] = out["date"].dt.year.astype("Int64").astype(str)
    out["stage"] = raw.get("stage", "unknown")
    out["team_scope"] = "national"
    out["home_team"] = raw.get("home_team")
    out["away_team"] = raw.get("away_team")
    out["home_goals"] = pd.to_numeric(raw.get("home_score", raw.get("home_goals")), errors="coerce")
    out["away_goals"] = pd.to_numeric(raw.get("away_score", raw.get("away_goals")), errors="coerce")
    out["neutral"] = pd.to_numeric(raw.get("neutral", 0), errors="coerce").fillna(0).astype(int)
    return out

def openfootball_json_to_matches(path: str | Path, competition: str = "openfootball", season: str = "unknown") -> pd.DataFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("matches", [])
    out_rows = []
    for i, r in enumerate(rows):
        out_rows.append({
            "match_id": r.get("match_id", f"openfootball_{i:05d}"),
            "date": r.get("date"),
            "competition": competition,
            "season": season,
            "stage": r.get("stage", "unknown"),
            "team_scope": "club",
            "home_team": r.get("home_team") or r.get("team1"),
            "away_team": r.get("away_team") or r.get("team2"),
            "home_goals": r.get("home_goals") or r.get("score1"),
            "away_goals": r.get("away_goals") or r.get("score2"),
            "neutral": r.get("neutral", 0),
        })
    df = pd.DataFrame(out_rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df
