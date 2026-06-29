from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from mundialytics.identity.normalization import canonical_team_name, normalize_text


def canonical_name(value: object) -> str:
    """Canonical string key used by the v0.21 statistical core.

    It normalizes accents, hyphens and punctuation. For team names, a small
    objective alias table is applied. This fixes manual-input mismatches such as
    "Álvaro"/"alvaro" and "Al-Dawsari"/"al dawsari" without using fuzzy
    matching globally.
    """
    base = normalize_text(value, keep_plus=True)
    return canonical_team_name(base) if base else ""


def stable_match_id(date: object, home_team: object, away_team: object) -> str:
    raw = f"{date}|{canonical_name(home_team)}|{canonical_name(away_team)}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"match_{digest}"


def read_csv_optional(path: str | Path | None) -> pd.DataFrame:
    if path is None or str(path).strip() == "":
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def standardize_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    if fixtures is None or fixtures.empty:
        raise ValueError("fixtures input is empty")
    df = fixtures.copy()
    rename = {}
    if "fixture_id" in df.columns and "match_id" not in df.columns:
        rename["fixture_id"] = "match_id"
    if "home" in df.columns and "home_team" not in df.columns:
        rename["home"] = "home_team"
    if "away" in df.columns and "away_team" not in df.columns:
        rename["away"] = "away_team"
    df = df.rename(columns=rename)
    missing = [c for c in ["home_team", "away_team"] if c not in df.columns]
    if missing:
        raise ValueError(f"fixtures missing required columns: {missing}")
    if "date" not in df.columns:
        df["date"] = "unknown"
    if "match_id" not in df.columns:
        df["match_id"] = [stable_match_id(r.get("date"), r.get("home_team"), r.get("away_team")) for _, r in df.iterrows()]
    for c in ["competition", "stage", "group", "team_scope", "team_type", "competition_context", "gender"]:
        if c not in df.columns:
            df[c] = "unknown"
    if "neutral" not in df.columns:
        df["neutral"] = 1
    df["neutral"] = df["neutral"].fillna(1).astype(int)
    df["home_team"] = df["home_team"].map(canonical_name)
    df["away_team"] = df["away_team"].map(canonical_name)
    df["match_id"] = df["match_id"].astype(str)
    return df


def fixture_team_context(fixtures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    f = standardize_fixtures(fixtures)
    for _, r in f.iterrows():
        base = r.to_dict()
        home = dict(base)
        home.update({"team": r["home_team"], "opponent": r["away_team"], "is_home": 1})
        rows.append(home)
        away = dict(base)
        away.update({"team": r["away_team"], "opponent": r["home_team"], "is_home": 0})
        rows.append(away)
    return pd.DataFrame(rows)


def standardize_current_players(
    lineups: pd.DataFrame | None,
    squads: pd.DataFrame | None,
    fixtures: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Return inference candidates from lineups first, squads second.

    This is the candidate gate: historical players are never candidates unless
    they are present in this current input.
    """
    warnings: list[str] = []
    fx = standardize_fixtures(fixtures)
    source = "lineups" if lineups is not None and not lineups.empty else "squads"
    df = lineups.copy() if source == "lineups" else (squads.copy() if squads is not None else pd.DataFrame())
    if df.empty:
        return pd.DataFrame(), ["no_current_lineups_or_squads_provided"]
    rename = {}
    if "fixture_id" in df.columns and "match_id" not in df.columns:
        rename["fixture_id"] = "match_id"
    if "name" in df.columns and "player" not in df.columns:
        rename["name"] = "player"
    if "minutes" in df.columns and "expected_minutes" not in df.columns:
        rename["minutes"] = "expected_minutes"
        warnings.append("minutes_column_interpreted_as_expected_minutes_for_manual_input")
    df = df.rename(columns=rename)
    for c in ["team", "player"]:
        if c not in df.columns:
            raise ValueError(f"{source} missing required column: {c}")
    df["team"] = df["team"].map(canonical_name)
    df["player"] = df["player"].map(canonical_name)
    if "position" not in df.columns:
        df["position"] = "UNK"
    if "started" not in df.columns:
        df["started"] = 1 if source == "lineups" else 0
    df["started"] = pd.to_numeric(df["started"], errors="coerce").fillna(0).astype(int)
    if "expected_minutes" not in df.columns:
        df["expected_minutes"] = np.where(df["started"] == 1, 75.0, 35.0)
    default_minutes = pd.Series(np.where(df["started"] == 1, 75.0, 35.0), index=df.index)
    df["expected_minutes"] = pd.to_numeric(df["expected_minutes"], errors="coerce").fillna(default_minutes)
    df["expected_minutes"] = df["expected_minutes"].clip(lower=1, upper=130)

    # Drop explicit retired/inactive players if a manual squad marks them.
    if "status" in df.columns:
        retired_mask = df["status"].astype(str).str.lower().isin({"retired", "inactive", "not_current"})
        n = int(retired_mask.sum())
        if n:
            warnings.append(f"dropped_non_current_players={n}")
            df = df.loc[~retired_mask].copy()

    if "match_id" not in df.columns:
        # Expand squad rows to every fixture involving that team.
        rows = []
        for _, p in df.iterrows():
            team_fixtures = fx[(fx["home_team"] == p["team"]) | (fx["away_team"] == p["team"])]
            for _, f in team_fixtures.iterrows():
                row = p.to_dict()
                row["match_id"] = f["match_id"]
                row["date"] = f["date"]
                row["opponent"] = f["away_team"] if f["home_team"] == p["team"] else f["home_team"]
                row["competition"] = f.get("competition", "unknown")
                row["stage"] = f.get("stage", "unknown")
                rows.append(row)
        df = pd.DataFrame(rows)
    else:
        df["match_id"] = df["match_id"].astype(str)
        context = fixture_team_context(fx)[["match_id", "team", "opponent", "date", "competition", "stage", "team_scope", "team_type", "competition_context", "gender"]]
        df = df.merge(context, on=["match_id", "team"], how="left", suffixes=("", "_fixture"))
        for c in ["opponent", "date", "competition", "stage", "team_scope", "team_type", "competition_context", "gender"]:
            fc = f"{c}_fixture"
            if fc in df.columns:
                if c not in df.columns:
                    df[c] = df[fc]
                else:
                    df[c] = df[c].fillna(df[fc])
                df = df.drop(columns=[fc])
    key = ["match_id", "team", "player"]
    dup = int(df.duplicated(key).sum()) if all(c in df.columns for c in key) else 0
    if dup:
        warnings.append(f"duplicate_current_player_candidates_removed={dup}")
        df = df.drop_duplicates(key, keep="first")
    df["candidate_source"] = source
    return df.reset_index(drop=True), warnings


def normalize_odds(odds: pd.DataFrame) -> pd.DataFrame:
    if odds is None or odds.empty:
        return pd.DataFrame(columns=["match_id", "market", "selection", "line", "odds_decimal", "bookmaker"])
    df = odds.copy()
    rename = {}
    if "fixture_id" in df.columns and "match_id" not in df.columns:
        rename["fixture_id"] = "match_id"
    if "market_type" in df.columns and "market" not in df.columns:
        rename["market_type"] = "market"
    if "odds" in df.columns and "odds_decimal" not in df.columns:
        rename["odds"] = "odds_decimal"
    df = df.rename(columns=rename)
    for c in ["match_id", "market", "selection", "odds_decimal"]:
        if c not in df.columns:
            raise ValueError(f"odds missing required column: {c}")
    if "line" not in df.columns:
        df["line"] = ""
    if "bookmaker" not in df.columns:
        df["bookmaker"] = "unknown"
    df["match_id"] = df["match_id"].astype(str)
    df["market"] = df["market"].astype(str).str.strip().str.lower()
    df["selection"] = df["selection"].astype(str).str.strip().str.lower()
    df["odds_decimal"] = pd.to_numeric(df["odds_decimal"], errors="coerce")
    df = df.dropna(subset=["odds_decimal"])
    df = df[df["odds_decimal"] > 1.0].copy()
    return df[["match_id", "market", "selection", "line", "odds_decimal", "bookmaker"]].reset_index(drop=True)


def assert_probability_columns(frame: pd.DataFrame, cols: Iterable[str], tolerance: float = 1e-6) -> list[str]:
    warnings: list[str] = []
    for c in cols:
        if c in frame.columns:
            bad = int(((frame[c] < -tolerance) | (frame[c] > 1 + tolerance) | frame[c].isna()).sum())
            if bad:
                warnings.append(f"probability_column_{c}_bad_rows={bad}")
    return warnings


def write_json(path: str | Path, data: dict) -> None:
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
