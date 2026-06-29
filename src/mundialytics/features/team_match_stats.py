from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import poisson

from mundialytics.data.competition_taxonomy import enrich_competition_metadata
from mundialytics.identity.normalization import canonical_team_name

TEAM_PROP_MARKETS = [
    "shots_for",
    "sot_for",
    "corners_for",
    "fouls_for",
    "yellow_cards_for",
]

MATCH_TOTAL_MARKETS = {
    "total_shots": "shots_for",
    "total_sot": "sot_for",
    "total_corners": "corners_for",
    "total_fouls": "fouls_for",
    "total_yellow_cards": "yellow_cards_for",
}


def _safe_sum(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0).sum())


def build_team_match_stats_from_player_events(player_events: pd.DataFrame) -> pd.DataFrame:
    """Build one row per team-match from player-event rows.

    Corners are only emitted when the input contains a `corners` column. This is
    intentional: the project should never invent corners when the source did not
    provide them.
    """
    pe = enrich_competition_metadata(player_events.copy(), overwrite=False)
    required = {"match_id", "team", "opponent"}
    missing = required - set(pe.columns)
    if missing:
        raise ValueError(f"player events missing required columns for team stats: {sorted(missing)}")
    if "date" in pe.columns:
        pe["date"] = pd.to_datetime(pe["date"], errors="coerce")
    value_cols = {
        "shots_for": "shots",
        "sot_for": "shots_on_target",
        "fouls_for": "fouls_committed",
        "yellow_cards_for": "yellow_cards",
        "red_cards_for": "red_cards",
        "goals_for": "goals",
        "corners_for": "corners",
    }
    rows: list[dict] = []
    group_cols = ["match_id", "team", "opponent"]
    for (match_id, team, opponent), g in pe.groupby(group_cols, dropna=False):
        row = {
            "match_id": match_id,
            "date": g["date"].dropna().min() if "date" in g.columns else None,
            "competition": g["competition"].dropna().iloc[0] if "competition" in g.columns and g["competition"].notna().any() else "unknown",
            "team_scope": g["team_scope"].dropna().iloc[0] if "team_scope" in g.columns and g["team_scope"].notna().any() else "unknown",
            "team_type": g["team_type"].dropna().iloc[0] if "team_type" in g.columns and g["team_type"].notna().any() else None,
            "competition_context": g["competition_context"].dropna().iloc[0] if "competition_context" in g.columns and g["competition_context"].notna().any() else None,
            "gender": g["gender"].dropna().iloc[0] if "gender" in g.columns and g["gender"].notna().any() else None,
            "team": canonical_team_name(team),
            "opponent": canonical_team_name(opponent),
        }
        for out_col, in_col in value_cols.items():
            if in_col in g.columns:
                row[out_col] = _safe_sum(g[in_col])
        rows.append(row)
    df = pd.DataFrame(rows)
    return add_opponent_against_columns(enrich_competition_metadata(df, overwrite=False)) if not df.empty else df


def build_team_match_stats_from_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Build team-match rows from wide match rows with home_/away_ totals."""
    rows = []
    m = enrich_competition_metadata(matches.copy(), overwrite=False)
    for _, r in m.iterrows():
        common = {
            "match_id": r.get("match_id"),
            "date": r.get("date"),
            "competition": r.get("competition", "unknown"),
            "team_scope": r.get("team_scope", "unknown"),
            "team_type": r.get("team_type"),
            "competition_context": r.get("competition_context"),
            "gender": r.get("gender"),
        }
        home = canonical_team_name(r.get("home_team"))
        away = canonical_team_name(r.get("away_team"))
        mappings = [
            (home, away, "home", "away"),
            (away, home, "away", "home"),
        ]
        for team, opponent, prefix, opp_prefix in mappings:
            row = {**common, "team": team, "opponent": opponent}
            for out, suffix in [
                ("shots_for", "shots"),
                ("sot_for", "sot"),
                ("corners_for", "corners"),
                ("fouls_for", "fouls"),
                ("yellow_cards_for", "yellow_cards"),
                ("goals_for", "goals"),
            ]:
                row[out] = r.get(f"{prefix}_{suffix}")
                row[out.replace("_for", "_against")] = r.get(f"{opp_prefix}_{suffix}")
            rows.append(row)
    df = pd.DataFrame(rows)
    return enrich_competition_metadata(df, overwrite=False) if not df.empty else df


def add_opponent_against_columns(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Attach against columns by joining the opponent row from the same match."""
    df = team_stats.copy()
    metrics = ["shots_for", "sot_for", "corners_for", "fouls_for", "yellow_cards_for", "red_cards_for", "goals_for"]
    opp_cols = [c for c in metrics if c in df.columns]
    if not opp_cols or df.empty:
        return df
    opp = df[["match_id", "team", *opp_cols]].copy().rename(columns={"team": "opponent", **{c: c.replace("_for", "_against") for c in opp_cols}})
    return df.merge(opp, on=["match_id", "opponent"], how="left")


def add_match_totals(team_stats: pd.DataFrame) -> pd.DataFrame:
    df = team_stats.copy()
    for total_col, for_col in MATCH_TOTAL_MARKETS.items():
        against_col = for_col.replace("_for", "_against")
        if for_col in df.columns and against_col in df.columns:
            df[total_col] = pd.to_numeric(df[for_col], errors="coerce") + pd.to_numeric(df[against_col], errors="coerce")
    return df


@dataclass
class TeamPropPredictionConfig:
    recent_window: int = 5
    min_team_matches: int = 3
    lambda_floor: float = 0.01
    lambda_cap: float = 40.0


def _clip(x: float, floor: float, cap: float) -> float:
    if x is None or pd.isna(x) or not math.isfinite(float(x)):
        return float("nan")
    return float(max(floor, min(cap, float(x))))


def _recent_mean(hist: pd.DataFrame, col: str, window: int) -> float | None:
    if col not in hist.columns or hist.empty:
        return None
    vals = pd.to_numeric(hist.sort_values(["date", "match_id"])[col], errors="coerce").dropna().tail(window)
    if vals.empty:
        return None
    return float(vals.mean())


def predict_team_props_simple(team_stats: pd.DataFrame, fixtures: pd.DataFrame, *, markets: Iterable[str] | None = None, config: TeamPropPredictionConfig | None = None) -> pd.DataFrame:
    """Predict team/match count props using leakage-safe recent averages.

    This is deliberately a conservative MVP model. It does not claim to be the
    final modelling layer; it gives auditable expectations until a calibrated
    Negative Binomial model is trained.
    """
    cfg = config or TeamPropPredictionConfig()
    markets = list(markets or TEAM_PROP_MARKETS)
    hist = enrich_competition_metadata(add_match_totals(team_stats.copy()), overwrite=False)
    if "date" in hist.columns:
        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    fx = enrich_competition_metadata(fixtures.copy(), overwrite=False)
    rows: list[dict] = []
    global_means = {m: float(pd.to_numeric(hist.get(m), errors="coerce").mean()) if m in hist.columns else float("nan") for m in markets}
    for _, f in fx.iterrows():
        teams = [(canonical_team_name(f.get("home_team")), canonical_team_name(f.get("away_team")), "home"), (canonical_team_name(f.get("away_team")), canonical_team_name(f.get("home_team")), "away")]
        fixture_date = pd.to_datetime(f.get("date"), errors="coerce")
        eligible = hist.copy()
        if pd.notna(fixture_date) and "date" in eligible.columns:
            eligible = eligible[pd.to_datetime(eligible["date"], errors="coerce") < fixture_date]
        per_team_expected: dict[tuple[str, str], dict[str, float]] = {}
        for team, opponent, side in teams:
            team_hist = eligible[eligible["team"].astype(str).eq(str(team))]
            opp_hist = eligible[eligible["team"].astype(str).eq(str(opponent))]
            base = {
                "fixture_id": f.get("fixture_id") or f.get("match_id"),
                "match_id": f.get("match_id") or f.get("fixture_id"),
                "date": f.get("date"),
                "competition": f.get("competition", "unknown"),
                "team_scope": f.get("team_scope", "unknown"),
                "team_type": f.get("team_type"),
                "competition_context": f.get("competition_context"),
                "gender": f.get("gender"),
                "team": team,
                "opponent": opponent,
                "side": side,
                "team_recent_matches": int(len(team_hist)),
                "opponent_recent_matches": int(len(opp_hist)),
            }
            expected_by_market: dict[str, float] = {}
            for market in markets:
                if market not in hist.columns:
                    base[f"expected_{market}"] = np.nan
                    base[f"{market}_confidence_flag"] = "unavailable"
                    base[f"{market}_warning"] = f"market_unavailable_in_training_data={market}"
                    continue
                against = market.replace("_for", "_against")
                team_for = _recent_mean(team_hist, market, cfg.recent_window)
                opp_against = _recent_mean(opp_hist, against, cfg.recent_window) if against in hist.columns else None
                global_mean = global_means.get(market)
                pieces = []
                if team_for is not None:
                    pieces.append((0.55, team_for))
                if opp_against is not None:
                    pieces.append((0.35, opp_against))
                if global_mean is not None and not pd.isna(global_mean):
                    pieces.append((0.10, global_mean))
                if not pieces:
                    lam = float("nan")
                else:
                    total_w = sum(w for w, _ in pieces)
                    lam = sum(w * v for w, v in pieces) / total_w
                lam = _clip(lam, cfg.lambda_floor, cfg.lambda_cap) if not pd.isna(lam) else lam
                base[f"expected_{market}"] = lam
                expected_by_market[market] = lam
                if len(team_hist) < cfg.min_team_matches or len(opp_hist) < cfg.min_team_matches:
                    base[f"{market}_confidence_flag"] = "low_sample"
                    base[f"{market}_warning"] = "low_team_or_opponent_history"
                else:
                    base[f"{market}_confidence_flag"] = "normal"
                    base[f"{market}_warning"] = ""
            per_team_expected[(team, opponent)] = expected_by_market
            rows.append(base)
        # match totals on both rows, using expected values from both teams.
        for team, opponent, _side in teams:
            pass
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Attach match totals by adding paired rows.
    for market in markets:
        exp_col = f"expected_{market}"
        if exp_col not in out.columns:
            continue
        opp = out[["match_id", "team", exp_col]].rename(columns={"team": "opponent", exp_col: f"opponent_{exp_col}"})
        out = out.merge(opp, on=["match_id", "opponent"], how="left")
        out[f"expected_match_total_{market}"] = pd.to_numeric(out[exp_col], errors="coerce") + pd.to_numeric(out[f"opponent_{exp_col}"], errors="coerce")
    return enrich_competition_metadata(out, overwrite=False)


def poisson_over_probability(expected_count: float, line: float) -> float:
    if expected_count is None or pd.isna(expected_count):
        return float("nan")
    # Betting line 8.5 => P(X > 8.5) = P(X >= 9).
    threshold = math.floor(float(line))
    return float(1 - poisson.cdf(threshold, float(expected_count)))
