from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mundialytics.statistical_core.distributions import probability_for_count_line, poisson_prob_over, poisson_prob_under
from mundialytics.statistical_core.schemas import canonical_name, normalize_odds


@dataclass
class BettingValueConfig:
    min_edge: float = 0.05
    min_ev: float = 0.03
    bankroll: float = 1000.0
    max_stake_fraction: float = 0.02
    kelly_fraction: float = 0.25


class BettingValueEngine:
    """Compare model probabilities with decimal odds in paper mode only."""

    def __init__(self, config: BettingValueConfig | None = None):
        self.config = config or BettingValueConfig()

    def evaluate(
        self,
        odds: pd.DataFrame,
        match_predictions: pd.DataFrame,
        team_stats_predictions: pd.DataFrame | None = None,
        player_event_predictions: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        odds_norm = normalize_odds(odds)
        if odds_norm.empty:
            return _empty_edges_frame()
        priced = _attach_no_vig_probabilities(odds_norm)
        rows: list[dict[str, Any]] = []
        for _, o in priced.iterrows():
            model_prob, source, warnings = self._lookup_model_probability(o, match_predictions, team_stats_predictions, player_event_predictions)
            if model_prob is None or not np.isfinite(model_prob):
                rows.append(_edge_row(o, np.nan, source, np.nan, np.nan, False, 0.0, "no_model_probability", "high", warnings))
                continue
            implied = float(o.get("implied_probability_no_vig", o["implied_probability_raw"]))
            edge = float(model_prob - implied)
            ev = float(model_prob * float(o["odds_decimal"]) - 1.0)
            confidence, risk, conf_warnings = _confidence_and_risk(o, model_prob, source, warnings)
            all_warnings = _clean_warnings(warnings + conf_warnings)
            recommended = bool(
                edge >= self.config.min_edge
                and ev >= self.config.min_ev
                and confidence != "low"
                and not _has_blocking_warning(all_warnings)
            )
            stake = _paper_stake(model_prob, float(o["odds_decimal"]), self.config) if recommended else 0.0
            reason = _reason(o, model_prob, implied, edge, ev, recommended, source, all_warnings)
            rows.append(_edge_row(o, model_prob, source, implied, edge, recommended, stake, confidence, risk, all_warnings, ev, reason))
        return pd.DataFrame(rows)

    def _lookup_model_probability(
        self,
        odds_row: pd.Series,
        match_predictions: pd.DataFrame,
        team_stats_predictions: pd.DataFrame | None,
        player_event_predictions: pd.DataFrame | None,
    ) -> tuple[float | None, str, list[str]]:
        market = str(odds_row["market"]).lower()
        selection = canonical_name(odds_row["selection"])
        match_id = str(odds_row["match_id"])
        line = odds_row.get("line", "")
        warnings: list[str] = []
        mp = match_predictions[match_predictions["match_id"].astype(str) == match_id] if match_predictions is not None and not match_predictions.empty else pd.DataFrame()
        if market in {"1x2", "match_winner", "winner", "result"} and not mp.empty:
            r = mp.iloc[0]
            home = canonical_name(r.get("home_team"))
            away = canonical_name(r.get("away_team"))
            if selection in {"home", "1", home}:
                return float(r["p_home_win"]), "match_model_1x2", warnings
            if selection in {"draw", "x"}:
                return float(r["p_draw"]), "match_model_1x2", warnings
            if selection in {"away", "2", away}:
                return float(r["p_away_win"]), "match_model_1x2", warnings
        if market in {"over_under_goals", "total_goals", "goals_total"} and not mp.empty:
            r = mp.iloc[0]
            total_lambda = float(r["lambda_home"]) + float(r["lambda_away"])
            if selection.startswith("under") or selection == "u":
                return poisson_prob_under(total_lambda, line), "match_model_total_goals", warnings
            return poisson_prob_over(total_lambda, line), "match_model_total_goals", warnings
        if market in {"btts", "both_teams_to_score"} and not mp.empty:
            p = float(mp.iloc[0]["p_btts"])
            if selection in {"yes", "y", "true", "1"}:
                return p, "match_model_btts", warnings
            if selection in {"no", "n", "false", "0"}:
                return 1.0 - p, "match_model_btts", warnings
        p_team = _team_market_probability(odds_row, team_stats_predictions)
        if p_team is not None:
            return p_team, "team_stats_model", warnings
        p_player, player_warnings = _player_market_probability(odds_row, player_event_predictions)
        if p_player is not None:
            return p_player, "player_event_model", player_warnings
        return None, "unmapped_market", ["unmapped_market_or_selection"]


def _attach_no_vig_probabilities(odds: pd.DataFrame) -> pd.DataFrame:
    out = odds.copy()
    out["implied_probability_raw"] = 1.0 / out["odds_decimal"].astype(float)
    group_cols = ["match_id", "market", "line", "bookmaker"]
    sums = out.groupby(group_cols, dropna=False)["implied_probability_raw"].transform("sum")
    # If the group contains several selections, normalize. If it is a single
    # one-way prop price, keep raw implied probability.
    counts = out.groupby(group_cols, dropna=False)["selection"].transform("count")
    out["implied_probability_no_vig"] = np.where(counts > 1, out["implied_probability_raw"] / sums, out["implied_probability_raw"])
    out["overround"] = np.where(counts > 1, sums - 1.0, np.nan)
    return out


def _team_market_probability(odds_row: pd.Series, team_stats: pd.DataFrame | None) -> float | None:
    if team_stats is None or team_stats.empty:
        return None
    market = str(odds_row["market"]).lower()
    mapping = {
        "team_shots": "shots",
        "shots": "shots",
        "team_shots_on_target": "shots_on_target",
        "shots_on_target": "shots_on_target",
        "team_fouls": "fouls",
        "fouls": "fouls",
        "team_yellow_cards": "yellow_cards",
        "yellow_cards": "yellow_cards",
        "team_corners": "corners",
        "corners": "corners",
        "total_shots": "total_shots",
        "total_shots_on_target": "total_shots_on_target",
        "total_fouls": "total_fouls",
        "total_yellow_cards": "total_yellow_cards",
        "total_corners": "total_corners",
    }
    stat_market = mapping.get(market)
    if stat_market is None:
        return None
    selection = canonical_name(odds_row["selection"])
    frame = team_stats[team_stats["match_id"].astype(str) == str(odds_row["match_id"])]
    if stat_market.startswith("total_"):
        frame = frame[frame["market"].astype(str) == stat_market]
    else:
        frame = frame[(frame["market"].astype(str) == stat_market) & (frame["team"].map(canonical_name) == selection)]
    if frame.empty:
        return None
    r = frame.iloc[0]
    if str(r.get("availability", "available")) != "available":
        return None
    lam = float(pd.to_numeric(r.get("expected_count"), errors="coerce"))
    sel = canonical_name(odds_row["selection"])
    # For team markets the selection usually names the team; use line/market as over by default.
    if sel in {"under", "u"}:
        return probability_for_count_line(lam, odds_row.get("line", 0.5), "under")
    return probability_for_count_line(lam, odds_row.get("line", 0.5), "over")


def _player_market_probability(odds_row: pd.Series, player_events: pd.DataFrame | None) -> tuple[float | None, list[str]]:
    if player_events is None or player_events.empty:
        return None, []
    market = str(odds_row["market"]).lower()
    if market not in set(player_events["market"].astype(str).str.lower()):
        return None, []
    selection = canonical_name(odds_row["selection"])
    frame = player_events[
        (player_events["match_id"].astype(str) == str(odds_row["match_id"]))
        & (player_events["market"].astype(str).str.lower() == market)
    ].copy()
    if frame.empty:
        return None, []
    # v0.21 can price odds written with the manual lineup name or the resolved
    # canonical historical name. This avoids missing Álvaro Morata / full-name
    # mismatches while preserving current-lineup candidate gating.
    name_cols = [c for c in ["player", "player_input_name", "canonical_player_name"] if c in frame.columns]
    if not name_cols:
        return None, []
    mask = False
    for c in name_cols:
        mask = mask | (frame[c].map(canonical_name) == selection)
    frame = frame[mask]
    if frame.empty:
        return None, []
    # Prefer exact manual input-name match, then highest sample size.
    frame["_input_match"] = frame.get("player_input_name", frame["player"]).map(canonical_name).eq(selection).astype(int)
    frame["_sample"] = pd.to_numeric(frame.get("sample_size_minutes", 0.0), errors="coerce").fillna(0.0)
    r = frame.sort_values(["_input_match", "_sample"], ascending=False).iloc[0]
    warnings = _clean_warnings(str(r.get("warnings", "")).split(";"))
    sample = float(pd.to_numeric(pd.Series([r.get("sample_size_minutes", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    if sample <= 0:
        warnings.append("sample_size_zero_no_player_pick")
    if str(r.get("identity_status", "matched")) != "matched":
        warnings.append(f"identity_{r.get('identity_status')}")
    if str(r.get("identity_match_level", "")) in {"unresolved", "ambiguous"}:
        warnings.append(f"identity_match_level_{r.get('identity_match_level')}")
    stored_line = str(r.get("line", "1+"))
    requested_line = str(odds_row.get("line", stored_line) or stored_line)
    if requested_line == stored_line or requested_line in {"", "nan"}:
        return float(r.get("safe_probability", r.get("raw_probability"))), _clean_warnings(warnings)
    lam = float(pd.to_numeric(r.get("expected_count"), errors="coerce"))
    return probability_for_count_line(lam, requested_line, "over"), _clean_warnings(warnings)


def _paper_stake(prob: float, odds: float, cfg: BettingValueConfig) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    kelly = (prob * odds - 1.0) / b
    frac = max(0.0, min(cfg.max_stake_fraction, cfg.kelly_fraction * kelly))
    return float(round(cfg.bankroll * frac, 2))


def _confidence_and_risk(odds_row: pd.Series, p: float, source: str, warnings: list[str]) -> tuple[str, str, list[str]]:
    warnings = _clean_warnings(warnings)
    joined = ";".join(warnings)
    if _has_blocking_warning(warnings) or "not_available" in joined or "unmapped" in joined:
        return "low", "high", []
    if "low_sample" in joined or "caution" in joined or source in {"team_stats_model", "player_event_model"}:
        return "medium", "medium", []
    if p < 0.08 or p > 0.92:
        return "medium", "high", ["extreme_probability_review"]
    return "high", "low", []



def _clean_warnings(warnings: list[str] | tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for w in warnings or []:
        text = str(w).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            continue
        if text not in out:
            out.append(text)
    return out


def _has_blocking_warning(warnings: list[str]) -> bool:
    joined = ";".join(_clean_warnings(warnings))
    blocking_terms = [
        "not_available",
        "unmapped",
        "sample_size_zero_no_player_pick",
        "identity_unresolved",
        "identity_ambiguous",
        "identity_match_level_unresolved",
        "identity_match_level_ambiguous",
        "role_guardrail_goalkeeper_attacking_prop_blocked",
    ]
    return any(term in joined for term in blocking_terms)


def _reason(o: pd.Series, p: float, implied: float, edge: float, ev: float, rec: bool, source: str, warnings: list[str]) -> str:
    status = "recommended" if rec else "not_recommended"
    warn = f" Warnings: {';'.join([w for w in warnings if w])}." if any(warnings) else ""
    return f"{status}: model {p:.1%} vs no-vig implied {implied:.1%}; edge {edge:.1%}; EV {ev:.1%}; source={source}.{warn}"


def _edge_row(
    odds_row: pd.Series,
    model_probability: float,
    source: str,
    implied: float,
    edge: float,
    recommended: bool,
    stake: float,
    confidence: str,
    risk: str,
    warnings: list[str],
    ev: float | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    ev = float(model_probability * float(odds_row["odds_decimal"]) - 1.0) if ev is None and np.isfinite(model_probability) else ev
    return {
        "match_id": odds_row["match_id"],
        "market": odds_row["market"],
        "selection": odds_row["selection"],
        "line": odds_row.get("line", ""),
        "bookmaker": odds_row.get("bookmaker", "unknown"),
        "odds_decimal": float(odds_row["odds_decimal"]),
        "model_probability": model_probability,
        "implied_probability": implied,
        "edge": edge,
        "ev": ev,
        "recommended": bool(recommended),
        "stake_virtual": stake,
        "confidence": confidence,
        "risk": risk,
        "probability_source": source,
        "paper_mode": True,
        "warnings": ";".join([w for w in warnings if w]),
        "reason": reason or "",
    }


def _empty_edges_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "match_id",
            "market",
            "selection",
            "line",
            "bookmaker",
            "odds_decimal",
            "model_probability",
            "implied_probability",
            "edge",
            "ev",
            "recommended",
            "stake_virtual",
            "confidence",
            "risk",
            "probability_source",
            "paper_mode",
            "warnings",
            "reason",
        ]
    )
