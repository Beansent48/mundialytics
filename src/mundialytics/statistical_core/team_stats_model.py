from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mundialytics.statistical_core.calibration import recency_weights, safe_count, weighted_mean
from mundialytics.statistical_core.schemas import canonical_name, fixture_team_context, standardize_fixtures


EVENT_ALIASES: dict[str, list[str]] = {
    "shots": ["shots", "shots_for"],
    "shots_on_target": ["shots_on_target", "sot", "sot_for"],
    "fouls": ["fouls", "fouls_committed", "fouls_for"],
    "yellow_cards": ["yellow_cards", "yellow_card", "yellow_cards_for"],
    "corners": ["corners", "corners_for"],
}

EVENT_DEFAULTS = {
    "shots": 10.0,
    "shots_on_target": 3.5,
    "fouls": 11.0,
    "yellow_cards": 1.8,
    "corners": 4.5,
}


@dataclass
class TeamStatProfile:
    team: str
    matches: int
    means_for: dict[str, float]
    means_against: dict[str, float]
    availability: dict[str, bool]
    warnings: str = ""


class TeamStatsModel:
    """Transparent team count model using own-for and opponent-against rates.

    v0.24 exposes a small, auditable configuration surface so the event model
    can be evaluated and hardened per market instead of being a fixed heuristic.
    """

    def __init__(
        self,
        own_weight: float = 0.55,
        recency_half_life_days: float = 365.0,
        profile_shrinkage_k: float = 0.0,
        low_sample_blend_k: float = 0.0,
    ):
        self.own_weight = float(own_weight)
        self.recency_half_life_days = float(recency_half_life_days)
        self.profile_shrinkage_k = float(profile_shrinkage_k)
        self.low_sample_blend_k = float(low_sample_blend_k)
        self.profiles: dict[str, TeamStatProfile] = {}
        self.global_means: dict[str, float] = EVENT_DEFAULTS.copy()
        self.availability: dict[str, bool] = {k: False for k in EVENT_ALIASES}
        self.audit: dict[str, Any] = {}

    @property
    def model_config(self) -> dict[str, float]:
        return {
            "own_weight": self.own_weight,
            "recency_half_life_days": self.recency_half_life_days,
            "profile_shrinkage_k": self.profile_shrinkage_k,
            "low_sample_blend_k": self.low_sample_blend_k,
        }

    def fit(self, historical_events: pd.DataFrame | None) -> "TeamStatsModel":
        team_match = build_team_match_stat_frame(historical_events)
        if team_match.empty:
            self.audit = {"team_stats_fit": "fallback_no_historical_events", "available_markets": []}
            self.availability = {k: False for k in EVENT_ALIASES}
            return self
        for event in EVENT_ALIASES:
            self.availability[event] = event in team_match.columns and pd.to_numeric(team_match[event], errors="coerce").notna().any()
            if self.availability[event]:
                self.global_means[event] = safe_count(pd.to_numeric(team_match[event], errors="coerce").mean(), event, EVENT_DEFAULTS[event])
        profiles: dict[str, TeamStatProfile] = {}
        for team, g in team_match.groupby("team"):
            weights = recency_weights(g["date"], half_life_days=self.recency_half_life_days) if "date" in g.columns else None
            sample_weight = 1.0
            if self.profile_shrinkage_k > 0:
                sample_weight = float(len(g) / (len(g) + self.profile_shrinkage_k))
            means_for: dict[str, float] = {}
            means_against: dict[str, float] = {}
            available: dict[str, bool] = {}
            for event in EVENT_ALIASES:
                available[event] = bool(self.availability.get(event, False))
                if not available[event]:
                    continue
                raw_for = weighted_mean(g[event], weights, self.global_means[event])
                means_for[event] = sample_weight * raw_for + (1.0 - sample_weight) * self.global_means[event]
                against_col = f"{event}_against"
                raw_against = weighted_mean(g[against_col], weights, self.global_means[event]) if against_col in g.columns else self.global_means[event]
                means_against[event] = sample_weight * raw_against + (1.0 - sample_weight) * self.global_means[event]
            profiles[str(team)] = TeamStatProfile(
                team=str(team),
                matches=int(len(g)),
                means_for=means_for,
                means_against=means_against,
                availability=available,
                warnings="" if len(g) >= 5 else "low_team_stat_sample",
            )
        self.profiles = profiles
        self.audit = {
            "team_stats_fit": "historical_event_team_counts",
            "teams": len(profiles),
            "available_markets": [k for k, v in self.availability.items() if v],
            "not_available_markets": [k for k, v in self.availability.items() if not v],
            "model_config": self.model_config,
        }
        return self

    def profile_for(self, team: str) -> TeamStatProfile:
        key = canonical_name(team)
        if key in self.profiles:
            return self.profiles[key]
        return TeamStatProfile(
            team=key,
            matches=0,
            means_for=self.global_means.copy(),
            means_against=self.global_means.copy(),
            availability=self.availability.copy(),
            warnings="fallback_team_stat_profile_no_history",
        )

    def predict_fixtures(self, fixtures: pd.DataFrame, match_predictions: pd.DataFrame | None = None) -> pd.DataFrame:
        contexts = fixture_team_context(standardize_fixtures(fixtures))
        rows: list[dict[str, Any]] = []
        for _, r in contexts.iterrows():
            team = self.profile_for(r["team"])
            opponent = self.profile_for(r["opponent"])
            for event in EVENT_ALIASES:
                warnings = []
                if not self.availability.get(event, False):
                    rows.append(
                        _prediction_row(r, event, np.nan, "not_available", "not_available", "market_not_available_in_historical_data")
                    )
                    continue
                own = team.means_for.get(event, self.global_means[event])
                conceded = opponent.means_against.get(event, self.global_means[event])
                w = float(np.clip(self.own_weight, 0.0, 1.0))
                expected = w * own + (1.0 - w) * conceded
                if self.low_sample_blend_k > 0:
                    team_conf = team.matches / (team.matches + self.low_sample_blend_k) if team.matches > 0 else 0.0
                    opp_conf = opponent.matches / (opponent.matches + self.low_sample_blend_k) if opponent.matches > 0 else 0.0
                    pair_conf = float(np.sqrt(max(team_conf, 0.0) * max(opp_conf, 0.0)))
                    expected = pair_conf * expected + (1.0 - pair_conf) * self.global_means[event]
                    if pair_conf < 0.60:
                        warnings.append(f"team_stat_shrunk_low_sample_pair_conf={pair_conf:.2f}")
                # SOT should never exceed shots too aggressively when both are predicted.
                if event == "shots_on_target":
                    shots_own = team.means_for.get("shots", self.global_means["shots"])
                    expected = min(expected, max(0.6, 0.55 * shots_own))
                expected = safe_count(expected, event, EVENT_DEFAULTS[event])
                if team.warnings:
                    warnings.append(team.warnings)
                if opponent.warnings:
                    warnings.append(f"opponent_{opponent.warnings}")
                confidence = "normal" if team.matches >= 5 and opponent.matches >= 5 else "caution"
                rows.append(_prediction_row(r, event, expected, "available", confidence, ";".join(warnings)))
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        total_rows = []
        for (match_id, event), g in out[out["availability"] == "available"].groupby(["match_id", "market"]):
            if len(g) >= 2:
                base = g.iloc[0].to_dict()
                total = float(pd.to_numeric(g["expected_count"], errors="coerce").sum())
                base.update(
                    {
                        "team": "match_total",
                        "opponent": "match_total",
                        "market": f"total_{event}",
                        "expected_count": total,
                        "confidence": "normal" if (g["confidence"] == "normal").all() else "caution",
                        "warnings": ";".join([str(x) for x in g["warnings"].dropna().unique() if str(x)]),
                    }
                )
                total_rows.append(base)
        if total_rows:
            out = pd.concat([out, pd.DataFrame(total_rows)], ignore_index=True)
        return out


def _prediction_row(context: pd.Series, market: str, expected: float, availability: str, confidence: str, warnings: str) -> dict[str, Any]:
    return {
        "match_id": str(context["match_id"]),
        "date": context.get("date", "unknown"),
        "competition": context.get("competition", "unknown"),
        "stage": context.get("stage", "unknown"),
        "team": context["team"],
        "opponent": context["opponent"],
        "is_home": int(context.get("is_home", 0)),
        "market": market,
        "expected_count": expected,
        "availability": availability,
        "confidence": confidence,
        "warnings": warnings,
        "model_type": "v020_team_count_profile",
    }


def build_team_match_stat_frame(historical_events: pd.DataFrame | None) -> pd.DataFrame:
    if historical_events is None or historical_events.empty:
        return pd.DataFrame()
    df = historical_events.copy()
    if "match_id" not in df.columns or "team" not in df.columns:
        return pd.DataFrame()
    df["team"] = df["team"].map(canonical_name)
    if "opponent" in df.columns:
        df["opponent"] = df["opponent"].map(canonical_name)
    else:
        df["opponent"] = "unknown"
    if "date" not in df.columns:
        df["date"] = pd.NaT
    agg = {"date": "first", "opponent": "first"}
    work = df[["match_id", "date", "team", "opponent"]].copy()
    for event, aliases in EVENT_ALIASES.items():
        col = next((c for c in aliases if c in df.columns), None)
        if col is not None and pd.to_numeric(df[col], errors="coerce").notna().any():
            work[event] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            agg[event] = "sum"
    if len(agg) <= 2:
        return pd.DataFrame()
    team_match = work.groupby(["match_id", "team"], dropna=False).agg(agg).reset_index()
    # Attach against by same match/opponent when the opponent has a row.
    for event in EVENT_ALIASES:
        if event not in team_match.columns:
            continue
        opp = team_match[["match_id", "team", event]].rename(columns={"team": "opponent", event: f"{event}_against"})
        team_match = team_match.merge(opp, on=["match_id", "opponent"], how="left")
    team_match["date"] = pd.to_datetime(team_match["date"], errors="coerce")
    return team_match
