from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson

from mundialytics.features.player_features import EVENT_COLUMNS, player_baselines
from mundialytics.identity.normalization import canonical_player_name, player_global_id, PlayerIdentityResolver, PlayerIdentityMatch


@dataclass
class PlayerEventPrediction:
    player: str
    market_type: str
    line: str
    expected_count: float
    probability: float
    expected_minutes: float
    sample_size: float
    explanation: str


class PlayerEventModel:
    """Transparent rate-based player props model.

    Operational constraints:
    - Historical/retired players may train the rates.
    - Matchday prediction candidates must still come from supplied current
      lineups/squads.
    - Backtests estimate pre-match expected minutes from historical rows only.
    - For national-team props, club history may inform player rates when the
      same player_id_global is present, but the model reports that sample mix.
    """

    MARKET_TO_EVENT = {
        "player_shots": "shots",
        "player_shots_on_target": "shots_on_target",
        "player_fouls_committed": "fouls_committed",
        "player_fouls_drawn": "fouls_drawn",
        "player_yellow_card": "yellow_cards",
        "player_goals": "goals",
        "player_assists": "assists",
    }

    def __init__(self, min_minutes_for_rate: int = 180):
        self.min_minutes_for_rate = min_minutes_for_rate
        self.baselines: pd.DataFrame | None = None
        self.team_type_baselines: pd.DataFrame | None = None
        self.position_baselines: pd.DataFrame | None = None
        self.context_position_baselines: pd.DataFrame | None = None
        self.position_minutes: pd.DataFrame | None = None
        self.context_position_minutes: pd.DataFrame | None = None
        self.identity_resolver: PlayerIdentityResolver | None = None

    @staticmethod
    def _player_key(player: str, player_id_global: str | None = None) -> str:
        if player_id_global is not None and str(player_id_global).strip() and str(player_id_global).lower() != "nan":
            return str(player_id_global)
        return player_global_id(canonical_player_name(player))

    def fit(self, player_events: pd.DataFrame) -> "PlayerEventModel":
        df = player_events.copy()
        if "player" in df.columns and "player_id_global" not in df.columns:
            df["player_id_global"] = df["player"].map(player_global_id)
        self.baselines = player_baselines(df)
        self.identity_resolver = PlayerIdentityResolver.from_baselines(self.baselines)
        if self.baselines.empty:
            self.position_baselines = pd.DataFrame()
            self.position_minutes = pd.DataFrame()
            self.team_type_baselines = pd.DataFrame()
            self.identity_resolver = PlayerIdentityResolver.from_baselines(self.baselines)
            return self

        if "player_id_global" in df.columns and "team_type" in df.columns:
            self.team_type_baselines = player_baselines(df, group_cols=["player_id_global", "team_type"])
        else:
            self.team_type_baselines = pd.DataFrame()

        self.position_baselines = (
            self.baselines.groupby("position")[[f"{c}_per90" for c in EVENT_COLUMNS]].median().reset_index()
            if "position" in self.baselines.columns else pd.DataFrame()
        )
        if "competition_context" in self.baselines.columns and "position" in self.baselines.columns:
            self.context_position_baselines = (
                self.baselines.groupby(["competition_context", "position"])[[f"{c}_per90" for c in EVENT_COLUMNS]]
                .median()
                .reset_index()
            )
        else:
            self.context_position_baselines = pd.DataFrame()
        if "position" in df.columns:
            tmp = df.copy()
            tmp["minutes"] = pd.to_numeric(tmp.get("minutes", 90), errors="coerce").fillna(0).clip(lower=0, upper=130)
            self.position_minutes = (
                tmp.groupby("position")["minutes"].median().reset_index().rename(columns={"minutes": "expected_minutes"})
            )
            if "competition_context" in tmp.columns:
                self.context_position_minutes = (
                    tmp.groupby(["competition_context", "position"])["minutes"]
                    .median()
                    .reset_index()
                    .rename(columns={"minutes": "expected_minutes"})
                )
            else:
                self.context_position_minutes = pd.DataFrame()
        else:
            self.position_minutes = pd.DataFrame()
            self.context_position_minutes = pd.DataFrame()
        return self

    @staticmethod
    def parse_line(line: str | float | int) -> int:
        if isinstance(line, (int, float)):
            return int(math.ceil(float(line)))
        text = str(line).replace("+", "").strip()
        return int(float(text))

    @staticmethod
    def probability_at_least(k: int, lam: float) -> float:
        if k <= 0:
            return 1.0
        return float(1 - poisson.cdf(k - 1, max(lam, 0.0)))

    def resolve_player_identity(self, player: str, player_id_global: str | None = None) -> PlayerIdentityMatch:
        if self.identity_resolver is None:
            return PlayerIdentityMatch(
                input_player=str(player),
                input_player_id_global=player_id_global,
                matched_player=canonical_player_name(player),
                matched_player_id_global=self._player_key(player, player_id_global),
                method="no_resolver",
                confidence=0.0,
                status="unmatched",
            )
        return self.identity_resolver.resolve(player, player_id_global)

    def _resolved_player_key(self, player: str, player_id_global: str | None = None) -> str:
        # Exact supplied IDs still win if they exist historically. Otherwise we
        # resolve short lineup names to full historical display names/IDs.
        match = self.resolve_player_identity(player, player_id_global)
        if match.status == "matched" and match.matched_player_id_global:
            return str(match.matched_player_id_global)
        return self._player_key(player, player_id_global)

    def _baseline_row(self, player: str, player_id_global: str | None = None) -> pd.DataFrame:
        if self.baselines is None or self.baselines.empty:
            return pd.DataFrame()
        key = self._resolved_player_key(player, player_id_global)
        if "player_id_global" in self.baselines.columns:
            row = self.baselines[self.baselines["player_id_global"].astype(str) == key]
            if not row.empty:
                return row
        cname = canonical_player_name(player)
        if "player" in self.baselines.columns:
            return self.baselines[self.baselines["player"].astype(str) == cname]
        return pd.DataFrame()

    def _team_type_row(self, player: str, player_id_global: str | None = None, team_type: str | None = None) -> pd.DataFrame:
        if not team_type or self.team_type_baselines is None or self.team_type_baselines.empty:
            return pd.DataFrame()
        key = self._resolved_player_key(player, player_id_global)
        if "player_id_global" not in self.team_type_baselines.columns or "team_type" not in self.team_type_baselines.columns:
            return pd.DataFrame()
        return self.team_type_baselines[
            (self.team_type_baselines["player_id_global"].astype(str) == key)
            & (self.team_type_baselines["team_type"].astype(str) == str(team_type))
        ]

    def player_sample_profile(self, player: str, player_id_global: str | None = None) -> dict[str, float]:
        row = self._baseline_row(player, player_id_global)
        if row.empty:
            return {"club_minutes_sample": 0.0, "national_minutes_sample": 0.0, "total_minutes_sample": 0.0}
        r = row.iloc[0]
        club = float(r.get("club_minutes_sample", 0.0) or 0.0)
        national = float(r.get("national_minutes_sample", 0.0) or 0.0)
        total = float(r.get("minutes_sample", club + national) or 0.0)
        return {"club_minutes_sample": club, "national_minutes_sample": national, "total_minutes_sample": total}

    def _baseline_rate(
        self,
        player: str,
        event: str,
        position: str | None = None,
        player_id_global: str | None = None,
        competition_context: str | None = None,
        team_type: str | None = None,
    ) -> tuple[float, str]:
        col = f"{event}_per90"
        target_row = self._team_type_row(player, player_id_global, team_type)
        if not target_row.empty and float(target_row.iloc[0].get("minutes_sample", 0)) >= self.min_minutes_for_rate:
            return float(target_row.iloc[0].get(col, 0.0)), f"player_{team_type}_history"

        global_row = self._baseline_row(player, player_id_global)
        if not global_row.empty:
            if float(global_row.iloc[0].get("minutes_sample", 0)) >= self.min_minutes_for_rate:
                # If predicting a national-team player with little national sample, this may use club evidence.
                source = "player_global_history"
                if team_type == "national_team" and float(global_row.iloc[0].get("club_minutes_sample", 0.0) or 0.0) > 0:
                    source = "player_global_history_includes_club_to_national"
                return float(global_row.iloc[0].get(col, 0.0)), source
            if not position and pd.notna(global_row.iloc[0].get("position")):
                position = str(global_row.iloc[0]["position"])

        if (
            self.context_position_baselines is not None
            and not self.context_position_baselines.empty
            and position is not None
            and competition_context is not None
        ):
            ctx_pos = self.context_position_baselines[
                (self.context_position_baselines["position"].astype(str) == str(position))
                & (self.context_position_baselines["competition_context"].astype(str) == str(competition_context))
            ]
            if not ctx_pos.empty and col in ctx_pos.columns:
                return float(ctx_pos.iloc[0][col]), "competition_context_position_prior"
        if self.position_baselines is not None and not self.position_baselines.empty and position is not None:
            pos = self.position_baselines[self.position_baselines["position"].astype(str) == str(position)]
            if not pos.empty and col in pos.columns:
                return float(pos.iloc[0][col]), "position_prior"
        return {"shots": 1.0, "shots_on_target": 0.35, "fouls_committed": 1.1, "fouls_drawn": 1.1, "yellow_cards": 0.18, "goals": 0.18, "assists": 0.12}.get(event, 0.5), "generic_prior"

    def _sample_size(self, player: str, player_id_global: str | None = None) -> float:
        row = self._baseline_row(player, player_id_global)
        if not row.empty:
            return float(row.iloc[0].get("minutes_sample", 0.0))
        return 0.0

    def expected_minutes_for_player(
        self,
        player: str,
        position: str | None = None,
        started: object | None = None,
        player_id_global: str | None = None,
        competition_context: str | None = None,
        team_type: str | None = None,
    ) -> tuple[float, str]:
        """Estimate pre-match minutes using only historical training rows."""
        target_row = self._team_type_row(player, player_id_global, team_type)
        if not target_row.empty and float(target_row.iloc[0].get("minutes_sample", 0)) >= self.min_minutes_for_rate:
            minutes = float(target_row.iloc[0].get("expected_minutes", 70.0))
            source = f"player_{team_type}_recent_history"
        else:
            row = self._baseline_row(player, player_id_global)
            if not row.empty and float(row.iloc[0].get("minutes_sample", 0)) >= self.min_minutes_for_rate:
                minutes = float(row.iloc[0].get("expected_minutes", 70.0))
                source = "player_global_recent_history"
                if team_type == "national_team" and float(row.iloc[0].get("club_minutes_sample", 0.0) or 0.0) > 0:
                    source = "player_global_recent_history_includes_club_to_national"
            elif (
                self.context_position_minutes is not None
                and not self.context_position_minutes.empty
                and position is not None
                and competition_context is not None
            ):
                pos = self.context_position_minutes[
                    (self.context_position_minutes["position"].astype(str) == str(position))
                    & (self.context_position_minutes["competition_context"].astype(str) == str(competition_context))
                ]
                if not pos.empty:
                    minutes = float(pos.iloc[0]["expected_minutes"])
                    source = "competition_context_position_median_history"
                else:
                    minutes = 65.0
                    source = "generic_prior"
            elif self.position_minutes is not None and not self.position_minutes.empty and position is not None:
                pos = self.position_minutes[self.position_minutes["position"].astype(str) == str(position)]
                minutes = float(pos.iloc[0]["expected_minutes"]) if not pos.empty else 65.0
                source = "position_median_history"
            else:
                minutes = 65.0
                source = "generic_prior"

        if started is not None and not pd.isna(started):
            s = str(started).strip().lower()
            is_start = s in {"1", "true", "yes", "starter", "started"}
            is_bench = s in {"0", "false", "no", "bench", "sub"}
            if is_start:
                minutes = max(minutes, 55.0)
                source += "+starter_floor"
            elif is_bench:
                minutes = min(minutes, 35.0)
                source += "+bench_cap"
        return float(np.clip(minutes, 1, 105)), source

    @staticmethod
    def context_multiplier(market_type: str, team_context: dict | None = None) -> float:
        ctx = team_context or {}
        elo_diff = float(ctx.get("elo_diff", 0.0))
        possession = float(ctx.get("expected_possession", 50.0))
        if market_type in {"player_shots", "player_shots_on_target", "player_goals", "player_assists"}:
            return float(np.clip(1 + elo_diff / 1000 + (possession - 50) / 160, 0.65, 1.45))
        if market_type == "player_fouls_committed":
            return float(np.clip(1 - elo_diff / 1100 + (50 - possession) / 140, 0.70, 1.55))
        if market_type == "player_fouls_drawn":
            return float(np.clip(1 + elo_diff / 1100 + (possession - 50) / 150, 0.70, 1.50))
        if market_type == "player_yellow_card":
            return float(np.clip(1 - elo_diff / 1300, 0.75, 1.40))
        return 1.0

    def predict_market(
        self,
        player: str,
        market_type: str,
        line: str | int | float,
        expected_minutes: float,
        team_context: dict | None = None,
        position: str | None = None,
        player_id_global: str | None = None,
        competition_context: str | None = None,
        team_type: str | None = None,
    ) -> PlayerEventPrediction:
        event = self.MARKET_TO_EVENT.get(market_type)
        if event is None:
            raise ValueError(f"Unsupported player market_type: {market_type}")
        player_name = canonical_player_name(player)
        base_rate, rate_source = self._baseline_rate(
            player_name, event, position=position, player_id_global=player_id_global,
            competition_context=competition_context, team_type=team_type,
        )
        mult = self.context_multiplier(market_type, team_context)
        lam = base_rate * max(float(expected_minutes), 1) / 90 * mult
        k = self.parse_line(line)
        prob = self.probability_at_least(k, lam)
        sample_size = self._sample_size(player_name, player_id_global=player_id_global)
        explanation = (
            f"{event} rate {base_rate:.2f}/90 [{rate_source}] × {expected_minutes:.0f} pre-match minutes × context {mult:.2f} "
            f"=> lambda {lam:.2f}, P({line})={prob:.1%}; sample={sample_size:.0f} min"
        )
        return PlayerEventPrediction(player_name, market_type, str(line), float(lam), float(prob), float(expected_minutes), float(sample_size), explanation)
