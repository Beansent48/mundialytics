from __future__ import annotations

from dataclasses import dataclass, field
import ast
from typing import Any

import numpy as np
import pandas as pd

from mundialytics.identity.player_resolver import PlayerIdentityResolution, PlayerIdentityResolver
from mundialytics.statistical_core.calibration import safe_probability
from mundialytics.statistical_core.distributions import probability_for_count_line
from mundialytics.statistical_core.schemas import canonical_name, standardize_current_players


PLAYER_MARKETS = {
    "player_shots": {"event": "shots", "line": "1+", "usable": "usable"},
    "player_shots_on_target": {"event": "shots_on_target", "line": "1+", "usable": "usable_with_caution"},
    "player_fouls_committed": {"event": "fouls", "line": "1+", "usable": "usable_with_caution"},
    "player_yellow_card": {"event": "yellow_cards", "line": "1+", "usable": "usable_with_caution_cap"},
}

POSITION_PRIORS = {
    "shots": {"st": 0.22, "cf": 0.22, "lw": 0.17, "rw": 0.17, "am": 0.14, "cm": 0.09, "dm": 0.05, "fb": 0.04, "cb": 0.04, "gk": 0.005},
    "shots_on_target": {"st": 0.24, "cf": 0.24, "lw": 0.17, "rw": 0.17, "am": 0.13, "cm": 0.08, "dm": 0.04, "fb": 0.03, "cb": 0.03, "gk": 0.001},
    "fouls": {"st": 0.09, "cf": 0.09, "lw": 0.08, "rw": 0.08, "am": 0.09, "cm": 0.12, "dm": 0.15, "fb": 0.12, "cb": 0.13, "gk": 0.01},
    "yellow_cards": {"st": 0.06, "cf": 0.06, "lw": 0.05, "rw": 0.05, "am": 0.06, "cm": 0.10, "dm": 0.15, "fb": 0.11, "cb": 0.13, "gk": 0.02},
}

EVENT_COL_ALIASES = {
    "shots": ["shots", "player_shots"],
    "shots_on_target": ["shots_on_target", "sot", "player_shots_on_target"],
    "fouls": ["fouls_committed", "fouls", "player_fouls_committed"],
    "yellow_cards": ["yellow_cards", "yellow_card", "player_yellow_card"],
}

# v0.34: squad rosters are useful before confirmed lineups exist, but they
# should not create an unlimited, equal-confidence player-prop board. Keep the
# current-candidate gate, then rank squad candidates by historical identity and
# sample size so the final market board can be conservative.
MAX_SQUAD_CANDIDATES_PER_TEAM = 16
LOW_CONFIDENCE_BASIC_MARKETS = {"player_shots", "player_fouls_committed", "player_yellow_card"}


@dataclass
class PlayerProfile:
    player: str
    current_team: str
    position: str
    minutes: float
    rates_per90: dict[str, float]
    shares: dict[str, float]
    canonical_player_name: str
    historical_teams_used: tuple[str, ...] = field(default_factory=tuple)
    identity_match_level: str = "unresolved"
    identity_status: str = "unresolved"
    identity_confidence: float = 0.0
    identity_warnings: tuple[str, ...] = field(default_factory=tuple)
    input_position: str = ""
    position_source: str = "provider_position"


class PlayerEventModel:
    """Current-candidate-only player prop model integrated with team totals.

    v0.21 resolves manual lineup names to historical identities and aggregates a
    player's valid historical teams for player profiling. It still gates
    inference strictly on current lineups/squads, so retired or historical-only
    players never become candidates by themselves.
    """

    def __init__(
        self,
        share_weight: float = 0.65,
        yellow_card_cap: float = 0.75,
        share_cap: float = 0.45,
        share_floor: float = 0.002,
    ):
        self.share_weight = float(share_weight)
        self.yellow_card_cap = float(yellow_card_cap)
        self.share_cap = float(share_cap)
        self.share_floor = float(share_floor)
        self.team_profiles: dict[str, PlayerProfile] = {}
        self.player_profiles: dict[str, PlayerProfile] = {}
        self.resolver = PlayerIdentityResolver(pd.DataFrame())
        self.position_rates: dict[str, dict[str, float]] = {}
        self.global_rates: dict[str, float] = {"shots": 1.0, "shots_on_target": 0.35, "fouls": 0.85, "yellow_cards": 0.15}
        self.audit: dict[str, Any] = {}

    def fit(self, historical_events: pd.DataFrame | None) -> "PlayerEventModel":
        if historical_events is None or historical_events.empty or "player" not in historical_events.columns:
            self.audit = {"player_event_fit": "fallback_no_historical_player_events", "players": 0, "identity_resolution": {"catalog_players": 0}}
            return self
        df = historical_events.copy()
        df["player"] = df["player"].map(canonical_name)
        df["team"] = df.get("team", "unknown")
        df["team"] = df["team"].map(canonical_name)
        if "position" not in df.columns:
            df["position"] = "UNK"
        if "minutes" not in df.columns:
            df["minutes"] = 90.0
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0).clip(lower=0)
        event_cols: dict[str, str] = {}
        for event, aliases in EVENT_COL_ALIASES.items():
            col = next((c for c in aliases if c in df.columns), None)
            if col is not None:
                df[event] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)
                event_cols[event] = event
        if not event_cols:
            self.audit = {"player_event_fit": "fallback_no_player_event_columns", "players": 0, "identity_resolution": {"catalog_players": 0}}
            return self

        self.resolver = PlayerIdentityResolver.from_historical_events(df)
        self.team_profiles = self._build_profiles(df, list(event_cols), group_cols=["player", "team"], team_specific=True)
        self.player_profiles = self._build_profiles(df, list(event_cols), group_cols=["player"], team_specific=False)
        self.position_rates = _position_rates(df, list(event_cols))
        for event in event_cols:
            total_min = max(float(df["minutes"].sum()), 1.0)
            self.global_rates[event] = 90.0 * float(df[event].sum()) / total_min
        self.audit = {
            "player_event_fit": "historical_player_events_with_identity_resolution",
            "players": len(self.player_profiles),
            "player_team_profiles": len(self.team_profiles),
            "events": list(event_cols),
            "identity_resolution": self.resolver.audit,
            "model_config": {
                "share_weight": self.share_weight,
                "yellow_card_cap": self.yellow_card_cap,
                "share_cap": self.share_cap,
                "share_floor": self.share_floor,
            },
        }
        return self

    def _build_profiles(self, df: pd.DataFrame, events: list[str], group_cols: list[str], team_specific: bool) -> dict[str, PlayerProfile]:
        profiles: dict[str, PlayerProfile] = {}
        for key_values, g in df.groupby(group_cols, dropna=False):
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            player = str(key_values[0])
            team = str(key_values[1]) if team_specific and len(key_values) > 1 else "any_team"
            minutes = float(g["minutes"].sum())
            pos = str(g["position"].mode().iloc[0]) if len(g["position"].dropna()) else "UNK"
            rates = {}
            shares = {}
            for event in events:
                total = float(g[event].sum())
                rates[event] = 90.0 * total / max(minutes, 1.0)
                shares[event] = _estimate_player_share(df, g, event)
            teams_used = tuple(sorted(t for t in g["team"].dropna().astype(str).unique() if t))
            dict_key = f"{player}|{team}" if team_specific else player
            profiles[dict_key] = PlayerProfile(
                player=player,
                current_team=team,
                position=pos,
                minutes=minutes,
                rates_per90=rates,
                shares=shares,
                canonical_player_name=player,
                historical_teams_used=teams_used,
                identity_match_level="historical_profile",
                identity_status="matched",
                identity_confidence=1.0,
                input_position=str(pos),
                position_source="historical_frequent_position",
            )
        return profiles

    def profile_for(self, player: str, team: str, position: str) -> PlayerProfile:
        input_player = canonical_name(player)
        current_team = canonical_name(team)
        input_position = _position_text(position)
        resolution = self.resolver.resolve(input_player, current_team)
        if resolution.identity_status == "matched" and resolution.canonical_player_key in self.player_profiles:
            base = self.player_profiles[resolution.canonical_player_key]
            resolved_position, position_source = _resolve_deployment_position(input_position, base.position)
            return PlayerProfile(
                player=input_player,
                current_team=current_team,
                position=resolved_position,
                minutes=base.minutes,
                rates_per90=base.rates_per90,
                shares=base.shares,
                canonical_player_name=base.canonical_player_name,
                historical_teams_used=resolution.historical_teams_used or base.historical_teams_used,
                identity_match_level=resolution.identity_match_level,
                identity_status=resolution.identity_status,
                identity_confidence=resolution.identity_confidence,
                identity_warnings=resolution.identity_warnings,
                input_position=str(input_position),
                position_source=position_source,
            )
        pos_key = _position_key(input_position)
        rates = {event: self.position_rates.get(pos_key, {}).get(event, self.global_rates.get(event, 0.2)) for event in EVENT_COL_ALIASES}
        shares = {event: POSITION_PRIORS.get(event, {}).get(pos_key, 0.08) for event in EVENT_COL_ALIASES}
        return PlayerProfile(
            player=input_player,
            current_team=current_team,
            position=str(input_position),
            minutes=0.0,
            rates_per90=rates,
            shares=shares,
            canonical_player_name=resolution.canonical_player_name,
            historical_teams_used=resolution.historical_teams_used,
            identity_match_level=resolution.identity_match_level,
            identity_status=resolution.identity_status,
            identity_confidence=resolution.identity_confidence,
            identity_warnings=resolution.identity_warnings,
            input_position=str(input_position),
            position_source="provider_position_unmatched",
        )

    def predict(
        self,
        fixtures: pd.DataFrame,
        lineups: pd.DataFrame | None,
        squads: pd.DataFrame | None,
        team_stats_predictions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        candidates, candidate_warnings = standardize_current_players(lineups, squads, fixtures)
        if candidates.empty:
            return pd.DataFrame(), candidate_warnings
        team_expected = _team_expected_lookup(team_stats_predictions)
        rows: list[dict[str, Any]] = []
        identity_levels: list[str] = []
        zero_sample_candidates: set[tuple[str, str]] = set()
        for _, p in candidates.iterrows():
            profile = self.profile_for(p["player"], p["team"], p.get("position", "UNK"))
            identity_levels.append(profile.identity_match_level)
            if profile.minutes <= 0:
                zero_sample_candidates.add((str(p["match_id"]), canonical_name(p["player"])))
            expected_minutes = float(p.get("expected_minutes", 75.0))
            pos_key = _position_key(profile.position)
            pos_group = _position_group(pos_key)
            candidate_source = str(p.get("candidate_source", "unknown"))
            player_input_source = str(p.get("player_input_source", candidate_source))
            player_selection_confidence = _player_selection_confidence(candidate_source, profile.minutes, profile.identity_status)
            candidate_score = _candidate_score(profile, expected_minutes, candidate_source, player_selection_confidence, pos_group)
            for market, meta in PLAYER_MARKETS.items():
                event = meta["event"]
                team_exp = team_expected.get((str(p["match_id"]), canonical_name(p["team"]), event), np.nan)
                warnings = list(profile.identity_warnings)
                if profile.identity_status != "matched":
                    warnings.append(f"identity_{profile.identity_status}")
                if profile.identity_match_level in {"ambiguous", "unresolved"}:
                    warnings.append(f"identity_match_level_{profile.identity_match_level}")
                if not np.isfinite(team_exp):
                    team_exp = _fallback_team_event_expected(event)
                    warnings.append("team_event_expectation_fallback")
                share = profile.shares.get(event, POSITION_PRIORS.get(event, {}).get(pos_key, 0.08))
                share = float(np.clip(share, self.share_floor, self.share_cap))
                rate_component = profile.rates_per90.get(event, self.global_rates.get(event, 0.2)) * expected_minutes / 90.0
                share_component = team_exp * share * expected_minutes / 75.0
                sw = float(np.clip(self.share_weight, 0.0, 1.0))
                expected_count = float(max(0.0, sw * share_component + (1.0 - sw) * rate_component))
                if pos_group == "goalkeeper" and market in {"player_shots", "player_shots_on_target"}:
                    expected_count = 0.0
                    warnings.append("role_guardrail_goalkeeper_attacking_prop_blocked")
                if event == "yellow_cards":
                    expected_count = min(expected_count, self.yellow_card_cap)
                raw_prob = probability_for_count_line(expected_count, meta["line"], "over")
                safe_prob, cap_warnings = safe_probability(raw_prob, market, sample_size=profile.minutes)
                if pos_group == "goalkeeper" and market in {"player_shots", "player_shots_on_target"}:
                    safe_prob = 0.0
                warnings.extend(cap_warnings)
                if profile.minutes <= 0:
                    warnings.append("sample_size_zero_no_player_pick")
                confidence = "normal" if profile.minutes >= 270 else ("very_low_sample" if profile.minutes < 90 else "low_sample")
                if meta["usable"] != "usable":
                    warnings.append(meta["usable"])
                rows.append(
                    {
                        "match_id": str(p["match_id"]),
                        "date": p.get("date", "unknown"),
                        "competition": p.get("competition", "unknown"),
                        "stage": p.get("stage", "unknown"),
                        "team": canonical_name(p["team"]),
                        "opponent": canonical_name(p.get("opponent", "")),
                        "player": canonical_name(p["player"]),
                        "player_input_name": canonical_name(p["player"]),
                        "canonical_player_name": profile.canonical_player_name,
                        "current_team": canonical_name(p["team"]),
                        "historical_teams_used": ";".join(profile.historical_teams_used),
                        "identity_match_level": profile.identity_match_level,
                        "identity_status": profile.identity_status,
                        "identity_confidence": profile.identity_confidence,
                        "position": profile.position,
                        "input_position": profile.input_position or p.get("position", "UNK"),
                        "resolved_position": profile.position,
                        "position_key": pos_key,
                        "position_group": pos_group,
                        "position_source": profile.position_source,
                        "candidate_source": candidate_source,
                        "player_input_source": player_input_source,
                        "player_selection_confidence": player_selection_confidence,
                        "candidate_score": float(candidate_score),
                        "market": market,
                        "line": meta["line"],
                        "team_expected_event": float(team_exp),
                        "player_share": float(share),
                        "historical_rate_per90": float(profile.rates_per90.get(event, self.global_rates.get(event, 0.0))),
                        "expected_minutes": expected_minutes,
                        "expected_count": expected_count,
                        "raw_probability": raw_prob,
                        "safe_probability": safe_prob,
                        "sample_size_minutes": float(profile.minutes),
                        "confidence_flag": confidence,
                        "warnings": ";".join(dict.fromkeys([w for w in warnings if w and str(w) != "nan"])),
                        "model_type": "v021_identity_resolved_team_integrated_player_props",
                    }
                )
        out = pd.DataFrame(rows)
        out = _apply_candidate_policy(out)
        self.audit["identity_resolution_runtime"] = {
            "candidates": int(len(candidates)),
            "zero_sample_candidates": int(len(zero_sample_candidates)),
            "match_levels": {k: int(v) for k, v in pd.Series(identity_levels).value_counts().to_dict().items()},
            "candidate_policy_counts": {k: int(v) for k, v in out.drop_duplicates(["match_id", "team", "player"]).get("candidate_policy", pd.Series(dtype=str)).value_counts().to_dict().items()} if not out.empty else {},
            "max_squad_candidates_per_team": MAX_SQUAD_CANDIDATES_PER_TEAM,
        }
        return out, candidate_warnings


def _position_text(position: object) -> str:
    if position is None:
        return "UNK"
    if isinstance(position, dict):
        for key in ("abbreviation", "displayName", "name", "shortName"):
            value = position.get(key)
            if value:
                return str(value)
        return "UNK"
    text = str(position).strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return _position_text(parsed)
        except Exception:
            pass
    return text or "UNK"


def _position_key(position: object) -> str:
    """Normalize provider/free-text positions into stable football role keys.

    StatsBomb-style exports often contain verbose strings such as ``left wing``
    or ``right center midfield``. Provider rosters can also contain nested
    dictionaries or dict-like strings (for example ESPN position payloads). This
    function unwraps those safely before mapping into compact tactical keys.
    """
    pos = canonical_name(_position_text(position) or "UNK")
    # ESPN sometimes uses single-letter abbreviations for soccer positions.
    if pos in {"goalkeeper", "keeper", "gk", "g", "portero"}:
        return "gk"
    if pos in {"left back", "right back", "full back", "fullback", "lb", "rb", "left fullback", "right fullback", "defender back", "b"}:
        return "fb"
    if pos in {"left wing back", "right wing back", "wing back", "lwb", "rwb"}:
        return "wb"
    if pos in {"center back", "centre back", "left center back", "right center back", "left centre back", "right centre back", "central defender", "defender", "d", "cb", "lcb", "rcb", "lc b"}:
        return "cb"
    if pos in {"defensive midfield", "center defensive midfield", "centre defensive midfield", "left defensive midfield", "right defensive midfield", "dm", "cdm"}:
        return "dm"
    if pos in {"center midfield", "centre midfield", "left center midfield", "right center midfield", "left centre midfield", "right centre midfield", "central midfield", "midfielder", "m", "cm", "mc"}:
        return "cm"
    if pos in {"attacking midfield", "center attacking midfield", "centre attacking midfield", "cam", "am"}:
        return "am"
    if pos in {"left wing", "lw", "left winger"}:
        return "lw"
    if pos in {"right wing", "rw", "right winger", "left midfield", "right midfield", "lm", "rm"}:
        return "rw" if pos in {"right wing", "rw", "right winger", "right midfield", "rm"} else "lw"
    if pos in {"striker", "center forward", "centre forward", "left center forward", "right center forward", "left centre forward", "right centre forward", "forward", "f", "st", "cf", "fw"}:
        return "st"
    return pos


def _position_group(position: object) -> str:
    """Coarser role group for calibration/deployment guardrails."""
    key = _position_key(position)
    if key == "gk":
        return "goalkeeper"
    if key == "cb":
        return "center_back"
    if key in {"fb", "wb"}:
        return "fullback_wingback"
    if key == "dm":
        return "defensive_midfield"
    if key == "cm":
        return "central_midfield"
    if key == "am":
        return "attacking_midfield"
    if key in {"lw", "rw"}:
        return "winger"
    if key in {"st", "cf", "fw"}:
        return "forward"
    return "unknown_outfield"


def _is_generic_provider_position(position: object) -> bool:
    """True for broad roster roles such as ESPN D/M/F that should not override history."""
    text = canonical_name(_position_text(position))
    key = _position_key(position)
    if not text or text in {"unk", "unknown", "nan", "none"}:
        return True
    return text in {
        "d", "defender", "defense",
        "m", "midfielder", "midfield",
        "f", "forward", "fw", "attacker",
        "b", "back",
    } or key in {"unknown", "unk"}


def _resolve_deployment_position(provider_position: object, historical_position: object) -> tuple[str, str]:
    """Choose the best tactical position for today-player inference.

    Confirmed lineups/providers may give exact roles, but roster feeds often only
    provide broad buckets (D/M/F/G). For matched players, using their most common
    historical StatsBomb position is a better tactical prior than treating every
    defender as centre-back, every midfielder as central midfielder and every
    forward as striker. Goalkeeper remains trusted from provider and history.
    """
    provider_text = _position_text(provider_position)
    historical_text = _position_text(historical_position)
    hist_group = _position_group(historical_text)
    provider_group = _position_group(provider_text)
    if hist_group != "unknown_outfield" and _is_generic_provider_position(provider_text):
        return historical_text, "historical_frequent_position_fallback"
    if provider_group == "unknown_outfield" and hist_group != "unknown_outfield":
        return historical_text, "historical_frequent_position_fallback"
    return provider_text, "provider_position"


def _candidate_score(profile: PlayerProfile, expected_minutes: float, candidate_source: str, confidence: str, position_group: str) -> float:
    """Rank current candidates for squad-fallback player props.

    This is not a starting-XI model. It is a conservative ordering so broad
    rosters do not produce a huge board of equally credible props. Confirmed
    lineups still bypass the strict squad filters downstream.
    """
    source = str(candidate_source or "unknown").lower()
    conf = str(confidence or "unknown").lower()
    score = float(np.log1p(max(profile.minutes, 0.0)))
    score += float(profile.identity_confidence) * 3.0
    score += float(np.clip(expected_minutes, 0, 130)) / 90.0
    if source == "lineups":
        score += 5.0
    score += {"high": 4.0, "medium": 3.0, "medium_low": 2.0, "low": 0.75, "very_low": -5.0}.get(conf, 0.0)
    if str(profile.identity_status) != "matched":
        score -= 10.0
    if max(profile.minutes, 0.0) <= 0:
        score -= 10.0
    if str(position_group) == "goalkeeper":
        score -= 0.4
    return float(score)


def _apply_candidate_policy(out: pd.DataFrame) -> pd.DataFrame:
    if out is None or out.empty:
        return out
    df = out.copy()
    key = ["match_id", "team", "player"]
    cand_cols = key + ["candidate_source", "player_selection_confidence", "candidate_score", "sample_size_minutes", "identity_status", "identity_match_level"]
    cand = df[cand_cols].drop_duplicates(key).copy()
    cand["candidate_score"] = pd.to_numeric(cand["candidate_score"], errors="coerce").fillna(-999.0)
    cand = cand.sort_values(key[:2] + ["candidate_score", "player"], ascending=[True, True, False, True]).copy()
    cand["candidate_rank_team"] = cand.groupby(["match_id", "team"]).cumcount() + 1
    cand["candidate_policy"] = cand.apply(_candidate_policy_row, axis=1)
    cand["candidate_reason"] = cand.apply(_candidate_reason_row, axis=1)
    cand = cand[key + ["candidate_rank_team", "candidate_policy", "candidate_reason"]]
    return df.merge(cand, on=key, how="left")


def _candidate_policy_row(row: pd.Series) -> str:
    source = str(row.get("candidate_source", "unknown")).lower()
    confidence = str(row.get("player_selection_confidence", "unknown")).lower()
    identity_status = str(row.get("identity_status", "")).lower()
    identity_level = str(row.get("identity_match_level", "")).lower()
    sample_minutes = float(pd.to_numeric(pd.Series([row.get("sample_size_minutes")]), errors="coerce").fillna(0.0).iloc[0])
    rank = int(row.get("candidate_rank_team", 9999) or 9999)
    if identity_status != "matched" or identity_level in {"unresolved", "ambiguous"} or sample_minutes <= 0:
        return "excluded_identity_or_sample"
    if source == "lineups":
        return "confirmed_lineup_candidate"
    if source == "squads":
        if rank > MAX_SQUAD_CANDIDATES_PER_TEAM:
            return "squad_excluded_rank_limit"
        if confidence == "very_low":
            return "squad_excluded_low_confidence"
        if confidence == "low":
            return "squad_low_confidence_basic_only"
        return "squad_fallback_candidate"
    return "unclassified_candidate"


def _candidate_reason_row(row: pd.Series) -> str:
    policy = str(row.get("candidate_policy", ""))
    source = str(row.get("candidate_source", "unknown"))
    confidence = str(row.get("player_selection_confidence", "unknown"))
    rank = row.get("candidate_rank_team", "")
    sample = row.get("sample_size_minutes", "")
    if policy == "excluded_identity_or_sample":
        return "identity_unresolved_or_zero_historical_sample"
    if policy == "squad_excluded_rank_limit":
        return f"squad_rank_above_{MAX_SQUAD_CANDIDATES_PER_TEAM};rank={rank}"
    if policy == "squad_excluded_low_confidence":
        return f"squad_confidence_too_low;rank={rank};sample_minutes={sample}"
    if policy == "squad_low_confidence_basic_only":
        return f"squad_low_confidence_basic_markets_only;rank={rank};sample_minutes={sample}"
    if policy == "squad_fallback_candidate":
        return f"squad_fallback_ranked_candidate;rank={rank};confidence={confidence}"
    if policy == "confirmed_lineup_candidate":
        return "confirmed_lineup_or_manual_lineup_candidate"
    return f"source={source};confidence={confidence};rank={rank}"


def _player_selection_confidence(candidate_source: str, sample_minutes: float, identity_status: str) -> str:
    source = str(candidate_source or "unknown").lower()
    if str(identity_status) != "matched":
        return "very_low"
    if source == "lineups":
        return "high" if sample_minutes >= 270 else "medium"
    if source == "squads":
        return "medium_low" if sample_minutes >= 900 else ("low" if sample_minutes >= 270 else "very_low")
    return "low"


def _position_rates(df: pd.DataFrame, events: list[str]) -> dict[str, dict[str, float]]:
    work = df.copy()
    work["position_key"] = work["position"].map(_position_key)
    out: dict[str, dict[str, float]] = {}
    for pos, g in work.groupby("position_key"):
        minutes = max(float(g["minutes"].sum()), 1.0)
        out[str(pos)] = {event: 90.0 * float(g[event].sum()) / minutes for event in events}
    return out


def _estimate_player_share(all_rows: pd.DataFrame, player_rows: pd.DataFrame, event: str) -> float:
    player_total = float(player_rows[event].sum())
    teams = player_rows["team"].dropna().unique().tolist()
    if not teams:
        return 0.08
    team_total = float(all_rows[all_rows["team"].isin(teams)][event].sum())
    if team_total <= 0:
        return 0.08
    return float(np.clip(player_total / team_total, 0.002, 0.45))


def _team_expected_lookup(team_stats: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    if team_stats is None or team_stats.empty:
        return out
    for _, r in team_stats.iterrows():
        if str(r.get("availability", "available")) != "available":
            continue
        market = str(r.get("market", ""))
        if market.startswith("total_"):
            continue
        value = pd.to_numeric(pd.Series([r.get("expected_count")]), errors="coerce").iloc[0]
        if not np.isfinite(value):
            continue
        out[(str(r.get("match_id")), canonical_name(r.get("team")), market)] = float(value)
    return out


def _fallback_team_event_expected(event: str) -> float:
    return {"shots": 10.0, "shots_on_target": 3.5, "fouls": 11.0, "yellow_cards": 1.8}.get(event, 1.0)
