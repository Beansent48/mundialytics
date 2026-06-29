from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mundialytics.identity.normalization import canonical_player_name, canonical_team_name, normalize_text, player_tokens


BUILTIN_PLAYER_ALIASES = {
    # High-value examples from the current matchday templates and common StatsBomb display names.
    "alvaro morata": "alvaro borja morata martin",
    "álvaro morata": "alvaro borja morata martin",
    "morata": "alvaro borja morata martin",
    "fede valverde": "federico santiago valverde dipetta",
    "f valverde": "federico santiago valverde dipetta",
    "federico valverde": "federico santiago valverde dipetta",
    "salem al-dawsari": "salem mohammed al dawsari",
    "salem al dawsari": "salem mohammed al dawsari",
    "salem aldawsari": "salem mohammed al dawsari",
    "lamine yamal": "lamine yamal",
}


@dataclass(frozen=True)
class PlayerIdentityResolution:
    input_player_name: str
    canonical_player_name: str
    current_team: str
    historical_teams_used: tuple[str, ...]
    identity_match_level: str
    identity_status: str
    identity_confidence: float
    identity_warnings: tuple[str, ...] = ()

    @property
    def canonical_player_key(self) -> str:
        return self.canonical_player_name


class PlayerIdentityResolver:
    """Conservative current-player -> historical-player resolver.

    It never creates inference candidates. It only resolves players already
    supplied in current_lineups/squads to historical identities. The resolver
    prefers exact/alias matches and only accepts token containment when it is
    unique enough to avoid confusing names such as "Salem" vs "Nasser Al Dawsari".
    """

    def __init__(self, player_catalog: pd.DataFrame | None = None, alias_map: dict[str, str] | None = None):
        self.catalog = self._prepare_catalog(player_catalog)
        self.alias_map = _load_alias_map(alias_map)
        self._player_lookup = {str(r["player"]): r for _, r in self.catalog.iterrows()} if not self.catalog.empty else {}
        self.audit: dict[str, Any] = {
            "catalog_players": int(len(self.catalog)),
            "alias_count": int(len(self.alias_map)),
        }

    @classmethod
    def from_historical_events(cls, historical_events: pd.DataFrame | None, alias_path: str | Path | None = None) -> "PlayerIdentityResolver":
        if historical_events is None or historical_events.empty or "player" not in historical_events.columns:
            return cls(pd.DataFrame())
        df = historical_events.copy()
        if "team" not in df.columns:
            df["team"] = "unknown"
        if "minutes" not in df.columns:
            df["minutes"] = 0.0
        df["player"] = df["player"].map(canonical_player_name)
        df["team"] = df["team"].map(canonical_team_name)
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0).clip(lower=0)
        catalog = (
            df.groupby(["player", "team"], dropna=False)["minutes"].sum().reset_index()
        )
        alias_map = _load_alias_csv(alias_path)
        return cls(catalog, alias_map=alias_map)

    @staticmethod
    def _prepare_catalog(catalog: pd.DataFrame | None) -> pd.DataFrame:
        if catalog is None or catalog.empty or "player" not in catalog.columns:
            return pd.DataFrame(columns=["player", "teams", "minutes", "tokens"])
        df = catalog.copy()
        if "team" not in df.columns:
            df["team"] = "unknown"
        if "minutes" not in df.columns:
            df["minutes"] = 0.0
        df["player"] = df["player"].map(canonical_player_name)
        df["team"] = df["team"].map(canonical_team_name)
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0).clip(lower=0)
        rows = []
        for player, g in df.groupby("player", dropna=False):
            teams = tuple(sorted(t for t in g["team"].dropna().astype(str).unique() if t))
            rows.append({
                "player": str(player),
                "teams": teams,
                "minutes": float(g["minutes"].sum()),
                "tokens": player_tokens(player),
            })
        return pd.DataFrame(rows).sort_values("minutes", ascending=False).reset_index(drop=True)

    def resolve(self, player: object, current_team: object) -> PlayerIdentityResolution:
        input_name = canonical_player_name(player)
        team = canonical_team_name(current_team)
        if self.catalog.empty:
            return self._fallback(input_name, team, "unresolved", "no_historical_player_catalog")

        # 1) Manual/built-in alias hit. This intentionally comes before exact
        # short-name hits so a known short lineup name can resolve to the richer
        # StatsBomb full-name career profile when both variants exist.
        alias_target = self.alias_map.get(normalize_text(input_name))
        if alias_target:
            row = self._player_lookup.get(alias_target)
            if row is not None:
                return self._resolution(input_name, team, row, "alias_player_team" if team in row["teams"] else "alias_player_any_team", 0.98)

        # 2) Exact normalized full-name hit.
        row = self._player_lookup.get(input_name)
        if row is not None:
            return self._resolution(input_name, team, row, "exact_player_team" if team in row["teams"] else "exact_player_any_team", 1.0)

        # 3) Conservative token containment. Accept only unique top candidates.
        q_tokens = set(player_tokens(input_name))
        candidates: list[tuple[float, pd.Series]] = []
        if len(q_tokens) >= 2:
            for _, row in self.catalog.iterrows():
                c_tokens = set(row["tokens"])
                if q_tokens.issubset(c_tokens):
                    # More query coverage + current-team support + larger sample wins.
                    coverage = len(q_tokens) / max(len(c_tokens), 1)
                    team_bonus = 0.08 if team in row["teams"] else 0.0
                    sample_bonus = min(float(row["minutes"]) / 5000.0, 0.05)
                    candidates.append((0.86 + 0.08 * coverage + team_bonus + sample_bonus, row))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_score, top_row = candidates[0]
            tied = [c for c in candidates if top_score - c[0] <= 0.025]
            if len(tied) == 1:
                level = "token_player_team" if team in top_row["teams"] else "token_player_any_team"
                return self._resolution(input_name, team, top_row, level, min(top_score, 0.96))
            tied_names = tuple(str(c[1]["player"]) for c in tied[:5])
            return PlayerIdentityResolution(
                input_player_name=input_name,
                canonical_player_name=input_name,
                current_team=team,
                historical_teams_used=(),
                identity_match_level="ambiguous",
                identity_status="ambiguous",
                identity_confidence=float(top_score),
                identity_warnings=("ambiguous_player_identity", f"candidates={','.join(tied_names)}"),
            )

        return self._fallback(input_name, team, "unresolved", "no_exact_alias_or_unique_token_match")

    def _fallback(self, input_name: str, team: str, status: str, warning: str) -> PlayerIdentityResolution:
        return PlayerIdentityResolution(
            input_player_name=input_name,
            canonical_player_name=input_name,
            current_team=team,
            historical_teams_used=(),
            identity_match_level=status,
            identity_status=status,
            identity_confidence=0.0,
            identity_warnings=(warning,),
        )

    def _resolution(self, input_name: str, team: str, row: pd.Series, level: str, confidence: float) -> PlayerIdentityResolution:
        teams = tuple(row["teams"] or ())
        warnings: list[str] = []
        adjusted_level = level
        if team not in teams:
            adjusted_level = "club_history_transfer" if teams else level
            warnings.append("current_team_not_seen_for_player_using_any_team_history")
        return PlayerIdentityResolution(
            input_player_name=input_name,
            canonical_player_name=str(row["player"]),
            current_team=team,
            historical_teams_used=teams,
            identity_match_level=adjusted_level,
            identity_status="matched",
            identity_confidence=float(confidence),
            identity_warnings=tuple(warnings),
        )


def _load_alias_csv(alias_path: str | Path | None) -> dict[str, str]:
    aliases: dict[str, str] = {}
    paths: list[Path] = []
    if alias_path:
        paths.append(Path(alias_path))
    # Project-local default. This makes the CLI work without another argument.
    here = Path(__file__).resolve()
    project_root = here.parents[3]
    paths.append(project_root / "data" / "identity" / "player_aliases.csv")
    for p in paths:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if not {"alias", "canonical_player"}.issubset(df.columns):
            continue
        for _, r in df.iterrows():
            a = normalize_text(r.get("alias"))
            c = canonical_player_name(r.get("canonical_player"))
            if a and c:
                aliases[a] = c
    return aliases


def _load_alias_map(extra_aliases: dict[str, str] | None = None) -> dict[str, str]:
    aliases = {normalize_text(a): canonical_player_name(c) for a, c in BUILTIN_PLAYER_ALIASES.items()}
    if extra_aliases:
        for a, c in extra_aliases.items():
            aliases[normalize_text(a)] = canonical_player_name(c)
    return aliases
