from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mundialytics.data.adapters.espn import (
    DEFAULT_WORLD_CUP_LEAGUE,
    ESPNClient,
    summary_response_to_lineups_df,
    team_payload_to_squad_df,
)
from mundialytics.data.adapters.sofascore import (
    SofaScoreClient,
    lineups_response_to_df,
    team_players_response_to_squad_df,
)


@dataclass
class PlayerInputFetchConfig:
    provider: str = "auto"
    espn_league: str = DEFAULT_WORLD_CUP_LEAGUE
    fetch_lineups: bool = True
    fetch_squads: bool = True
    fail_soft: bool = True


def _fixture_by_provider_id(fixtures: pd.DataFrame, provider_match_id: Any) -> dict[str, Any]:
    if fixtures is None or fixtures.empty:
        return {}
    pid = str(provider_match_id)
    for _, row in fixtures.iterrows():
        if str(row.get("provider_match_id", row.get("fixture_id", ""))) == pid or str(row.get("fixture_id", "")) == pid:
            return row.to_dict()
    return {}


def _raw_rows_for_final_fixtures(provider_fixtures: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    if provider_fixtures is None or provider_fixtures.empty or fixtures is None or fixtures.empty:
        return pd.DataFrame()
    ids = {str(x) for x in fixtures.get("provider_match_id", pd.Series(dtype=str)).dropna().astype(str)}
    ids |= {str(x) for x in fixtures.get("fixture_id", pd.Series(dtype=str)).dropna().astype(str)}
    work = provider_fixtures.copy()
    mask = work.get("provider_match_id", pd.Series(index=work.index, dtype=object)).astype(str).isin(ids)
    mask = mask | work.get("fixture_id", pd.Series(index=work.index, dtype=object)).astype(str).isin(ids)
    return work.loc[mask].copy()


def fetch_player_inputs_for_fixtures(
    provider_fixtures: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    config: PlayerInputFetchConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fetch best-available current lineups and/or squad fallbacks for fixtures.

    Free public sources are best-effort. Lineups usually appear around kickoff;
    squads/rosters are broader and should be treated as fallback candidates only.
    Network errors are captured in the audit instead of failing the whole matchday.
    """
    cfg = config or PlayerInputFetchConfig()
    rows = _raw_rows_for_final_fixtures(provider_fixtures, fixtures)
    lineups: list[pd.DataFrame] = []
    squads: list[pd.DataFrame] = []
    attempts: list[dict[str, Any]] = []

    sofa = SofaScoreClient()
    espn = ESPNClient()

    for _, raw in rows.iterrows():
        provider = str(raw.get("provider") or "").lower()
        event_id = raw.get("provider_match_id") or raw.get("fixture_id")
        fixture_row = _fixture_by_provider_id(fixtures, event_id) or raw.to_dict()
        if not event_id:
            attempts.append({"provider": provider or "unknown", "status": "skipped", "reason": "missing_provider_match_id"})
            continue

        if cfg.provider != "auto" and provider and provider != cfg.provider:
            continue

        if provider == "sofascore":
            if cfg.fetch_lineups:
                try:
                    payload = sofa.event_lineups(event_id)
                    df = lineups_response_to_df(payload, fixture_row=fixture_row, provider_match_id=event_id)
                    if not df.empty:
                        lineups.append(df)
                    attempts.append({"provider": provider, "fixture_id": str(event_id), "kind": "lineups", "status": "ok", "rows": int(len(df))})
                except Exception as exc:
                    attempts.append({"provider": provider, "fixture_id": str(event_id), "kind": "lineups", "status": "error", "error": repr(exc)})
                    if not cfg.fail_soft:
                        raise
            if cfg.fetch_squads:
                for side, team_id_col, team_col in [("home", "home_team_id", "home_team"), ("away", "away_team_id", "away_team")]:
                    team_id = raw.get(team_id_col)
                    team_name = raw.get(team_col) or fixture_row.get(team_col)
                    if not team_id or pd.isna(team_id):
                        continue
                    try:
                        payload = sofa.team_players(team_id)
                        df = team_players_response_to_squad_df(payload, team_name=str(team_name), fixture_row=fixture_row, provider_match_id=event_id)
                        if not df.empty:
                            squads.append(df)
                        attempts.append({"provider": provider, "fixture_id": str(event_id), "kind": f"{side}_squad", "status": "ok", "rows": int(len(df))})
                    except Exception as exc:
                        attempts.append({"provider": provider, "fixture_id": str(event_id), "kind": f"{side}_squad", "status": "error", "error": repr(exc)})
                        if not cfg.fail_soft:
                            raise

        elif provider == "espn":
            if cfg.fetch_lineups:
                try:
                    payload = espn.summary(event_id, league=cfg.espn_league)
                    df = summary_response_to_lineups_df(payload, fixture_row=fixture_row, provider_match_id=event_id)
                    if not df.empty:
                        lineups.append(df)
                    attempts.append({"provider": provider, "fixture_id": str(event_id), "kind": "summary_lineups", "status": "ok", "rows": int(len(df))})
                except Exception as exc:
                    attempts.append({"provider": provider, "fixture_id": str(event_id), "kind": "summary_lineups", "status": "error", "error": repr(exc)})
                    if not cfg.fail_soft:
                        raise
            if cfg.fetch_squads:
                for side, team_id_col, team_col in [("home", "home_team_id", "home_team"), ("away", "away_team_id", "away_team")]:
                    team_id = raw.get(team_id_col)
                    team_name = raw.get(team_col) or fixture_row.get(team_col)
                    if not team_id or pd.isna(team_id):
                        continue
                    for endpoint_kind, fetcher in [
                        ("team_roster", lambda tid=team_id: espn.team_roster(tid, league=cfg.espn_league)),
                        ("team", lambda tid=team_id: espn.team(tid, league=cfg.espn_league)),
                    ]:
                        try:
                            payload = fetcher()
                            df = team_payload_to_squad_df(payload, team_name=str(team_name), fixture_row=fixture_row, provider_match_id=event_id)
                            if not df.empty:
                                squads.append(df)
                                attempts.append({"provider": provider, "fixture_id": str(event_id), "kind": f"{side}_{endpoint_kind}", "status": "ok", "rows": int(len(df))})
                                break
                            attempts.append({"provider": provider, "fixture_id": str(event_id), "kind": f"{side}_{endpoint_kind}", "status": "ok_empty", "rows": 0})
                        except Exception as exc:
                            attempts.append({"provider": provider, "fixture_id": str(event_id), "kind": f"{side}_{endpoint_kind}", "status": "error", "error": repr(exc)})
                            if not cfg.fail_soft:
                                raise
        else:
            attempts.append({"provider": provider or "unknown", "fixture_id": str(event_id), "status": "unsupported_provider"})

    lineup_df = pd.concat(lineups, ignore_index=True).drop_duplicates(subset=["match_id", "team", "player"]) if lineups else pd.DataFrame()
    squad_df = pd.concat(squads, ignore_index=True).drop_duplicates(subset=["team", "player"]) if squads else pd.DataFrame()
    audit = {
        "status": "player_inputs_fetched" if (not lineup_df.empty or not squad_df.empty) else "no_player_inputs_available_yet",
        "lineups_rows": int(len(lineup_df)),
        "squads_rows": int(len(squad_df)),
        "attempts": attempts,
        "notes": [
            "Confirmed lineups may only appear close to kickoff.",
            "Squad/roster fallback is broader than an official lineup and should be treated as lower confidence.",
        ],
    }
    return lineup_df, squad_df, audit
