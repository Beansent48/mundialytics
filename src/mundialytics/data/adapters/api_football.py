from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from mundialytics.data.competition_taxonomy import enrich_competition_metadata
from mundialytics.data.identity import canonical_team_name
from mundialytics.data.provider_identity import canonical_provider_player_id, normalize_provider_player_id

API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
PROVIDER = "api_football"

@dataclass
class ApiFootballClient:
    api_key: str | None = None
    base_url: str = API_FOOTBALL_BASE_URL
    timeout: int = 30

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("API_FOOTBALL_KEY") or os.getenv("APISPORTS_KEY")
        if not self.api_key:
            raise ValueError("API-Football key not found. Set API_FOOTBALL_KEY or pass api_key.")

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        ep = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        resp = requests.get(
            f"{self.base_url}{ep}",
            headers={"x-apisports-key": str(self.api_key)},
            params=params or {},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("errors"):
            # API-Football returns errors inside a 200 payload for quota/parameter issues.
            errors = data.get("errors")
            if errors not in ({}, []):
                raise RuntimeError(f"API-Football returned errors: {errors}")
        return data

    def fixtures(self, **params: Any) -> dict[str, Any]:
        return self.get("/fixtures", params=params)

    def fixture_lineups(self, fixture_id: int | str) -> dict[str, Any]:
        return self.get("/fixtures/lineups", params={"fixture": fixture_id})

    def fixture_statistics(self, fixture_id: int | str) -> dict[str, Any]:
        return self.get("/fixtures/statistics", params={"fixture": fixture_id})


def _response_rows(payload: dict[str, Any]) -> list[Any]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("response")
    return rows if isinstance(rows, list) else []


def fixtures_response_to_df(payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in _response_rows(payload):
        fixture = item.get("fixture") or {}
        league = item.get("league") or {}
        teams = item.get("teams") or {}
        goals = item.get("goals") or {}
        score = item.get("score") or {}
        rows.append({
            "provider": PROVIDER,
            "provider_match_id": fixture.get("id"),
            "fixture_id": fixture.get("id"),
            "match_id": f"api_football:{fixture.get('id')}",
            "date": fixture.get("date"),
            "fixture_timezone": fixture.get("timezone"),
            "timestamp": fixture.get("timestamp"),
            "venue_id": (fixture.get("venue") or {}).get("id"),
            "venue_name": (fixture.get("venue") or {}).get("name"),
            "status_short": (fixture.get("status") or {}).get("short"),
            "status_long": (fixture.get("status") or {}).get("long"),
            "league_id": league.get("id"),
            "competition": league.get("name"),
            "season": league.get("season"),
            "round": league.get("round"),
            "home_team_id": (teams.get("home") or {}).get("id"),
            "away_team_id": (teams.get("away") or {}).get("id"),
            "home_team": canonical_team_name((teams.get("home") or {}).get("name")),
            "away_team": canonical_team_name((teams.get("away") or {}).get("name")),
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
            "home_penalty": ((score.get("penalty") or {}).get("home")),
            "away_penalty": ((score.get("penalty") or {}).get("away")),
            "neutral": None,
        })
    df = pd.DataFrame(rows)
    return enrich_competition_metadata(df, overwrite=True) if not df.empty else df


def lineups_response_to_df(payload: dict[str, Any], *, fixture_id: str | int | None = None, date: str | None = None, competition: str | None = None) -> pd.DataFrame:
    """Parse API-Football /fixtures/lineups response into current_lineups format.

    API-Football lineup objects contain team, formation, startXI and substitutes.
    We keep provider ids as first-class columns; these should be used as source
    of truth before fuzzy name matching.
    """
    teams_payload = _response_rows(payload)
    team_names = [canonical_team_name(((t.get("team") or {}).get("name"))) for t in teams_payload]
    rows: list[dict[str, Any]] = []
    match_id = f"api_football:{fixture_id}" if fixture_id is not None else None
    for team_obj in teams_payload:
        team = team_obj.get("team") or {}
        team_name = canonical_team_name(team.get("name"))
        opponent = next((t for t in team_names if t != team_name), None)
        formation = team_obj.get("formation")
        for bucket, started in [("startXI", 1), ("substitutes", 0)]:
            for item in team_obj.get(bucket) or []:
                player = item.get("player") or {}
                pid = normalize_provider_player_id(player.get("id"))
                pname = player.get("name")
                if not pid and not pname:
                    continue
                pos = player.get("pos") or player.get("position")
                rows.append({
                    "provider": PROVIDER,
                    "provider_match_id": fixture_id,
                    "match_id": match_id,
                    "date": date,
                    "competition": competition,
                    "team": team_name,
                    "opponent": opponent,
                    "team_provider_id": team.get("id"),
                    "player": pname,
                    "provider_player_name": pname,
                    "provider_player_id": pid,
                    "api_football_player_id": pid,
                    "canonical_player_id": canonical_provider_player_id(PROVIDER, pid),
                    "number": player.get("number"),
                    "position": pos,
                    "grid": player.get("grid"),
                    "formation": formation,
                    "started": started,
                    "expected_minutes": 75 if started else 25,
                })
    df = pd.DataFrame(rows)
    return enrich_competition_metadata(df, overwrite=True) if not df.empty else df


def statistics_response_to_team_stats(payload: dict[str, Any], *, fixture_id: str | int | None = None, date: str | None = None, competition: str | None = None) -> pd.DataFrame:
    """Parse team fixture statistics from API-Football.

    This is intentionally lightweight; v0.18 team stats can build on it.
    """
    rows: list[dict[str, Any]] = []
    for item in _response_rows(payload):
        team = item.get("team") or {}
        stats = item.get("statistics") or []
        row = {
            "provider": PROVIDER,
            "provider_match_id": fixture_id,
            "match_id": f"api_football:{fixture_id}" if fixture_id is not None else None,
            "date": date,
            "competition": competition,
            "team": canonical_team_name(team.get("name")),
            "team_provider_id": team.get("id"),
        }
        for stat in stats:
            typ = str(stat.get("type") or "").strip().lower().replace(" ", "_")
            val = stat.get("value")
            if typ:
                row[f"api_stat_{typ}"] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    return enrich_competition_metadata(df, overwrite=True) if not df.empty else df
