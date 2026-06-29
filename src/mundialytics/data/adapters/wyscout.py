from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from mundialytics.data.identity import add_player_identity_columns, add_team_identity_columns, canonical_team_name, canonical_player_name

# Common Wyscout public-data tag IDs. The public dataset uses integer tags;
# keeping the IDs here avoids scattering magic numbers across the codebase.
WYS_TAG_GOAL = 101
WYS_TAG_OWN_GOAL = 102
WYS_TAG_ASSIST = 301
WYS_TAG_KEY_PASS = 302
WYS_TAG_YELLOW_CARD = 1702
WYS_TAG_SECOND_YELLOW = 1703
WYS_TAG_RED_CARD = 1701
WYS_TAG_ACCURATE = 1801


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _as_list(obj: Any) -> list[dict]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # Several mirrors wrap data under a top-level key.
        for key in ("events", "matches", "players", "teams", "competitions"):
            if isinstance(obj.get(key), list):
                return obj[key]
    raise ValueError("Expected a JSON list or a dict containing a list of records.")


def _id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value)
    return text[:-2] if text.endswith(".0") else text


def _tags(event: dict) -> set[int]:
    vals: set[int] = set()
    for tag in event.get("tags") or []:
        if isinstance(tag, dict) and "id" in tag:
            try:
                vals.add(int(tag["id"]))
            except (TypeError, ValueError):
                pass
        else:
            try:
                vals.add(int(tag))
            except (TypeError, ValueError):
                pass
    return vals


def _players_map(players_json: str | Path | None) -> dict[str, str]:
    if not players_json:
        return {}
    players = _as_list(_read_json(players_json))
    out = {}
    for p in players:
        pid = _id(p.get("wyId") or p.get("playerId") or p.get("id"))
        if not pid:
            continue
        name = p.get("shortName") or p.get("name") or p.get("lastName") or pid
        out[pid] = canonical_player_name(name)
    return out


def _teams_map(teams_json: str | Path | None) -> dict[str, str]:
    if not teams_json:
        return {}
    teams = _as_list(_read_json(teams_json))
    out = {}
    for t in teams:
        tid = _id(t.get("wyId") or t.get("teamId") or t.get("id"))
        if not tid:
            continue
        name = t.get("name") or t.get("officialName") or tid
        out[tid] = canonical_team_name(name)
    return out


def _matches_meta(matches_json: str | Path | None, teams: dict[str, str]) -> dict[str, dict]:
    if not matches_json:
        return {}
    matches = _as_list(_read_json(matches_json))
    meta: dict[str, dict] = {}
    for m in matches:
        mid = _id(m.get("wyId") or m.get("matchId") or m.get("id"))
        if not mid:
            continue
        team_ids = []
        teams_data = m.get("teamsData") or {}
        if isinstance(teams_data, dict):
            team_ids = [_id(k) for k in teams_data.keys()]
        label = m.get("label") or ""
        date = m.get("dateutc") or m.get("date")
        home_team = away_team = None
        for tid, info in teams_data.items() if isinstance(teams_data, dict) else []:
            side = (info or {}).get("side")
            name = teams.get(_id(tid) or "", _id(tid) or "unknown")
            if side == "home":
                home_team = name
            elif side == "away":
                away_team = name
        if not home_team and len(team_ids) >= 1:
            home_team = teams.get(team_ids[0], team_ids[0])
        if not away_team and len(team_ids) >= 2:
            away_team = teams.get(team_ids[1], team_ids[1])
        meta[mid] = {
            "date": date,
            "label": label,
            "home_team": home_team,
            "away_team": away_team,
            "team_ids": [t for t in team_ids if t],
            "raw": m,
        }
    return meta


def wyscout_events_to_player_events(
    events_json: str | Path,
    *,
    matches_json: str | Path | None = None,
    players_json: str | Path | None = None,
    teams_json: str | Path | None = None,
    competition: str = "Wyscout Public Data",
    season: str | None = None,
    team_scope: str = "club",
    default_minutes: float = 90.0,
) -> pd.DataFrame:
    """Aggregate Wyscout public events into player-match rows.

    The public Wyscout dataset is event-level data. It does not always provide
    reliable minutes in the event file itself, so this function uses
    ``default_minutes`` unless a lineup file is later merged in. Use
    :func:`wyscout_matches_to_lineups` to estimate minutes from match metadata.
    """
    events = _as_list(_read_json(events_json))
    players = _players_map(players_json)
    teams = _teams_map(teams_json)
    matches = _matches_meta(matches_json, teams)
    rows: dict[tuple[str, str, str], dict] = {}

    def row_for(match_id: str, team_id: str, player_id: str) -> dict:
        team = teams.get(team_id, team_id)
        player = players.get(player_id, player_id)
        meta = matches.get(match_id, {})
        opponent = None
        for tid in meta.get("team_ids", []):
            if tid != team_id:
                opponent = teams.get(tid, tid)
                break
        key = (match_id, team, player)
        if key not in rows:
            rows[key] = {
                "match_id": match_id,
                "date": meta.get("date"),
                "competition": competition,
                "season": season,
                "team_scope": team_scope,
                "source": "wyscout_public",
                "team": team,
                "opponent": opponent,
                "player": player,
                "position": None,
                "minutes": default_minutes,
                "shots": 0,
                "shots_on_target": 0,
                "fouls_committed": 0,
                "fouls_drawn": 0,
                "yellow_cards": 0,
                "red_cards": 0,
                "goals": 0,
                "assists": 0,
                "key_passes": 0,
                "passes": 0,
                "accurate_passes": 0,
                "crosses": 0,
                "duels": 0,
                "defensive_duels": 0,
                "interceptions": 0,
                "recoveries": 0,
            }
        return rows[key]

    for ev in events:
        match_id = _id(ev.get("matchId") or ev.get("match_id") or ev.get("wyId"))
        team_id = _id(ev.get("teamId") or ev.get("team_id"))
        player_id = _id(ev.get("playerId") or ev.get("player_id"))
        if not match_id or not team_id or not player_id or player_id == "0":
            continue
        r = row_for(match_id, team_id, player_id)
        event_name = str(ev.get("eventName") or ev.get("event") or "")
        sub_name = str(ev.get("subEventName") or ev.get("sub_event") or "")
        tags = _tags(ev)

        if event_name == "Shot" or "shot" in sub_name.lower():
            r["shots"] += 1
            if WYS_TAG_GOAL in tags or WYS_TAG_ACCURATE in tags:
                r["shots_on_target"] += 1
            if WYS_TAG_GOAL in tags:
                r["goals"] += 1
        elif event_name == "Pass":
            r["passes"] += 1
            if WYS_TAG_ACCURATE in tags:
                r["accurate_passes"] += 1
            if WYS_TAG_ASSIST in tags:
                r["assists"] += 1
            if WYS_TAG_KEY_PASS in tags:
                r["key_passes"] += 1
            if "cross" in sub_name.lower():
                r["crosses"] += 1
        elif event_name == "Foul":
            r["fouls_committed"] += 1
            if WYS_TAG_YELLOW_CARD in tags or WYS_TAG_SECOND_YELLOW in tags:
                r["yellow_cards"] += 1
            if WYS_TAG_RED_CARD in tags:
                r["red_cards"] += 1
        elif event_name == "Duel":
            r["duels"] += 1
            if "defensive" in sub_name.lower():
                r["defensive_duels"] += 1
        elif event_name == "Others on the ball":
            if "interception" in sub_name.lower():
                r["interceptions"] += 1
            if "recovery" in sub_name.lower() or "ball out" not in sub_name.lower():
                r["recoveries"] += 1

    out = pd.DataFrame(rows.values())
    if out.empty:
        return pd.DataFrame(columns=[
            "match_id", "date", "competition", "season", "team_scope", "source", "team", "opponent", "player",
            "position", "minutes", "shots", "shots_on_target", "fouls_committed", "fouls_drawn", "yellow_cards",
            "red_cards", "goals", "assists", "key_passes", "passes", "accurate_passes", "crosses", "duels",
            "defensive_duels", "interceptions", "recoveries", "player_id_global", "player_context_id",
        ])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return add_player_identity_columns(out)


def wyscout_events_to_team_events(
    events_json: str | Path,
    *,
    matches_json: str | Path | None = None,
    teams_json: str | Path | None = None,
    competition: str = "Wyscout Public Data",
    season: str | None = None,
    team_scope: str = "club",
) -> pd.DataFrame:
    """Aggregate Wyscout events into team-match event totals."""
    player_events = wyscout_events_to_player_events(
        events_json,
        matches_json=matches_json,
        teams_json=teams_json,
        competition=competition,
        season=season,
        team_scope=team_scope,
    )
    if player_events.empty:
        return pd.DataFrame()
    agg_cols = [
        "shots", "shots_on_target", "fouls_committed", "yellow_cards", "red_cards", "goals", "assists",
        "key_passes", "passes", "accurate_passes", "crosses", "duels", "defensive_duels", "interceptions", "recoveries",
    ]
    out = player_events.groupby(["match_id", "date", "competition", "season", "team_scope", "team", "opponent"], dropna=False)[agg_cols].sum().reset_index()
    out = out.rename(columns={
        "shots": "shots_for",
        "shots_on_target": "sot_for",
        "fouls_committed": "fouls_for",
        "yellow_cards": "yellow_cards_for",
        "red_cards": "red_cards_for",
        "goals": "goals_for",
    })
    return add_team_identity_columns(out)


def wyscout_matches_to_lineups(
    matches_json: str | Path,
    *,
    players_json: str | Path | None = None,
    teams_json: str | Path | None = None,
    competition: str = "Wyscout Public Data",
    season: str | None = None,
    team_scope: str = "club",
    match_length: int = 90,
) -> pd.DataFrame:
    """Extract approximate lineups/substitutions from Wyscout match metadata.

    Wyscout match records usually contain ``teamsData -> formation`` with
    ``lineup``, ``bench`` and ``substitutions``. This function converts that to
    the internal lineup schema used by the minutes and Sustituto+ modules.
    """
    matches = _as_list(_read_json(matches_json))
    players = _players_map(players_json)
    teams = _teams_map(teams_json)
    rows: list[dict] = []

    for m in matches:
        match_id = _id(m.get("wyId") or m.get("matchId") or m.get("id"))
        if not match_id:
            continue
        date = m.get("dateutc") or m.get("date")
        teams_data = m.get("teamsData") or {}
        if not isinstance(teams_data, dict):
            continue
        for team_id_raw, info in teams_data.items():
            team_id = _id(team_id_raw) or str(team_id_raw)
            team = teams.get(team_id, team_id)
            formation = (info or {}).get("formation") or {}
            lineup = formation.get("lineup") or []
            substitutions = formation.get("substitutions") or []
            subs_by_out = {}
            subs_by_in = {}
            for sub in substitutions:
                p_out = _id(sub.get("playerOut") or sub.get("player_out"))
                p_in = _id(sub.get("playerIn") or sub.get("player_in"))
                minute = sub.get("minute") or sub.get("matchMinute") or sub.get("min")
                try:
                    minute = int(float(minute))
                except (TypeError, ValueError):
                    minute = None
                if p_out:
                    subs_by_out[p_out] = (p_in, minute)
                if p_in:
                    subs_by_in[p_in] = (p_out, minute)
            # starters
            for starter in lineup:
                pid = _id(starter.get("playerId") or starter.get("player_id") or starter.get("wyId"))
                if not pid:
                    continue
                p_in, minute = subs_by_out.get(pid, (None, None))
                rows.append({
                    "match_id": match_id,
                    "date": date,
                    "competition": competition,
                    "season": season,
                    "team_scope": team_scope,
                    "team": team,
                    "player": players.get(pid, pid),
                    "position": starter.get("position") or starter.get("role") or None,
                    "started": 1,
                    "minutes": float(minute if minute is not None else match_length),
                    "replaced_by": players.get(p_in, p_in) if p_in else None,
                    "replacement_minute": minute,
                    "source": "wyscout_public",
                })
            # substitutes who entered
            for pid, (p_out, minute) in subs_by_in.items():
                rows.append({
                    "match_id": match_id,
                    "date": date,
                    "competition": competition,
                    "season": season,
                    "team_scope": team_scope,
                    "team": team,
                    "player": players.get(pid, pid),
                    "position": None,
                    "started": 0,
                    "minutes": float(max(match_length - (minute or match_length), 0)),
                    "replaced_by": None,
                    "replacement_minute": minute,
                    "replaced_player": players.get(p_out, p_out) if p_out else None,
                    "source": "wyscout_public",
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["match_id", "date", "team", "player", "position", "started", "minutes", "replaced_by", "replacement_minute"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = add_team_identity_columns(out)
    return add_player_identity_columns(out)
