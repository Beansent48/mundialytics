from __future__ import annotations

"""Utilities for importing bookmaker-relevant extra match statistics.

The module is intentionally conservative:
- It creates corners/saves targets when present in source data.
- Football-Data saves can be explicitly derived from SOT against minus goals against, but are flagged.
- It labels source and quality so downstream models know whether a stat is real or derived.
- It normalises to team-match rows, because team and match total lines can be built from the
  same table.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Any

import pandas as pd


def norm_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    txt = str(value).strip().lower()
    for ch in ["_", "-", ".", ",", "'", "’", "(", ")", "/"]:
        txt = txt.replace(ch, " ")
    return " ".join(txt.split())


def _to_num(value: object) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return float("nan")
        if isinstance(value, str):
            # API-Football sometimes returns values like '53%'.
            value = value.strip().replace("%", "")
        return float(value)
    except Exception:
        return float("nan")


def _first_present(row: pd.Series, names: Iterable[str]) -> object:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return None


def _date_parse(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    txt = str(value).strip()
    # Football-Data uses dd/mm/yy or dd/mm/YYYY; StatsBomb/API rows commonly use ISO.
    # Parse explicitly to avoid pandas dayfirst warnings and silent month/day flips.
    if "/" in txt:
        dt = pd.to_datetime(txt, errors="coerce", dayfirst=True)
    else:
        dt = pd.to_datetime(txt, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        return txt
    return dt.strftime("%Y-%m-%d")


def _empty_team_match_stats() -> pd.DataFrame:
    cols = [
        "match_id", "date", "competition", "season", "home_team", "away_team", "team", "opponent", "is_home",
        "goals_for", "goals_against", "shots_for", "shots_against", "shots_on_target_for", "shots_on_target_against",
        "corners_for", "corners_against", "saves_for", "saves_against", "saves_data_quality_flag", "yellow_cards_for", "yellow_cards_against",
        "red_cards_for", "red_cards_against", "fouls_for", "fouls_against", "data_source", "data_quality_flag",
    ]
    return pd.DataFrame(columns=cols)


def parse_football_data_csvs(paths: Iterable[Path | str], source_name: str = "football-data.co.uk", derive_saves_from_sot: bool = False) -> pd.DataFrame:
    """Parse Football-Data.co.uk season CSVs into team-match stats rows.

    Useful columns when present:
    FTHG/FTAG goals, HS/AS shots, HST/AST shots on target, HC/AC corners,
    HF/AF fouls, HY/AY yellow cards, HR/AR red cards.

    If derive_saves_from_sot=True, team goalkeeper saves are explicitly estimated as:
    opponent shots_on_target - goals_against. This is useful for bookmaker-style
    goalkeeper-save research but is lower quality than provider saves, so it is flagged.
    """
    rows: list[dict[str, Any]] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin1")
        except Exception:
            continue
        # Keep empty trailing rows out.
        df = df.dropna(how="all")
        for i, r in df.iterrows():
            home = _first_present(r, ["HomeTeam", "Home", "home_team"])
            away = _first_present(r, ["AwayTeam", "Away", "away_team"])
            if not home or not away or pd.isna(home) or pd.isna(away):
                continue
            date = _date_parse(_first_present(r, ["Date", "date", "MatchDate"]))
            div = _first_present(r, ["Div", "League", "competition", "Competition"])
            season = _first_present(r, ["Season", "season"])
            match_id = _first_present(r, ["match_id", "MatchID", "fixture_id"])
            if not match_id:
                match_id = f"fd:{norm_text(div)}:{date}:{norm_text(home)}:{norm_text(away)}"
            vals = {
                "home_goals": _to_num(_first_present(r, ["FTHG", "HG", "HomeGoals"])),
                "away_goals": _to_num(_first_present(r, ["FTAG", "AG", "AwayGoals"])),
                "home_shots": _to_num(_first_present(r, ["HS", "HomeShots", "home_shots"])),
                "away_shots": _to_num(_first_present(r, ["AS", "AwayShots", "away_shots"])),
                "home_sot": _to_num(_first_present(r, ["HST", "HomeShotsOnTarget", "home_shots_on_target"])),
                "away_sot": _to_num(_first_present(r, ["AST", "AwayShotsOnTarget", "away_shots_on_target"])),
                "home_corners": _to_num(_first_present(r, ["HC", "HomeCorners", "home_corners"])),
                "away_corners": _to_num(_first_present(r, ["AC", "AwayCorners", "away_corners"])),
                "home_fouls": _to_num(_first_present(r, ["HF", "HomeFouls", "home_fouls"])),
                "away_fouls": _to_num(_first_present(r, ["AF", "AwayFouls", "away_fouls"])),
                "home_yellow": _to_num(_first_present(r, ["HY", "HomeYellowCards", "home_yellow_cards"])),
                "away_yellow": _to_num(_first_present(r, ["AY", "AwayYellowCards", "away_yellow_cards"])),
                "home_red": _to_num(_first_present(r, ["HR", "HomeRedCards", "home_red_cards"])),
                "away_red": _to_num(_first_present(r, ["AR", "AwayRedCards", "away_red_cards"])),
                # Football-Data does not usually include goalkeeper saves.
                "home_saves": _to_num(_first_present(r, ["HSaves", "HomeSaves", "home_saves", "home_goalkeeper_saves"])),
                "away_saves": _to_num(_first_present(r, ["ASaves", "AwaySaves", "away_saves", "away_goalkeeper_saves"])),
            }
            saves_quality = "provider_saves_real"
            if derive_saves_from_sot:
                # Explicitly derived approximation: goalkeeper saves for a team are the opponent's
                # shots on target minus the goals conceded by that team. Keep it non-negative and
                # flag it so downstream evaluation can separate it from real provider saves.
                if pd.isna(vals["home_saves"]) and pd.notna(vals["away_sot"]) and pd.notna(vals["away_goals"]):
                    vals["home_saves"] = max(0.0, vals["away_sot"] - vals["away_goals"])
                    saves_quality = "derived_saves_from_sot_minus_goals"
                if pd.isna(vals["away_saves"]) and pd.notna(vals["home_sot"]) and pd.notna(vals["home_goals"]):
                    vals["away_saves"] = max(0.0, vals["home_sot"] - vals["home_goals"])
                    saves_quality = "derived_saves_from_sot_minus_goals"
            if not derive_saves_from_sot and pd.isna(vals["home_saves"]) and pd.isna(vals["away_saves"]):
                saves_quality = "saves_not_available"
            for side in ["home", "away"]:
                opp = "away" if side == "home" else "home"
                row_quality = "provider_boxscore_real_stats"
                if saves_quality == "derived_saves_from_sot_minus_goals":
                    row_quality += ";goalkeeper_saves_derived_sot_minus_goals"
                rows.append({
                    "match_id": str(match_id),
                    "date": date,
                    "competition": str(div) if div is not None and not pd.isna(div) else "",
                    "season": str(season) if season is not None and not pd.isna(season) else "",
                    "home_team": norm_text(home),
                    "away_team": norm_text(away),
                    "team": norm_text(home if side == "home" else away),
                    "opponent": norm_text(away if side == "home" else home),
                    "is_home": int(side == "home"),
                    "goals_for": vals[f"{side}_goals"],
                    "goals_against": vals[f"{opp}_goals"],
                    "shots_for": vals[f"{side}_shots"],
                    "shots_against": vals[f"{opp}_shots"],
                    "shots_on_target_for": vals[f"{side}_sot"],
                    "shots_on_target_against": vals[f"{opp}_sot"],
                    "corners_for": vals[f"{side}_corners"],
                    "corners_against": vals[f"{opp}_corners"],
                    "saves_for": vals[f"{side}_saves"],
                    "saves_against": vals[f"{opp}_saves"],
                    "saves_data_quality_flag": saves_quality,
                    "yellow_cards_for": vals[f"{side}_yellow"],
                    "yellow_cards_against": vals[f"{opp}_yellow"],
                    "red_cards_for": vals[f"{side}_red"],
                    "red_cards_against": vals[f"{opp}_red"],
                    "fouls_for": vals[f"{side}_fouls"],
                    "fouls_against": vals[f"{opp}_fouls"],
                    "data_source": source_name,
                    "data_quality_flag": row_quality,
                })
    out = pd.DataFrame(rows) if rows else _empty_team_match_stats()
    return clean_team_match_market_stats(out)


def _api_stat_map(stats: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in stats or []:
        name = norm_text(item.get("type") or item.get("name") or item.get("displayName"))
        value = _to_num(item.get("value"))
        if "corner" in name:
            out["corners"] = value
        elif "goalkeeper" in name and "save" in name:
            out["saves"] = value
        elif name == "saves" or "saves" == name:
            out["saves"] = value
        elif "shots on goal" in name or "shots on target" in name:
            out["shots_on_target"] = value
        elif name in {"total shots", "shots total", "shots"}:
            out["shots"] = value
        elif "yellow" in name:
            out["yellow_cards"] = value
        elif "red" in name:
            out["red_cards"] = value
        elif "foul" in name:
            out["fouls"] = value
    return out


def parse_api_football_fixture_stats_json(paths: Iterable[Path | str], source_name: str = "api-football") -> pd.DataFrame:
    """Parse saved API-Football fixture statistics JSON into team-match stats.

    Expected shape can be either raw API response with response=[{team:{name}, statistics:[...]}], or
    a list/dict containing fixture/match metadata and team statistics.
    """
    rows: list[dict[str, Any]] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        fixture_id = ""
        date = ""
        competition = ""
        # `season` is emitted in the row dict below; without this initialisation
        # the function raised NameError on its first fixture, so the whole
        # API-Football import path (scripts/import_provider_fixture_stats.py)
        # was dead on arrival. Read from the league block like the sibling
        # parsers in this module do.
        season = ""
        teams_meta: dict[str, str] = {}
        # Direct API-Football endpoint response.
        if isinstance(payload, dict):
            fixture = payload.get("fixture") or payload.get("match") or {}
            if isinstance(fixture, dict):
                fixture_id = str(fixture.get("id") or fixture.get("fixture_id") or path.stem)
                date = _date_parse(fixture.get("date") or fixture.get("timestamp"))
            league = payload.get("league") or payload.get("competition") or {}
            if isinstance(league, dict):
                competition = str(league.get("name") or league.get("id") or "")
                season = str(league.get("season_name") or league.get("season") or "")
            teams = payload.get("teams") or {}
            if isinstance(teams, dict):
                home = teams.get("home") or {}
                away = teams.get("away") or {}
                if isinstance(home, dict):
                    teams_meta["home_team"] = norm_text(home.get("name"))
                if isinstance(away, dict):
                    teams_meta["away_team"] = norm_text(away.get("name"))
            response = payload.get("response") if isinstance(payload.get("response"), list) else None
        else:
            response = payload if isinstance(payload, list) else None
        if response is None:
            response = []
        parsed_teams = []
        for block in response:
            if not isinstance(block, dict):
                continue
            team_obj = block.get("team") or {}
            team_name = team_obj.get("name") if isinstance(team_obj, dict) else block.get("team")
            stats = block.get("statistics") or block.get("stats") or []
            parsed_teams.append((norm_text(team_name), _api_stat_map(stats)))
        if len(parsed_teams) < 2:
            continue
        # Use first two teams. API-Football response is home/away ordered in practice.
        home_team = teams_meta.get("home_team") or parsed_teams[0][0]
        away_team = teams_meta.get("away_team") or parsed_teams[1][0]
        stats_by_team = {parsed_teams[0][0]: parsed_teams[0][1], parsed_teams[1][0]: parsed_teams[1][1]}
        for team, opp, is_home in [(home_team, away_team, 1), (away_team, home_team, 0)]:
            st = stats_by_team.get(team, {})
            op = stats_by_team.get(opp, {})
            rows.append({
                "match_id": str(fixture_id or f"api:{path.stem}"),
                "date": date,
                "competition": competition,
                "season": season,
                "home_team": home_team,
                "away_team": away_team,
                "team": team,
                "opponent": opp,
                "is_home": is_home,
                "goals_for": float("nan"),
                "goals_against": float("nan"),
                "shots_for": st.get("shots", float("nan")),
                "shots_against": op.get("shots", float("nan")),
                "shots_on_target_for": st.get("shots_on_target", float("nan")),
                "shots_on_target_against": op.get("shots_on_target", float("nan")),
                "corners_for": st.get("corners", float("nan")),
                "corners_against": op.get("corners", float("nan")),
                "saves_for": st.get("saves", float("nan")),
                "saves_against": op.get("saves", float("nan")),
                "saves_data_quality_flag": "provider_saves_real" if pd.notna(st.get("saves", float("nan"))) else "saves_not_available",
                "yellow_cards_for": st.get("yellow_cards", float("nan")),
                "yellow_cards_against": op.get("yellow_cards", float("nan")),
                "red_cards_for": st.get("red_cards", float("nan")),
                "red_cards_against": op.get("red_cards", float("nan")),
                "fouls_for": st.get("fouls", float("nan")),
                "fouls_against": op.get("fouls", float("nan")),
                "data_source": source_name,
                "data_quality_flag": "provider_fixture_stats_real_stats",
            })
    out = pd.DataFrame(rows) if rows else _empty_team_match_stats()
    return clean_team_match_market_stats(out)


def _get_name(obj: Any, *keys: str) -> str:
    cur = obj
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return ""
    if isinstance(cur, dict):
        return norm_text(cur.get("name") or cur.get("id"))
    return norm_text(cur)


def parse_statsbomb_event_json(paths: Iterable[Path | str], source_name: str = "statsbomb_raw_events") -> pd.DataFrame:
    """Parse StatsBomb raw event JSON files for corners and goalkeeper saves.

    This does not need shot/SOT proxies. Corners are counted from Pass type Corner.
    Saves are counted from goalkeeper events whose type/outcome contains saved/save.
    If metadata is missing, match_id comes from filename and teams are inferred from event team names.
    """
    match_rows: list[dict[str, Any]] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        try:
            events = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sidecar = _load_sidecar_metadata(path)
        if isinstance(events, dict) and isinstance(events.get("events"), list):
            meta = {**sidecar, **events}
            events_list = events.get("events", [])
            match_id = str(meta.get("match_id") or meta.get("id") or path.stem.replace("fixture_", ""))
            date = _date_parse(meta.get("match_date") or meta.get("date"))
            competition = str(meta.get("competition_name") or meta.get("competition") or "")
            season = str(meta.get("season_name") or meta.get("season") or "")
            home_team_meta = norm_text(meta.get("home_team") or meta.get("home_team_name") or "")
            away_team_meta = norm_text(meta.get("away_team") or meta.get("away_team_name") or "")
        elif isinstance(events, list):
            events_list = events
            meta = sidecar
            match_id = str(meta.get("match_id") or path.stem.replace("fixture_", ""))
            date = _date_parse(meta.get("match_date") or meta.get("date"))
            competition = str(meta.get("competition_name") or meta.get("competition") or "")
            season = str(meta.get("season_name") or meta.get("season") or "")
            home_team_meta = norm_text(meta.get("home_team") or meta.get("home_team_name") or "")
            away_team_meta = norm_text(meta.get("away_team") or meta.get("away_team_name") or "")
        else:
            continue
        event_teams = sorted({norm_text(_get_name(e, "team", "name")) for e in events_list if isinstance(e, dict) and _get_name(e, "team", "name")})
        if home_team_meta and away_team_meta:
            teams = [home_team_meta, away_team_meta]
        else:
            teams = event_teams
        if len(teams) < 2:
            continue
        stats = {t: {"corners_for": 0, "saves_for": 0, "shots_for": 0, "shots_on_target_for": 0} for t in teams}
        for e in events_list:
            if not isinstance(e, dict):
                continue
            team = norm_text(_get_name(e, "team", "name"))
            if not team or team not in stats:
                continue
            typ = norm_text(_get_name(e, "type", "name"))
            if typ == "pass":
                pass_type = norm_text(_get_name(e, "pass", "type", "name"))
                if pass_type == "corner" or "corner" in pass_type:
                    stats[team]["corners_for"] += 1
            if typ == "shot":
                stats[team]["shots_for"] += 1
                outcome = norm_text(_get_name(e, "shot", "outcome", "name"))
                # StatsBomb shot outcomes include Saved, Saved to Post, Goal etc. Any saved shot was on target.
                if outcome in {"goal", "saved", "saved to post", "saved off target"} or "saved" in outcome:
                    stats[team]["shots_on_target_for"] += 1
            if typ in {"goal keeper", "goalkeeper"}:
                gk_type = norm_text(_get_name(e, "goalkeeper", "type", "name"))
                outcome = norm_text(_get_name(e, "goalkeeper", "outcome", "name"))
                text = f"{gk_type} {outcome}"
                if "saved" in text or "save" in text:
                    stats[team]["saves_for"] += 1
        # If more than two teams somehow exist, use first two with most events.
        if len(teams) > 2:
            event_counts = {t: sum(1 for e in events_list if isinstance(e, dict) and norm_text(_get_name(e, "team", "name")) == t) for t in teams}
            teams = sorted(teams, key=lambda t: event_counts.get(t, 0), reverse=True)[:2]
        home_team, away_team = teams[0], teams[1]
        for team, opp, is_home in [(home_team, away_team, 1), (away_team, home_team, 0)]:
            st = stats.get(team, {})
            op = stats.get(opp, {})
            match_rows.append({
                "match_id": str(match_id),
                "date": date,
                "competition": competition,
                "season": "",
                "home_team": home_team,
                "away_team": away_team,
                "team": team,
                "opponent": opp,
                "is_home": is_home,
                "goals_for": float("nan"),
                "goals_against": float("nan"),
                "shots_for": st.get("shots_for", float("nan")),
                "shots_against": op.get("shots_for", float("nan")),
                "shots_on_target_for": st.get("shots_on_target_for", float("nan")),
                "shots_on_target_against": op.get("shots_on_target_for", float("nan")),
                "corners_for": st.get("corners_for", float("nan")),
                "corners_against": op.get("corners_for", float("nan")),
                "saves_for": st.get("saves_for", float("nan")),
                "saves_against": op.get("saves_for", float("nan")),
                "saves_data_quality_flag": "raw_event_goalkeeper_saves",
                "yellow_cards_for": float("nan"),
                "yellow_cards_against": float("nan"),
                "red_cards_for": float("nan"),
                "red_cards_against": float("nan"),
                "fouls_for": float("nan"),
                "fouls_against": float("nan"),
                "data_source": source_name,
                "data_quality_flag": "raw_event_derived_real_events",
            })
    out = pd.DataFrame(match_rows) if match_rows else _empty_team_match_stats()
    return clean_team_match_market_stats(out)




def _empty_goalkeeper_match_stats() -> pd.DataFrame:
    cols = [
        "match_id", "date", "competition", "season", "home_team", "away_team",
        "team", "opponent", "goalkeeper", "player", "goalkeeper_id", "is_home",
        "saves", "shots_on_target_against", "goals_against", "team_saves_total",
        "data_source", "data_quality_flag", "saves_data_quality_flag",
    ]
    return pd.DataFrame(columns=cols)


def _statsbomb_event_type(e: dict[str, Any]) -> str:
    return norm_text(_get_name(e, "type", "name"))


def _statsbomb_player_name(e: dict[str, Any]) -> str:
    return norm_text(_get_name(e, "player", "name"))


def _statsbomb_player_id(e: dict[str, Any]) -> str:
    cur = e.get("player") if isinstance(e, dict) else None
    if isinstance(cur, dict):
        return str(cur.get("id") or "")
    return ""


def _load_sidecar_metadata(path: Path) -> dict[str, Any]:
    """Load optional metadata saved by the StatsBomb downloader.

    Raw StatsBomb event files are just a list of events and do not contain match date,
    home/away labels or competition. The downloader writes fixture_123.metadata.json next
    to the event file when possible. This function also accepts 123.metadata.json.
    """
    candidates = [
        path.with_suffix(".metadata.json"),
        path.with_name(f"{path.stem}.metadata.json"),
        path.with_name(f"fixture_{path.stem}.metadata.json"),
    ]
    for c in candidates:
        if c.exists():
            try:
                return json.loads(c.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def parse_statsbomb_goalkeeper_match_json(paths: Iterable[Path | str], source_name: str = "statsbomb_raw_events") -> pd.DataFrame:
    """Parse StatsBomb raw event JSON files into goalkeeper-match saves rows.

    Unlike team-level derived saves, this function identifies the goalkeeper when the raw
    event feed contains Starting XI or Goal Keeper events. Rows with zero saves are added
    for starting goalkeepers when Starting XI is present. This prevents a biased dataset
    containing only goalkeepers who made at least one save.
    """
    rows: list[dict[str, Any]] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sidecar = _load_sidecar_metadata(path)
        if isinstance(payload, dict) and isinstance(payload.get("events"), list):
            events_list = payload.get("events", [])
            meta = {**sidecar, **payload}
        elif isinstance(payload, list):
            events_list = payload
            meta = sidecar
        else:
            continue
        match_id = str(meta.get("match_id") or meta.get("id") or path.stem.replace("fixture_", ""))
        date = _date_parse(meta.get("match_date") or meta.get("date"))
        competition = str(meta.get("competition_name") or meta.get("competition") or "")
        season = str(meta.get("season_name") or meta.get("season") or "")
        home_team = norm_text(meta.get("home_team") or meta.get("home_team_name") or "")
        away_team = norm_text(meta.get("away_team") or meta.get("away_team_name") or "")
        event_teams = sorted({norm_text(_get_name(e, "team", "name")) for e in events_list if isinstance(e, dict) and _get_name(e, "team", "name")})
        if not home_team or not away_team:
            if len(event_teams) >= 2:
                home_team = home_team or event_teams[0]
                away_team = away_team or event_teams[1]
        teams = [t for t in [home_team, away_team] if t]
        if len(teams) < 2:
            teams = event_teams[:2]
        if len(teams) < 2:
            continue
        home_team, away_team = teams[0], teams[1]
        # Team attacking summary needed to contextualise goalkeeper saves.
        team_attack = {home_team: {"sot": 0, "goals": 0}, away_team: {"sot": 0, "goals": 0}}
        starting_gks: dict[str, dict[str, str]] = {home_team: {}, away_team: {}}
        saves_by_gk: dict[tuple[str, str, str], int] = {}
        for e in events_list:
            if not isinstance(e, dict):
                continue
            team = norm_text(_get_name(e, "team", "name"))
            if team not in team_attack:
                continue
            typ = _statsbomb_event_type(e)
            if typ == "starting xi":
                tactics = e.get("tactics") if isinstance(e.get("tactics"), dict) else {}
                lineup = tactics.get("lineup") if isinstance(tactics.get("lineup"), list) else []
                for item in lineup:
                    if not isinstance(item, dict):
                        continue
                    pos = norm_text(_get_name(item, "position", "name"))
                    if "goalkeeper" not in pos:
                        continue
                    player_obj = item.get("player") if isinstance(item.get("player"), dict) else {}
                    name = norm_text(player_obj.get("name"))
                    pid = str(player_obj.get("id") or "")
                    if name:
                        starting_gks.setdefault(team, {})[pid or name] = name
            elif typ == "shot":
                outcome = norm_text(_get_name(e, "shot", "outcome", "name"))
                if outcome in {"goal", "saved", "saved to post", "saved off target"} or "saved" in outcome:
                    team_attack[team]["sot"] += 1
                if outcome == "goal":
                    team_attack[team]["goals"] += 1
            elif typ in {"goal keeper", "goalkeeper"}:
                gk_type = norm_text(_get_name(e, "goalkeeper", "type", "name"))
                outcome = norm_text(_get_name(e, "goalkeeper", "outcome", "name"))
                text = f"{gk_type} {outcome}"
                if "saved" in text or "save" in text:
                    gk_name = _statsbomb_player_name(e)
                    gk_id = _statsbomb_player_id(e)
                    key = (team, gk_id or gk_name, gk_name)
                    saves_by_gk[key] = saves_by_gk.get(key, 0) + 1
        # Candidate GK rows: all starting GKs plus any GK who recorded a save.
        candidates: dict[str, dict[str, tuple[str, str]]] = {home_team: {}, away_team: {}}
        for team, mapping in starting_gks.items():
            for pid_or_name, name in mapping.items():
                candidates.setdefault(team, {})[pid_or_name] = (pid_or_name if pid_or_name != name else "", name)
        for (team, pid_or_name, name), count in saves_by_gk.items():
            if name:
                candidates.setdefault(team, {})[pid_or_name or name] = (pid_or_name if pid_or_name != name else "", name)
        for team, opp, is_home in [(home_team, away_team, 1), (away_team, home_team, 0)]:
            opp_sot = float(team_attack.get(opp, {}).get("sot", 0))
            opp_goals = float(team_attack.get(opp, {}).get("goals", 0))
            team_total_saves = sum(v for (t, _pid, _name), v in saves_by_gk.items() if t == team)
            for pid_key, (pid, name) in candidates.get(team, {}).items():
                saves = 0
                for (t, pkey, pname), v in saves_by_gk.items():
                    if t == team and (pkey == pid_key or pname == name):
                        saves += int(v)
                rows.append({
                    "match_id": str(match_id),
                    "date": date,
                    "competition": competition,
                    "season": season,
                    "home_team": home_team,
                    "away_team": away_team,
                    "team": team,
                    "opponent": opp,
                    "goalkeeper": name,
                    "player": name,
                    "goalkeeper_id": str(pid or pid_key or ""),
                    "is_home": int(is_home),
                    "saves": float(saves),
                    "shots_on_target_against": opp_sot,
                    "goals_against": opp_goals,
                    "team_saves_total": float(team_total_saves),
                    "data_source": source_name,
                    "data_quality_flag": "raw_event_goalkeeper_saves;starting_xi_zero_rows_when_available",
                    "saves_data_quality_flag": "raw_event_goalkeeper_saves",
                })
    out = pd.DataFrame(rows) if rows else _empty_goalkeeper_match_stats()
    return clean_goalkeeper_match_stats(out)


def parse_api_football_fixture_player_stats_json(paths: Iterable[Path | str], source_name: str = "api-football-fixture-players") -> pd.DataFrame:
    """Parse API-Football fixture player statistics into goalkeeper saves rows.

    The exact payload shape varies slightly by API endpoint. This parser accepts common
    fixtures/players responses and looks for player position G/GK/Goalkeeper and saves
    fields in nested statistics.
    """
    rows: list[dict[str, Any]] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        fixture = payload.get("fixture") if isinstance(payload, dict) and isinstance(payload.get("fixture"), dict) else {}
        league = payload.get("league") if isinstance(payload, dict) and isinstance(payload.get("league"), dict) else {}
        teams_meta = payload.get("teams") if isinstance(payload, dict) and isinstance(payload.get("teams"), dict) else {}
        match_id = str(fixture.get("id") or payload.get("fixture_id") if isinstance(payload, dict) else path.stem)
        if not match_id or match_id == "None":
            match_id = path.stem
        date = _date_parse(fixture.get("date"))
        competition = str(league.get("name") or league.get("id") or "")
        season = str(league.get("season") or "")
        home_team = norm_text((teams_meta.get("home") or {}).get("name")) if isinstance(teams_meta.get("home"), dict) else ""
        away_team = norm_text((teams_meta.get("away") or {}).get("name")) if isinstance(teams_meta.get("away"), dict) else ""
        response = payload.get("response") if isinstance(payload, dict) and isinstance(payload.get("response"), list) else payload if isinstance(payload, list) else []
        for team_block in response:
            if not isinstance(team_block, dict):
                continue
            team_obj = team_block.get("team") if isinstance(team_block.get("team"), dict) else {}
            team = norm_text(team_obj.get("name") or team_block.get("team"))
            players = team_block.get("players") if isinstance(team_block.get("players"), list) else []
            if not team:
                continue
            if not home_team and len(response) >= 1:
                home_team = team
            for player_block in players:
                if not isinstance(player_block, dict):
                    continue
                player_obj = player_block.get("player") if isinstance(player_block.get("player"), dict) else {}
                player_name = norm_text(player_obj.get("name") or player_block.get("player"))
                player_id = str(player_obj.get("id") or "")
                stats_list = player_block.get("statistics") if isinstance(player_block.get("statistics"), list) else []
                for st in stats_list or [{}]:
                    games = st.get("games") if isinstance(st.get("games"), dict) else {}
                    pos = norm_text(games.get("position") or st.get("position"))
                    if pos and pos not in {"g", "gk", "goalkeeper", "keeper"} and "goalkeeper" not in pos:
                        continue
                    goals = st.get("goals") if isinstance(st.get("goals"), dict) else {}
                    saves = _to_num(goals.get("saves") or st.get("saves") or st.get("goalkeeper_saves"))
                    if pd.isna(saves):
                        continue
                    rows.append({
                        "match_id": str(match_id),
                        "date": date,
                        "competition": competition,
                        "season": season,
                        "home_team": home_team,
                        "away_team": away_team,
                        "team": team,
                        "opponent": "",
                        "goalkeeper": player_name,
                        "player": player_name,
                        "goalkeeper_id": player_id,
                        "is_home": int(team == home_team) if home_team else float("nan"),
                        "saves": saves,
                        "shots_on_target_against": float("nan"),
                        "goals_against": float("nan"),
                        "team_saves_total": saves,
                        "data_source": source_name,
                        "data_quality_flag": "provider_player_goalkeeper_saves",
                        "saves_data_quality_flag": "provider_player_goalkeeper_saves_real",
                    })
        # Fill opponent from known home/away.
        for r in rows:
            if r.get("match_id") == str(match_id) and not r.get("opponent") and home_team and away_team:
                r["opponent"] = away_team if r.get("team") == home_team else home_team
                r["away_team"] = away_team
                r["home_team"] = home_team
    out = pd.DataFrame(rows) if rows else _empty_goalkeeper_match_stats()
    return clean_goalkeeper_match_stats(out)


def clean_goalkeeper_match_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_goalkeeper_match_stats()
    out = df.copy()
    for c in _empty_goalkeeper_match_stats().columns:
        if c not in out.columns:
            out[c] = "" if c in {"match_id", "date", "competition", "season", "home_team", "away_team", "team", "opponent", "goalkeeper", "player", "goalkeeper_id", "data_source", "data_quality_flag", "saves_data_quality_flag"} else float("nan")
    for c in ["home_team", "away_team", "team", "opponent", "goalkeeper", "player"]:
        out[c] = out[c].map(norm_text)
    for c in ["is_home", "saves", "shots_on_target_against", "goals_against", "team_saves_total"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out[_empty_goalkeeper_match_stats().columns].copy()
    out = out.drop_duplicates(["match_id", "team", "goalkeeper"], keep="last")
    return out

def clean_team_match_market_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_team_match_stats()
    out = df.copy()
    for c in _empty_team_match_stats().columns:
        if c not in out.columns:
            out[c] = "" if c in {"match_id", "date", "competition", "season", "home_team", "away_team", "team", "opponent", "data_source", "data_quality_flag", "saves_data_quality_flag"} else float("nan")
    for c in ["home_team", "away_team", "team", "opponent"]:
        out[c] = out[c].map(norm_text)
    for c in [
        "goals_for", "goals_against", "shots_for", "shots_against", "shots_on_target_for", "shots_on_target_against",
        "corners_for", "corners_against", "saves_for", "saves_against", "yellow_cards_for", "yellow_cards_against",
        "red_cards_for", "red_cards_against", "fouls_for", "fouls_against", "is_home",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out[_empty_team_match_stats().columns].copy()
    out = out.drop_duplicates(["match_id", "team", "opponent", "is_home"], keep="last")
    return out


def combine_team_match_stats(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    valid = [f for f in frames if f is not None and not f.empty]
    if not valid:
        return _empty_team_match_stats()
    out = pd.concat(valid, ignore_index=True, sort=False)
    return clean_team_match_market_stats(out)
