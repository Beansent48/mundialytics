from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from mundialytics.data.identity import add_player_identity_columns, add_team_identity_columns, canonical_team_name


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _event_type(ev: dict) -> str | None:
    return (ev.get("type") or {}).get("name")


def _team(ev: dict) -> str | None:
    return (ev.get("team") or {}).get("name")


def _player(ev: dict) -> str | None:
    return (ev.get("player") or {}).get("name")


def _player_id(ev: dict) -> str | None:
    pid = (ev.get("player") or {}).get("id")
    return None if pid is None else str(pid)


# StatsBomb pitch is 120x80 (yards), opponent goal at (120, 40). Used to
# derive ball-PROGRESSION metrics (progressive passes/carries, entries into
# the final third / penalty area) from each event's start/end coordinates.
# These capture "plays the team forward", which the duel/volume/finishing
# stats all miss for deep controllers (Modrić, Xavi, Kroos) and ball-playing
# defenders alike -- position-general, not just a midfielder fix.
_GOAL_XY = (120.0, 40.0)
_PROG_PASS_MIN = 10.0   # yards a completed pass must gain toward goal to count
_PROG_CARRY_MIN = 5.0   # yards a carry must gain toward goal to count (carries are shorter)


def _dist_to_goal(loc: Any) -> float | None:
    if not loc or len(loc) < 2:
        return None
    return ((loc[0] - _GOAL_XY[0]) ** 2 + (loc[1] - _GOAL_XY[1]) ** 2) ** 0.5


def _in_box(loc: Any) -> bool:
    """True if a StatsBomb [x, y] location is inside the opponent penalty area."""
    return bool(loc) and len(loc) >= 2 and loc[0] >= 102.0 and 18.0 <= loc[1] <= 62.0


def statsbomb_open_data_match_metadata(data_root: str | Path) -> dict[str, dict]:
    """Scan a StatsBomb Open Data root and return match_id -> metadata.

    Expected root can be either the repository ``data`` directory or a parent
    containing ``data/matches``. This lets build_event_datasets.py attach real
    match dates/competition names to event aggregates instead of leaving dates
    empty.
    """
    root = Path(data_root)
    if (root / "data" / "matches").exists():
        root = root / "data"
    matches_dir = root / "matches"
    out: dict[str, dict] = {}
    if not matches_dir.exists():
        return out
    competitions = {}
    comp_file = root / "competitions.json"
    if comp_file.exists():
        try:
            for row in _read_json(comp_file):
                competitions[(str(row.get("competition_id")), str(row.get("season_id")))] = {
                    "competition": row.get("competition_name"),
                    "season": row.get("season_name"),
                }
        except Exception:
            competitions = {}
    for fp in matches_dir.glob("*/*.json"):
        comp_id = fp.parent.name
        season_id = fp.stem
        comp_meta = competitions.get((comp_id, season_id), {})
        try:
            matches = _read_json(fp)
        except Exception:
            continue
        for m in matches if isinstance(matches, list) else []:
            mid = m.get("match_id")
            if mid is None:
                continue
            out[str(mid)] = {
                "match_id": str(mid),
                "date": m.get("match_date"),
                "kick_off": m.get("kick_off"),
                "competition": comp_meta.get("competition") or ((m.get("competition") or {}).get("competition_name")),
                "season": comp_meta.get("season") or ((m.get("season") or {}).get("season_name")),
                "home_team": ((m.get("home_team") or {}).get("home_team_name")),
                "away_team": ((m.get("away_team") or {}).get("away_team_name")),
            }
    return out


def statsbomb_events_to_player_events(events_json: str | Path, *, match_id: str | int, date: str | None = None, team_scope: str = "club", competition: str = "StatsBomb Open Data") -> pd.DataFrame:
    """Aggregate a StatsBomb events JSON file into player-event rows.

    It extracts freely available event markets: shots, shots on target, fouls
    committed/drawn, cards, goals, assists, passes and pressure/duel proxies.
    Minutes should be merged from ``statsbomb_events_to_lineups`` when possible.
    """
    events = _read_json(events_json)
    rows: dict[tuple[str, str], dict] = {}

    def get_row(team: str, player: str) -> dict:
        key = (canonical_team_name(team), str(player))
        if key not in rows:
            rows[key] = {
                "match_id": match_id,
                "date": date,
                "competition": competition,
                "team_scope": team_scope,
                "source": "statsbomb_open_data",
                "provider": "statsbomb",
                "team": key[0],
                "opponent": None,
                "player": player,
                "provider_player_id": None,
                "statsbomb_player_id": None,
                "position": None,
                "minutes": float("nan"),
                "shots": 0,
                "shots_on_target": 0,
                "fouls_committed": 0,
                "fouls_drawn": 0,
                "yellow_cards": 0,
                "red_cards": 0,
                "goals": 0,
                "assists": 0,
                "passes": 0,
                "complete_passes": 0,
                "key_passes": 0,
                "pressures": 0,
                "duels": 0,
                "duels_won": 0,
                "duels_lost": 0,
                "dribbles": 0,
                "successful_dribbles": 0,
                "tackles": 0,
                "interceptions": 0,
                "ball_recoveries": 0,
                "dribbled_past": 0,
                "clearances": 0,
                "blocks": 0,
                "progressive_passes": 0,
                "passes_into_final_third": 0,
                "passes_into_box": 0,
                "through_balls": 0,
                "carries": 0,
                "progressive_carries": 0,
                "passes_under_pressure": 0,
                "complete_passes_under_pressure": 0,
                "aerials_won": 0,
                "aerials_lost": 0,
                "crosses": 0,
                "cut_backs": 0,
            }
        return rows[key]

    teams_seen = sorted({canonical_team_name(_team(ev)) for ev in events if _team(ev)})

    for ev in events:
        team = _team(ev)
        player = _player(ev)
        if not team or not player:
            continue
        r = get_row(team, player)
        pid = _player_id(ev)
        if pid and not r.get("provider_player_id"):
            r["provider_player_id"] = pid
            r["statsbomb_player_id"] = pid
        if ev.get("position"):
            r["position"] = ev["position"].get("name")
        typ = _event_type(ev)
        if typ == "Shot":
            r["shots"] += 1
            outcome = ((ev.get("shot") or {}).get("outcome") or {}).get("name")
            # Bookmaker-like SOT criterion. Betfair/Opta count goals and
            # shots saved by the goalkeeper. StatsBomb does not reliably expose
            # "last defender block that prevents a goal" in the open JSON, so
            # ordinary Blocked shots are NOT counted as SOT here.
            if outcome in {"Goal", "Saved", "Saved to Post"}:
                r["shots_on_target"] += 1
            if outcome == "Goal":
                r["goals"] += 1
        elif typ == "Foul Committed":
            r["fouls_committed"] += 1
            card = ((ev.get("foul_committed") or {}).get("card") or {}).get("name")
            if card and "Yellow" in card:
                r["yellow_cards"] += 1
            if card and "Red" in card:
                r["red_cards"] += 1
        elif typ == "Foul Won":
            r["fouls_drawn"] += 1
        elif typ == "Bad Behaviour":
            card = ((ev.get("bad_behaviour") or {}).get("card") or {}).get("name")
            if card and "Yellow" in card:
                r["yellow_cards"] += 1
            if card and "Red" in card:
                r["red_cards"] += 1
        elif typ == "Pass":
            p = ev.get("pass") or {}
            r["passes"] += 1
            # Passing UNDER PRESSURE -- composure signal (retention when
            # pressed), a key separator of elite deep controllers that plain
            # pass completion misses.
            if ev.get("under_pressure"):
                r["passes_under_pressure"] += 1
                if p.get("outcome") is None:
                    r["complete_passes_under_pressure"] += 1
            if p.get("outcome") is None:
                r["complete_passes"] += 1
                # Ball progression from a COMPLETED pass only (an incomplete
                # forward ball didn't actually advance possession).
                start, end = ev.get("location"), p.get("end_location")
                d0, d1 = _dist_to_goal(start), _dist_to_goal(end)
                if d0 is not None and d1 is not None and d0 - d1 >= _PROG_PASS_MIN:
                    r["progressive_passes"] += 1
                if start and end and len(start) >= 2 and len(end) >= 2 and start[0] < 80.0 <= end[0]:
                    r["passes_into_final_third"] += 1
                if _in_box(end) and not _in_box(start):
                    r["passes_into_box"] += 1
            # Through balls: creative intent, counted regardless of completion.
            if ((p.get("technique") or {}).get("name")) == "Through Ball":
                r["through_balls"] += 1
            # Crosses / cut-backs -- wide-play signals for winger / attacking-
            # full-back roles (see the archetype system).
            if p.get("cross") is True:
                r["crosses"] += 1
            if p.get("cut_back") is True:
                r["cut_backs"] += 1
            if p.get("goal_assist") is True:
                r["assists"] += 1
            if p.get("shot_assist") is True:
                r["key_passes"] += 1
        elif typ == "Carry":
            r["carries"] += 1
            start, end = ev.get("location"), (ev.get("carry") or {}).get("end_location")
            d0, d1 = _dist_to_goal(start), _dist_to_goal(end)
            if (d0 is not None and d1 is not None and d0 - d1 >= _PROG_CARRY_MIN) \
                    or (_in_box(end) and not _in_box(start)):
                r["progressive_carries"] += 1
        elif typ == "Pressure":
            r["pressures"] += 1
        elif typ == "Duel":
            r["duels"] += 1
            duel_type = (((ev.get("duel") or {}).get("type") or {}).get("name") or "").lower()
            if "tackle" in duel_type:
                r["tackles"] += 1
            # "Aerial Lost" is the only aerial DUEL sub-type; aerials WON are
            # flagged on the underlying Pass/Shot/Clearance event instead (below).
            if "aerial" in duel_type:
                r["aerials_lost"] += 1
            # Outcome is missing for a large share of duels (untagged
            # sub-type, mostly off-ball challenges) -- those are left out of
            # both won/lost so the win rate isn't diluted by unknowns.
            duel_outcome = (((ev.get("duel") or {}).get("outcome") or {}).get("name") or "")
            if duel_outcome in ("Won", "Success In Play", "Success Out"):
                r["duels_won"] += 1
            elif duel_outcome in ("Lost In Play", "Lost Out"):
                r["duels_lost"] += 1
        elif typ == "Dribble":
            r["dribbles"] += 1
            outcome = (((ev.get("dribble") or {}).get("outcome") or {}).get("name") or "").lower()
            if outcome == "complete":
                r["successful_dribbles"] += 1
        elif typ == "Interception":
            r["interceptions"] += 1
        elif typ == "Ball Recovery":
            r["ball_recoveries"] += 1
        elif typ == "Dribbled Past":
            # Player field is the defender who got beaten, not the dribbler.
            r["dribbled_past"] += 1
        elif typ == "Clearance":
            r["clearances"] += 1
        elif typ == "Block":
            r["blocks"] += 1

        # Aerial duels WON are flagged (aerial_won=True) on the underlying
        # action event (Pass/Shot/Clearance/Miscontrol), not as their own event
        # -- checked outside the elif chain so it catches every carrier type.
        for _sub in (ev.get("pass"), ev.get("shot"), ev.get("clearance"), ev.get("miscontrol")):
            if isinstance(_sub, dict) and _sub.get("aerial_won") is True:
                r["aerials_won"] += 1
                break

    for r in rows.values():
        opponents = [t for t in teams_seen if t != r["team"]]
        r["opponent"] = opponents[0] if opponents else None

    out = pd.DataFrame(rows.values())
    return add_player_identity_columns(out)


def statsbomb_events_to_team_events(events_json: str | Path, *, match_id: str | int, date: str | None = None, team_scope: str = "club", competition: str = "StatsBomb Open Data") -> pd.DataFrame:
    """Aggregate StatsBomb events into team-match event totals."""
    pe = statsbomb_events_to_player_events(events_json, match_id=match_id, date=date, team_scope=team_scope, competition=competition)
    if pe.empty:
        return pd.DataFrame()
    agg_cols = [
        "shots", "shots_on_target", "fouls_committed", "fouls_drawn", "yellow_cards", "red_cards", "goals",
        "assists", "passes", "complete_passes", "key_passes", "pressures", "duels", "dribbles", "successful_dribbles",
        "tackles", "interceptions", "ball_recoveries",
        "duels_won", "duels_lost", "dribbled_past", "clearances", "blocks",
        "progressive_passes", "passes_into_final_third", "passes_into_box",
        "through_balls", "carries", "progressive_carries",
        "passes_under_pressure", "complete_passes_under_pressure", "aerials_won", "aerials_lost",
        "crosses", "cut_backs",
    ]
    out = pe.groupby(["match_id", "date", "competition", "team_scope", "team", "opponent"], dropna=False)[agg_cols].sum().reset_index()
    out = out.rename(columns={
        "shots": "shots_for",
        "shots_on_target": "sot_for",
        "fouls_committed": "fouls_for",
        "fouls_drawn": "fouls_drawn_for",
        "yellow_cards": "yellow_cards_for",
        "red_cards": "red_cards_for",
        "goals": "goals_for",
    })
    return add_team_identity_columns(out)


def statsbomb_events_to_lineups(events_json: str | Path, *, match_id: str | int, date: str | None = None, team_scope: str = "club", competition: str = "StatsBomb Open Data", match_length: int = 90) -> pd.DataFrame:
    """Extract Starting XI, substitutions and minutes from StatsBomb events.

    StatsBomb event files contain ``Starting XI`` events with formations/players
    and ``Substitution`` events with replacements. Tactical-shift events are
    extracted separately by ``statsbomb_events_to_tactical_shifts``.
    """
    events = _read_json(events_json)
    rows: list[dict] = []
    starters: dict[tuple[str, str], dict] = {}
    substitutions: list[dict] = []

    for ev in events:
        typ = _event_type(ev)
        team = _team(ev)
        if not team:
            continue
        if typ == "Starting XI":
            tactics = ev.get("tactics") or {}
            formation = tactics.get("formation")
            lineup = tactics.get("lineup") or []
            for item in lineup:
                player = ((item.get("player") or {}).get("name"))
                if not player:
                    continue
                pos = ((item.get("position") or {}).get("name"))
                key = (canonical_team_name(team), player)
                starters[key] = {
                    "match_id": match_id,
                    "date": date,
                    "competition": competition,
                    "team_scope": team_scope,
                    "source": "statsbomb_open_data",
                    "provider": "statsbomb",
                    "team": team,
                    "player": player,
                    "provider_player_id": str((item.get("player") or {}).get("id")) if (item.get("player") or {}).get("id") is not None else None,
                    "statsbomb_player_id": str((item.get("player") or {}).get("id")) if (item.get("player") or {}).get("id") is not None else None,
                    "position": pos,
                    "formation": formation,
                    "started": 1,
                    "minutes": float(match_length),
                    "replaced_by": None,
                    "replacement_minute": None,
                }
        elif typ == "Substitution":
            player_out = _player(ev)
            repl_obj = ((ev.get("substitution") or {}).get("replacement") or {})
            repl = repl_obj.get("name")
            repl_id = repl_obj.get("id")
            minute = ev.get("minute")
            try:
                minute = int(float(minute))
            except (TypeError, ValueError):
                minute = None
            substitutions.append({"team": team, "player_out": player_out, "player_in": repl, "player_in_id": repl_id, "minute": minute})

    for sub in substitutions:
        team_can = canonical_team_name(sub["team"])
        out_player = sub.get("player_out")
        in_player = sub.get("player_in")
        minute = sub.get("minute")
        if out_player:
            key = (team_can, out_player)
            if key in starters:
                starters[key]["minutes"] = float(minute if minute is not None else match_length)
                starters[key]["replaced_by"] = in_player
                starters[key]["replacement_minute"] = minute
        if in_player:
            rows.append({
                "match_id": match_id,
                "date": date,
                "competition": competition,
                "team_scope": team_scope,
                "source": "statsbomb_open_data",
                "provider": "statsbomb",
                "team": sub["team"],
                "player": in_player,
                "provider_player_id": str(sub.get("player_in_id")) if sub.get("player_in_id") is not None else None,
                "statsbomb_player_id": str(sub.get("player_in_id")) if sub.get("player_in_id") is not None else None,
                "position": None,
                "formation": None,
                "started": 0,
                "minutes": float(max(match_length - (minute or match_length), 0)),
                "replaced_by": None,
                "replacement_minute": minute,
                "replaced_player": out_player,
            })

    rows.extend(starters.values())
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["match_id", "team", "player", "started", "minutes", "replaced_by", "replacement_minute"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = add_team_identity_columns(out)
    return add_player_identity_columns(out)


def statsbomb_events_to_tactical_shifts(events_json: str | Path, *, match_id: str | int, date: str | None = None, team_scope: str = "club", competition: str = "StatsBomb Open Data") -> pd.DataFrame:
    """Extract formation changes from Starting XI and Tactical Shift events."""
    events = _read_json(events_json)
    rows: list[dict] = []
    for ev in events:
        typ = _event_type(ev)
        if typ not in {"Starting XI", "Tactical Shift"}:
            continue
        tactics = ev.get("tactics") or {}
        rows.append({
            "match_id": match_id,
            "date": date,
            "competition": competition,
            "team_scope": team_scope,
            "source": "statsbomb_open_data",
            "team": _team(ev),
            "minute": ev.get("minute", 0),
            "second": ev.get("second", 0),
            "event_type": typ,
            "formation": tactics.get("formation"),
            "players_in_shape": len(tactics.get("lineup") or []),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["match_id", "team", "minute", "event_type", "formation"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return add_team_identity_columns(out)
