from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import math
import time

import pandas as pd
import requests

from mundialytics.data.identity import canonical_team_name
from mundialytics.providers.api_config import ProviderConfigError, endpoint_spec, provider_runtime, render_template

PROVIDER_NAME = "creativesdev_live_football"
DEFAULT_BASE_URL = "https://free-api-live-football-data.p.rapidapi.com"
DEFAULT_RAPIDAPI_HOST = "free-api-live-football-data.p.rapidapi.com"


class CreativesDevAccessError(RuntimeError):
    pass


class CreativesDevRateLimitError(RuntimeError):
    pass


def _now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_params(params: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (params or {}).items():
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if value == "":
            continue
        out[key] = value
    return out


def _cache_name(endpoint: str, params: dict[str, Any]) -> str:
    blob = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24] + ".json"


def _unwrap_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        # RapidAPI providers often use one of these wrappers. Keep raw if wrapper is ambiguous.
        for key in ("data", "response", "result", "results", "payload"):
            value = payload.get(key)
            if isinstance(value, (list, dict)) and len(payload) <= 6:
                return value
    return payload


def _as_list(payload: Any) -> list[Any]:
    payload = _unwrap_payload(payload)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "response", "results", "fixtures", "matches", "events", "lineups", "statistics", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                nested = _as_list(value)
                if nested:
                    return nested
        return [payload]
    return []


def _get_nested(obj: dict[str, Any], paths: list[str], default: Any = "") -> Any:
    for path in paths:
        cur: Any = obj
        ok = True
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur not in (None, ""):
            return cur
    return default


def _to_iso_utc(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        if isinstance(value, (int, float)) or str(value).isdigit():
            # Accept seconds or milliseconds.
            n = int(float(value))
            if n > 10_000_000_000:
                n = int(n / 1000)
            return datetime.fromtimestamp(n, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        pass
    text = str(value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return str(value)


@dataclass
class CreativesDevClient:
    rapidapi_key: str | None = None
    rapidapi_host: str | None = None
    base_url: str | None = None
    timeout: int = 30
    cache_dir: str | Path | None = None
    ledger_path: str | Path | None = None
    min_interval_sec: float = 0.25
    max_calls: int = 25
    monthly_budget: int | None = None
    user_agent: str = "MundialyticsBettingEngine/0.44 (+personal research)"
    calls_made: int = 0
    cache_hits: int = 0
    _last_call_ts: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url or DEFAULT_BASE_URL
        self.rapidapi_host = self.rapidapi_host or DEFAULT_RAPIDAPI_HOST

    @classmethod
    def from_config(cls, *, provider_config: str | Path | None = None, max_calls: int | None = None, monthly_budget: int | None = None, cache_dir: str | Path | None = None, ledger_path: str | Path | None = None) -> "CreativesDevClient":
        runtime = provider_runtime(PROVIDER_NAME, provider_config, required=False)
        resolved_max_calls = max_calls if max_calls is not None else (runtime.max_calls_per_run or 25)
        resolved_budget = monthly_budget if monthly_budget is not None else runtime.monthly_budget
        return cls(
            rapidapi_key=runtime.api_key,
            rapidapi_host=runtime.host or DEFAULT_RAPIDAPI_HOST,
            base_url=runtime.base_url or DEFAULT_BASE_URL,
            cache_dir=cache_dir or runtime.cache_dir,
            ledger_path=ledger_path or runtime.ledger_path,
            min_interval_sec=runtime.min_interval_sec,
            max_calls=int(resolved_max_calls),
            monthly_budget=resolved_budget,
        )

    def _headers(self) -> dict[str, str]:
        if not self.rapidapi_key:
            raise RuntimeError("Creativesdev RapidAPI requires a key. Set RAPIDAPI_KEY or configure rapidapi.key/key_env.")
        if not self.rapidapi_host:
            raise RuntimeError("Creativesdev RapidAPI requires rapidapi.host from the RapidAPI code snippet.")
        return {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
            "X-RapidAPI-Key": self.rapidapi_key,
            "X-RapidAPI-Host": self.rapidapi_host,
        }

    def _monthly_calls_used(self) -> int:
        if not self.ledger_path:
            return 0
        path = Path(self.ledger_path)
        if not path.exists():
            return 0
        month = datetime.now(tz=timezone.utc).strftime("%Y-%m")
        used = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except Exception:
                continue
            if item.get("provider") == PROVIDER_NAME and str(item.get("ts", ""))[:7] == month and item.get("counted", True):
                used += 1
        return used

    def _write_ledger(self, record: dict[str, Any]) -> None:
        if not self.ledger_path:
            return
        path = Path(self.ledger_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def get(self, endpoint: str, params: dict[str, Any] | None = None, *, force: bool = False, unwrap: bool = True) -> Any:
        ep = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        query = _clean_params(params)
        cache_path: Path | None = None
        if self.cache_dir:
            cache_root = Path(self.cache_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_path = cache_root / _cache_name(ep, query)
            if cache_path.exists() and not force:
                self.cache_hits += 1
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                return _unwrap_payload(payload) if unwrap else payload
        if self.calls_made >= self.max_calls:
            raise RuntimeError(f"Creativesdev call budget exceeded: {self.calls_made}/{self.max_calls}.")
        if self.monthly_budget is not None and self._monthly_calls_used() >= self.monthly_budget:
            raise RuntimeError(f"Creativesdev monthly budget exceeded: {self._monthly_calls_used()}/{self.monthly_budget}.")
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)
        url = f"{str(self.base_url).rstrip('/')}{ep}"
        response = requests.get(url, params=query, headers=self._headers(), timeout=self.timeout)
        self._last_call_ts = time.monotonic()
        self.calls_made += 1
        self._write_ledger({
            "ts": _now_utc_iso(),
            "provider": PROVIDER_NAME,
            "endpoint": ep,
            "status_code": response.status_code,
            "counted": True,
            "rate_limit_limit": response.headers.get("X-RateLimit-Limit"),
            "rate_limit_remaining": response.headers.get("X-RateLimit-Remaining") or response.headers.get("x-ratelimit-requests-remaining"),
            "rate_limit_reset": response.headers.get("X-RateLimit-Reset"),
            "retry_after": response.headers.get("Retry-After"),
        })
        if response.status_code == 429:
            raise CreativesDevRateLimitError(f"Creativesdev RapidAPI rate limited (429). Retry-After={response.headers.get('Retry-After')}")
        if response.status_code in {401, 403}:
            raise CreativesDevAccessError(f"Creativesdev access error {response.status_code}: {response.text[:300]}")
        response.raise_for_status()
        payload = response.json()
        if cache_path is not None:
            cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return _unwrap_payload(payload) if unwrap else payload

    def get_configured_endpoint(self, endpoint_key: str, *, provider_config: str | Path | None = None, variables: dict[str, Any] | None = None, force: bool = False) -> Any:
        runtime = provider_runtime(PROVIDER_NAME, provider_config, required=True)
        spec = endpoint_spec(runtime, endpoint_key)
        if not spec.get("path"):
            raise ProviderConfigError(f"Endpoint '{endpoint_key}' has no path. Copy the path from the RapidAPI code snippet into the external config.")
        variables = variables or {}
        path = render_template(spec.get("path"), variables)
        params = render_template(spec.get("params", {}), variables)
        return self.get(str(path), params if isinstance(params, dict) else {}, force=force)



FIXTURE_COLUMNS = [
    "provider",
    "provider_fixture_id",
    "provider_match_id",
    "provider_league_id",
    "kickoff_utc",
    "home_team",
    "away_team",
    "home_team_canonical",
    "away_team_canonical",
    "competition",
    "tournament_stage",
    "status",
    "status_short",
    "home_score",
    "away_score",
    "raw_score",
    "raw_keys",
]

LINEUP_COLUMNS = [
    "match_id",
    "provider",
    "team",
    "player",
    "provider_player_id",
    "position",
    "started",
    "raw_team",
]

STATS_COLUMNS = [
    "match_id",
    "provider",
    "team",
    "stat_name",
    "stat_value",
]

EVENT_COLUMNS = [
    "match_id",
    "provider",
    "minute",
    "team",
    "player",
    "event_type",
    "detail",
]


def _blank_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _text_or_empty(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, dict):
        for key in ("name", "long", "short", "displayName", "title", "label"):
            if value.get(key) not in (None, ""):
                return str(value[key])
        return ""
    return str(value)


def _parse_score_str(value: Any) -> tuple[str, str]:
    text = str(value or "").strip()
    if not text:
        return "", ""
    for sep in (" - ", "-", ":"):
        if sep in text:
            left, right = text.split(sep, 1)
            return left.strip(), right.strip()
    return "", ""


def _non_empty(*values: Any) -> bool:
    return any(v not in (None, "") for v in values)


def _fixture_id(item: dict[str, Any]) -> Any:
    return _get_nested(item, ["fixture.id", "id", "event.id", "match.id", "match_id", "fixture_id"])


def normalize_fixtures(payload: Any) -> pd.DataFrame:
    """Normalize Creativesdev fixture/date endpoint output.

    The RapidAPI endpoint `football-get-matches-by-date` returns fixture-like rows where
    kickoff is nested under `status.utcTime`, not top-level `date`. v0.45 reads that field
    and avoids creating fake event rows from the same payload.
    """
    rows: list[dict[str, Any]] = []
    for item in _as_list(payload):
        if not isinstance(item, dict):
            continue
        fixture_id = _fixture_id(item)
        home = _get_nested(item, [
            "teams.home.name", "home.name", "homeTeam.name", "home_team.name",
            "homeTeam", "home", "localteam.name",
        ])
        away = _get_nested(item, [
            "teams.away.name", "away.name", "awayTeam.name", "away_team.name",
            "awayTeam", "away", "visitorteam.name",
        ])
        kickoff = _get_nested(item, [
            "fixture.date", "status.utcTime", "utcTime", "date", "event_date",
            "kickoff", "startTime", "time.utcTime", "time.starting_at.date_time", "timeTS",
        ])
        league = _get_nested(item, [
            "league.name", "competition.name", "tournament.name", "league", "competition",
        ])
        league_id = _get_nested(item, ["league.id", "leagueId", "competition.id", "tournament.id"])
        tournament_stage = _get_nested(item, ["tournamentStage", "stage.name", "stage", "round"])
        raw_status = _get_nested(item, ["fixture.status.long", "status.reason.long", "status.long", "status"])
        status_short = _get_nested(item, ["fixture.status.short", "status.reason.short", "status.short", "statusId"])
        raw_score = _get_nested(item, ["goals.scoreStr", "scoreStr", "status.scoreStr", "score.fulltime", "scores.fulltime"])
        home_score = _get_nested(item, ["goals.home", "score.home", "scores.home", "homeScore.current", "homeScore", "home.score"])
        away_score = _get_nested(item, ["goals.away", "score.away", "scores.away", "awayScore.current", "awayScore", "away.score"])
        if not _non_empty(home_score, away_score):
            home_score, away_score = _parse_score_str(raw_score)
        if not _non_empty(fixture_id, home, away):
            continue
        rows.append({
            "provider": PROVIDER_NAME,
            "provider_fixture_id": fixture_id,
            "provider_match_id": fixture_id,
            "provider_league_id": league_id,
            "kickoff_utc": _to_iso_utc(kickoff),
            "home_team": _text_or_empty(home),
            "away_team": _text_or_empty(away),
            "home_team_canonical": canonical_team_name(home),
            "away_team_canonical": canonical_team_name(away),
            "competition": _text_or_empty(league),
            "tournament_stage": _text_or_empty(tournament_stage),
            "status": _text_or_empty(raw_status),
            "status_short": _text_or_empty(status_short),
            "home_score": home_score,
            "away_score": away_score,
            "raw_score": _text_or_empty(raw_score),
            "raw_keys": ";".join(sorted(item.keys())),
        })
    return pd.DataFrame(rows, columns=FIXTURE_COLUMNS)


def normalize_lineups(payload: Any, *, match_id: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in _as_list(payload):
        if not isinstance(item, dict):
            continue
        team = _get_nested(item, ["team.name", "team", "club.name"])
        players = item.get("players") or item.get("lineup") or item.get("startXI") or item.get("substitutes")
        if isinstance(players, list):
            for p in players:
                pobj = p.get("player", p) if isinstance(p, dict) else {}
                if not isinstance(pobj, dict):
                    continue
                player = _get_nested(pobj, ["name", "player_name"])
                if not player:
                    continue
                rows.append({
                    "match_id": match_id or _fixture_id(item),
                    "provider": PROVIDER_NAME,
                    "team": _text_or_empty(team),
                    "player": _text_or_empty(player),
                    "provider_player_id": _get_nested(pobj, ["id", "player_id"]),
                    "position": _get_nested(pobj, ["pos", "position", "grid"]),
                    "started": True if "start" in str(item.keys()).lower() or item.get("type") == "starting" else "",
                    "raw_team": _text_or_empty(team),
                })
        else:
            player = _get_nested(item, ["player.name", "name", "player_name"])
            # Do not treat fixture rows as one-player lineups just because they have home/away names.
            if player and not _non_empty(_get_nested(item, ["home.name", "home", "away.name", "away"])):
                rows.append({
                    "match_id": match_id or _fixture_id(item),
                    "provider": PROVIDER_NAME,
                    "team": _text_or_empty(team),
                    "player": _text_or_empty(player),
                    "provider_player_id": _get_nested(item, ["player.id", "id", "player_id"]),
                    "position": _get_nested(item, ["player.pos", "position", "pos"]),
                    "started": "",
                    "raw_team": _text_or_empty(team),
                })
    return pd.DataFrame(rows, columns=LINEUP_COLUMNS)


def normalize_match_statistics(payload: Any, *, match_id: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in _as_list(payload):
        if not isinstance(item, dict):
            continue
        team = _get_nested(item, ["team.name", "team", "name"])
        stats = item.get("statistics") or item.get("stats") or []
        # Avoid interpreting a fixture list wrapper as statistics just because it has a `data` key.
        if isinstance(stats, dict):
            stats = [{"type": k, "value": v} for k, v in stats.items()]
        for stat in stats if isinstance(stats, list) else []:
            if not isinstance(stat, dict):
                continue
            stat_name = _get_nested(stat, ["type", "name", "key"])
            if not stat_name:
                continue
            rows.append({
                "match_id": match_id or _fixture_id(item),
                "provider": PROVIDER_NAME,
                "team": _text_or_empty(team),
                "stat_name": _text_or_empty(stat_name),
                "stat_value": _get_nested(stat, ["value", "displayValue"]),
            })
    return pd.DataFrame(rows, columns=STATS_COLUMNS)


def normalize_events(payload: Any, *, match_id: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in _as_list(payload):
        if not isinstance(item, dict):
            continue
        event_type = _get_nested(item, ["type", "event_type", "incidentType", "event.type", "detail"])
        minute = _get_nested(item, ["time.elapsed", "minute", "time.minute", "time"])
        team = _get_nested(item, ["team.name", "team"])
        player = _get_nested(item, ["player.name", "player"])
        detail = _get_nested(item, ["detail", "comments", "description", "text"])
        # Fixture rows from football-get-matches-by-date contain id/home/away/status but no actual event.
        if not _non_empty(event_type, minute, team, player, detail):
            continue
        if isinstance(team, dict):
            team = _text_or_empty(team)
        if isinstance(player, dict):
            player = _text_or_empty(player)
        rows.append({
            "match_id": match_id or _fixture_id(item),
            "provider": PROVIDER_NAME,
            "minute": minute,
            "team": _text_or_empty(team),
            "player": _text_or_empty(player),
            "event_type": _text_or_empty(event_type),
            "detail": _text_or_empty(detail),
        })
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)
