from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import math
import os
import time

import pandas as pd
import requests

from mundialytics.betting.odds_contract import ODDS_INPUT_COLUMNS, norm_key, norm_text, standard_odds_input_frame
from mundialytics.data.identity import canonical_team_name
from mundialytics.providers.api_config import provider_runtime

ODDSPAPI_DIRECT_BASE_URL = "https://api.oddspapi.io"
ODDSPAPI_RAPIDAPI_BASE_URL = "https://odds-api1.p.rapidapi.com"
ODDSPAPI_RAPIDAPI_HOST = "odds-api1.p.rapidapi.com"
SOCCER_SPORT_ID = 10
PROVIDER_NAME = "oddspapi"


class OddsPapiAccessError(RuntimeError):
    """Raised when the key/plan cannot access an OddsPapi endpoint."""


class OddsPapiRateLimitError(RuntimeError):
    """Raised when OddsPapi or RapidAPI returns a rate-limit response."""


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


def _cache_name(endpoint: str, params: dict[str, Any], mode: str) -> str:
    # Do not include raw API key in cache filenames; keep only a stable marker.
    scrubbed = dict(params)
    if "apiKey" in scrubbed:
        scrubbed["apiKey"] = "__redacted__"
    blob = json.dumps({"endpoint": endpoint, "params": scrubbed, "mode": mode}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24] + ".json"


def _dt_to_epoch_seconds(value: Any, *, end_of_day: bool = False) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError("Empty date/time value")
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        suffix = "T23:59:59+00:00" if end_of_day else "T00:00:00+00:00"
        text = text + suffix
    text = text.replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp())


def epoch_millis_to_iso(value: Any) -> str:
    try:
        ms = int(float(value))
    except Exception:
        return ""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def epoch_seconds_to_iso(value: Any) -> str:
    try:
        sec = int(float(value))
    except Exception:
        return ""
    return datetime.fromtimestamp(sec, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _unwrap_payload(payload: Any) -> Any:
    """Handle direct v5 and marketplace wrappers without losing the core object/list."""
    if isinstance(payload, dict):
        # Common wrappers: {data: ...}, {result: ...}, {payload: ...}
        for key in ("data", "result", "payload"):
            val = payload.get(key)
            if isinstance(val, (list, dict)) and len(payload.keys()) <= 4:
                return val
    return payload


@dataclass
class OddsPapiClient:
    """Defensive OddsPapi client with cache, persistent ledger and hard call budget.

    Modes:
    - direct: https://v5.oddspapi.io/en + apiKey query parameter.
    - rapidapi: RapidAPI proxy + X-RapidAPI-Key / X-RapidAPI-Host headers.

    The default RapidAPI host/base URL matches the public RapidAPI listing slug
    `odds-api1`, but both can be overridden from the RapidAPI code snippet.
    """

    api_key: str | None = None
    mode: str = "direct"
    base_url: str | None = None
    rapidapi_key: str | None = None
    rapidapi_host: str | None = None
    timeout: int = 30
    cache_dir: str | Path | None = None
    ledger_path: str | Path | None = None
    min_interval_sec: float = 0.25
    max_calls: int = 25
    monthly_budget: int | None = None
    user_agent: str = "MundialyticsBettingEngine/0.46 (+personal research)"
    endpoints: dict[str, Any] = field(default_factory=dict)
    calls_made: int = 0
    cache_hits: int = 0
    _last_call_ts: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.mode = (self.mode or "direct").lower().strip()
        if self.mode == "rapidapi":
            self.base_url = self.base_url or ODDSPAPI_RAPIDAPI_BASE_URL
            self.rapidapi_host = self.rapidapi_host or ODDSPAPI_RAPIDAPI_HOST
        else:
            self.base_url = self.base_url or ODDSPAPI_DIRECT_BASE_URL

    @classmethod
    def from_env(
        cls,
        *,
        mode: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path | None = None,
        ledger_path: str | Path | None = None,
        max_calls: int | None = 25,
        monthly_budget: int | None = None,
        provider_config: str | Path | None = None,
    ) -> "OddsPapiClient":
        """Build a client from env plus optional external provider config.

        Precedence: explicit CLI args > env vars > config/mundialytics_api_config.local.yaml > built-in defaults.
        This keeps API keys outside update ZIPs while leaving old commands compatible.
        """
        runtime = provider_runtime("oddspapi", provider_config, required=False)
        resolved_mode = mode or os.getenv("ODDSPAPI_MODE") or runtime.mode or "direct"
        env_budget = os.getenv("ODDSPAPI_MONTHLY_BUDGET")
        env_max_calls = os.getenv("ODDSPAPI_MAX_CALLS_PER_RUN")
        resolved_max_calls = max_calls
        if resolved_max_calls is None:
            if env_max_calls and env_max_calls.isdigit():
                resolved_max_calls = int(env_max_calls)
            elif runtime.max_calls_per_run is not None:
                resolved_max_calls = runtime.max_calls_per_run
            else:
                resolved_max_calls = 25
        resolved_budget = monthly_budget
        if resolved_budget is None:
            if env_budget and env_budget.isdigit():
                resolved_budget = int(env_budget)
            else:
                resolved_budget = runtime.monthly_budget
        return cls(
            api_key=os.getenv("ODDSPAPI_API_KEY") or (runtime.api_key if resolved_mode != "rapidapi" else None),
            rapidapi_key=os.getenv("RAPIDAPI_ODDSPAPI_KEY") or os.getenv("RAPIDAPI_KEY") or (runtime.api_key if resolved_mode == "rapidapi" else None),
            rapidapi_host=os.getenv("RAPIDAPI_ODDSPAPI_HOST") or runtime.host,
            mode=resolved_mode,
            base_url=base_url or os.getenv("ODDSPAPI_BASE_URL") or runtime.base_url,
            cache_dir=cache_dir or runtime.cache_dir,
            ledger_path=ledger_path or os.getenv("ODDSPAPI_LEDGER_PATH") or runtime.ledger_path,
            max_calls=int(resolved_max_calls),
            monthly_budget=resolved_budget,
            min_interval_sec=runtime.min_interval_sec,
            endpoints=runtime.endpoints or {},
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": self.user_agent}
        if self.mode == "rapidapi":
            if not self.rapidapi_key:
                raise RuntimeError("RapidAPI mode requires RAPIDAPI_KEY or RAPIDAPI_ODDSPAPI_KEY.")
            if not self.rapidapi_host:
                raise RuntimeError("RapidAPI mode requires RAPIDAPI_ODDSPAPI_HOST from the RapidAPI code snippet.")
            headers["X-RapidAPI-Key"] = self.rapidapi_key
            headers["X-RapidAPI-Host"] = self.rapidapi_host
        return headers

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
            if str(item.get("ts", ""))[:7] == month and item.get("counted", True):
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
        if self.mode == "direct":
            key = self.api_key or os.getenv("ODDSPAPI_API_KEY")
            if not key:
                raise RuntimeError("Direct OddsPapi mode requires ODDSPAPI_API_KEY.")
            query.setdefault("apiKey", key)
        elif self.mode != "rapidapi":
            raise ValueError("mode must be 'direct' or 'rapidapi'")

        cache_path: Path | None = None
        if self.cache_dir:
            cache_root = Path(self.cache_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_path = cache_root / _cache_name(ep, query, self.mode)
            if cache_path.exists() and not force:
                self.cache_hits += 1
                return _unwrap_payload(json.loads(cache_path.read_text(encoding="utf-8"))) if unwrap else json.loads(cache_path.read_text(encoding="utf-8"))

        if self.calls_made >= self.max_calls:
            raise RuntimeError(f"OddsPapi call budget exceeded: {self.calls_made}/{self.max_calls}. Increase --max-api-calls intentionally.")
        if self.monthly_budget is not None:
            used = self._monthly_calls_used()
            if used >= self.monthly_budget:
                raise RuntimeError(f"OddsPapi monthly budget would be exceeded: {used}/{self.monthly_budget}. Use cache or increase intentionally.")

        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)

        url = f"{str(self.base_url).rstrip('/')}{ep}"
        response = requests.get(url, params=query, headers=self._headers(), timeout=self.timeout)
        self._last_call_ts = time.monotonic()
        self.calls_made += 1
        record = {
            "ts": _now_utc_iso(),
            "mode": self.mode,
            "endpoint": ep,
            "status_code": response.status_code,
            "counted": True,
            "rate_limit_limit": response.headers.get("X-RateLimit-Limit"),
            "rate_limit_remaining": response.headers.get("X-RateLimit-Remaining") or response.headers.get("x-ratelimit-requests-remaining"),
            "rate_limit_reset": response.headers.get("X-RateLimit-Reset"),
            "retry_after": response.headers.get("Retry-After"),
        }
        self._write_ledger(record)

        if response.status_code == 429:
            retry = response.headers.get("Retry-After") or "unknown"
            raise OddsPapiRateLimitError(f"OddsPapi rate limited (429). Retry-After={retry}. Lower --max-api-calls or wait.")
        if response.status_code in {401, 403}:
            raise OddsPapiAccessError(f"OddsPapi access error {response.status_code}: {response.text[:300]}")
        response.raise_for_status()
        payload = response.json()
        if cache_path is not None:
            cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return _unwrap_payload(payload) if unwrap else payload

    def _endpoint(self, key: str, default: str) -> str:
        spec = self.endpoints.get(key) if isinstance(self.endpoints, dict) else None
        if isinstance(spec, str) and spec.strip():
            return spec.strip()
        if isinstance(spec, dict):
            path = spec.get("path")
            if isinstance(path, str) and path.strip():
                return path.strip()
        return default

    def sports(self) -> Any:
        return self.get(self._endpoint("sports", "/v4/sports"), {"language": "en"})

    def bookmakers(self, *, player_props: bool | None = None, bookmakers: str | None = None) -> Any:
        params: dict[str, Any] = {"bookmakers": bookmakers}
        if player_props is not None:
            params["playerProps"] = str(player_props).lower()
        return self.get(self._endpoint("bookmakers", "/v4/bookmakers"), params)

    def markets(self, *, sport_id: int = SOCCER_SPORT_ID, market_ids: str | None = None, outcome_ids: str | None = None) -> Any:
        return self.get(self._endpoint("markets", "/v4/markets"), {"sportId": sport_id, "marketIds": market_ids, "outcomeIds": outcome_ids, "language": "en"})

    def players(self, *, sport_id: int | None = None, participant_id: int | None = None, player_ids: str | None = None) -> Any:
        return self.get(self._endpoint("players", "/v4/players"), {"sportId": sport_id, "participantId": participant_id, "playerIds": player_ids, "language": "en"})

    def fixtures(self, *, sport_id: int = SOCCER_SPORT_ID, start_time_from: int | None = None, start_time_to: int | None = None, tournament_id: int | None = None, status_id: int | None = None, bookmakers: str | None = None, fixture_ids: str | None = None) -> Any:
        return self.get(self._endpoint("fixtures", "/v4/fixtures"), {"sportId": sport_id, "from": epoch_seconds_to_iso(start_time_from) if start_time_from else None, "to": epoch_seconds_to_iso(start_time_to) if start_time_to else None, "tournamentId": tournament_id, "statusId": status_id, "bookmakers": bookmakers, "fixtureIds": fixture_ids, "hasOdds": "true" if bookmakers else None, "language": "en"})

    def fixture_odds(self, *, fixture_id: str, bookmakers: str | None = None, main_line: bool | None = None, since: int | None = None) -> Any:
        return self.get(self._endpoint("fixture_odds", "/v4/odds"), {"fixtureId": fixture_id, "bookmakers": bookmakers, "oddsFormat": "decimal", "language": "en", "verbosity": 3})

    def fixture_main_odds(self, *, fixture_ids: str | None = None, tournament_id: int | None = None, bookmakers: str | None = None, since: int | None = None) -> Any:
        # Docs require exactly one of tournamentId or fixtureIds.
        if bool(fixture_ids) == bool(tournament_id):
            raise ValueError("fixture_main_odds requires exactly one of fixture_ids or tournament_id")
        return self.get(self._endpoint("fixture_main_odds", "/v4/odds-by-tournaments"), {"tournamentIds": str(tournament_id) if tournament_id else None, "fixtureIds": fixture_ids, "bookmaker": bookmakers, "oddsFormat": "decimal", "language": "en", "verbosity": 3})

    def fixture_historical_odds(self, *, fixture_id: str, bookmaker: str | None = None, odds_ids: str | None = None) -> Any:
        # OddsPapi rule: if oddsIds is not supplied, exactly one bookmaker is required.
        if not odds_ids and not bookmaker:
            raise ValueError("fixture_historical_odds requires bookmaker when odds_ids is not provided.")
        return self.get(self._endpoint("fixture_historical_odds", "/v4/historical-odds"), {"fixtureId": fixture_id, "bookmakers": bookmaker, "oddsIds": odds_ids})

    def fixture_clv(self, *, fixture_id: str, bookmakers: str | None = None, odds_ids: str | None = None) -> Any:
        return self.get(self._endpoint("fixture_clv", "/v4/clv"), {"fixtureId": fixture_id, "bookmakers": bookmakers, "oddsIds": odds_ids})


def _as_list(payload: Any) -> list[Any]:
    payload = _unwrap_payload(payload)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("items", "results", "sports", "bookmakers", "markets", "fixtures"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
    return []


def sports_to_frame(payload: Any) -> pd.DataFrame:
    return pd.DataFrame(_as_list(payload))


def bookmakers_to_frame(payload: Any) -> pd.DataFrame:
    items = _as_list(payload)
    if items:
        return pd.DataFrame(items)
    # Some endpoints may return a dict keyed by slug.
    payload = _unwrap_payload(payload)
    if isinstance(payload, dict):
        rows = []
        for slug, val in payload.items():
            row = {"bookmaker": slug}
            if isinstance(val, dict):
                row.update(val)
            rows.append(row)
        return pd.DataFrame(rows)
    return pd.DataFrame()


def markets_to_frame(payload: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in _as_list(payload):
        if not isinstance(item, dict):
            continue
        base = {k: item.get(k) for k in ["marketId", "marketLength", "sportId", "playerProp", "handicap", "period", "marketType", "marketName", "marketNameShort"]}
        outcomes = item.get("outcomes") or []
        if isinstance(outcomes, dict):
            outcomes = list(outcomes.values())
        if outcomes:
            for out in outcomes:
                row = dict(base)
                if isinstance(out, dict):
                    row["outcomeId"] = out.get("outcomeId") or out.get("id")
                    row["outcomeName"] = out.get("outcomeName") or out.get("name")
                else:
                    row["outcomeId"] = ""
                    row["outcomeName"] = str(out)
                rows.append(row)
        else:
            rows.append(base)
    return pd.DataFrame(rows)


def fixtures_to_frame(payload: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fixtures = _as_list(payload)
    for fx in fixtures:
        if not isinstance(fx, dict):
            continue
        participants = fx.get("participants") or {}
        if isinstance(participants, list):
            p1 = participants[0] if len(participants) > 0 and isinstance(participants[0], dict) else {}
            p2 = participants[1] if len(participants) > 1 and isinstance(participants[1], dict) else {}
            participants = {
                "participant1Name": p1.get("participantName") or p1.get("name"),
                "participant2Name": p2.get("participantName") or p2.get("name"),
                "participant1Id": p1.get("participantId") or p1.get("id"),
                "participant2Id": p2.get("participantId") or p2.get("id"),
            }
        tournament = fx.get("tournament") or {}
        season = fx.get("season") or {}
        sport = fx.get("sport") or {}
        status = fx.get("status") or {}
        external = fx.get("externalProviders") or fx.get("external") or {}
        start_time = fx.get("startTime") or fx.get("commenceTime") or fx.get("kickoff")
        if isinstance(start_time, str) and not start_time.isdigit():
            try:
                start_time = _dt_to_epoch_seconds(start_time)
            except Exception:
                pass
        rows.append({
            "provider": PROVIDER_NAME,
            "provider_fixture_id": fx.get("fixtureId") or fx.get("id") or fx.get("eventId"),
            "fixture_id": fx.get("fixtureId") or fx.get("id") or fx.get("eventId"),
            "sport_id": sport.get("sportId") or fx.get("sportId"),
            "sport_name": sport.get("sportName") or sport.get("name"),
            "tournament_id": tournament.get("tournamentId") or fx.get("tournamentId"),
            "tournament_name": tournament.get("tournamentName") or tournament.get("name"),
            "category_name": tournament.get("categoryName"),
            "season_id": season.get("seasonId") or fx.get("seasonId"),
            "season_name": season.get("seasonName") or season.get("name"),
            "start_time_epoch": start_time,
            "kickoff_utc": epoch_seconds_to_iso(start_time),
            "true_start_time": fx.get("trueStartTime"),
            "status_id": status.get("statusId") or fx.get("statusId"),
            "status_name": status.get("statusName") or status.get("name"),
            "home_team": participants.get("participant1Name") or fx.get("homeTeam") or fx.get("home_team"),
            "away_team": participants.get("participant2Name") or fx.get("awayTeam") or fx.get("away_team"),
            "home_team_canonical": canonical_team_name(participants.get("participant1Name") or fx.get("homeTeam") or fx.get("home_team")),
            "away_team_canonical": canonical_team_name(participants.get("participant2Name") or fx.get("awayTeam") or fx.get("away_team")),
            "participant1_id": participants.get("participant1Id"),
            "participant2_id": participants.get("participant2Id"),
            "sofascore_id": external.get("sofascoreId"),
            "flashscore_id": external.get("flashscoreId"),
            "pinnacle_id": external.get("pinnacleId"),
            "betradar_id": external.get("betradarId"),
        })
    return pd.DataFrame(rows)


def infer_internal_market_from_oddspapi_market(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    market_type = norm_key(row.get("marketType"))
    period = norm_key(row.get("period"))
    market_name = norm_text(row.get("marketName"))
    outcome = norm_text(row.get("outcomeName"))
    handicap = row.get("handicap")
    raw_player_prop = row.get("playerProp")
    player_prop = bool(raw_player_prop) if not isinstance(raw_player_prop, str) else raw_player_prop.lower() in {"1", "true", "yes"}
    side = ""
    scope = "match"
    market_key = ""
    confidence = "needs_manual_review"
    notes = "unmapped_market"

    fulltime = period in {"", "fulltime", "result", "regular_time", "ft"}

    if market_type in {"1x2", "moneyline"} and fulltime:
        market_key = "1x2"
        side = {"1": "home", "home": "home", "x": "draw", "draw": "draw", "2": "away", "away": "away"}.get(outcome, outcome)
        confidence = "high" if side in {"home", "draw", "away"} else "medium"
        notes = "soccer_fulltime_result"
    elif market_type in {"bothteamsscore", "both_teams_score", "btts"} or "both teams" in market_name:
        market_key = "btts"
        side = {"yes": "yes", "no": "no"}.get(outcome, outcome)
        confidence = "high" if side in {"yes", "no"} else "medium"
        notes = "both_teams_to_score"
    elif market_type in {"totals", "over_under", "total"} and fulltime and not any(x in market_name for x in ["corner", "card", "foul", "shot", "save"]):
        market_key = "goals"
        side = {"over": "over", "under": "under"}.get(outcome, outcome)
        confidence = "high" if side in {"over", "under"} else "medium"
        notes = "fulltime_total_goals"
    elif market_type.startswith("teamtotals") and not any(x in market_name for x in ["corner", "card", "foul", "shot", "save"]):
        market_key = "team_goals"
        scope = "team"
        side = {"over": "over", "under": "under"}.get(outcome, outcome)
        confidence = "medium"
        notes = f"team_total_goals_from_{market_type}"
    elif player_prop:
        if "shot on target" in market_name or "shots on target" in market_name or "sot" in market_name:
            market_key = "player_shots_on_target"
        elif "shot" in market_name:
            market_key = "player_shots"
        elif "foul" in market_name:
            market_key = "player_fouls_committed"
        elif "yellow" in market_name or "card" in market_name:
            market_key = "player_yellow_card"
        elif "save" in market_name:
            market_key = "goalkeeper_saves"
        side = {"over": "over", "under": "under", "yes": "yes", "no": "no"}.get(outcome, outcome)
        scope = "player"
        confidence = "medium" if market_key else "needs_manual_review"
        notes = "player_prop_name_based_mapping_review_required"
    else:
        # Match/team secondary stat markets. OddsPapi market names are normalized but still provider-controlled; review mappings.
        wants_team = "team" in market_name or market_type.startswith("team") or "team1" in market_type or "team2" in market_type
        if "corner" in market_name:
            market_key = "team_corners" if wants_team else "corners"
        elif "shot on target" in market_name or "shots on target" in market_name or "sot" in market_name:
            market_key = "team_shots_on_target" if wants_team else "shots_on_target"
        elif "shot" in market_name:
            market_key = "team_shots" if wants_team else "shots"
        elif "foul" in market_name:
            market_key = "team_fouls" if wants_team else "fouls"
        elif "yellow" in market_name or "card" in market_name or "booking" in market_name:
            market_key = "team_yellow_cards" if wants_team else "yellow_cards"
        elif "save" in market_name:
            market_key = "team_goalkeeper_saves" if wants_team else "goalkeeper_saves"
        if market_key:
            side = {"over": "over", "under": "under", "yes": "yes", "no": "no", "1": "home", "2": "away"}.get(outcome, outcome)
            scope = "team" if market_key.startswith("team_") else "match"
            confidence = "low_review_required" if scope == "team" else "medium_review_required"
            notes = "secondary_stat_market_name_based_mapping_review_required"

    try:
        line = float(handicap)
    except Exception:
        line = float("nan")
    if market_key in {"1x2", "btts"}:
        line = float("nan")
    return {
        "internal_market_key": market_key,
        "internal_scope": scope,
        "internal_side": side,
        "internal_line": line,
        "mapping_confidence": confidence,
        "mapping_notes": notes,
        "provider_market_type": market_type,
        "provider_period": period,
    }


def build_market_mapping_frame(markets_df: pd.DataFrame) -> pd.DataFrame:
    if markets_df is None or markets_df.empty:
        return pd.DataFrame()
    rows = []
    for _, row in markets_df.iterrows():
        inferred = infer_internal_market_from_oddspapi_market(row)
        out = row.to_dict()
        out.update(inferred)
        rows.append(out)
    return pd.DataFrame(rows)


def _market_lookup(markets_df: pd.DataFrame | None) -> dict[int, dict[str, Any]]:
    if markets_df is None or markets_df.empty:
        return {}
    lookup: dict[int, dict[str, Any]] = {}
    for _, row in markets_df.iterrows():
        # outcomeId is the precise selection key. marketId is only a fallback and
        # must not overwrite a real outcome row from the same market.
        oid = row.get("outcomeId")
        try:
            lookup[int(float(oid))] = row.to_dict()
        except Exception:
            pass
        mid = row.get("marketId")
        try:
            lookup.setdefault(int(float(mid)), row.to_dict())
        except Exception:
            pass
    return lookup


def _as_int_or_none(value: Any) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return None


def _iter_v5_odds(odds_obj: Any) -> Iterable[tuple[str, str, dict[str, Any]]]:
    if not isinstance(odds_obj, dict):
        return []
    rows = []
    for bookmaker, prices_by_id in odds_obj.items():
        if not isinstance(prices_by_id, dict):
            continue
        for odds_id, price_or_timeline in prices_by_id.items():
            if isinstance(price_or_timeline, dict) and "price" in price_or_timeline:
                item = dict(price_or_timeline)
                item.setdefault("bookmaker", bookmaker)
                item.setdefault("oddsId", odds_id)
                rows.append((str(bookmaker), str(odds_id), item))
            elif isinstance(price_or_timeline, dict):
                for _, item in price_or_timeline.items():
                    if isinstance(item, dict) and "price" in item:
                        row = dict(item)
                        row.setdefault("bookmaker", bookmaker)
                        row.setdefault("oddsId", odds_id)
                        rows.append((str(bookmaker), str(odds_id), row))
            elif isinstance(price_or_timeline, list):
                for item in price_or_timeline:
                    if isinstance(item, dict) and "price" in item:
                        row = dict(item)
                        row.setdefault("bookmaker", bookmaker)
                        row.setdefault("oddsId", odds_id)
                        rows.append((str(bookmaker), str(odds_id), row))
    return rows


def _iter_bookmakers_market_tree(bookmakers_obj: Any) -> Iterable[tuple[str, str, dict[str, Any]]]:
    """Flatten historical tree shape: bookmakers -> markets -> outcomes -> players -> snapshots."""
    if not isinstance(bookmakers_obj, dict):
        return []
    rows = []
    for bookmaker, bdata in bookmakers_obj.items():
        if not isinstance(bdata, dict):
            continue
        markets = bdata.get("markets") or {}
        if isinstance(markets, list):
            markets = {str(m.get("marketId", i)): m for i, m in enumerate(markets) if isinstance(m, dict)}
        for market_id, mdata in markets.items():
            if not isinstance(mdata, dict):
                continue
            outcomes = mdata.get("outcomes") or {}
            if isinstance(outcomes, list):
                outcomes = {str(o.get("outcomeId", i)): o for i, o in enumerate(outcomes) if isinstance(o, dict)}
            for outcome_id, odata in outcomes.items():
                if not isinstance(odata, dict):
                    continue
                players = odata.get("players")
                if isinstance(players, dict):
                    for player_id, timeline in players.items():
                        snapshots = timeline if isinstance(timeline, list) else [timeline]
                        for snap in snapshots:
                            if isinstance(snap, dict) and "price" in snap:
                                item = dict(snap)
                                item.setdefault("bookmaker", bookmaker)
                                item.setdefault("marketId", market_id)
                                item.setdefault("outcomeId", outcome_id)
                                item.setdefault("playerId", player_id)
                                rows.append((str(bookmaker), f"tree:{bookmaker}:{outcome_id}:{player_id}", item))
                else:
                    snapshots = odata if isinstance(odata, list) else [odata]
                    for snap in snapshots:
                        if isinstance(snap, dict) and "price" in snap:
                            item = dict(snap)
                            item.setdefault("bookmaker", bookmaker)
                            item.setdefault("marketId", market_id)
                            item.setdefault("outcomeId", outcome_id)
                            item.setdefault("playerId", 0)
                            rows.append((str(bookmaker), f"tree:{bookmaker}:{outcome_id}:0", item))
    return rows


def _iter_legacy_bookmaker_odds(bookmaker_odds: Any) -> Iterable[tuple[str, str, dict[str, Any]]]:
    """Support older OddsPapi/RapidAPI examples using bookmakerOdds."""
    if not isinstance(bookmaker_odds, dict):
        return []
    rows = []
    for bookmaker, bdata in bookmaker_odds.items():
        if not isinstance(bdata, dict):
            continue
        markets = bdata.get("markets") or {}
        for market_id, mdata in (markets.items() if isinstance(markets, dict) else []):
            outcomes = mdata.get("outcomes") if isinstance(mdata, dict) else {}
            if isinstance(outcomes, dict):
                iter_outcomes = outcomes.items()
            elif isinstance(outcomes, list):
                iter_outcomes = [(str(o.get("outcomeId", i)), o) for i, o in enumerate(outcomes) if isinstance(o, dict)]
            else:
                iter_outcomes = []
            for outcome_id, outcome_data in iter_outcomes:
                if isinstance(outcome_data, dict) and "price" in outcome_data:
                    item = dict(outcome_data)
                    item.setdefault("bookmaker", bookmaker)
                    item.setdefault("marketId", market_id)
                    item.setdefault("outcomeId", item.get("outcomeId") or outcome_id)
                    item.setdefault("playerId", item.get("playerId") or 0)
                    rows.append((str(bookmaker), f"legacy:{bookmaker}:{outcome_id}:0", item))
                elif isinstance(outcome_data, dict) and "players" in outcome_data:
                    for player_id, pdata in (outcome_data.get("players") or {}).items():
                        snapshots = pdata if isinstance(pdata, list) else [pdata]
                        for snap in snapshots:
                            if isinstance(snap, dict) and "price" in snap:
                                item = dict(snap)
                                item.setdefault("bookmaker", bookmaker)
                                item.setdefault("marketId", market_id)
                                item.setdefault("outcomeId", outcome_id)
                                item.setdefault("playerId", player_id)
                                rows.append((str(bookmaker), f"legacy:{bookmaker}:{outcome_id}:{player_id}", item))
    return rows


def _extract_price_rows(payload: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    rows = []
    rows.extend(_iter_v5_odds(payload.get("odds")))
    rows.extend(_iter_bookmakers_market_tree(payload.get("bookmakers")))
    rows.extend(_iter_legacy_bookmaker_odds(payload.get("bookmakerOdds")))
    return rows


def _fixture_core(payload: dict[str, Any]) -> dict[str, Any]:
    data = _unwrap_payload(payload)
    if not isinstance(data, dict):
        data = payload
    participants = data.get("participants") or {}
    if isinstance(participants, list):
        p1 = participants[0] if len(participants) > 0 and isinstance(participants[0], dict) else {}
        p2 = participants[1] if len(participants) > 1 and isinstance(participants[1], dict) else {}
        participants = {
            "participant1Name": p1.get("participantName") or p1.get("name"),
            "participant2Name": p2.get("participantName") or p2.get("name"),
        }
    start_epoch = data.get("startTime") or data.get("commenceTime") or data.get("kickoff")
    if isinstance(start_epoch, str) and not start_epoch.isdigit():
        try:
            start_epoch = _dt_to_epoch_seconds(start_epoch)
        except Exception:
            pass
    return {
        "fixture_id": data.get("fixtureId") or data.get("id") or data.get("eventId"),
        "start_epoch": start_epoch,
        "home": participants.get("participant1Name") or data.get("homeTeam") or data.get("home_team") or "",
        "away": participants.get("participant2Name") or data.get("awayTeam") or data.get("away_team") or "",
        "status": data.get("status") or {},
    }


def _subject_team_from_market(inferred: dict[str, Any], market_row: dict[str, Any], home: str, away: str, item: dict[str, Any]) -> str:
    market_type = inferred.get("provider_market_type") or norm_key(market_row.get("marketType"))
    market_name = norm_text(market_row.get("marketName"))
    outcome = norm_text(market_row.get("outcomeName") or item.get("outcomeName"))
    if "team1" in market_type or "team 1" in market_name:
        return home
    if "team2" in market_type or "team 2" in market_name:
        return away
    if outcome in {"1", "home"}:
        return home
    if outcome in {"2", "away"}:
        return away
    return item.get("teamName") or item.get("participantName") or ""


def flatten_oddspapi_odds_response(
    payload: dict[str, Any],
    *,
    markets_df: pd.DataFrame | None = None,
    snapshot_policy: str = "all",
    pre_kickoff_seconds: int = 3600,
    include_unmapped: bool = True,
) -> pd.DataFrame:
    """Flatten OddsPapi current/historical fixture odds to Mundialytics odds schema.

    Supports v5 REST shapes (`odds`, `bookmakers` market tree) and older RapidAPI
    `bookmakerOdds` examples. `snapshot_policy` controls leakage:
    - all: all snapshots.
    - closing: latest price before kickoff.
    - pre_kickoff: latest price at least N seconds before kickoff.
    """
    payload = _unwrap_payload(payload)
    if not isinstance(payload, dict):
        return pd.DataFrame(columns=ODDS_INPUT_COLUMNS)
    markets_lookup = _market_lookup(markets_df)
    core = _fixture_core(payload)
    fixture_id = core["fixture_id"]
    start_epoch = core["start_epoch"]
    kickoff_ms = None
    try:
        kickoff_ms = int(float(start_epoch)) * 1000 if start_epoch is not None else None
    except Exception:
        kickoff_ms = None
    date = epoch_seconds_to_iso(start_epoch)[:10] if start_epoch is not None else ""
    home = core["home"]
    away = core["away"]
    price_rows = _extract_price_rows(payload)
    raw_rows: list[dict[str, Any]] = []
    for bookmaker, odds_id, item in price_rows:
        outcome_id = item.get("outcomeId") or item.get("id")
        outcome_id_int = _as_int_or_none(outcome_id)
        market_id_int = _as_int_or_none(item.get("marketId"))
        market_row = markets_lookup.get(outcome_id_int, {}) if outcome_id_int is not None else {}
        if not market_row and market_id_int is not None:
            market_row = markets_lookup.get(market_id_int, {})
        inferred = infer_internal_market_from_oddspapi_market(market_row) if market_row else {}
        if not inferred and not include_unmapped:
            continue
        market_key = inferred.get("internal_market_key", "")
        side = inferred.get("internal_side", "") or norm_key(market_row.get("outcomeName", item.get("outcomeName", "")))
        line = inferred.get("internal_line", float("nan"))
        player_name = item.get("playerName") or item.get("player") or item.get("description") or ""
        player_id = item.get("playerId")
        if player_id in (0, "0", None):
            player_id = ""
        # Docs use changedAt for current odds and createdAt in some historical examples.
        changed_at = item.get("changedAt") or item.get("createdAt") or item.get("bookmakerChangedAt") or item.get("timestamp")
        if isinstance(changed_at, str) and not changed_at.isdigit():
            try:
                changed_at = int(datetime.fromisoformat(changed_at.replace("Z", "+00:00")).timestamp() * 1000)
            except Exception:
                pass
        subject_team = ""
        if inferred.get("internal_scope") == "team":
            subject_team = _subject_team_from_market(inferred, market_row, home, away, item)
        raw_rows.append({
            "snapshot_time_utc": epoch_millis_to_iso(changed_at),
            "bookmaker": item.get("bookmaker") or bookmaker,
            "provider": PROVIDER_NAME,
            "provider_event_id": fixture_id,
            "internal_match_id": "",
            "match_id": "",
            "date": date,
            "home_team": home,
            "away_team": away,
            "market_key": market_key,
            "market": market_key or market_row.get("marketName", ""),
            "scope": inferred.get("internal_scope", ""),
            "subject_team": subject_team,
            "subject_player": str(player_name) if player_name else "",
            "line": line,
            "side": side,
            "bookmaker_odds": item.get("price"),
            "is_live": bool((core.get("status") or {}).get("live")),
            "source_url": "",
            "notes": f"odds_id={odds_id};outcome_id={outcome_id};player_id={player_id};mapping={inferred.get('mapping_confidence','unmapped')};provider_market={market_row.get('marketName','')}",
            "_odds_id": odds_id,
            "_changed_at_ms": pd.to_numeric(changed_at, errors="coerce"),
            "_kickoff_ms": kickoff_ms,
        })
    if not raw_rows:
        return pd.DataFrame(columns=ODDS_INPUT_COLUMNS)
    df = pd.DataFrame(raw_rows)
    if snapshot_policy != "all":
        df = _select_snapshot_rows(df, policy=snapshot_policy, pre_kickoff_seconds=pre_kickoff_seconds)
    return standard_odds_input_frame(df)


def _select_snapshot_rows(df: pd.DataFrame, *, policy: str, pre_kickoff_seconds: int) -> pd.DataFrame:
    work = df.copy()
    changed = pd.to_numeric(work["_changed_at_ms"], errors="coerce")
    kickoff = pd.to_numeric(work["_kickoff_ms"], errors="coerce")
    if policy == "closing":
        valid = changed.le(kickoff).fillna(True)
    elif policy == "pre_kickoff":
        valid = changed.le(kickoff - int(pre_kickoff_seconds) * 1000).fillna(True)
    else:
        raise ValueError("snapshot_policy must be all, closing or pre_kickoff")
    work = work[valid].copy()
    if work.empty:
        return work
    work = work.sort_values(["_odds_id", "_changed_at_ms"])
    return work.groupby("_odds_id", as_index=False).tail(1)


def build_fixture_windows_from_model_lines(
    model_lines: pd.DataFrame,
    *,
    chunk_days: int = 7,
    pad_hours: int = 12,
    max_windows: int | None = None,
) -> pd.DataFrame:
    if model_lines is None or model_lines.empty:
        return pd.DataFrame(columns=["window_id", "start_date", "end_date", "startTimeFrom", "startTimeTo", "expected_matches"])
    work = model_lines.copy()
    date_col = "kickoff_utc" if "kickoff_utc" in work.columns and work["kickoff_utc"].notna().any() else "date"
    work["_date"] = pd.to_datetime(work[date_col], errors="coerce", utc=True).dt.date
    matches = work.dropna(subset=["_date"]).drop_duplicates(["match_id", "_date"])
    if matches.empty:
        return pd.DataFrame(columns=["window_id", "start_date", "end_date", "startTimeFrom", "startTimeTo", "expected_matches"])
    min_date = pd.Timestamp(matches["_date"].min())
    max_date = pd.Timestamp(matches["_date"].max())
    windows = []
    current = min_date
    wid = 1
    while current <= max_date:
        end = min(current + pd.Timedelta(days=max(1, chunk_days) - 1), max_date)
        start_ts = pd.Timestamp(current).tz_localize("UTC") - pd.Timedelta(hours=pad_hours)
        end_ts = pd.Timestamp(end).tz_localize("UTC") + pd.Timedelta(days=1, hours=pad_hours) - pd.Timedelta(seconds=1)
        mask = (pd.to_datetime(matches["_date"]) >= current) & (pd.to_datetime(matches["_date"]) <= end)
        windows.append({
            "window_id": wid,
            "start_date": str(current.date()),
            "end_date": str(end.date()),
            "startTimeFrom": int(start_ts.timestamp()),
            "startTimeTo": int(end_ts.timestamp()),
            "expected_matches": int(mask.sum()),
        })
        wid += 1
        current = end + pd.Timedelta(days=1)
        if max_windows is not None and len(windows) >= max_windows:
            break
    return pd.DataFrame(windows)


def match_model_lines_to_provider_fixtures(model_lines: pd.DataFrame, fixtures_df: pd.DataFrame, *, max_date_diff_days: int = 1) -> pd.DataFrame:
    """Create fuzzy match candidates between internal matches and OddsPapi fixtures."""
    import difflib

    if model_lines.empty or fixtures_df.empty:
        return pd.DataFrame()
    matches = model_lines.drop_duplicates("match_id").copy()
    date_col = "kickoff_utc" if "kickoff_utc" in matches.columns and matches["kickoff_utc"].notna().any() else "date"
    matches["_date"] = pd.to_datetime(matches[date_col], errors="coerce", utc=True).dt.date
    fixtures = fixtures_df.copy()
    fixtures["_date"] = pd.to_datetime(fixtures.get("kickoff_utc", fixtures.get("date", "")), errors="coerce", utc=True).dt.date
    rows = []
    for _, m in matches.iterrows():
        mdate = m.get("_date")
        if pd.isna(mdate):
            continue
        for _, f in fixtures.iterrows():
            fdate = f.get("_date")
            if pd.isna(fdate):
                continue
            ddays = abs((pd.Timestamp(mdate) - pd.Timestamp(fdate)).days)
            if ddays > max_date_diff_days:
                continue
            home_score = difflib.SequenceMatcher(None, norm_text(m.get("home_team")), norm_text(f.get("home_team"))).ratio()
            away_score = difflib.SequenceMatcher(None, norm_text(m.get("away_team")), norm_text(f.get("away_team"))).ratio()
            swapped_home = difflib.SequenceMatcher(None, norm_text(m.get("home_team")), norm_text(f.get("away_team"))).ratio()
            swapped_away = difflib.SequenceMatcher(None, norm_text(m.get("away_team")), norm_text(f.get("home_team"))).ratio()
            direct = (home_score + away_score) / 2.0
            swapped = (swapped_home + swapped_away) / 2.0
            rows.append({
                "match_id": m.get("match_id"),
                "date": m.get("date"),
                "home_team": m.get("home_team"),
                "away_team": m.get("away_team"),
                "provider": PROVIDER_NAME,
                "provider_fixture_id": f.get("provider_fixture_id") or f.get("fixture_id"),
                "provider_home_team": f.get("home_team"),
                "provider_away_team": f.get("away_team"),
                "provider_kickoff_utc": f.get("kickoff_utc"),
                "date_diff_days": ddays,
                "direct_team_score": direct,
                "swapped_team_score": swapped,
                "best_score": max(direct, swapped),
                "orientation": "direct" if direct >= swapped else "swapped",
                "auto_match": bool(max(direct, swapped) >= 0.86 and ddays <= max_date_diff_days),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["match_id", "auto_match", "best_score"], ascending=[True, False, False])
    return out
