
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import re

import pandas as pd

from mundialytics.data_quality.team_registry import normalize_provider_name, provider_alias_map


ADVANCED_DATA_VERSION = "v0.50.3_fbref_team_match_normalization_repair"

ADVANCED_MATCH_COLUMNS = [
    "provider",
    "provider_match_id",
    "date",
    "competition",
    "season",
    "stage",
    "team_scope",
    "neutral",
    "venue_country",
    "venue_city",
    "home_team_country",
    "away_team_country",
    "home_team_city",
    "away_team_city",
    "home_team",
    "away_team",
    "home_xg",
    "away_xg",
    "home_npxg",
    "away_npxg",
    "home_xa",
    "away_xa",
    "home_npxg_plus_xa",
    "away_npxg_plus_xa",
    "home_shots",
    "away_shots",
    "home_sot",
    "away_sot",
    "home_shots_inside_box",
    "away_shots_inside_box",
    "home_shots_outside_box",
    "away_shots_outside_box",
    "home_avg_shot_distance",
    "away_avg_shot_distance",
    "home_xg_per_shot",
    "away_xg_per_shot",
    "home_header_shots",
    "away_header_shots",
    "home_left_foot_shots",
    "away_left_foot_shots",
    "home_right_foot_shots",
    "away_right_foot_shots",
    "home_header_xg",
    "away_header_xg",
    "home_left_foot_xg",
    "away_left_foot_xg",
    "home_right_foot_xg",
    "away_right_foot_xg",
    "home_penalty_xg",
    "away_penalty_xg",
    "home_open_play_xg",
    "away_open_play_xg",
    "home_set_piece_xg",
    "away_set_piece_xg",
    "home_corner_xg",
    "away_corner_xg",
    "home_free_kick_xg",
    "away_free_kick_xg",
    "home_counterattack_xg",
    "away_counterattack_xg",
    "home_big_chances",
    "away_big_chances",
    "home_corners",
    "away_corners",
    "home_fouls",
    "away_fouls",
    "home_yellow_cards",
    "away_yellow_cards",
    "home_red_cards",
    "away_red_cards",
    "home_possession",
    "away_possession",
    "home_field_tilt",
    "away_field_tilt",
    "home_ppda",
    "away_ppda",
    "home_touches_attacking_third",
    "away_touches_attacking_third",
    "home_touches_box",
    "away_touches_box",
    "home_final_third_entries",
    "away_final_third_entries",
    "home_deep_completions",
    "away_deep_completions",
    "home_progressive_passes",
    "away_progressive_passes",
    "home_progressive_carries",
    "away_progressive_carries",
    "home_passes_into_final_third",
    "away_passes_into_final_third",
    "home_passes_into_penalty_area",
    "away_passes_into_penalty_area",
    "home_crosses_into_penalty_area",
    "away_crosses_into_penalty_area",
    "home_through_balls",
    "away_through_balls",
    "home_shot_creating_actions",
    "away_shot_creating_actions",
    "home_goal_creating_actions",
    "away_goal_creating_actions",
    "home_tackles",
    "away_tackles",
    "home_interceptions",
    "away_interceptions",
    "home_blocks",
    "away_blocks",
    "home_clearances",
    "away_clearances",
    "home_pressures",
    "away_pressures",
    "home_ball_recoveries",
    "away_ball_recoveries",
    "home_errors",
    "away_errors",
    "home_keeper_saves",
    "away_keeper_saves",
    "home_keeper_psxg",
    "away_keeper_psxg",
    "home_keeper_goals_against",
    "away_keeper_goals_against",
    "home_keeper_save_pct",
    "away_keeper_save_pct",
    "source_confidence",
    "join_method",
]

PLAYER_MATCH_COLUMNS = [
    "provider",
    "provider_match_id",
    "match_id",
    "date",
    "competition",
    "season",
    "team",
    "opponent",
    "player",
    "player_id",
    "position",
    "started",
    "minutes",
    "goals",
    "assists",
    "xg",
    "npxg",
    "xa",
    "xg_chain",
    "xg_buildup",
    "shots",
    "sot",
    "key_passes",
    "sca",
    "gca",
    "passes",
    "passes_completed",
    "progressive_passes",
    "progressive_carries",
    "carries_into_final_third",
    "carries_into_penalty_area",
    "touches",
    "touches_box",
    "tackles",
    "interceptions",
    "blocks",
    "pressures",
    "fouls",
    "fouled",
    "yellow_cards",
    "red_cards",
    "saves",
    "psxg",
]

SHOT_EVENT_COLUMNS = [
    "provider",
    "provider_match_id",
    "match_id",
    "date",
    "competition",
    "season",
    "team",
    "opponent",
    "player",
    "player_id",
    "minute",
    "second",
    "xg",
    "psxg",
    "outcome",
    "body_part",
    "situation",
    "shot_type",
    "assisted_by",
    "x",
    "y",
    "distance_to_goal",
    "is_penalty",
    "is_goal",
    "is_header",
    "is_left_foot",
    "is_right_foot",
    "is_inside_box",
    "is_outside_box",
]

LINEUP_COLUMNS = [
    "provider",
    "provider_match_id",
    "match_id",
    "date",
    "competition",
    "season",
    "team",
    "opponent",
    "player",
    "player_id",
    "position",
    "started",
    "bench",
    "minutes_played",
    "shirt_number",
    "formation_position",
    "provider_position_id",
]


@dataclass(frozen=True)
class AdvancedImportOutputs:
    advanced_matches: pd.DataFrame
    player_matches: pd.DataFrame
    shot_events: pd.DataFrame
    lineups: pd.DataFrame
    report: dict[str, Any]


@dataclass(frozen=True)
class AdvancedMergeOutputs:
    canonical_advanced_matches: pd.DataFrame
    provider_priority_report: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class AdvancedEnrichmentOutputs:
    enriched_matches: pd.DataFrame
    canonical_advanced_matches: pd.DataFrame
    join_report: pd.DataFrame
    summary: dict[str, Any]


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("home_team_name") or value.get("away_team_name") or "")
    return str(value or "")


def _is_event_type(event: dict[str, Any], name: str) -> bool:
    typ = event.get("type")
    return isinstance(typ, dict) and str(typ.get("name") or "").lower() == name.lower()


def _side(team: str, home: str, away: str) -> str | None:
    team_n = normalize_provider_name(team)
    if team_n == normalize_provider_name(home):
        return "home"
    if team_n == normalize_provider_name(away):
        return "away"
    return None


def _safe_numeric(value: Any) -> float | pd.NA:
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")


def _pick_column(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    lower = {str(c).lower().strip(): c for c in df.columns}
    normalized = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        key = str(alias).lower().strip()
        if key in lower:
            return lower[key]
        norm = re.sub(r"[^a-z0-9]+", "", key)
        if norm in normalized:
            return normalized[norm]
    return None


def _make_match_key(df: pd.DataFrame, *, home_col: str = "home_team", away_col: str = "away_team") -> pd.Series:
    dates = pd.to_datetime(df.get("date"), errors="coerce").dt.date.astype("string").fillna("")
    home = df.get(home_col, pd.Series([""] * len(df), index=df.index)).map(normalize_provider_name).fillna("")
    away = df.get(away_col, pd.Series([""] * len(df), index=df.index)).map(normalize_provider_name).fillna("")
    return dates + "|" + home + "|" + away


def _build_team_join_alias_map(
    registry: pd.DataFrame | None,
    provider_alias_column: str,
    manual_aliases: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Build a provider-name -> canonical join-name map.

    The join layer must map *both* Football-Data short names and provider names
    to the same canonical key. Registry-only mapping is often incomplete for
    FBref/StatsBomb, so v0.50.2 accepts a small manual alias CSV with columns
    like ``football_data_name,provider_name,canonical_name``.
    """
    alias_map: dict[str, str] = {}

    def add_alias(alias: Any, canonical: Any) -> None:
        alias_norm = normalize_provider_name(alias)
        canonical_norm = normalize_provider_name(canonical)
        if alias_norm and canonical_norm:
            alias_map[alias_norm] = canonical_norm

    if registry is not None and not registry.empty:
        provider_columns = [
            provider_alias_column,
            "canonical_team_name",
            "football_data_name",
            "clubelo_name",
            "understat_name",
            "statsbomb_name",
            "fbref_name",
            "provider_name",
        ]
        provider_columns = [c for c in dict.fromkeys(provider_columns) if c in registry.columns]
        for _, row in registry.iterrows():
            canonical = row.get("football_data_name") or row.get("canonical_team_name")
            if not canonical:
                continue
            add_alias(canonical, canonical)
            for col in provider_columns:
                add_alias(row.get(col), canonical)

    if manual_aliases is not None and not manual_aliases.empty:
        manual_columns = [c for c in ["football_data_name", "provider_name", "canonical_name"] if c in manual_aliases.columns]
        for _, row in manual_aliases.iterrows():
            canonical = row.get("canonical_name") or row.get("football_data_name") or row.get("provider_name")
            if not canonical:
                continue
            for col in manual_columns:
                add_alias(row.get(col), canonical)
            add_alias(canonical, canonical)

    return alias_map


def _apply_team_join_aliases(
    df: pd.DataFrame,
    alias_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    alias_map = alias_map or {}
    for side in ["home", "away"]:
        col = f"{side}_team"
        out[f"{col}_join_name"] = out[col].map(
            lambda v: alias_map.get(normalize_provider_name(v), normalize_provider_name(v))
        )
    return out


def _apply_registry_aliases(
    df: pd.DataFrame,
    registry: pd.DataFrame | None,
    provider_alias_column: str,
    manual_aliases: pd.DataFrame | None = None,
) -> pd.DataFrame:
    alias_map = _build_team_join_alias_map(registry, provider_alias_column, manual_aliases)
    return _apply_team_join_aliases(df, alias_map)


def _normalise_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _venue_value(row: pd.Series, *names: str) -> str:
    for name in names:
        if name in row and pd.notna(row.get(name)) and str(row.get(name)).strip():
            return str(row.get(name)).strip()
    return ""


def venue_advantage_side(row: pd.Series) -> str:
    """Return which listed side should receive venue/home advantage.

    Normal league home matches return ``home``. Neutral fixtures return ``none``
    unless the venue country/city matches one of the listed teams/countries.
    This supports cases like a Champions League final in Madrid involving Real
    Madrid, or a World Cup match in the USA involving the USA.
    """
    neutral = int(pd.to_numeric(pd.Series([row.get("neutral", 0)]), errors="coerce").fillna(0).iloc[0])
    if neutral == 0:
        return "home"

    venue_country = _normalise_text(_venue_value(row, "venue_country", "host_country", "country"))
    venue_city = _normalise_text(_venue_value(row, "venue_city", "city"))

    home_country = _normalise_text(_venue_value(row, "home_team_country", "home_country"))
    away_country = _normalise_text(_venue_value(row, "away_team_country", "away_country"))
    home_city = _normalise_text(_venue_value(row, "home_team_city", "home_city"))
    away_city = _normalise_text(_venue_value(row, "away_team_city", "away_city"))
    home_team = _normalise_text(_venue_value(row, "home_team"))
    away_team = _normalise_text(_venue_value(row, "away_team"))

    home_match = bool((venue_country and venue_country == home_country) or (venue_city and venue_city in {home_city, home_team}))
    away_match = bool((venue_country and venue_country == away_country) or (venue_city and venue_city in {away_city, away_team}))

    if home_match and not away_match:
        return "home"
    if away_match and not home_match:
        return "away"
    return "none"


def venue_home_advantage_factor(row: pd.Series) -> float:
    side = venue_advantage_side(row)
    if side == "home":
        return 1.0
    if side == "away":
        return -1.0
    return 0.0


def _is_inside_statsbomb_penalty_box(x: Any, y: Any) -> bool:
    """Approximate StatsBomb penalty-box flag on a 120x80 pitch."""
    try:
        xf = float(x)
        yf = float(y)
    except (TypeError, ValueError):
        return False
    return xf >= 102.0 and 18.0 <= yf <= 62.0


def _distance_to_goal_statsbomb(x: Any, y: Any) -> float | pd.NA:
    try:
        xf = float(x)
        yf = float(y)
    except (TypeError, ValueError):
        return pd.NA
    return ((120.0 - xf) ** 2 + (40.0 - yf) ** 2) ** 0.5


TEAM_ROW_STAT_ALIASES: dict[str, list[str]] = {
    "xg": ["xg", "expected_goals", "xg_for", "gf_xg"],
    "npxg": ["npxg", "np_xg", "non_penalty_xg"],
    "xa": ["xa", "xag", "expected_assists"],
    "shots": ["shots", "sh", "total_shots"],
    "sot": ["sot", "shots_on_target", "shotsontarget"],
    "corners": ["corners", "corner_kicks", "ck"],
    "fouls": ["fouls", "fls"],
    "yellow_cards": ["yellow_cards", "cards_yellow", "yellow"],
    "red_cards": ["red_cards", "cards_red", "red"],
    "possession": ["possession", "poss"],
    "field_tilt": ["field_tilt"],
    "ppda": ["ppda"],
    "touches_box": ["touches_box", "touches_att_pen_area", "touches_att_penalty_area"],
    "progressive_passes": ["progressive_passes", "prgp", "prog_passes"],
    "progressive_carries": ["progressive_carries", "prgc", "prog_carries"],
    "passes_into_final_third": ["passes_into_final_third", "1/3", "passes_1/3"],
    "passes_into_penalty_area": ["passes_into_penalty_area", "ppa"],
    "crosses_into_penalty_area": ["crosses_into_penalty_area", "crspa"],
    "shot_creating_actions": ["shot_creating_actions", "sca"],
    "goal_creating_actions": ["goal_creating_actions", "gca"],
    "tackles": ["tackles", "tkl"],
    "interceptions": ["interceptions", "int"],
    "blocks": ["blocks", "blocks_total"],
    "clearances": ["clearances", "clr"],
    "pressures": ["pressures", "press"],
    "keeper_saves": ["keeper_saves", "saves", "save"],
    "keeper_psxg": ["keeper_psxg", "psxg", "post_shot_xg"],
    "keeper_goals_against": ["keeper_goals_against", "goals_against", "ga"],
    "keeper_save_pct": ["keeper_save_pct", "save_pct", "savepct"],
}


def _copy_first_existing_numeric(df: pd.DataFrame, target: str, candidates: list[str]) -> pd.DataFrame:
    """Copy the first present candidate column into target without overwriting existing values.

    Some soccerdata/FBref exports flatten multi-index columns into ambiguous
    labels such as ``Standard.1`` or ``Performance.2``.  This helper lets us
    create explicit canonical aliases before the generic team-match importer
    runs.
    """
    if target in df.columns and df[target].notna().any():
        return df
    for candidate in candidates:
        if candidate in df.columns:
            df[target] = pd.to_numeric(df[candidate], errors="coerce")
            return df
    return df


def _with_fbref_team_match_aliases(df: pd.DataFrame, *, provider: str) -> pd.DataFrame:
    """Add explicit metric aliases for soccerdata FBref team-match exports.

    soccerdata's ``FBref.read_team_match_stats`` can emit flattened columns like
    ``Standard``, ``Standard.1`` or ``Performance.2`` instead of semantic names.
    In the observed raw exports:

    * shooting: Standard=goals, Standard.1=shots, Standard.2=shots on target,
      Standard.3=SoT%, Standard.4=G/Sh, Standard.5=G/SoT, Standard.6=FK,
      Standard.7=PK.  xG/npxG are not present in this raw file.
    * misc: Performance=CrdY, Performance.1=CrdR, Performance.3=Fls,
      Performance.4=Fld, Performance.6=Crs, Performance.7=Int,
      Performance.8=TklW.
    * keeper: Performance=SoTA, Performance.1=GA, Performance.2=Saves,
      Performance.3=Save%, Performance.4=CS.

    The mapping is intentionally conservative: counts are mapped, but derived
    percentages are only used for keeper_save_pct and xG is left null when the
    export does not contain it.
    """
    out = df.copy()
    provider_token = str(provider or "").lower()

    if "fbref" not in provider_token:
        return out

    if "shooting" in provider_token:
        out = _copy_first_existing_numeric(out, "shots", ["Sh", "sh", "Standard.1"])
        out = _copy_first_existing_numeric(out, "sot", ["SoT", "sot", "Standard.2"])
        # These are counts, not xG.  Keep xG/npxG empty unless a semantic xG
        # column is really present in the raw file.
        out = _copy_first_existing_numeric(out, "free_kick_shots", ["FK", "Standard.6"])
        out = _copy_first_existing_numeric(out, "penalty_goals", ["PK", "Standard.7"])

    if "misc" in provider_token:
        out = _copy_first_existing_numeric(out, "yellow_cards", ["CrdY", "Performance"])
        out = _copy_first_existing_numeric(out, "red_cards", ["CrdR", "Performance.1"])
        out = _copy_first_existing_numeric(out, "fouls", ["Fls", "Performance.3"])
        out = _copy_first_existing_numeric(out, "interceptions", ["Int", "Performance.7"])
        out = _copy_first_existing_numeric(out, "tackles", ["TklW", "Performance.8"])
        # FBref Misc ``Crs`` is all crosses, not crosses into the box.  Store it
        # only if a future canonical field is added; do not mislabel it as box crosses.

    if "keeper" in provider_token:
        out = _copy_first_existing_numeric(out, "keeper_goals_against", ["GA", "Performance.1"])
        out = _copy_first_existing_numeric(out, "keeper_saves", ["Saves", "Save", "Performance.2"])
        out = _copy_first_existing_numeric(out, "keeper_save_pct", ["Save%", "SavePct", "Performance.3"])

    if "schedule" in provider_token:
        out = _copy_first_existing_numeric(out, "possession", ["Poss", "possession"])

    return out



def _is_team_match_provider_csv(df: pd.DataFrame) -> bool:
    team = _pick_column(df, ["team", "squad"])
    opponent = _pick_column(df, ["opponent", "opp"])
    venue = _pick_column(df, ["venue", "home_away", "homeaway"])
    return bool(team and opponent and venue)


def _canonicalize_team_match_provider_csv(df: pd.DataFrame, *, provider: str) -> pd.DataFrame:
    """Normalize one-row-per-team match stats, common in FBref/soccerdata exports."""
    df = _with_fbref_team_match_aliases(df, provider=provider)
    team_col = _pick_column(df, ["team", "squad"])
    opp_col = _pick_column(df, ["opponent", "opp"])
    venue_col = _pick_column(df, ["venue", "home_away", "homeaway"])
    date_col = _pick_column(df, ["date", "match_date", "datetime", "kickoff"])
    comp_col = _pick_column(df, ["competition", "league", "comp", "competition_name"])
    season_col = _pick_column(df, ["season", "season_name", "year"])
    match_id_col = _pick_column(df, ["provider_match_id", "match_id", "game_id", "id", "fixture_id"])

    if not team_col or not opp_col or not venue_col or not date_col:
        return _canonicalize_match_provider_csv(df, provider=provider)

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        venue = str(row.get(venue_col, "")).lower()
        is_home = venue.startswith("home") or venue in {"h", "1", "true"}
        is_away = venue.startswith("away") or venue in {"a", "0", "false"}
        if not is_home and not is_away:
            continue
        side = "home" if is_home else "away"
        other = "away" if is_home else "home"
        rec: dict[str, Any] = {
            "provider": provider,
            "provider_match_id": row.get(match_id_col) if match_id_col else pd.NA,
            "date": row.get(date_col),
            "competition": row.get(comp_col) if comp_col else pd.NA,
            "season": row.get(season_col) if season_col else pd.NA,
            "home_team": row.get(team_col) if is_home else row.get(opp_col),
            "away_team": row.get(opp_col) if is_home else row.get(team_col),
            "source_confidence": "provider_team_match_csv",
            "join_method": "date_team",
        }
        for stat, aliases in TEAM_ROW_STAT_ALIASES.items():
            c = _pick_column(df, aliases)
            if c:
                rec[f"{side}_{stat}"] = row.get(c)
            # FBref often carries against columns in the same row.
            against_c = _pick_column(df, [f"{aliases[0]}_against", f"{stat}_against", f"{stat}_allowed", f"{stat}_ga"])
            if against_c:
                rec[f"{other}_{stat}"] = row.get(against_c)
        records.append(rec)

    if not records:
        return _canonicalize_match_provider_csv(df, provider=provider)

    tmp = pd.DataFrame(records)
    tmp["match_join_key"] = _make_match_key(tmp)
    combined_rows: list[dict[str, Any]] = []
    for _, g in tmp.groupby("match_join_key", dropna=False, sort=False):
        out: dict[str, Any] = {}
        for col in ADVANCED_MATCH_COLUMNS:
            vals = g[col] if col in g.columns else pd.Series(dtype=object)
            non_null = vals.dropna()
            out[col] = non_null.iloc[0] if len(non_null) else pd.NA
        combined_rows.append(out)
    out_df = pd.DataFrame(combined_rows)
    for col in ADVANCED_MATCH_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = pd.NA
    return _coerce_advanced_match_frame(out_df, provider=provider)


PLAYER_STAT_ALIASES: dict[str, list[str]] = {
    "provider_match_id": ["provider_match_id", "match_id", "game_id", "id", "fixture_id"],
    "date": ["date", "match_date", "datetime", "kickoff"],
    "competition": ["competition", "league", "comp", "competition_name"],
    "season": ["season", "season_name", "year"],
    "team": ["team", "squad", "club", "team_name"],
    "opponent": ["opponent", "opp"],
    "player": ["player", "player_name", "name"],
    "player_id": ["player_id", "id", "understat_id", "fbref_id"],
    "position": ["position", "pos"],
    "started": ["started", "start", "is_starter"],
    "minutes": ["minutes", "min", "time"],
    "goals": ["goals", "g", "gls"],
    "assists": ["assists", "a", "ast"],
    "xg": ["xg"],
    "npxg": ["npxg", "np_xg"],
    "xa": ["xa", "xag"],
    "xg_chain": ["xgchain", "xg_chain"],
    "xg_buildup": ["xgbuildup", "xg_buildup"],
    "shots": ["shots", "sh"],
    "sot": ["sot", "shots_on_target"],
    "key_passes": ["key_passes", "kp"],
    "sca": ["sca", "shot_creating_actions"],
    "gca": ["gca", "goal_creating_actions"],
    "passes": ["passes", "pass_attempts"],
    "passes_completed": ["passes_completed", "cmp"],
    "progressive_passes": ["progressive_passes", "prgp"],
    "progressive_carries": ["progressive_carries", "prgc"],
    "carries_into_final_third": ["carries_into_final_third"],
    "carries_into_penalty_area": ["carries_into_penalty_area"],
    "touches": ["touches"],
    "touches_box": ["touches_box", "touches_att_pen_area"],
    "tackles": ["tackles", "tkl"],
    "interceptions": ["interceptions", "int"],
    "blocks": ["blocks"],
    "pressures": ["pressures"],
    "fouls": ["fouls", "fls"],
    "fouled": ["fouled", "fld"],
    "yellow_cards": ["yellow_cards", "yellow"],
    "red_cards": ["red_cards", "red"],
    "saves": ["saves"],
    "psxg": ["psxg", "post_shot_xg"],
}


def _is_player_match_provider_csv(df: pd.DataFrame) -> bool:
    return bool(_pick_column(df, ["player", "player_name", "name"]) and _pick_column(df, ["team", "squad", "club", "team_name"]))


def _canonicalize_player_match_provider_csv(df: pd.DataFrame, *, provider: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in PLAYER_MATCH_COLUMNS:
        if col == "provider":
            out[col] = provider
            continue
        c = _pick_column(df, PLAYER_STAT_ALIASES.get(col, [col]))
        out[col] = df[c] if c else pd.NA
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype("string")
    for col in ["provider_match_id", "match_id", "competition", "season", "team", "opponent", "player", "player_id", "position"]:
        out[col] = out[col].fillna("").astype(str)
    for col in [c for c in PLAYER_MATCH_COLUMNS if c not in {"provider", "provider_match_id", "match_id", "date", "competition", "season", "team", "opponent", "player", "player_id", "position"}]:
        if col in {"started"}:
            out[col] = out[col].map(lambda v: str(v).lower() in {"true", "1", "yes", "y", "starter", "start"} if pd.notna(v) else pd.NA)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[PLAYER_MATCH_COLUMNS].copy()


SHOT_EVENT_ALIASES: dict[str, list[str]] = {
    "provider_match_id": ["provider_match_id", "match_id", "game_id", "id", "fixture_id"],
    "match_id": ["canonical_match_id"],
    "date": ["date", "match_date", "datetime"],
    "competition": ["competition", "league"],
    "season": ["season", "year"],
    "team": ["team", "squad", "club"],
    "opponent": ["opponent", "opp"],
    "player": ["player", "player_name", "shooter"],
    "player_id": ["player_id", "shooter_id"],
    "minute": ["minute", "min"],
    "second": ["second", "sec"],
    "xg": ["xg", "shot_xg"],
    "psxg": ["psxg", "post_shot_xg"],
    "outcome": ["outcome", "result"],
    "body_part": ["body_part", "shot_body_part"],
    "situation": ["situation", "shot_situation"],
    "shot_type": ["shot_type", "type"],
    "assisted_by": ["assisted_by", "assist_player", "key_pass_player"],
    "x": ["x", "x_coord", "X"],
    "y": ["y", "y_coord", "Y"],
    "distance_to_goal": ["distance_to_goal", "shot_distance"],
}


def _is_shot_event_provider_csv(df: pd.DataFrame) -> bool:
    return bool(_pick_column(df, ["xg", "shot_xg"]) and _pick_column(df, ["player", "player_name", "shooter"]) and _pick_column(df, ["x", "x_coord", "X"]))


def _canonicalize_shot_event_provider_csv(df: pd.DataFrame, *, provider: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in SHOT_EVENT_COLUMNS:
        if col == "provider":
            out[col] = provider
            continue
        c = _pick_column(df, SHOT_EVENT_ALIASES.get(col, [col]))
        out[col] = df[c] if c else pd.NA
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype("string")
    for col in ["provider_match_id", "match_id", "competition", "season", "team", "opponent", "player", "player_id", "outcome", "body_part", "situation", "shot_type", "assisted_by"]:
        out[col] = out[col].fillna("").astype(str)
    for col in ["minute", "second", "xg", "psxg", "x", "y", "distance_to_goal"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    body = out["body_part"].astype(str).str.lower()
    out["is_header"] = out["is_header"].combine_first(body.str.contains("head", na=False)) if "is_header" in out.columns else body.str.contains("head", na=False)
    out["is_left_foot"] = out["is_left_foot"].combine_first(body.str.contains("left", na=False)) if "is_left_foot" in out.columns else body.str.contains("left", na=False)
    out["is_right_foot"] = out["is_right_foot"].combine_first(body.str.contains("right", na=False)) if "is_right_foot" in out.columns else body.str.contains("right", na=False)
    situation = out["situation"].astype(str).str.lower()
    out["is_penalty"] = out["is_penalty"].combine_first(situation.str.contains("penalty", na=False)) if "is_penalty" in out.columns else situation.str.contains("penalty", na=False)
    outcome = out["outcome"].astype(str).str.lower()
    out["is_goal"] = out["is_goal"].combine_first(outcome.eq("goal")) if "is_goal" in out.columns else outcome.eq("goal")
    if out["is_inside_box"].isna().all() and {"x", "y"}.issubset(out.columns):
        out["is_inside_box"] = [_is_inside_statsbomb_penalty_box(x, y) for x, y in zip(out["x"], out["y"], strict=False)]
    out["is_outside_box"] = out["is_outside_box"].combine_first(~out["is_inside_box"].fillna(False))
    return out[SHOT_EVENT_COLUMNS].copy()


def _coerce_advanced_match_frame(out: pd.DataFrame, *, provider: str) -> pd.DataFrame:
    for col in ADVANCED_MATCH_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["provider"] = out["provider"].replace("", provider).fillna(provider)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype("string")
    for col in [
        "home_team", "away_team", "competition", "season", "stage", "team_scope",
        "provider_match_id", "venue_country", "venue_city", "home_team_country",
        "away_team_country", "home_team_city", "away_team_city",
    ]:
        out[col] = out[col].fillna("").astype(str)
    numeric_cols = [
        c for c in ADVANCED_MATCH_COLUMNS
        if c.startswith(("home_", "away_")) and c not in [
            "home_team", "away_team", "home_team_country", "away_team_country",
            "home_team_city", "away_team_city",
        ]
    ] + ["neutral"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["source_confidence"] = out["source_confidence"].fillna("provider_csv")
    out["join_method"] = out["join_method"].fillna("date_team")
    return out[ADVANCED_MATCH_COLUMNS].copy()


def _canonicalize_match_provider_csv(df: pd.DataFrame, *, provider: str) -> pd.DataFrame:
    aliases: dict[str, list[str]] = {
        "provider_match_id": ["provider_match_id", "match_id", "game_id", "id", "fixture_id"],
        "date": ["date", "Date", "match_date", "datetime", "kickoff"],
        "competition": ["competition", "league", "comp", "competition_name", "Div"],
        "season": ["season", "season_name", "year"],
        "stage": ["stage", "round", "matchweek"],
        "team_scope": ["team_scope"],
        "neutral": ["neutral", "is_neutral", "neutral_venue"],
        "venue_country": ["venue_country", "host_country", "country"],
        "venue_city": ["venue_city", "city"],
        "home_team_country": ["home_team_country", "home_country"],
        "away_team_country": ["away_team_country", "away_country"],
        "home_team_city": ["home_team_city", "home_city"],
        "away_team_city": ["away_team_city", "away_city"],
        "home_team": ["home_team", "HomeTeam", "home", "home_name", "home_team_name", "h_team", "h"],
        "away_team": ["away_team", "AwayTeam", "away", "away_name", "away_team_name", "a_team", "a"],

        # Chance quality / shot volume.
        "home_xg": ["home_xg", "hxg", "h_xg", "xg_home", "home_exp_goals", "home_expected_goals"],
        "away_xg": ["away_xg", "axg", "a_xg", "xg_away", "away_exp_goals", "away_expected_goals"],
        "home_npxg": ["home_npxg", "hnpxg", "home_np_xg", "home_non_penalty_xg"],
        "away_npxg": ["away_npxg", "anpxg", "away_np_xg", "away_non_penalty_xg"],
        "home_xa": ["home_xa", "hxa", "home_xag", "home_xg_assisted", "home_expected_assists"],
        "away_xa": ["away_xa", "axa", "away_xag", "away_xg_assisted", "away_expected_assists"],
        "home_shots": ["home_shots", "HS", "h_shots", "home_sh", "home_total_shots", "home_sh"],
        "away_shots": ["away_shots", "AS", "a_shots", "away_sh", "away_total_shots", "away_sh"],
        "home_sot": ["home_sot", "HST", "home_shots_on_target", "h_sot", "home_sota"],
        "away_sot": ["away_sot", "AST", "away_shots_on_target", "a_sot", "away_sota"],
        "home_shots_inside_box": ["home_shots_inside_box", "home_shots_in_box", "home_box_shots"],
        "away_shots_inside_box": ["away_shots_inside_box", "away_shots_in_box", "away_box_shots"],
        "home_shots_outside_box": ["home_shots_outside_box", "home_shots_out_box", "home_outside_box_shots"],
        "away_shots_outside_box": ["away_shots_outside_box", "away_shots_out_box", "away_outside_box_shots"],
        "home_avg_shot_distance": ["home_avg_shot_distance", "home_average_shot_distance"],
        "away_avg_shot_distance": ["away_avg_shot_distance", "away_average_shot_distance"],
        "home_header_shots": ["home_header_shots", "home_headed_shots"],
        "away_header_shots": ["away_header_shots", "away_headed_shots"],
        "home_left_foot_shots": ["home_left_foot_shots"],
        "away_left_foot_shots": ["away_left_foot_shots"],
        "home_right_foot_shots": ["home_right_foot_shots"],
        "away_right_foot_shots": ["away_right_foot_shots"],
        "home_header_xg": ["home_header_xg", "home_headed_xg"],
        "away_header_xg": ["away_header_xg", "away_headed_xg"],
        "home_left_foot_xg": ["home_left_foot_xg"],
        "away_left_foot_xg": ["away_left_foot_xg"],
        "home_right_foot_xg": ["home_right_foot_xg"],
        "away_right_foot_xg": ["away_right_foot_xg"],
        "home_penalty_xg": ["home_penalty_xg", "home_pen_xg"],
        "away_penalty_xg": ["away_penalty_xg", "away_pen_xg"],
        "home_open_play_xg": ["home_open_play_xg"],
        "away_open_play_xg": ["away_open_play_xg"],
        "home_set_piece_xg": ["home_set_piece_xg"],
        "away_set_piece_xg": ["away_set_piece_xg"],
        "home_corner_xg": ["home_corner_xg"],
        "away_corner_xg": ["away_corner_xg"],
        "home_free_kick_xg": ["home_free_kick_xg", "home_freekick_xg"],
        "away_free_kick_xg": ["away_free_kick_xg", "away_freekick_xg"],
        "home_counterattack_xg": ["home_counterattack_xg", "home_counter_xg"],
        "away_counterattack_xg": ["away_counterattack_xg", "away_counter_xg"],
        "home_big_chances": ["home_big_chances"],
        "away_big_chances": ["away_big_chances"],

        # Football-Data box score / discipline.
        "home_corners": ["home_corners", "HC", "home_corner_kicks"],
        "away_corners": ["away_corners", "AC", "away_corner_kicks"],
        "home_fouls": ["home_fouls", "HF"],
        "away_fouls": ["away_fouls", "AF"],
        "home_yellow_cards": ["home_yellow_cards", "HY", "home_yellows"],
        "away_yellow_cards": ["away_yellow_cards", "AY", "away_yellows"],
        "home_red_cards": ["home_red_cards", "HR", "home_reds"],
        "away_red_cards": ["away_red_cards", "AR", "away_reds"],

        # Possession / territory / progression.
        "home_possession": ["home_possession", "home_poss", "possession_home"],
        "away_possession": ["away_possession", "away_poss", "possession_away"],
        "home_field_tilt": ["home_field_tilt", "field_tilt_home"],
        "away_field_tilt": ["away_field_tilt", "field_tilt_away"],
        "home_ppda": ["home_ppda", "ppda_home"],
        "away_ppda": ["away_ppda", "ppda_away"],
        "home_touches_attacking_third": ["home_touches_attacking_third", "home_att_third_touches"],
        "away_touches_attacking_third": ["away_touches_attacking_third", "away_att_third_touches"],
        "home_touches_box": ["home_touches_box", "home_touches_penalty_area", "home_touches_att_pen_area"],
        "away_touches_box": ["away_touches_box", "away_touches_penalty_area", "away_touches_att_pen_area"],
        "home_final_third_entries": ["home_final_third_entries"],
        "away_final_third_entries": ["away_final_third_entries"],
        "home_deep_completions": ["home_deep_completions"],
        "away_deep_completions": ["away_deep_completions"],
        "home_progressive_passes": ["home_progressive_passes", "home_prgp"],
        "away_progressive_passes": ["away_progressive_passes", "away_prgp"],
        "home_progressive_carries": ["home_progressive_carries", "home_prgc"],
        "away_progressive_carries": ["away_progressive_carries", "away_prgc"],
        "home_passes_into_final_third": ["home_passes_into_final_third", "home_passes_1/3"],
        "away_passes_into_final_third": ["away_passes_into_final_third", "away_passes_1/3"],
        "home_passes_into_penalty_area": ["home_passes_into_penalty_area", "home_ppa"],
        "away_passes_into_penalty_area": ["away_passes_into_penalty_area", "away_ppa"],
        "home_crosses_into_penalty_area": ["home_crosses_into_penalty_area", "home_crspa"],
        "away_crosses_into_penalty_area": ["away_crosses_into_penalty_area", "away_crspa"],
        "home_through_balls": ["home_through_balls"],
        "away_through_balls": ["away_through_balls"],

        # Defensive / keeper.
        "home_shot_creating_actions": ["home_shot_creating_actions", "home_sca"],
        "away_shot_creating_actions": ["away_shot_creating_actions", "away_sca"],
        "home_goal_creating_actions": ["home_goal_creating_actions", "home_gca"],
        "away_goal_creating_actions": ["away_goal_creating_actions", "away_gca"],
        "home_tackles": ["home_tackles", "home_tkl"],
        "away_tackles": ["away_tackles", "away_tkl"],
        "home_interceptions": ["home_interceptions", "home_int"],
        "away_interceptions": ["away_interceptions", "away_int"],
        "home_blocks": ["home_blocks"],
        "away_blocks": ["away_blocks"],
        "home_clearances": ["home_clearances", "home_clr"],
        "away_clearances": ["away_clearances", "away_clr"],
        "home_pressures": ["home_pressures"],
        "away_pressures": ["away_pressures"],
        "home_ball_recoveries": ["home_ball_recoveries", "home_recoveries"],
        "away_ball_recoveries": ["away_ball_recoveries", "away_recoveries"],
        "home_errors": ["home_errors", "home_err"],
        "away_errors": ["away_errors", "away_err"],
        "home_keeper_saves": ["home_keeper_saves", "home_saves"],
        "away_keeper_saves": ["away_keeper_saves", "away_saves"],
        "home_keeper_psxg": ["home_keeper_psxg", "home_psxg", "home_post_shot_xg"],
        "away_keeper_psxg": ["away_keeper_psxg", "away_psxg", "away_post_shot_xg"],
        "home_keeper_goals_against": ["home_keeper_goals_against", "home_gk_ga"],
        "away_keeper_goals_against": ["away_keeper_goals_against", "away_gk_ga"],
        "home_keeper_save_pct": ["home_keeper_save_pct", "home_save_pct"],
        "away_keeper_save_pct": ["away_keeper_save_pct", "away_save_pct"],
    }
    out = pd.DataFrame(index=df.index)
    for col in ADVANCED_MATCH_COLUMNS:
        if col == "provider":
            out[col] = provider
            continue
        c = _pick_column(df, aliases.get(col, [col]))
        out[col] = df[c] if c else pd.NA

    for side in ["home", "away"]:
        xg = pd.to_numeric(out.get(f"{side}_xg"), errors="coerce")
        npxg = pd.to_numeric(out.get(f"{side}_npxg"), errors="coerce")
        xa = pd.to_numeric(out.get(f"{side}_xa"), errors="coerce")
        shots = pd.to_numeric(out.get(f"{side}_shots"), errors="coerce")
        if f"{side}_npxg_plus_xa" in out.columns:
            out[f"{side}_npxg_plus_xa"] = pd.to_numeric(out[f"{side}_npxg_plus_xa"], errors="coerce").combine_first(npxg + xa)
        if f"{side}_xg_per_shot" in out.columns:
            out[f"{side}_xg_per_shot"] = pd.to_numeric(out[f"{side}_xg_per_shot"], errors="coerce").combine_first(xg / shots.where(shots > 0))

    return _coerce_advanced_match_frame(out, provider=provider)

def import_advanced_provider_csv(input_csv: str | Path, *, provider: str = "provider_csv") -> AdvancedImportOutputs:
    df = pd.read_csv(input_csv)
    matches = _empty(ADVANCED_MATCH_COLUMNS)
    players = _empty(PLAYER_MATCH_COLUMNS)
    shots = _empty(SHOT_EVENT_COLUMNS)
    mode_parts: list[str] = []

    if _is_shot_event_provider_csv(df):
        shots = _canonicalize_shot_event_provider_csv(df, provider=provider)
        mode_parts.append("shot_event_csv_import")
    if _is_player_match_provider_csv(df):
        players = _canonicalize_player_match_provider_csv(df, provider=provider)
        mode_parts.append("player_match_csv_import")
    if _is_team_match_provider_csv(df):
        matches = _canonicalize_team_match_provider_csv(df, provider=provider)
        mode_parts.append("team_match_provider_csv_import")
    elif not mode_parts or {"home_team", "HomeTeam", "h_team", "home"}.intersection(set(map(str, df.columns))):
        matches = _canonicalize_match_provider_csv(df, provider=provider)
        mode_parts.append("advanced_match_provider_csv_import")

    report = {
        "version": ADVANCED_DATA_VERSION,
        "provider": provider,
        "mode": "+".join(mode_parts) if mode_parts else "csv_import_unrecognized",
        "status": "ok" if (len(matches) or len(players) or len(shots)) else "warning",
        "input_rows": int(len(df)),
        "advanced_match_rows": int(len(matches)),
        "player_match_rows": int(len(players)),
        "shot_event_rows": int(len(shots)),
        "lineup_rows": 0,
        "raw_data_changed": False,
    }
    return AdvancedImportOutputs(matches, players, shots, _empty(LINEUP_COLUMNS), report)

def _load_competitions(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "competitions.json"
    if not path.exists():
        raise FileNotFoundError(f"StatsBomb competitions.json not found: {path}")
    return pd.DataFrame(_read_json(path))


def _iter_match_files(data_dir: Path, competition_ids: set[str] | None, season_ids: set[str] | None) -> list[Path]:
    matches_dir = data_dir / "matches"
    if not matches_dir.exists():
        raise FileNotFoundError(f"StatsBomb matches directory not found: {matches_dir}")
    files: list[Path] = []
    for comp_dir in sorted(matches_dir.iterdir()):
        if not comp_dir.is_dir():
            continue
        if competition_ids and comp_dir.name not in competition_ids:
            continue
        for season_file in sorted(comp_dir.glob("*.json")):
            if season_ids and season_file.stem not in season_ids:
                continue
            files.append(season_file)
    return files


def import_statsbomb_open_advanced(
    data_dir: str | Path,
    *,
    competition_ids: Iterable[str] | None = None,
    season_ids: Iterable[str] | None = None,
    max_matches: int | None = None,
) -> AdvancedImportOutputs:
    """Import StatsBomb Open Data into match, player, shot-event and lineup contracts.

    StatsBomb Open Data is event-rich and includes lineups, but coverage is
    partial. The importer keeps post-match observations separate from
    leakage-safe rolling features, which are built later by the snapshot builder.
    """
    data_dir = Path(data_dir)
    competitions = _load_competitions(data_dir)
    meta_rows: list[dict[str, Any]] = []
    for file in _iter_match_files(data_dir, set(map(str, competition_ids or [])) or None, set(map(str, season_ids or [])) or None):
        comp_id = file.parent.name
        season_id = file.stem
        comp_meta = competitions[
            (competitions["competition_id"].astype(str) == comp_id)
            & (competitions["season_id"].astype(str) == season_id)
        ]
        comp_name = str(comp_meta["competition_name"].iloc[0]) if len(comp_meta) else ""
        season_name = str(comp_meta["season_name"].iloc[0]) if len(comp_meta) else season_id
        for match in _read_json(file):
            meta_rows.append({
                "provider_match_id": str(match.get("match_id")),
                "date": match.get("match_date"),
                "competition": comp_name,
                "season": season_name,
                "home_team": _as_name(match.get("home_team")),
                "away_team": _as_name(match.get("away_team")),
                "home_score": match.get("home_score"),
                "away_score": match.get("away_score"),
            })
    if max_matches is not None:
        meta_rows = meta_rows[:max_matches]
    meta = pd.DataFrame(meta_rows)

    match_acc: dict[str, dict[str, Any]] = {}
    player_rows: list[dict[str, Any]] = []
    shot_rows: list[dict[str, Any]] = []
    lineup_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    if meta.empty:
        report = {
            "version": ADVANCED_DATA_VERSION,
            "provider": "statsbomb_open_data",
            "mode": "statsbomb_open_advanced_import",
            "status": "warning",
            "matches_discovered": 0,
            "advanced_match_rows": 0,
            "player_match_rows": 0,
            "shot_event_rows": 0,
            "lineup_rows": 0,
            "failures": [],
            "raw_data_changed": False,
        }
        return AdvancedImportOutputs(_empty(ADVANCED_MATCH_COLUMNS), _empty(PLAYER_MATCH_COLUMNS), _empty(SHOT_EVENT_COLUMNS), _empty(LINEUP_COLUMNS), report)

    for _, m in meta.iterrows():
        pid = str(m["provider_match_id"])
        events_file = data_dir / "events" / f"{pid}.json"
        lineups_file = data_dir / "lineups" / f"{pid}.json"
        base = {
            "provider": "statsbomb_open_data",
            "provider_match_id": pid,
            "date": m["date"],
            "competition": m["competition"],
            "season": m["season"],
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "source_confidence": "official_open_event_aggregate",
            "join_method": "date_team",
        }
        acc = {**base}
        for side in ["home", "away"]:
            for stat in [
                "xg", "npxg", "xa", "npxg_plus_xa", "shots", "sot", "shots_inside_box", "shots_outside_box",
                "shot_distance_sum", "avg_shot_distance", "xg_per_shot", "header_shots", "left_foot_shots",
                "right_foot_shots", "header_xg", "left_foot_xg", "right_foot_xg", "penalty_xg",
                "open_play_xg", "set_piece_xg", "corner_xg", "free_kick_xg", "counterattack_xg",
                "big_chances", "passes_into_final_third", "passes_into_penalty_area", "deep_completions",
                "final_third_entries", "progressive_passes", "progressive_carries", "touches_box",
                "touches_attacking_third", "shot_creating_actions", "goal_creating_actions", "tackles",
                "interceptions", "blocks", "clearances", "pressures", "ball_recoveries", "errors",
                "keeper_saves"
            ]:
                acc[f"{side}_{stat}"] = 0.0
        try:
            events = _read_json(events_file)
        except Exception as exc:
            failures.append({"provider_match_id": pid, "error": str(exc)})
            events = []

        # Lineups are a separate high-value contract for future player/lineup rating.
        try:
            lineup_payload = _read_json(lineups_file) if lineups_file.exists() else []
            for team_entry in lineup_payload:
                team = _as_name(team_entry.get("team_name") or team_entry.get("team"))
                side = _side(team, m["home_team"], m["away_team"])
                opponent = m["away_team"] if side == "home" else m["home_team"] if side == "away" else ""
                for player_entry in team_entry.get("lineup", []) or []:
                    positions = player_entry.get("positions") or []
                    first_pos = positions[0] if positions else {}
                    lineup_rows.append({
                        "provider": "statsbomb_open_data",
                        "provider_match_id": pid,
                        "match_id": pd.NA,
                        "date": m["date"],
                        "competition": m["competition"],
                        "season": m["season"],
                        "team": team,
                        "opponent": opponent,
                        "player": player_entry.get("player_name") or player_entry.get("name"),
                        "player_id": player_entry.get("player_id"),
                        "position": first_pos.get("position") if isinstance(first_pos, dict) else pd.NA,
                        "started": bool(positions),
                        "bench": not bool(positions),
                        "minutes_played": pd.NA,
                        "shirt_number": player_entry.get("jersey_number"),
                        "formation_position": first_pos.get("from") if isinstance(first_pos, dict) else pd.NA,
                        "provider_position_id": first_pos.get("position_id") if isinstance(first_pos, dict) else pd.NA,
                    })
        except Exception as exc:
            failures.append({"provider_match_id": pid, "lineup_error": str(exc)})

        players: dict[tuple[str, str], dict[str, Any]] = {}

        for event in events:
            team = _as_name(event.get("team"))
            player = _as_name(event.get("player"))
            side = _side(team, m["home_team"], m["away_team"])
            if not side:
                continue
            opponent = m["away_team"] if side == "home" else m["home_team"]

            if player:
                key = (team, player)
                players.setdefault(key, {
                    "provider": "statsbomb_open_data",
                    "provider_match_id": pid,
                    "match_id": pd.NA,
                    "date": m["date"],
                    "competition": m["competition"],
                    "season": m["season"],
                    "team": team,
                    "opponent": opponent,
                    "player": player,
                    "player_id": (event.get("player") or {}).get("id") if isinstance(event.get("player"), dict) else pd.NA,
                    "position": pd.NA,
                    "started": False,
                    "minutes": pd.NA,
                    "goals": 0,
                    "assists": 0,
                    "xg": 0.0,
                    "npxg": 0.0,
                    "xa": pd.NA,
                    "xg_chain": pd.NA,
                    "xg_buildup": pd.NA,
                    "shots": 0,
                    "sot": 0,
                    "key_passes": 0,
                    "sca": 0,
                    "gca": 0,
                    "passes": 0,
                    "passes_completed": 0,
                    "progressive_passes": 0,
                    "progressive_carries": 0,
                    "carries_into_final_third": pd.NA,
                    "carries_into_penalty_area": pd.NA,
                    "touches": 0,
                    "touches_box": 0,
                    "tackles": 0,
                    "interceptions": 0,
                    "blocks": 0,
                    "pressures": 0,
                    "fouls": 0,
                    "fouled": 0,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "saves": 0,
                    "psxg": pd.NA,
                })

            event_type = _as_name(event.get("type"))
            loc = event.get("location") or [pd.NA, pd.NA]
            x = loc[0] if len(loc) > 0 else pd.NA
            y = loc[1] if len(loc) > 1 else pd.NA

            if event_type.lower() == "shot":
                shot = event.get("shot") or {}
                xg = float(shot.get("statsbomb_xg") or 0.0)
                outcome = _as_name(shot.get("outcome"))
                body = _as_name(shot.get("body_part"))
                situation = _as_name(shot.get("type"))
                is_penalty = situation.lower() == "penalty"
                is_header = "head" in body.lower()
                is_left = "left" in body.lower()
                is_right = "right" in body.lower()
                is_inside_box = _is_inside_statsbomb_penalty_box(x, y)
                distance = _distance_to_goal_statsbomb(x, y)
                situation_l = situation.lower()

                acc[f"{side}_xg"] += xg
                acc[f"{side}_shots"] += 1
                acc[f"{side}_shot_distance_sum"] += 0.0 if pd.isna(distance) else float(distance)
                acc[f"{side}_shots_inside_box"] += int(is_inside_box)
                acc[f"{side}_shots_outside_box"] += int(not is_inside_box)
                acc[f"{side}_header_shots"] += int(is_header)
                acc[f"{side}_left_foot_shots"] += int(is_left)
                acc[f"{side}_right_foot_shots"] += int(is_right)
                acc[f"{side}_header_xg"] += xg if is_header else 0.0
                acc[f"{side}_left_foot_xg"] += xg if is_left else 0.0
                acc[f"{side}_right_foot_xg"] += xg if is_right else 0.0
                if not is_penalty:
                    acc[f"{side}_npxg"] += xg
                if is_penalty:
                    acc[f"{side}_penalty_xg"] += xg
                elif "free kick" in situation_l:
                    acc[f"{side}_free_kick_xg"] += xg
                    acc[f"{side}_set_piece_xg"] += xg
                elif "corner" in situation_l:
                    acc[f"{side}_corner_xg"] += xg
                    acc[f"{side}_set_piece_xg"] += xg
                elif "counter" in situation_l:
                    acc[f"{side}_counterattack_xg"] += xg
                    acc[f"{side}_open_play_xg"] += xg
                else:
                    acc[f"{side}_open_play_xg"] += xg
                if outcome.lower() in {"goal", "saved", "saved to post"}:
                    acc[f"{side}_sot"] += 1

                if player:
                    prow = players[(team, player)]
                    prow["shots"] += 1
                    prow["xg"] += xg
                    if not is_penalty:
                        prow["npxg"] += xg
                    if outcome.lower() == "goal":
                        prow["goals"] += 1
                    if outcome.lower() in {"goal", "saved", "saved to post"}:
                        prow["sot"] += 1

                shot_rows.append({
                    "provider": "statsbomb_open_data",
                    "provider_match_id": pid,
                    "match_id": pd.NA,
                    "date": m["date"],
                    "competition": m["competition"],
                    "season": m["season"],
                    "team": team,
                    "opponent": opponent,
                    "player": player,
                    "player_id": (event.get("player") or {}).get("id") if isinstance(event.get("player"), dict) else pd.NA,
                    "minute": event.get("minute"),
                    "second": event.get("second"),
                    "xg": xg,
                    "psxg": pd.NA,
                    "outcome": outcome,
                    "body_part": body,
                    "situation": situation,
                    "shot_type": situation,
                    "assisted_by": _as_name((shot.get("key_pass_id") or "")),
                    "x": x,
                    "y": y,
                    "distance_to_goal": distance,
                    "is_penalty": is_penalty,
                    "is_goal": outcome.lower() == "goal",
                    "is_header": is_header,
                    "is_left_foot": is_left,
                    "is_right_foot": is_right,
                    "is_inside_box": is_inside_box,
                    "is_outside_box": not is_inside_box,
                })
            elif event_type.lower() == "pass":
                acc[f"{side}_progressive_passes"] += 1 if (event.get("pass") or {}).get("length", 0) else 0
                if player:
                    players[(team, player)]["passes"] += 1
                    outcome = _as_name((event.get("pass") or {}).get("outcome"))
                    if not outcome:
                        players[(team, player)]["passes_completed"] += 1
            elif event_type.lower() == "carry":
                acc[f"{side}_progressive_carries"] += 1
                if player:
                    players[(team, player)]["touches"] += 1
            elif event_type.lower() == "pressure":
                acc[f"{side}_pressures"] += 1
                if player:
                    players[(team, player)]["pressures"] += 1
            elif event_type.lower() == "duel":
                duel = event.get("duel") or {}
                dtyp = _as_name(duel.get("type"))
                if "tackle" in dtyp.lower():
                    acc[f"{side}_tackles"] += 1
                    if player:
                        players[(team, player)]["tackles"] += 1
            elif event_type.lower() == "interception":
                acc[f"{side}_interceptions"] += 1
                if player:
                    players[(team, player)]["interceptions"] += 1
            elif event_type.lower() == "block":
                acc[f"{side}_blocks"] += 1
                if player:
                    players[(team, player)]["blocks"] += 1
            elif event_type.lower() == "clearance":
                acc[f"{side}_clearances"] += 1
            elif event_type.lower() == "ball recovery":
                acc[f"{side}_ball_recoveries"] += 1
            elif event_type.lower() == "error":
                acc[f"{side}_errors"] += 1
            elif event_type.lower() == "foul committed":
                if player:
                    players[(team, player)]["fouls"] += 1
                card = _as_name((event.get("foul_committed") or {}).get("card"))
                if card.lower() == "yellow card" and player:
                    players[(team, player)]["yellow_cards"] += 1
                if "red" in card.lower() and player:
                    players[(team, player)]["red_cards"] += 1
            elif event_type.lower() == "foul won" and player:
                players[(team, player)]["fouled"] += 1
            elif event_type.lower() == "goal keeper":
                gk = event.get("goalkeeper") or {}
                if _as_name(gk.get("outcome")).lower() in {"saved", "success"}:
                    acc[f"{side}_keeper_saves"] = acc.get(f"{side}_keeper_saves", 0) + 1
                    if player:
                        players[(team, player)]["saves"] += 1

        for side in ["home", "away"]:
            shots = acc.get(f"{side}_shots", 0.0)
            xg = acc.get(f"{side}_xg", 0.0)
            if shots:
                acc[f"{side}_avg_shot_distance"] = acc.get(f"{side}_shot_distance_sum", 0.0) / shots
                acc[f"{side}_xg_per_shot"] = xg / shots
            acc.pop(f"{side}_shot_distance_sum", None)
            npxg = acc.get(f"{side}_npxg", 0.0)
            xa = acc.get(f"{side}_xa", 0.0)
            acc[f"{side}_npxg_plus_xa"] = npxg + xa if pd.notna(xa) else pd.NA

        match_acc[pid] = acc
        player_rows.extend(players.values())

    match_df = pd.DataFrame(match_acc.values())
    for col in ADVANCED_MATCH_COLUMNS:
        if col not in match_df.columns:
            match_df[col] = pd.NA
    player_df = pd.DataFrame(player_rows)
    for col in PLAYER_MATCH_COLUMNS:
        if col not in player_df.columns:
            player_df[col] = pd.NA
    shots_df = pd.DataFrame(shot_rows)
    for col in SHOT_EVENT_COLUMNS:
        if col not in shots_df.columns:
            shots_df[col] = pd.NA
    lineups_df = pd.DataFrame(lineup_rows)
    for col in LINEUP_COLUMNS:
        if col not in lineups_df.columns:
            lineups_df[col] = pd.NA

    report = {
        "version": ADVANCED_DATA_VERSION,
        "provider": "statsbomb_open_data",
        "mode": "statsbomb_open_advanced_event_lineup_aggregate",
        "status": "ok" if len(match_df) else "warning",
        "matches_discovered": int(len(meta)),
        "advanced_match_rows": int(len(match_df)),
        "player_match_rows": int(len(player_df)),
        "shot_event_rows": int(len(shots_df)),
        "lineup_rows": int(len(lineups_df)),
        "failures": failures[:20],
        "failure_count": int(len(failures)),
        "competitions": sorted(meta.get("competition", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())[:100],
        "seasons": sorted(meta.get("season", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())[:100],
        "coverage_note": "StatsBomb Open Data is official/free but partial; Big-5 modern coverage is not guaranteed.",
        "raw_data_changed": False,
        "lineup_policy": "lineups are stored as canonical evidence for future lineup-strength ratings; not used as pre-match features unless lineup timing is explicit",
    }
    return AdvancedImportOutputs(
        match_df[ADVANCED_MATCH_COLUMNS].copy(),
        player_df[PLAYER_MATCH_COLUMNS].copy(),
        shots_df[SHOT_EVENT_COLUMNS].copy(),
        lineups_df[LINEUP_COLUMNS].copy(),
        report,
    )

def merge_advanced_match_sources(
    sources: list[tuple[str, pd.DataFrame]],
    *,
    provider_priority: list[str] | None = None,
) -> AdvancedMergeOutputs:
    """Merge advanced providers by match key and column-level priority.

    Earlier v0.50 selected one provider row per match. That was too lossy: a
    source with xG but no shots could wipe out another source with shots/SOT.
    This version keeps the provider priority but fills each metric from the
    first provider that has a non-null value for that metric.
    """
    frames: list[pd.DataFrame] = []
    for provider, df in sources:
        if df is None or len(df) == 0:
            continue
        if set(ADVANCED_MATCH_COLUMNS).issubset(df.columns):
            canon = df[ADVANCED_MATCH_COLUMNS].copy()
        elif _is_team_match_provider_csv(df):
            canon = _canonicalize_team_match_provider_csv(df, provider=provider)
        else:
            canon = _canonicalize_match_provider_csv(df, provider=provider)
        canon["provider"] = canon["provider"].replace("", provider).fillna(provider)
        canon["match_join_key"] = _make_match_key(canon)
        frames.append(canon)

    if not frames:
        summary = {
            "version": ADVANCED_DATA_VERSION,
            "status": "warning",
            "reason": "no_provider_rows",
            "input_sources": [name for name, _ in sources],
            "canonical_rows": 0,
        }
        return AdvancedMergeOutputs(_empty(ADVANCED_MATCH_COLUMNS), pd.DataFrame(), summary)

    combined = pd.concat(frames, ignore_index=True)
    priority = provider_priority or [
        "fbref",
        "soccerdata_fbref",
        "worldfootballr_fbref",
        "kaggle_understat",
        "rapidapi_football_xg",
        "statsbomb_open_data",
        "understat",
        "provider_csv",
    ]
    priority_map = {p: i for i, p in enumerate(priority)}
    combined["provider_priority"] = combined["provider"].map(lambda p: priority_map.get(str(p), len(priority_map) + 99))
    combined = combined.sort_values(["match_join_key", "provider_priority", "provider_match_id"])

    identity_cols = {
        "provider_match_id", "date", "competition", "season", "stage", "team_scope",
        "neutral", "venue_country", "venue_city", "home_team_country", "away_team_country",
        "home_team_city", "away_team_city", "home_team", "away_team", "source_confidence", "join_method",
    }
    selected_rows: list[dict[str, Any]] = []
    field_provider_rows: list[dict[str, Any]] = []
    for key, g in combined.groupby("match_join_key", dropna=False, sort=False):
        out: dict[str, Any] = {}
        used_providers: set[str] = set()
        for col in ADVANCED_MATCH_COLUMNS:
            if col == "provider":
                continue
            vals = g[[col, "provider"]] if col in g.columns else pd.DataFrame(columns=[col, "provider"])
            non_null = vals[vals[col].notna()]
            if len(non_null):
                out[col] = non_null.iloc[0][col]
                used_provider = str(non_null.iloc[0]["provider"])
                used_providers.add(used_provider)
                if col not in identity_cols:
                    field_provider_rows.append({
                        "match_join_key": key,
                        "column": col,
                        "provider": used_provider,
                    })
            else:
                out[col] = pd.NA
        out["provider"] = "+".join(sorted(used_providers)) if used_providers else str(g["provider"].iloc[0])
        selected_rows.append(out)

    canonical = pd.DataFrame(selected_rows)
    for col in ADVANCED_MATCH_COLUMNS:
        if col not in canonical.columns:
            canonical[col] = pd.NA
    canonical = canonical[ADVANCED_MATCH_COLUMNS].copy()

    provider_counts = combined["provider"].value_counts(dropna=False).reset_index()
    provider_counts.columns = ["provider", "input_rows"]
    selected_provider_counts = pd.Series(
        [p for providers in canonical["provider"].fillna("").astype(str) for p in providers.split("+") if p]
    ).value_counts().reset_index()
    selected_provider_counts.columns = ["provider", "selected_or_contributed_rows"]
    provider_report = provider_counts.merge(selected_provider_counts, on="provider", how="outer").fillna(0)
    provider_report["input_rows"] = provider_report["input_rows"].astype(int)
    provider_report["selected_or_contributed_rows"] = provider_report["selected_or_contributed_rows"].astype(int)

    field_provider_df = pd.DataFrame(field_provider_rows)
    if not field_provider_df.empty:
        field_summary = (
            field_provider_df.groupby(["column", "provider"])
            .size()
            .reset_index(name="filled_values")
            .sort_values(["column", "filled_values"], ascending=[True, False])
        )
    else:
        field_summary = pd.DataFrame(columns=["column", "provider", "filled_values"])

    summary = {
        "version": ADVANCED_DATA_VERSION,
        "status": "ok",
        "merge_strategy": "column_level_priority_first_non_null",
        "input_sources": [name for name, _ in sources],
        "input_rows": int(len(combined)),
        "canonical_rows": int(len(canonical)),
        "provider_priority": priority,
        "providers_selected": provider_report.to_dict(orient="records"),
        "field_provider_examples": field_summary.head(80).to_dict(orient="records"),
        "raw_data_changed": False,
        "leakage_policy": "advanced_match_stats_are_post_match_observations; only prior rolling features may be model inputs",
    }
    return AdvancedMergeOutputs(canonical, provider_report, summary)

def enrich_matches_with_advanced_stats(
    matches: pd.DataFrame,
    advanced_matches: pd.DataFrame,
    *,
    registry: pd.DataFrame | None = None,
    manual_aliases: pd.DataFrame | None = None,
    provider_alias_column: str = "football_data_name",
    dataset_name: str = "advanced_enriched_matches",
) -> AdvancedEnrichmentOutputs:
    """Attach advanced stats without destroying base match stats.

    Critical policy: base canonical columns such as Football-Data shots/SOT,
    corners, fouls and cards are preserved when an advanced provider has nulls.
    Advanced values fill missing base values or add new metrics.
    """
    match_df = matches.copy()
    if "match_id" not in match_df.columns or "date" not in match_df.columns:
        summary = {
            "version": ADVANCED_DATA_VERSION,
            "dataset_name": dataset_name,
            "status": "blocked",
            "reason": "matches_missing_match_id_or_date",
            "input_rows": int(len(matches)),
        }
        return AdvancedEnrichmentOutputs(match_df, _empty(ADVANCED_MATCH_COLUMNS), pd.DataFrame(), summary)

    if set(ADVANCED_MATCH_COLUMNS).issubset(advanced_matches.columns):
        canon = advanced_matches[ADVANCED_MATCH_COLUMNS].copy()
    elif _is_team_match_provider_csv(advanced_matches):
        canon = _canonicalize_team_match_provider_csv(advanced_matches, provider="advanced_provider")
    else:
        canon = _canonicalize_match_provider_csv(advanced_matches, provider="advanced_provider")
    alias_map = _build_team_join_alias_map(registry, provider_alias_column, manual_aliases)
    match_df = _apply_team_join_aliases(match_df, alias_map)
    canon = _apply_team_join_aliases(canon, alias_map)

    match_df["date"] = pd.to_datetime(match_df["date"], errors="coerce")
    match_df["match_join_key"] = _make_match_key(
        match_df,
        home_col="home_team_join_name",
        away_col="away_team_join_name",
    )
    canon["match_join_key"] = _make_match_key(
        canon,
        home_col="home_team_join_name",
        away_col="away_team_join_name",
    )

    enrichment_cols = [c for c in ADVANCED_MATCH_COLUMNS if c not in {"date", "competition", "season", "home_team", "away_team"}]
    right_cols = ["match_join_key", *enrichment_cols]
    right = canon[right_cols].drop_duplicates("match_join_key").copy()
    merged = match_df.merge(right, on="match_join_key", how="left", suffixes=("", "__advanced"), validate="many_to_one")

    advanced_value_cols: list[str] = []
    for col in enrichment_cols:
        adv_col = f"{col}__advanced"
        if adv_col in merged.columns and col in merged.columns:
            merged[col] = merged[adv_col].combine_first(merged[col])
            advanced_value_cols.append(adv_col)
        elif adv_col in merged.columns:
            merged[col] = merged[adv_col]
            advanced_value_cols.append(adv_col)
        elif col not in merged.columns:
            merged[col] = pd.NA

    joined_keys = set(canon["match_join_key"].dropna().astype(str))
    merged["advanced_data_available"] = merged["match_join_key"].astype(str).isin(joined_keys)

    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("__advanced")] + ["match_join_key"], errors="ignore")

    join_report = match_df[["match_id", "date", "home_team", "away_team", "match_join_key"]].copy()
    join_report["advanced_data_available"] = join_report["match_join_key"].astype(str).isin(joined_keys)
    join_report = join_report.drop(columns=["match_join_key"])

    coverage = float(merged["advanced_data_available"].mean()) if len(merged) else 0.0
    summary = {
        "version": ADVANCED_DATA_VERSION,
        "dataset_name": dataset_name,
        "status": "ok" if coverage > 0 else "blocked",
        "input_match_rows": int(len(matches)),
        "advanced_rows": int(len(canon)),
        "output_rows": int(len(merged)),
        "matches_with_advanced_data": int(merged["advanced_data_available"].sum()),
        "coverage_rate": coverage,
        "providers": sorted(canon.get("provider", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
        "manual_alias_rows": int(len(manual_aliases)) if manual_aliases is not None else 0,
        "team_join_alias_count": int(len(alias_map)),
        "preservation_policy": "existing canonical match stats are preserved when advanced provider values are null",
        "unmatched_examples": join_report.loc[~join_report["advanced_data_available"], ["match_id", "home_team", "away_team"]].head(20).to_dict(orient="records"),
        "raw_data_changed": False,
        "model_logic_changed": False,
        "leakage_policy": "advanced current-match columns are observations/targets; only prior rolling features may be model inputs",
    }
    return AdvancedEnrichmentOutputs(merged, canon, join_report, summary)

def audit_advanced_data_coverage(matches: pd.DataFrame, *, dataset_name: str = "advanced_data_coverage") -> dict[str, Any]:
    feature_groups = {
        "venue_context": ["neutral"],
        "xg": ["home_xg", "away_xg"],
        "npxg": ["home_npxg", "away_npxg"],
        "xa": ["home_xa", "away_xa"],
        "shot_volume": ["home_shots", "away_shots", "home_sot", "away_sot"],
        "shot_location": ["home_shots_inside_box", "away_shots_inside_box", "home_shots_outside_box", "away_shots_outside_box"],
        "shot_body_part": ["home_header_shots", "away_header_shots", "home_left_foot_shots", "away_left_foot_shots", "home_right_foot_shots", "away_right_foot_shots"],
        "shot_quality": ["home_xg", "away_xg", "home_shots", "away_shots", "home_xg_per_shot", "away_xg_per_shot"],
        "set_pieces": ["home_corners", "away_corners", "home_set_piece_xg", "away_set_piece_xg"],
        "discipline": ["home_fouls", "away_fouls", "home_yellow_cards", "away_yellow_cards", "home_red_cards", "away_red_cards"],
        "possession": ["home_possession", "away_possession"],
        "territory": ["home_field_tilt", "away_field_tilt", "home_touches_box", "away_touches_box", "home_final_third_entries", "away_final_third_entries"],
        "progression": ["home_progressive_passes", "away_progressive_passes", "home_progressive_carries", "away_progressive_carries"],
        "defense": ["home_tackles", "away_tackles", "home_interceptions", "away_interceptions", "home_blocks", "away_blocks"],
        "goalkeeping": ["home_keeper_psxg", "away_keeper_psxg", "home_keeper_saves", "away_keeper_saves"],
    }
    rows = []
    for name, cols in feature_groups.items():
        available_cols = [c for c in cols if c in matches.columns]
        if available_cols:
            full = matches[available_cols].notna().all(axis=1)
            any_cov = matches[available_cols].notna().any(axis=1)
            full_count = int(full.sum())
            any_count = int(any_cov.sum())
        else:
            full_count = 0
            any_count = 0
        rows.append({
            "feature_group": name,
            "required_columns": ",".join(cols),
            "available_columns": ",".join(available_cols),
            "missing_columns": ",".join([c for c in cols if c not in matches.columns]),
            "rows": int(len(matches)),
            "rows_with_full_group": full_count,
            "rows_with_any_group": any_count,
            "full_coverage_rate": float(full_count / len(matches)) if len(matches) else 0.0,
            "any_coverage_rate": float(any_count / len(matches)) if len(matches) else 0.0,
            "status": "available" if full_count else ("partial" if any_count else "unavailable"),
        })
    coverage = pd.DataFrame(rows)
    summary = {
        "version": ADVANCED_DATA_VERSION,
        "dataset_name": dataset_name,
        "status": "ok" if (coverage["rows_with_any_group"] > 0).any() else "blocked",
        "rows": int(len(matches)),
        "feature_groups": coverage.to_dict(orient="records"),
        "providers": sorted(matches.get("provider", matches.get("xg_provider", pd.Series(dtype=str))).dropna().astype(str).unique().tolist()) if len(matches) else [],
        "recommendations": [
            "Treat current-match advanced columns as targets/diagnostics, not model features.",
            "Use model_ready snapshots to transform historical advanced stats into prior rolling features.",
            "Prioritize xG/shots/SOT coverage before supervised advanced models.",
            "Keep field_tilt once in the contract; do not duplicate it in possession and territory groups.",
            "Do not block baseline models when advanced provider coverage is partial.",
        ],
        "raw_data_changed": False,
    }
    return {"summary": summary, "coverage": coverage}

