from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd

from mundialytics.betting.odds_contract import norm_key, norm_text

TARGET_MARKET_KEYS = {
    "1x2",
    "btts",
    "goals",
    "team_goals",
    "corners",
    "team_corners",
    "yellow_cards",
    "team_yellow_cards",
    "shots",
    "team_shots",
    "shots_on_target",
    "team_shots_on_target",
    "fouls",
    "team_fouls",
    "goalkeeper_saves",
    "team_goalkeeper_saves",
    "player_shots",
    "player_shots_on_target",
    "player_fouls_committed",
    "player_yellow_card",
}

SNAPSHOT_OFFSETS_SECONDS = {
    "t24h": 24 * 3600,
    "t6h": 6 * 3600,
    "t1h": 3600,
    "t10m": 10 * 60,
    "closing": 0,
}


def read_csv_safely(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False)


def ensure_datetime_utc(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return ts




def parse_datetime_series_utc(values: Any) -> pd.Series:
    """Parse a pandas Series/list/scalar into UTC timestamps robustly.

    Handles normal ISO strings, date-only strings, yyyymmdd integers/strings,
    Unix seconds and Unix milliseconds. Returns a Series so downstream code can
    safely use .dt even when all values are missing.
    """
    if isinstance(values, pd.Series):
        raw = values.copy()
    elif isinstance(values, (list, tuple, np.ndarray, pd.Index)):
        raw = pd.Series(values)
    else:
        raw = pd.Series([values])

    text = raw.astype(str).str.strip()

    # yyyymmdd integers/strings, e.g. 20260623. Do this before generic
    # pandas parsing because pd.to_datetime(20260623) can be treated as
    # nanoseconds after epoch.
    ymd_mask = raw.notna() & text.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")
    if ymd_mask.any():
        parsed.loc[ymd_mask] = pd.to_datetime(text.loc[ymd_mask], format="%Y%m%d", errors="coerce", utc=True)

    # Normal pandas parsing for everything else.
    generic_mask = parsed.isna() & raw.notna()
    if generic_mask.any():
        parsed.loc[generic_mask] = pd.to_datetime(raw.loc[generic_mask], errors="coerce", utc=True)

    # Unix epoch seconds/milliseconds if still unresolved and numeric.
    remaining = parsed.isna() & raw.notna()
    if remaining.any():
        numeric = pd.to_numeric(raw, errors="coerce")
        sec_mask = remaining & numeric.between(946684800, 4102444800, inclusive="both")  # 2000-2100 seconds
        if sec_mask.any():
            parsed.loc[sec_mask] = pd.to_datetime(numeric.loc[sec_mask], unit="s", errors="coerce", utc=True)
        remaining = parsed.isna() & raw.notna()
        ms_mask = remaining & numeric.between(946684800000, 4102444800000, inclusive="both")  # 2000-2100 ms
        if ms_mask.any():
            parsed.loc[ms_mask] = pd.to_datetime(numeric.loc[ms_mask], unit="ms", errors="coerce", utc=True)

    return pd.Series(parsed, index=raw.index, dtype="datetime64[ns, UTC]")


def date_column_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    """Small diagnostic payload for scripts when internal match parsing fails."""
    if df is None or df.empty:
        return {"rows": 0, "columns": []}
    cols = list(df.columns)
    likely = [c for c in cols if any(token in c.lower() for token in ["date", "time", "kickoff", "match", "home", "away", "team"])]
    preview_cols = likely[:12] or cols[:12]
    preview = df[preview_cols].head(3).astype(str).to_dict(orient="records")
    return {"rows": int(len(df)), "columns": cols, "likely_columns": likely, "preview": preview}

def iso_utc(ts: Any) -> str:
    t = ensure_datetime_utc(ts)
    if pd.isna(t):
        return ""
    return t.isoformat().replace("+00:00", "Z")


def epoch_sec(ts: Any) -> int | None:
    t = ensure_datetime_utc(ts)
    if pd.isna(t):
        return None
    return int(t.timestamp())


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_internal_matches(matches: pd.DataFrame, *, min_date: str = "2026-01-01") -> pd.DataFrame:
    """Normalize historical internal matches into a fixture-matching table.

    Required logical fields: match_id, date/kickoff_utc, home_team, away_team.
    The function is deliberately permissive about column names so it can use
    outputs from different Mundialytics versions. It is also defensive about
    date formats because some large backtest outputs store dates as text,
    yyyymmdd integers, date-only strings, Unix seconds, or empty kickoff fields.
    """
    empty_cols = [
        "match_id", "date", "kickoff_utc", "home_team", "away_team",
        "competition", "season", "gender", "team_type", "competition_context",
    ]
    if matches is None or matches.empty:
        return pd.DataFrame(columns=empty_cols)
    work = matches.copy()
    id_col = _first_existing_column(work, ["match_id", "internal_match_id", "game_id", "fixture_id", "id"])
    home_col = _first_existing_column(work, ["home_team", "home", "team_home", "home_name", "homeTeam", "home_team_name"])
    away_col = _first_existing_column(work, ["away_team", "away", "team_away", "away_name", "awayTeam", "away_team_name"])
    kickoff_col = _first_existing_column(work, [
        "kickoff_utc", "kickoff", "start_time", "startTime", "start_time_utc",
        "date_time", "datetime", "match_time", "event_time", "utcTime",
    ])
    date_col = _first_existing_column(work, ["date", "match_date", "utc_date", "game_date", "event_date"])
    if not id_col or not home_col or not away_col or not (kickoff_col or date_col):
        diag = date_column_diagnostics(work)
        raise ValueError(
            "Internal match input must include match_id, home_team, away_team and date/kickoff_utc-like columns. "
            f"Detected columns: {diag.get('likely_columns') or diag.get('columns')}"
        )

    out = pd.DataFrame(index=work.index)
    out["match_id"] = work[id_col].astype(str)

    # Build a proper datetime Series first; never call .dt on object dtype.
    if kickoff_col:
        kickoff_ts = parse_datetime_series_utc(work[kickoff_col])
    else:
        kickoff_ts = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns, UTC]")
    if date_col:
        date_ts = parse_datetime_series_utc(work[date_col])
        kickoff_ts = kickoff_ts.where(kickoff_ts.notna(), date_ts)
    kickoff_ts = parse_datetime_series_utc(kickoff_ts)

    out["date"] = kickoff_ts.dt.strftime("%Y-%m-%d")
    out["kickoff_utc"] = kickoff_ts.map(iso_utc)
    out["home_team"] = work[home_col].astype(str).str.strip()
    out["away_team"] = work[away_col].astype(str).str.strip()
    out["home_team_canonical"] = out["home_team"].map(norm_text)
    out["away_team_canonical"] = out["away_team"].map(norm_text)
    for col in ["competition", "season", "gender", "team_type", "competition_context"]:
        out[col] = work[col] if col in work.columns else ""

    # Drop rows without a valid date/team; if the caller used a market-line file,
    # duplicates per match are expected and removed below.
    out = out[out["date"].notna() & out["home_team"].ne("") & out["away_team"].ne("")]
    if min_date and not out.empty:
        min_ts = pd.to_datetime(min_date, errors="coerce")
        out = out[pd.to_datetime(out["date"], errors="coerce") >= min_ts]
    out = out.drop_duplicates("match_id").sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return out


def build_fixture_request_windows(
    matches: pd.DataFrame,
    *,
    chunk_hours: int = 24,
    pad_hours: int = 4,
    max_windows: int | None = None,
    api_max_window_hours: float = 239.0,
    clamp_to_internal_range: bool = True,
) -> pd.DataFrame:
    """Build OddsPapi fixture discovery windows synced to internal matches.

    OddsPapi fixture discovery with sportId + from/to allows both times only
    when they are under 10 days apart. This planner therefore caps the
    effective request span (chunk + 2*pad) below that limit by default.
    Bigger caller-provided chunk sizes are reduced rather than allowed to
    silently create invalid API requests.
    """
    columns = [
        "window_id", "from", "to", "startTimeFrom", "startTimeTo",
        "expected_matches", "raw_from", "raw_to", "span_hours",
        "raw_span_hours", "requested_chunk_hours", "effective_chunk_hours",
        "pad_hours", "api_max_window_hours", "chunk_was_capped",
    ]
    if matches is None or matches.empty:
        return pd.DataFrame(columns=columns)
    work = matches.copy()
    work["_kickoff"] = pd.to_datetime(work.get("kickoff_utc", work.get("date")), errors="coerce", utc=True)
    work = work.dropna(subset=["_kickoff"])
    if work.empty:
        return pd.DataFrame(columns=columns)

    requested_chunk_hours = max(1.0, float(chunk_hours))
    pad_hours_f = max(0.0, float(pad_hours))
    max_span = max(1.0, float(api_max_window_hours))
    max_effective_chunk_hours = max(1.0, max_span - 2 * pad_hours_f)
    effective_chunk_hours = min(requested_chunk_hours, max_effective_chunk_hours)
    chunk_was_capped = bool(effective_chunk_hours < requested_chunk_hours)

    start = work["_kickoff"].min().floor("D")
    end = work["_kickoff"].max().ceil("D")
    chunk = pd.Timedelta(hours=effective_chunk_hours)
    pad = pd.Timedelta(hours=pad_hours_f)
    rows = []
    cursor = start
    wid = 1
    while cursor < end:
        raw_end = min(cursor + chunk, end)
        frm = cursor - pad
        to = raw_end + pad
        if clamp_to_internal_range:
            frm = max(frm, start)
            to = min(to, end)
        # Final safety clamp: never let a docs_v4 window reach/exceed the
        # configured max API window span. This matters when float rounding or
        # future parameter changes alter chunk/pad values.
        span_hours = (to - frm).total_seconds() / 3600.0
        if span_hours > max_span:
            to = frm + pd.Timedelta(hours=max_span)
            span_hours = (to - frm).total_seconds() / 3600.0
        mask = work["_kickoff"].between(frm, to, inclusive="both")
        rows.append({
            "window_id": wid,
            "from": iso_utc(frm),
            "to": iso_utc(to),
            "startTimeFrom": int(frm.timestamp()),
            "startTimeTo": int(to.timestamp()),
            "expected_matches": int(mask.sum()),
            "raw_from": iso_utc(cursor),
            "raw_to": iso_utc(raw_end),
            "span_hours": round(span_hours, 4),
            "raw_span_hours": round((raw_end - cursor).total_seconds() / 3600.0, 4),
            "requested_chunk_hours": requested_chunk_hours,
            "effective_chunk_hours": effective_chunk_hours,
            "pad_hours": pad_hours_f,
            "api_max_window_hours": max_span,
            "chunk_was_capped": chunk_was_capped,
            "clamp_to_internal_range": bool(clamp_to_internal_range),
        })
        wid += 1
        cursor = raw_end
        if max_windows is not None and len(rows) >= max_windows:
            break
    return pd.DataFrame(rows, columns=columns)


def summarize_fixture_plan(matches: pd.DataFrame, windows: pd.DataFrame) -> dict[str, Any]:
    """Return date/range diagnostics for a fixture-discovery plan."""
    summary: dict[str, Any] = {
        "prepared_matches": int(len(matches)) if matches is not None else 0,
        "request_windows": int(len(windows)) if windows is not None else 0,
    }
    if matches is not None and not matches.empty:
        dt = pd.to_datetime(matches.get("kickoff_utc", matches.get("date")), errors="coerce", utc=True).dropna()
        if not dt.empty:
            summary.update({
                "internal_min_kickoff_utc": iso_utc(dt.min()),
                "internal_max_kickoff_utc": iso_utc(dt.max()),
                "internal_min_date": dt.min().strftime("%Y-%m-%d"),
                "internal_max_date": dt.max().strftime("%Y-%m-%d"),
            })
    if windows is not None and not windows.empty:
        spans = pd.to_numeric(windows.get("span_hours"), errors="coerce")
        summary.update({
            "first_window_from": str(windows.iloc[0].get("from", "")),
            "last_window_to": str(windows.iloc[-1].get("to", "")),
            "max_window_span_hours": float(spans.max()) if spans.notna().any() else None,
            "min_window_span_hours": float(spans.min()) if spans.notna().any() else None,
            "total_expected_match_window_hits": int(pd.to_numeric(windows.get("expected_matches"), errors="coerce").fillna(0).sum()),
            "any_chunk_capped": bool(windows.get("chunk_was_capped", pd.Series(dtype=bool)).astype(bool).any()) if "chunk_was_capped" in windows.columns else False,
        })
    return summary


def stable_json_name(prefix: str, parts: list[Any]) -> str:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(parts[0]))[:80] if parts else prefix
    return f"{prefix}_{safe}_{digest}.json"


def match_internal_to_provider(
    internal_matches: pd.DataFrame,
    provider_fixtures: pd.DataFrame,
    *,
    max_hours_diff: float = 30.0,
    auto_threshold: float = 0.86,
) -> pd.DataFrame:
    """Fuzzy match internal historical matches to OddsPapi provider fixtures."""
    import difflib

    if internal_matches.empty or provider_fixtures.empty:
        return pd.DataFrame()
    m = internal_matches.copy()
    f = provider_fixtures.copy()
    m["_kickoff"] = pd.to_datetime(m.get("kickoff_utc", m.get("date")), errors="coerce", utc=True)
    f["_kickoff"] = pd.to_datetime(f.get("kickoff_utc", f.get("date")), errors="coerce", utc=True)
    rows: list[dict[str, Any]] = []
    for _, im in m.dropna(subset=["_kickoff"]).iterrows():
        lo = im["_kickoff"] - pd.Timedelta(hours=max_hours_diff)
        hi = im["_kickoff"] + pd.Timedelta(hours=max_hours_diff)
        candidates = f[f["_kickoff"].between(lo, hi, inclusive="both")].copy()
        for _, pf in candidates.iterrows():
            home_direct = difflib.SequenceMatcher(None, norm_text(im.get("home_team")), norm_text(pf.get("home_team"))).ratio()
            away_direct = difflib.SequenceMatcher(None, norm_text(im.get("away_team")), norm_text(pf.get("away_team"))).ratio()
            home_swap = difflib.SequenceMatcher(None, norm_text(im.get("home_team")), norm_text(pf.get("away_team"))).ratio()
            away_swap = difflib.SequenceMatcher(None, norm_text(im.get("away_team")), norm_text(pf.get("home_team"))).ratio()
            direct = (home_direct + away_direct) / 2
            swapped = (home_swap + away_swap) / 2
            best = max(direct, swapped)
            delta_h = abs((im["_kickoff"] - pf["_kickoff"]).total_seconds()) / 3600.0
            # penalize large time differences lightly; useful when only date is known internally
            confidence = max(0.0, best - min(delta_h / max(max_hours_diff, 1), 1) * 0.08)
            rows.append({
                "match_id": im.get("match_id"),
                "date": im.get("date"),
                "home_team": im.get("home_team"),
                "away_team": im.get("away_team"),
                "kickoff_utc": im.get("kickoff_utc"),
                "provider": pf.get("provider", "oddspapi"),
                "provider_fixture_id": pf.get("provider_fixture_id") or pf.get("fixture_id"),
                "provider_home_team": pf.get("home_team"),
                "provider_away_team": pf.get("away_team"),
                "provider_kickoff_utc": pf.get("kickoff_utc"),
                "provider_tournament_id": pf.get("tournament_id"),
                "provider_tournament_name": pf.get("tournament_name"),
                "direct_team_score": direct,
                "swapped_team_score": swapped,
                "orientation": "direct" if direct >= swapped else "swapped",
                "hours_diff": delta_h,
                "match_confidence": confidence,
                "auto_match": bool(confidence >= auto_threshold),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["match_id", "auto_match", "match_confidence", "hours_diff"], ascending=[True, False, False, True])
    return out


def select_best_fixture_matches(candidates: pd.DataFrame, *, auto_threshold: float = 0.86) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return pd.DataFrame()
    work = candidates.copy()
    work = work[work["match_confidence"].ge(auto_threshold)].copy()
    if work.empty:
        return pd.DataFrame(columns=candidates.columns)
    work = work.sort_values(["match_id", "match_confidence", "hours_diff"], ascending=[True, False, True])
    return work.groupby("match_id", as_index=False).head(1).reset_index(drop=True)


def filter_target_market_mapping(mapping: pd.DataFrame, *, include_review_required: bool = True) -> pd.DataFrame:
    if mapping is None or mapping.empty:
        return pd.DataFrame()
    work = mapping.copy()
    if "internal_market_key" not in work.columns:
        return pd.DataFrame()
    work["internal_market_key"] = work["internal_market_key"].astype(str).map(norm_key)
    work = work[work["internal_market_key"].isin(TARGET_MARKET_KEYS)].copy()
    if not include_review_required and "mapping_confidence" in work.columns:
        work = work[work["mapping_confidence"].isin(["high", "medium"])]
    return work.reset_index(drop=True)


def build_snapshot_rows(
    odds_ticks: pd.DataFrame,
    kickoff_lookup: pd.DataFrame,
    *,
    snapshot_offsets: dict[str, int] | None = None,
    allow_closing_at_kickoff: bool = False,
) -> pd.DataFrame:
    """Build leakage-safe snapshots from normalized odds ticks.

    For each market selection and snapshot label, choose the latest observed price with
    snapshot_time <= kickoff - offset. For closing, offset=0 but still pre-kickoff unless
    allow_closing_at_kickoff=True.
    """
    if odds_ticks is None or odds_ticks.empty:
        return pd.DataFrame()
    offsets = snapshot_offsets or SNAPSHOT_OFFSETS_SECONDS
    odds = odds_ticks.copy()
    odds["_snapshot_ts"] = pd.to_datetime(odds["snapshot_time_utc"], errors="coerce", utc=True)
    if "kickoff_utc" not in odds.columns:
        lookup = kickoff_lookup.copy()
        if "provider_fixture_id" in lookup.columns and "provider_event_id" in odds.columns:
            lk = lookup.drop_duplicates("provider_fixture_id").set_index("provider_fixture_id")
            odds["kickoff_utc"] = odds["provider_event_id"].map(lk.get("provider_kickoff_utc", lk.get("kickoff_utc", pd.Series(dtype=str))))
        elif "match_id" in lookup.columns:
            lk = lookup.drop_duplicates("match_id").set_index("match_id")
            odds["kickoff_utc"] = odds["match_id"].map(lk.get("provider_kickoff_utc", lk.get("kickoff_utc", pd.Series(dtype=str))))
        else:
            odds["kickoff_utc"] = ""
    odds["_kickoff_ts"] = pd.to_datetime(odds["kickoff_utc"], errors="coerce", utc=True)
    odds = odds.dropna(subset=["_snapshot_ts", "_kickoff_ts"])
    odds["bookmaker_odds"] = pd.to_numeric(odds["bookmaker_odds"], errors="coerce")
    odds = odds[odds["bookmaker_odds"].gt(1.0)].copy()
    group_cols = [
        "match_id", "provider_event_id", "bookmaker", "market_key", "scope", "subject_team", "subject_player", "line", "side"
    ]
    for c in group_cols:
        if c not in odds.columns:
            odds[c] = ""
    rows: list[pd.DataFrame] = []
    for label, offset in offsets.items():
        cutoff = odds["_kickoff_ts"] - pd.to_timedelta(int(offset), unit="s")
        if label == "closing" and not allow_closing_at_kickoff:
            valid = odds["_snapshot_ts"].lt(cutoff)
        else:
            valid = odds["_snapshot_ts"].le(cutoff)
        sub = odds[valid].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(group_cols + ["_snapshot_ts"])
        selected = sub.groupby(group_cols, dropna=False, as_index=False).tail(1).copy()
        selected["snapshot_label"] = label
        selected["target_cutoff_utc"] = cutoff.loc[selected.index].map(iso_utc).values
        selected["seconds_before_kickoff"] = (selected["_kickoff_ts"] - selected["_snapshot_ts"]).dt.total_seconds()
        rows.append(selected)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep = [c for c in odds_ticks.columns if c in out.columns] + ["kickoff_utc", "snapshot_label", "target_cutoff_utc", "seconds_before_kickoff"]
    # Deduplicate preserving order.
    seen = set(); cols = []
    for c in keep:
        if c not in seen:
            cols.append(c); seen.add(c)
    return out[cols].sort_values(["match_id", "market_key", "scope", "line", "side", "snapshot_label"]).reset_index(drop=True)


def audit_backfill_coverage(mapping: pd.DataFrame, odds_ticks: pd.DataFrame, snapshots: pd.DataFrame | None = None) -> dict[str, Any]:
    mapped_matches = int(mapping["match_id"].nunique()) if mapping is not None and not mapping.empty and "match_id" in mapping.columns else 0
    priced_matches = int(odds_ticks["match_id"].nunique()) if odds_ticks is not None and not odds_ticks.empty and "match_id" in odds_ticks.columns else 0
    markets = sorted(odds_ticks["market_key"].dropna().astype(str).unique().tolist()) if odds_ticks is not None and not odds_ticks.empty and "market_key" in odds_ticks.columns else []
    target_markets_found = sorted(set(markets) & TARGET_MARKET_KEYS)
    summary = {
        "mapped_matches": mapped_matches,
        "priced_matches": priced_matches,
        "priced_match_coverage": round(priced_matches / mapped_matches, 4) if mapped_matches else 0.0,
        "odds_tick_rows": int(len(odds_ticks)) if odds_ticks is not None else 0,
        "snapshot_rows": int(len(snapshots)) if snapshots is not None else 0,
        "markets_found": markets,
        "target_markets_found": target_markets_found,
        "target_markets_missing": sorted(TARGET_MARKET_KEYS - set(target_markets_found)),
    }
    if odds_ticks is not None and not odds_ticks.empty and "market_key" in odds_ticks.columns:
        summary["market_rows"] = odds_ticks.groupby("market_key").size().sort_values(ascending=False).to_dict()
    if snapshots is not None and not snapshots.empty and "snapshot_label" in snapshots.columns:
        summary["snapshot_label_rows"] = snapshots.groupby("snapshot_label").size().to_dict()
    return summary
