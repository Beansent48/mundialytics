from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.adapters.sofascore import SofaScoreClient, fetch_scheduled_events_window, scheduled_events_response_to_df


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _today_in_timezone(timezone: str) -> tuple[str, str | None]:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(timezone)).date().isoformat(), None
    except Exception as exc:  # pragma: no cover
        return datetime.now().date().isoformat(), (
            f"Could not resolve timezone '{timezone}' locally ({exc!r}); used machine local date. "
            "Pass --date YYYY-MM-DD to avoid ambiguity."
        )


def _filter_df_by_local_date(df: pd.DataFrame, *, local_date: str | None, timezone: str) -> tuple[pd.DataFrame, str | None]:
    if df.empty or not local_date:
        return df, None
    try:
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            kickoff = pd.to_datetime(df["timestamp"], unit="s", errors="coerce", utc=True)
        else:
            kickoff = pd.to_datetime(df["date"], errors="coerce", utc=True)
        local = kickoff.dt.tz_convert(timezone)
        mask = local.dt.date.astype(str).eq(str(local_date))
        out = df.loc[mask].copy()
        if not out.empty:
            out["kickoff_local_date"] = local.loc[mask].dt.date.astype(str).values
            out["kickoff_local_time"] = local.loc[mask].dt.strftime("%H:%M %Z").values
        removed = int(len(df) - len(out))
        return out, None if removed == 0 else f"post_filtered_by_local_date_removed_rows={removed}"
    except Exception as exc:  # pragma: no cover
        return df, f"Could not post-filter by local date in timezone {timezone!r}: {exc!r}"


def _filter_contains(df: pd.DataFrame, *, include: list[str], exclude: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    if df.empty:
        return df, {"include_removed_rows": 0, "exclude_removed_rows": 0}
    search_cols = [c for c in ["competition", "tournament_name", "unique_tournament_name", "category", "round"] if c in df.columns]
    if not search_cols:
        return df, {"include_removed_rows": 0, "exclude_removed_rows": 0}
    haystack = df[search_cols].fillna("").astype(str).agg(" ".join, axis=1).str.casefold()
    before = len(df)
    if include:
        include_mask = False
        for needle in include:
            include_mask = include_mask | haystack.str.contains(str(needle).casefold(), regex=False)
        df = df.loc[include_mask].copy()
        haystack = haystack.loc[df.index]
    after_include = len(df)
    if exclude and not df.empty:
        exclude_mask = False
        for needle in exclude:
            exclude_mask = exclude_mask | haystack.str.contains(str(needle).casefold(), regex=False)
        df = df.loc[~exclude_mask].copy()
    return df, {"include_removed_rows": before - after_include, "exclude_removed_rows": after_include - len(df)}


def _format_table(df: pd.DataFrame, *, max_rows: int) -> str:
    if df.empty:
        return "No fixtures returned."
    shown = df.copy()
    if "kickoff_local_date" in shown.columns and "kickoff_local_time" in shown.columns:
        shown["kickoff"] = shown["kickoff_local_date"].astype(str) + " " + shown["kickoff_local_time"].astype(str)
    else:
        shown["kickoff"] = pd.to_datetime(shown["date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")
    cols = ["fixture_id", "kickoff", "competition", "home_team", "away_team", "status_short", "status_long"]
    cols = [c for c in cols if c in shown.columns]
    sort_cols = [c for c in ["timestamp", "fixture_id"] if c in shown.columns]
    shown = shown.sort_values(sort_cols).head(max_rows) if sort_cols else shown.head(max_rows)
    return shown[cols].to_string(index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch free football fixtures from SofaScore public scheduled-events endpoint.")
    date_group = p.add_mutually_exclusive_group()
    date_group.add_argument("--date", default=None, help="Local calendar date to keep, YYYY-MM-DD.")
    date_group.add_argument("--today", action="store_true")
    date_group.add_argument("--tomorrow", action="store_true")
    p.add_argument("--timezone", default="America/New_York", help="IANA timezone for selecting/printing the local day.")
    p.add_argument("--world-cup", action="store_true", help="Convenience filter for senior FIFA World Cup fixtures.")
    p.add_argument("--competition-contains", action="append", default=[], help="Post-filter competition/tournament text. Can be repeated.")
    p.add_argument("--exclude-contains", action="append", default=[], help="Exclude competition/tournament text. Can be repeated.")
    p.add_argument("--days-before", type=int, default=1, help="Fetch this many extra SofaScore schedule dates before local date.")
    p.add_argument("--days-after", type=int, default=1, help="Fetch this many extra SofaScore schedule dates after local date.")
    p.add_argument("--no-local-date-post-filter", action="store_true")
    p.add_argument("--out", default="outputs/sofascore_fixtures.csv")
    p.add_argument("--raw-out", default=None)
    p.add_argument("--print-table", action="store_true", default=True)
    p.add_argument("--no-print-table", dest="print_table", action="store_false")
    p.add_argument("--max-print-rows", type=int, default=80)
    args = p.parse_args()

    resolved_date = args.date
    timezone_warning = None
    if args.today or args.tomorrow:
        resolved_date, timezone_warning = _today_in_timezone(args.timezone)
        if args.tomorrow:
            resolved_date = (datetime.fromisoformat(resolved_date) + timedelta(days=1)).date().isoformat()
    if not resolved_date:
        raise SystemExit("Specify --today, --tomorrow, or --date YYYY-MM-DD.")

    include = list(args.competition_contains or [])
    exclude = list(args.exclude_contains or [])
    if args.world_cup:
        # SofaScore has used names such as 'World Cup' and 'World Championship'.
        include.extend(["world cup", "world championship"])
        exclude.extend(["women", "u17", "u19", "u20", "club", "qualification", "qualifying", "qualifier"])

    client = SofaScoreClient()
    payloads, fetched_dates = fetch_scheduled_events_window(client, center_date=resolved_date, days_before=args.days_before, days_after=args.days_after)

    raw_path = _resolve(args.raw_out) if args.raw_out else None
    if raw_path:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps({"fetched_dates": fetched_dates, "payloads": payloads}, indent=2, ensure_ascii=False), encoding="utf-8")

    df = scheduled_events_response_to_df(payloads)
    rows_before_filters = int(len(df))
    df, contains_report = _filter_contains(df, include=include, exclude=exclude)
    local_date_filter_warning = None
    if not args.no_local_date_post_filter:
        df, local_date_filter_warning = _filter_df_by_local_date(df, local_date=resolved_date, timezone=args.timezone)

    out_path = _resolve(args.out)
    assert out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    report = {
        "status": "SOFASCORE_FIXTURES_FETCHED",
        "source_note": "SofaScore scheduled-events is a public, unofficial endpoint; keep raw JSON and expect possible changes.",
        "resolved_local_date": resolved_date,
        "timezone": args.timezone,
        "fetched_sofascore_dates": fetched_dates,
        "world_cup_filter": bool(args.world_cup),
        "include_filters": include,
        "exclude_filters": exclude,
        "rows_before_filters": rows_before_filters,
        "rows": int(len(df)),
        "out": str(out_path),
        "raw_out": str(raw_path) if raw_path else None,
        "timezone_warning": timezone_warning,
        "contains_filter_report": contains_report,
        "local_date_filter_warning": local_date_filter_warning,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.print_table:
        print("\nFixtures:")
        print(_format_table(df, max_rows=args.max_print_rows))


if __name__ == "__main__":
    main()
