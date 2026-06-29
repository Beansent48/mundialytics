from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.adapters.api_football import ApiFootballClient, fixtures_response_to_df

API_FOOTBALL_WORLD_CUP_LEAGUE_ID = "1"
API_FOOTBALL_WORLD_CUP_SEASON = "2026"


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _today_in_timezone(timezone: str) -> tuple[str, str | None]:
    """Return YYYY-MM-DD for the requested IANA timezone.

    On Windows, zoneinfo may require the optional tzdata package. If the local
    environment does not have IANA timezone data installed, we fall back to the
    machine's local date and emit a warning in the command report. Users can
    always pass --date explicitly to avoid ambiguity.
    """
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(timezone)).date().isoformat(), None
    except Exception as exc:  # pragma: no cover - platform dependent
        return datetime.now().date().isoformat(), (
            f"Could not resolve timezone '{timezone}' locally ({exc!r}); "
            "used machine local date. Pass --date YYYY-MM-DD for an exact date."
        )



def _filter_df_by_local_date(df: pd.DataFrame, *, local_date: str | None, timezone: str) -> tuple[pd.DataFrame, str | None]:
    """Keep only fixtures whose kickoff calendar date matches local_date in timezone.

    API-Football returns kickoff timestamps. Relying only on the API's `date=`
    parameter can be confusing when the user is thinking in a US timezone but
    the machine/API payload is handled in UTC or local European time. This
    post-filter is the safety net: after fetching, we recompute the kickoff date
    in the requested IANA timezone and drop anything outside that local day.
    """
    if df.empty or not local_date:
        return df, None
    try:
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            kickoff = pd.to_datetime(df["timestamp"], unit="s", errors="coerce", utc=True)
        elif "date" in df.columns:
            kickoff = pd.to_datetime(df["date"], errors="coerce", utc=True)
        else:
            return df, "Could not post-filter by local date: no timestamp/date column."
        local = kickoff.dt.tz_convert(timezone)
        mask = local.dt.date.astype(str).eq(str(local_date))
        out = df.loc[mask].copy()
        if not out.empty:
            out["kickoff_local_date"] = local.loc[mask].dt.date.astype(str).values
            out["kickoff_local_time"] = local.loc[mask].dt.strftime("%H:%M %Z").values
        removed = int(len(df) - len(out))
        warning = None if removed == 0 else f"post_filtered_by_local_date_removed_rows={removed}"
        return out, warning
    except Exception as exc:  # pragma: no cover - platform/pandas timezone edge case
        return df, f"Could not post-filter by local date in timezone {timezone!r}: {exc!r}"

def _format_fixture_table(df: pd.DataFrame, *, max_rows: int = 80) -> str:
    if df.empty:
        return "No fixtures returned."
    shown = df.copy()
    if "kickoff_local_date" in shown.columns and "kickoff_local_time" in shown.columns:
        shown["kickoff"] = shown["kickoff_local_date"].astype(str) + " " + shown["kickoff_local_time"].astype(str)
    elif "date" in shown.columns:
        dt = pd.to_datetime(shown["date"], errors="coerce")
        # Keep the API-provided timezone/offset in the source CSV, but use a
        # readable local clock in the console table.
        shown["kickoff"] = dt.dt.strftime("%Y-%m-%d %H:%M %Z").fillna(shown["date"].astype(str))
    else:
        shown["kickoff"] = ""
    cols = [
        "fixture_id",
        "kickoff",
        "competition",
        "home_team",
        "away_team",
        "status_short",
        "status_long",
    ]
    cols = [c for c in cols if c in shown.columns]
    shown = shown.sort_values([c for c in ["date", "fixture_id"] if c in shown.columns]).head(max_rows)
    return shown[cols].to_string(index=False)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Fetch API-Football fixtures. Use --today with --timezone America/New_York "
            "to get today's games in US Eastern time and print fixture IDs."
        )
    )
    date_group = p.add_mutually_exclusive_group()
    date_group.add_argument("--date", default=None, help="Fixture date in YYYY-MM-DD. Interpreted by API-Football with --timezone when provided.")
    date_group.add_argument("--today", action="store_true", help="Use today's date in the requested --timezone.")
    date_group.add_argument("--tomorrow", action="store_true", help="Use tomorrow's date in the requested --timezone.")
    p.add_argument("--timezone", default="America/New_York", help="IANA timezone for API-Football timestamps, e.g. America/New_York, America/Los_Angeles.")
    p.add_argument("--league", default=None, help="Optional API-Football league id.")
    p.add_argument("--season", default=None, help="Optional season year, often required with --league.")
    p.add_argument("--world-cup", action="store_true", help="Shortcut for FIFA World Cup 2026: --league 1 --season 2026.")
    p.add_argument("--competition-contains", default=None, help="Optional post-filter on competition name, case-insensitive, e.g. 'World Cup'.")
    p.add_argument("--no-local-date-post-filter", action="store_true", help="Disable post-filtering fixtures to the requested local calendar date.")
    p.add_argument("--team", default=None, help="Optional API-Football team id.")
    p.add_argument("--from-date", dest="from_date", default=None)
    p.add_argument("--to-date", dest="to_date", default=None)
    p.add_argument("--out", default="outputs/api_football_fixtures.csv")
    p.add_argument("--raw-out", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--print-table", action="store_true", help="Print a compact table with fixture IDs and kickoff times.")
    p.add_argument("--max-print-rows", type=int, default=80)
    args = p.parse_args()

    resolved_date = args.date
    warning = None
    if args.today or args.tomorrow:
        resolved_date, warning = _today_in_timezone(args.timezone)
        if args.tomorrow:
            resolved_date = (datetime.fromisoformat(resolved_date) + timedelta(days=1)).date().isoformat()

    league = args.league
    season = args.season
    if args.world_cup:
        league = league or API_FOOTBALL_WORLD_CUP_LEAGUE_ID
        season = season or API_FOOTBALL_WORLD_CUP_SEASON

    params = {
        k: v
        for k, v in {
            "date": resolved_date,
            "league": league,
            "season": season,
            "team": args.team,
            "from": args.from_date,
            "to": args.to_date,
            "timezone": args.timezone,
        }.items()
        if v is not None
    }

    if not any(k in params for k in ["date", "from", "to", "league", "team"]):
        raise SystemExit("Specify --today, --date, --from-date/--to-date, --league, or --team. For today's games use: --today --timezone America/New_York")

    out_path = _resolve(args.out)
    raw_path = _resolve(args.raw_out) if args.raw_out else None
    assert out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client = ApiFootballClient(api_key=args.api_key)
    payload = client.fixtures(**params)
    if raw_path:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    df = fixtures_response_to_df(payload)
    rows_before_post_filter = int(len(df))
    competition_filter_warning = None
    if args.competition_contains and not df.empty and "competition" in df.columns:
        needle = str(args.competition_contains).casefold()
        df = df[df["competition"].fillna("").astype(str).str.casefold().str.contains(needle, regex=False)].copy()
        competition_filter_warning = f"competition_contains_removed_rows={rows_before_post_filter - len(df)}"

    local_date_filter_warning = None
    if not args.no_local_date_post_filter and resolved_date:
        df, local_date_filter_warning = _filter_df_by_local_date(df, local_date=resolved_date, timezone=args.timezone)

    df.to_csv(out_path, index=False)

    report = {
        "status": "API_FOOTBALL_FIXTURES_FETCHED",
        "params": params,
        "world_cup_shortcut": bool(args.world_cup),
        "rows_before_post_filter": rows_before_post_filter,
        "rows": int(len(df)),
        "out": str(out_path),
        "raw_out": str(raw_path) if raw_path else None,
        "timezone_warning": warning,
        "competition_filter_warning": competition_filter_warning,
        "local_date_filter_warning": local_date_filter_warning,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.print_table:
        print("\nFixtures:")
        print(_format_fixture_table(df, max_rows=args.max_print_rows))


if __name__ == "__main__":
    main()
