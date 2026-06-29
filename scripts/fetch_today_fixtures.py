from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.adapters.espn import DEFAULT_WORLD_CUP_LEAGUE, ESPNClient, scoreboard_response_to_df
from mundialytics.data.adapters.sofascore import SofaScoreClient, fetch_scheduled_events_window, scheduled_events_response_to_df
from mundialytics.data.free_fixtures import (
    filter_by_local_date,
    filter_contains,
    format_fixtures_table,
    resolved_local_date,
    world_cup_filters,
)


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _fetch_sofascore(local_date: str, *, days_before: int, days_after: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    client = SofaScoreClient()
    payloads, fetched_dates = fetch_scheduled_events_window(client, center_date=local_date, days_before=days_before, days_after=days_after)
    df = scheduled_events_response_to_df(payloads)
    return df, {"provider": "sofascore", "raw": {"fetched_dates": fetched_dates, "payloads": payloads}, "rows_raw": int(len(df))}


def _fetch_espn(local_date: str, *, espn_league: str, espn_limit: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    client = ESPNClient()
    dates = local_date.replace("-", "")
    payload = client.scoreboard(league=espn_league, dates=dates, limit=espn_limit)
    df = scoreboard_response_to_df(payload, league_slug=espn_league)
    return df, {"provider": "espn", "raw": payload, "rows_raw": int(len(df)), "espn_league": espn_league, "dates": dates}


def _apply_competition_filter(df: pd.DataFrame, competition: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not competition or competition == "all":
        return df, {"competition_filter": "all"}
    if competition in {"world_cup", "fifa_world_cup"}:
        include, exclude = world_cup_filters()
        out, report = filter_contains(df, include=include, exclude=exclude)
        report.update({"competition_filter": competition, "include": include, "exclude": exclude})
        return out, report
    # Generic contains filter for future extension.
    out, report = filter_contains(df, include=[competition.replace("_", " ")], exclude=[])
    report.update({"competition_filter": competition})
    return out, report


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch today's football fixtures from free/keyless providers. Defaults to World Cup in US Eastern time.")
    date_group = p.add_mutually_exclusive_group()
    date_group.add_argument("--date", default=None, help="Local calendar date, YYYY-MM-DD.")
    date_group.add_argument("--today", action="store_true")
    date_group.add_argument("--tomorrow", action="store_true")
    p.add_argument("--competition", default="world_cup", help="world_cup or all. Default: world_cup.")
    p.add_argument("--provider", choices=["auto", "sofascore", "espn"], default="auto")
    p.add_argument("--timezone", default="America/New_York")
    p.add_argument("--days-before", type=int, default=1, help="SofaScore only: fetch extra date-window days before selected local date.")
    p.add_argument("--days-after", type=int, default=1, help="SofaScore only: fetch extra date-window days after selected local date.")
    p.add_argument("--espn-league", default=DEFAULT_WORLD_CUP_LEAGUE, help="ESPN soccer league slug, e.g. fifa.world.")
    p.add_argument("--espn-limit", type=int, default=500)
    p.add_argument("--out", default="outputs/free_today_fixtures.csv")
    p.add_argument("--raw-out", default="outputs/free_today_fixtures_raw.json")
    p.add_argument("--report-out", default=None)
    p.add_argument("--print-table", action="store_true", default=True)
    p.add_argument("--no-print-table", dest="print_table", action="store_false")
    p.add_argument("--max-print-rows", type=int, default=80)
    args = p.parse_args()

    if not (args.date or args.today or args.tomorrow):
        args.today = True
    local_date, timezone_warning = resolved_local_date(date=args.date, today=args.today, tomorrow=args.tomorrow, timezone=args.timezone)

    providers = [args.provider] if args.provider != "auto" else ["sofascore", "espn"]
    attempts: list[dict[str, Any]] = []
    chosen_df = pd.DataFrame()
    chosen_raw: dict[str, Any] | None = None
    chosen_provider: str | None = None

    for provider in providers:
        try:
            if provider == "sofascore":
                df, raw = _fetch_sofascore(local_date, days_before=args.days_before, days_after=args.days_after)
            elif provider == "espn":
                df, raw = _fetch_espn(local_date, espn_league=args.espn_league, espn_limit=args.espn_limit)
            else:  # pragma: no cover
                raise ValueError(f"unknown provider {provider}")
            before_competition = len(df)
            df, comp_report = _apply_competition_filter(df, args.competition)
            df, local_filter_warning = filter_by_local_date(df, local_date=local_date, timezone=args.timezone)
            attempt = {
                "provider": provider,
                "status": "ok",
                "rows_raw": int(raw.get("rows_raw", before_competition)),
                "rows_after_competition_filter": int(len(df) + comp_report.get("exclude_removed_rows", 0)),
                "rows_final": int(len(df)),
                "competition_report": comp_report,
                "local_date_filter_warning": local_filter_warning,
            }
            attempts.append(attempt)
            if not df.empty or args.provider != "auto":
                chosen_df = df
                chosen_raw = raw
                chosen_provider = provider
                break
        except Exception as exc:
            attempts.append({"provider": provider, "status": "error", "error": repr(exc)})
            if args.provider != "auto":
                raise

    out_path = _resolve(args.out)
    assert out_path is not None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chosen_df.to_csv(out_path, index=False)

    raw_path = _resolve(args.raw_out) if args.raw_out else None
    if raw_path:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            json.dumps({"chosen_provider": chosen_provider, "local_date": local_date, "timezone": args.timezone, "attempts": attempts, "chosen_raw": chosen_raw}, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    report = {
        "status": "FREE_FIXTURES_FETCHED" if chosen_provider else "FREE_FIXTURES_NOT_FOUND",
        "chosen_provider": chosen_provider,
        "competition": args.competition,
        "resolved_local_date": local_date,
        "timezone": args.timezone,
        "rows": int(len(chosen_df)),
        "out": str(out_path),
        "raw_out": str(raw_path) if raw_path else None,
        "attempts": attempts,
        "timezone_warning": timezone_warning,
        "source_notes": [
            "SofaScore scheduled-events is a public unofficial endpoint; keep raw JSON and expect possible changes.",
            "ESPN site scoreboard is a public undocumented endpoint; useful as a fallback, not a contracted feed.",
        ],
    }
    report_path = _resolve(args.report_out) if args.report_out else None
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if args.print_table:
        print("\nFixtures:")
        print(format_fixtures_table(chosen_df, max_rows=args.max_print_rows))


if __name__ == "__main__":
    main()
