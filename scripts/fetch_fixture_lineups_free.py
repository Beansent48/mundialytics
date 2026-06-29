from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.adapters.sofascore import SofaScoreClient, lineups_response_to_df


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _fixture_row_from_args(args) -> dict:
    row = {}
    if args.fixtures:
        fx = pd.read_csv(_resolve(args.fixtures))
        key = "fixture_id" if "fixture_id" in fx.columns else "provider_match_id"
        hit = fx[fx[key].astype(str).eq(str(args.fixture_id))]
        if hit.empty:
            raise SystemExit(f"fixture_id {args.fixture_id!r} not found in {args.fixtures}")
        row = hit.iloc[0].to_dict()
    else:
        row = {
            "fixture_id": args.fixture_id,
            "match_id": f"sofascore:{args.fixture_id}",
            "date": args.date,
            "competition": args.competition,
            "home_team": args.home_team,
            "away_team": args.away_team,
            "team_scope": args.team_scope,
            "team_type": args.team_type,
            "competition_context": args.competition_context,
            "gender": args.gender,
        }
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch lineups for one SofaScore fixture/event id and output current_lineups-compatible CSV.")
    p.add_argument("--fixture-id", required=True, help="SofaScore event id from fetch_today_fixtures.py output.")
    p.add_argument("--fixtures", default=None, help="Optional fixtures CSV to attach teams/competition metadata.")
    p.add_argument("--date", default=None)
    p.add_argument("--competition", default="FIFA World Cup")
    p.add_argument("--home-team", default=None)
    p.add_argument("--away-team", default=None)
    p.add_argument("--team-scope", default="national")
    p.add_argument("--team-type", default="national_team")
    p.add_argument("--competition-context", default="international_national_tournament")
    p.add_argument("--gender", default="men")
    p.add_argument("--out", default="outputs/sofascore_fixture_lineups.csv")
    p.add_argument("--raw-out", default="outputs/sofascore_fixture_lineups_raw.json")
    args = p.parse_args()

    fixture_row = _fixture_row_from_args(args)
    client = SofaScoreClient()
    payload = client.event_lineups(args.fixture_id)
    raw_path = _resolve(args.raw_out)
    if raw_path:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    df = lineups_response_to_df(payload, fixture_row=fixture_row, provider_match_id=args.fixture_id)
    out_path = _resolve(args.out)
    assert out_path is not None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    report = {
        "status": "FREE_FIXTURE_LINEUPS_FETCHED",
        "provider": "sofascore",
        "fixture_id": args.fixture_id,
        "rows": int(len(df)),
        "out": str(out_path),
        "raw_out": str(raw_path) if raw_path else None,
        "lineup_status_counts": df.get("lineup_status", pd.Series(dtype=str)).value_counts(dropna=False).to_dict() if not df.empty else {},
        "note": "SofaScore lineups are public/unofficial; lineups may be unavailable until near kickoff.",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
