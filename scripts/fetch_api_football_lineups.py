from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.adapters.api_football import ApiFootballClient, lineups_response_to_df


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch API-Football fixture lineups into current_lineups/provider-player format.")
    p.add_argument("--fixture-id", required=True, help="API-Football fixture id")
    p.add_argument("--date", default=None)
    p.add_argument("--competition", default=None)
    p.add_argument("--out", default="outputs/api_football_current_lineups.csv")
    p.add_argument("--raw-out", default=None)
    p.add_argument("--api-key", default=None, help="Optional; otherwise reads API_FOOTBALL_KEY/APISPORTS_KEY")
    args = p.parse_args()
    out_path = _resolve(args.out)
    raw_path = _resolve(args.raw_out) if args.raw_out else None
    assert out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client = ApiFootballClient(api_key=args.api_key)
    payload = client.fixture_lineups(args.fixture_id)
    if raw_path:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    df = lineups_response_to_df(payload, fixture_id=args.fixture_id, date=args.date, competition=args.competition)
    df.to_csv(out_path, index=False)
    print(json.dumps({"status": "API_FOOTBALL_LINEUPS_FETCHED", "fixture_id": args.fixture_id, "rows": int(len(df)), "players": int(df["provider_player_id"].nunique()) if not df.empty else 0, "out": str(out_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
