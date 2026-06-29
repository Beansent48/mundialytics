from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.adapters import football_data_uk_to_matches, international_results_to_matches, openfootball_json_to_matches


def main() -> None:
    p = argparse.ArgumentParser(description="Convert public source files into Mundialytics canonical match CSVs.")
    p.add_argument("--source", required=True, choices=["football-data-uk", "international-results", "openfootball"])
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--season", default=None)
    p.add_argument("--competition", default=None)
    p.add_argument("--team-scope", default="club", choices=["club", "national"])
    args = p.parse_args()

    inp = ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    if args.source == "football-data-uk":
        df = football_data_uk_to_matches(inp, season=args.season)
    elif args.source == "international-results":
        df = international_results_to_matches(inp)
    elif args.source == "openfootball":
        df = openfootball_json_to_matches(inp, competition=args.competition or "openfootball", season=args.season or "unknown", team_scope=args.team_scope)
    else:  # pragma: no cover
        raise ValueError(args.source)

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Built canonical dataset: {out}")
    teams = sorted(set(df["home_team"].astype(str)).union(df["away_team"].astype(str)))
    print(f"rows={len(df)}, teams={len(teams)}, scope={sorted(df['team_scope'].unique())}")


if __name__ == "__main__":
    main()
