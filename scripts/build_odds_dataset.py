from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.adapters import football_data_uk_to_match_odds


def main() -> None:
    p = argparse.ArgumentParser(description="Extract canonical odds rows from public source files.")
    p.add_argument("--source", required=True, choices=["football-data-uk"])
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    inp = ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    if args.source == "football-data-uk":
        df = football_data_uk_to_match_odds(inp)
    else:  # pragma: no cover
        raise ValueError(args.source)
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Built odds dataset: {out}")
    print(f"rows={len(df)}, bookmakers={sorted(df['bookmaker'].unique().tolist()) if len(df) else []}")


if __name__ == "__main__":
    main()
