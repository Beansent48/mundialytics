from __future__ import annotations

import argparse
import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.loaders import download_file

DEFAULT_INTERNATIONAL_RESULTS = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def main() -> None:
    p = argparse.ArgumentParser(description="Reproducible download helper for public football data.")
    sub = p.add_subparsers(dest="cmd", required=True)

    intl = sub.add_parser("international-results", help="Download martj42 international results.csv")
    intl.add_argument("--url", default=DEFAULT_INTERNATIONAL_RESULTS)
    intl.add_argument("--out", default="data/raw/international_results/results.csv")

    fdu = sub.add_parser("football-data-uk", help="Download one Football-Data.co.uk CSV by direct URL")
    fdu.add_argument("--url", required=True, help="Direct CSV URL, e.g. https://www.football-data.co.uk/mmz4281/2526/E0.csv")
    fdu.add_argument("--out", required=True, help="Output CSV path, e.g. data/raw/football_data_uk/2526_E0.csv")

    ce = sub.add_parser("clubelo", help="Download ClubElo CSV API endpoint")
    ce.add_argument("--url", default="http://api.clubelo.com/2026-06-01")
    ce.add_argument("--out", default="data/raw/clubelo/clubelo.csv")

    sb = sub.add_parser("statsbomb-open-data", help="Download and extract full StatsBomb Open Data repository data folder")
    sb.add_argument("--out", default="data/raw/statsbomb/open-data/data")
    sb.add_argument("--zip", default=None, help="Optional local GitHub zip; use this if automatic download fails")

    raw = sub.add_parser("raw-url", help="Download any raw public file with a URL")
    raw.add_argument("--url", required=True)
    raw.add_argument("--out", required=True)

    args = p.parse_args()
    if args.cmd == "statsbomb-open-data":
        cmd = [sys.executable, str(ROOT / "scripts" / "setup_statsbomb_open_data.py"), "--out", args.out]
        if args.zip:
            cmd += ["--zip", args.zip]
        subprocess.run(cmd, check=True)
        return

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    download_file(args.url, out)
    print(f"Downloaded {args.url} -> {out}")


if __name__ == "__main__":
    main()
