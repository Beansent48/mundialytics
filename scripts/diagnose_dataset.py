from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.loaders import load_matches
from mundialytics.data.quality import data_quality_report


def main() -> None:
    p = argparse.ArgumentParser(description="Run data-quality diagnostics on a canonical match CSV.")
    p.add_argument("--matches", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    path = ROOT / args.matches if not Path(args.matches).is_absolute() else Path(args.matches)
    report = data_quality_report(load_matches(path))
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
