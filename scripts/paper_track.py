from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.reports.paper_tracker import append_picks, ledger_summary, settle_ledger


def main() -> None:
    p = argparse.ArgumentParser(description="Append or settle paper-betting picks.")
    sub = p.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("append")
    add.add_argument("--picks", required=True)
    add.add_argument("--ledger", default="outputs/paper_ledger.csv")
    add.add_argument("--stake", type=float, default=1.0)
    add.add_argument("--created-at", default=None, help="Optional ISO timestamp for reproducible paper ledgers.")
    settle = sub.add_parser("settle")
    settle.add_argument("--ledger", default="outputs/paper_ledger.csv")
    settle.add_argument("--outcomes", required=True)
    args = p.parse_args()

    if args.cmd == "append":
        picks = pd.read_csv(ROOT / args.picks if not Path(args.picks).is_absolute() else args.picks)
        ledger_path = ROOT / args.ledger if not Path(args.ledger).is_absolute() else Path(args.ledger)
        created_at = args.created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        ledger = append_picks(picks, ledger_path, created_at, stake=args.stake)
    else:
        ledger_path = ROOT / args.ledger if not Path(args.ledger).is_absolute() else Path(args.ledger)
        ledger = pd.read_csv(ledger_path)
        outcomes = pd.read_csv(ROOT / args.outcomes if not Path(args.outcomes).is_absolute() else args.outcomes)
        ledger = settle_ledger(ledger, outcomes)
        ledger.to_csv(ledger_path, index=False)
    print(json.dumps(ledger_summary(ledger), indent=2))


if __name__ == "__main__":
    main()
