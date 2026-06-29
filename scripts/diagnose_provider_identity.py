from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.provider_identity import load_identity_map, resolve_lineup_row_with_identity_map


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose provider ID -> historical player mapping for lineups.")
    p.add_argument("--identity-map", required=True)
    p.add_argument("--lineups", default=None)
    p.add_argument("--players", nargs="*", default=[])
    args = p.parse_args()
    identity_map = load_identity_map(_resolve(args.identity_map))
    rows = []
    if args.lineups:
        lineups = pd.read_csv(_resolve(args.lineups))
        for _, r in lineups.iterrows():
            res = resolve_lineup_row_with_identity_map(r, identity_map)
            rows.append({**dict(r), **res.__dict__})
    for name in args.players:
        res = resolve_lineup_row_with_identity_map({"provider":"api_football", "player": name}, identity_map)
        rows.append({"player": name, **res.__dict__})
    df = pd.DataFrame(rows)
    if df.empty:
        print(json.dumps({"status": "NO_ROWS", "identity_rows": int(len(identity_map))}, indent=2))
    else:
        cols = [c for c in ["player","provider","provider_player_id","canonical_player_id","status","historical_player_id_global","historical_player_name","method","confidence","reason"] if c in df.columns]
        print(df[cols].to_string(index=False))
        print(json.dumps({"identity_rows": int(len(identity_map)), "status_counts": df["status"].value_counts(dropna=False).to_dict() if "status" in df else {}}, indent=2))


if __name__ == "__main__":
    main()
