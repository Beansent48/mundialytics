from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.inference.safe_props import prepare_player_events_for_model
from mundialytics.models.player_event_model import PlayerEventModel
from mundialytics.data.identity import player_global_id


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose current-lineup player identity matching against historical player_events.")
    ap.add_argument("--player-events", required=True)
    ap.add_argument("--players", nargs="+", required=True, help="Player display names to search/resolve")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    events = pd.read_csv(_resolve(args.player_events))
    hist = prepare_player_events_for_model(events)
    model = PlayerEventModel().fit(hist)
    rows = []
    for name in args.players:
        guessed = player_global_id(name)
        match = model.resolve_player_identity(name, guessed)
        profile = model.player_sample_profile(name, match.matched_player_id_global if match.status == "matched" else guessed)
        rows.append({
            "input_player": name,
            "input_player_id_global": guessed,
            "matched_player_name": match.matched_player,
            "matched_player_id_global": match.matched_player_id_global,
            "player_match_status": match.status,
            "player_match_method": match.method,
            "player_match_confidence": match.confidence,
            "reason": match.reason,
            **profile,
        })
    payload = {"status": "PLAYER_IDENTITY_DIAGNOSIS_COMPLETE", "rows": rows}
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    if args.out:
        out = _resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
