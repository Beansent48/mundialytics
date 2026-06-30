#!/usr/bin/env python3
"""Recompute real per-player assist counts and merge them into the profiles CSV.

player_profiles_with_positions.csv ships an assists_per_match column that is
0.0 for every single player (assists_sample mirrors shots_sample exactly,
suggesting whatever ad-hoc script built the CSV copy-pasted the shots
aggregation and never summed real assists). StatsBomb's raw event JSON
already exposes a reliable `pass.goal_assist` flag — see
statsbomb_events_to_player_events() in
src/mundialytics/data/adapters/statsbomb.py — so this rebuilds assists from
the source events instead of trusting the broken column.

Run:
    python scripts/fix_player_assists.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.adapters.statsbomb import statsbomb_events_to_player_events

PROFILES_PATH = ROOT / "data/processed/player_profiles_with_positions.csv"
EVENTS_DIR = ROOT / "data/raw/statsbomb/open-data/data/events"


def main() -> None:
    files = sorted(EVENTS_DIR.glob("*.json"))
    print(f"Scanning {len(files)} match event files for assists...")

    totals: dict[str, int] = {}
    for i, f in enumerate(files):
        try:
            df = statsbomb_events_to_player_events(f, match_id=f.stem)
        except Exception as exc:
            print(f"  skip {f.name}: {exc}")
            continue
        if df.empty:
            continue
        for player, assists in zip(df["player"], df["assists"]):
            if assists:
                totals[player] = totals.get(player, 0) + int(assists)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)} files processed...")

    print(f"Found assists for {len(totals)} players")

    profiles = pd.read_csv(PROFILES_PATH)
    assists_series = profiles["player"].map(totals).fillna(0.0)
    matches = profiles["matches"].clip(lower=1)
    profiles["assists_per_match"] = assists_series / matches

    profiles.to_csv(PROFILES_PATH, index=False)
    nonzero = int((profiles["assists_per_match"] > 0).sum())
    print(f"{nonzero}/{len(profiles)} players now have nonzero assists_per_match")
    print(f"Wrote {PROFILES_PATH}")


if __name__ == "__main__":
    main()
