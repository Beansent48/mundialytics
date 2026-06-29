#!/usr/bin/env python3
"""Build player-level goalkeeper saves from StatsBomb raw event JSON files.

This uses real Goal Keeper save events and Starting XI goalkeepers when available, so it can
include zero-save rows instead of only rows where a goalkeeper made a save.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.extra_match_stats import parse_statsbomb_goalkeeper_match_json


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def _is_event_json(path: Path) -> bool:
    name = path.name.lower()
    if not name.endswith(".json"):
        return False
    if name.endswith(".metadata.json"):
        return False
    if "summary" in name or "dry_run" in name or name.startswith("competitions") or name.startswith("matches"):
        return False
    return True


def _discover(paths: list[str], dirs: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in paths or []:
        p = _resolve(item)
        if p.exists() and _is_event_json(p):
            out.append(p)
    for d in dirs or []:
        dp = _resolve(d)
        if dp.exists():
            out.extend([p for p in sorted(dp.rglob("*.json")) if _is_event_json(p)])
    return list(dict.fromkeys(out))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build player-level goalkeeper saves from StatsBomb raw event JSON files")
    ap.add_argument("--event-json", action="append", default=[], help="One StatsBomb event JSON. Can be repeated.")
    ap.add_argument("--event-json-dir", action="append", default=[], help="Directory with StatsBomb event JSON files. Can be repeated.")
    ap.add_argument("--out", default="data/processed/goalkeeper_match_stats.csv")
    args = ap.parse_args(argv)

    files = _discover(args.event_json, args.event_json_dir)
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = parse_statsbomb_goalkeeper_match_json(files)
    df.to_csv(out_path, index=False)
    summary = {
        "version": "v0.38_statsbomb_goalkeeper_match_stats",
        "source": "statsbomb_raw_events",
        "input_files": [str(p) for p in files],
        "rows": int(len(df)),
        "matches": int(df["match_id"].nunique()) if not df.empty else 0,
        "goalkeepers": int(df["goalkeeper"].nunique()) if not df.empty else 0,
        "saves_rows": int(df["saves"].notna().sum()) if "saves" in df.columns else 0,
        "zero_save_rows": int((df["saves"] == 0).sum()) if "saves" in df.columns else 0,
        "output": str(out_path),
        "hard_rule": "player goalkeeper saves come from raw goalkeeper save events/provider player stats, not SOT-goals proxy",
    }
    (out_path.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("MUNDIALYTICS STATSBOMB GOALKEEPER STATS")
    print(f"Input files: {len(files)}")
    print(f"Rows: {len(df)} | matches: {summary['matches']} | goalkeepers: {summary['goalkeepers']}")
    print(f"Output: {out_path}")
    print(f"Saves rows: {summary['saves_rows']} | Zero-save rows: {summary['zero_save_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
