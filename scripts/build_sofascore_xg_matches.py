from __future__ import annotations

"""Live match-level team xG for the CURRENT season from Sofascore.

Why this exists: Understat (the historical xG source, see [[project_xg_pipeline]])
froze at 2026-05-24, leaving 2026/27 with 0% xG. FBref direct is 403; FBref via
soccerdata bypasses the 403 but is served a degraded page WITHOUT the xG columns
(see [[project_soccerdata_env_change]]). Sofascore's JSON API, by contrast, is
reachable via the tls-client soccerdata ships, and its per-match `shotmap`
endpoint carries per-shot xG (Opta-grade). Reconnecting fresh xG form is worth
~0.0039 RPS (5/5 folds, scripts/validate_walkforward.py).

Pipeline (mirrors scripts/build_understat_xg_matches.py so the output drops into
the same schema the foundation xG-augment already consumes):

  soccerdata.Sofascore.read_schedule  -> game_ids + team names + score (played?)
  api.sofascore.com/.../{id}/shotmap  -> per-shot xg / situation / shotType
  aggregate per (game, team)          -> home/away xg, npxg, xg_op, xg_sp, shots
  to_foundation_name                  -> football-data names, for a registry-free join

Outputs the same two files as the Understat builder but under sofascore_* names, so
update_season's augment_foundation_with_xg can concat Understat (history) +
Sofascore (current) and merge on (date, home_team_fd, away_team_fd).

Set-piece / open-play split follows the Understat convention; only home_xg/away_xg
feed the deployed xG-rate model (op/sp are gated off), so the split is best-effort.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.enrichment.xg import CANONICAL_XG_COLUMNS  # noqa: E402
from mundialytics.enrichment.sofascore_team_aliases import to_foundation_name  # noqa: E402

LEAGUES = ["ENG-Premier League", "ESP-La Liga", "ITA-Serie A", "GER-Bundesliga", "FRA-Ligue 1"]
DEFAULT_OUT_CANONICAL = "data/external/xg/sofascore/sofascore_xg_matches.csv"
DEFAULT_OUT_TEAM_MATCH = "data/processed/sofascore_team_match_xg.csv"
SHOTMAP_URL = "https://api.sofascore.com/api/v1/event/{gid}/shotmap"

# Sofascore `situation` buckets. Penalties are excluded from non-penalty xG; the
# open-play vs set-piece split mirrors the Understat convention.
PENALTY_SITUATIONS = {"penalty"}
OPEN_PLAY_SITUATIONS = {"regular", "assisted", "fast-break"}
SET_PIECE_SITUATIONS = {"corner", "free-kick", "set-piece", "throw-in-set-piece"}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _fetch_shotmap(client, gid: int, retries: int = 3, pause: float = 0.4) -> list[dict] | None:
    """Return the shotmap list for a game id, or None if unavailable."""
    for attempt in range(retries):
        try:
            r = client.get(SHOTMAP_URL.format(gid=gid), timeout=20)
            if r.status_code == 200:
                return r.json().get("shotmap", []) or []
            if r.status_code == 404:
                return None  # no shotmap for this match (not played / no data)
        except Exception:
            pass
        time.sleep(pause * (attempt + 1))
    return None


def _aggregate(shots: list[dict]) -> dict[str, dict[str, float]] | None:
    """Aggregate a shotmap into per-side {home,away} xg/npxg/op/sp/shots/goals."""
    if not shots:
        return None
    agg = {side: {"xg": 0.0, "npxg": 0.0, "xg_op": 0.0, "xg_sp": 0.0, "shots": 0, "goals": 0}
           for side in ("home", "away")}
    for s in shots:
        side = "home" if s.get("isHome") else "away"
        xg = float(s.get("xg") or 0.0)
        situ = str(s.get("situation") or "").strip().lower()
        is_pen = situ in PENALTY_SITUATIONS
        d = agg[side]
        d["xg"] += xg
        d["shots"] += 1
        if not is_pen:
            d["npxg"] += xg
        if situ in OPEN_PLAY_SITUATIONS:
            d["xg_op"] += xg
        elif situ in SET_PIECE_SITUATIONS:
            d["xg_sp"] += xg
        if str(s.get("shotType") or "").lower() == "goal":
            d["goals"] += 1
    return agg


def build(seasons: int, leagues: list[str], pause: float, limit: int | None) -> pd.DataFrame:
    import soccerdata as sd
    import tls_requests

    client = tls_requests.Client()
    rows: list[dict[str, Any]] = []
    n_played = n_ok = 0
    for lg in leagues:
        sched = sd.Sofascore(leagues=lg, seasons=seasons).read_schedule().reset_index()
        played = sched.dropna(subset=["home_score", "away_score"])
        print(f"  {lg}: {len(sched)} fixtures, {len(played)} played", flush=True)
        for _, r in played.iterrows():
            if limit is not None and n_ok >= limit:
                break
            n_played += 1
            gid = int(r["game_id"])
            shots = _fetch_shotmap(client, gid, pause=pause)
            agg = _aggregate(shots)
            time.sleep(pause)
            if agg is None:
                continue
            h, a = agg["home"], agg["away"]
            date10 = pd.to_datetime(r["date"], errors="coerce")
            date10 = date10.strftime("%Y-%m-%d") if pd.notna(date10) else None
            rows.append({
                "provider": "sofascore",
                "provider_match_id": gid,
                "date": date10,
                "competition": lg,
                "season": "2627",
                "home_team": str(r["home_team"]),
                "away_team": str(r["away_team"]),
                "home_team_fd": to_foundation_name(str(r["home_team"])),
                "away_team_fd": to_foundation_name(str(r["away_team"])),
                "home_xg": round(h["xg"], 4), "away_xg": round(a["xg"], 4),
                "home_npxg": round(h["npxg"], 4), "away_npxg": round(a["npxg"], 4),
                "home_xg_op": round(h["xg_op"], 4), "away_xg_op": round(a["xg_op"], 4),
                "home_xg_sp": round(h["xg_sp"], 4), "away_xg_sp": round(a["xg_sp"], 4),
                "home_shots": h["shots"], "away_shots": a["shots"],
                "home_goals": h["goals"], "away_goals": a["goals"],
                "xg_match_confidence": "sofascore_shotmap_aggregation",
            })
            n_ok += 1
    print(f"  shotmaps: {n_ok}/{n_played} matches with xG", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, default=2026, help="Season start year (2026 = 2026/27).")
    parser.add_argument("--leagues", nargs="*", default=LEAGUES)
    parser.add_argument("--pause", type=float, default=0.4, help="Seconds between API calls.")
    parser.add_argument("--limit", type=int, default=None, help="Cap matches (for a quick smoke test).")
    parser.add_argument("--out-canonical", default=DEFAULT_OUT_CANONICAL)
    parser.add_argument("--out-team-match", default=DEFAULT_OUT_TEAM_MATCH)
    args = parser.parse_args()

    team_match = build(args.seasons, args.leagues, args.pause, args.limit)
    if team_match.empty:
        print(json.dumps({"error": "no rows produced"}))
        sys.exit(1)

    canonical = team_match.copy()
    canonical["home_team"] = canonical["home_team_fd"]
    canonical["away_team"] = canonical["away_team_fd"]
    canonical = canonical.reindex(columns=CANONICAL_XG_COLUMNS)

    out_canonical = _resolve(args.out_canonical)
    out_team_match = _resolve(args.out_team_match)
    out_canonical.parent.mkdir(parents=True, exist_ok=True)
    out_team_match.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(out_canonical, index=False)
    team_match.to_csv(out_team_match, index=False)

    summary = {
        "matches_out": int(len(team_match)),
        "leagues": args.leagues,
        "home_xg_mean": round(float(team_match["home_xg"].mean()), 3),
        "away_xg_mean": round(float(team_match["away_xg"].mean()), 3),
        "date_max": str(team_match["date"].max()),
        "outputs": {"canonical": str(out_canonical), "team_match": str(out_team_match)},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
