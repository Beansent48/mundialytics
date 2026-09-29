"""How close is text xG (ESPN commentary) to Understat's xG, match by match?

Shots are parsed from data/external/advanced/espn/commentary/<season>.jsonl, the model is
cross-fitted (5 folds by match, so no match is scored by a model that saw it), summed per
team-match and joined to data/processed/understat_team_match_xg.csv on date (+-1 day) and
the canonical home team. Reports shot coverage against ESPN's own totals, the share of
shots with no recognised zone, and correlation / bias / MAE against Understat.

    python scripts/espn_xg/validate_text_xg.py 2025-2026
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from log_lineup_pass import _canon  # noqa: E402
from mundialytics.enrichment.text_xg import TextXG, shots_from_commentary  # noqa: E402
from mundialytics.statistical_core.schemas import canonical_name  # noqa: E402

COMM = ROOT / "data/external/advanced/espn/commentary"


def load_shots(seasons) -> tuple[pd.DataFrame, pd.DataFrame]:
    canon = _canon()
    shots, games = [], []
    for season in seasons:
        for line in (COMM / f"{season}.jsonl").read_text(encoding="utf-8").splitlines():
            g = json.loads(line)
            s = shots_from_commentary([c["text"] for c in g["commentary"]], g["home"], g["away"])
            box = {k: int(v.get("totalShots") or 0) for k, v in g["box"].items()}
            games.append({"event_id": g["event_id"], "season": season, "date": g["date"],
                          "competition": g["competition"], "home": canon(g["home"]), "away": canon(g["away"]),
                          "home_raw": g["home"], "away_raw": g["away"],
                          "home_goals": g["home_goals"], "away_goals": g["away_goals"],
                          "box_shots": sum(box.values()), "parsed": len(s)})
            if len(s):
                s["event_id"] = g["event_id"]
                shots.append(s)
    return pd.concat(shots, ignore_index=True), pd.DataFrame(games)


def cross_fit(shots: pd.DataFrame, k: int = 5) -> np.ndarray:
    ev = shots["event_id"].unique()
    rng = np.random.default_rng(0)
    fold = dict(zip(ev, rng.integers(0, k, len(ev))))
    f = shots["event_id"].map(fold).values
    xg = np.zeros(len(shots))
    for i in range(k):
        m = TextXG().fit(shots[f != i])
        xg[f == i] = m.predict(shots[f == i])
    return xg


def main() -> int:
    seasons = sys.argv[1:] or ["2025-2026"]
    shots, games = load_shots(seasons)
    print(f"matches {len(games)}, attempts parsed {len(shots)}; parsed/box shots "
          f"{games.parsed.sum() / games.box_shots.sum():.3f}; side unresolved {(shots.side == '?').mean():.2%}; "
          f"zone unknown {(shots.zone == 'unknown').mean():.2%}")
    shots["xg"] = cross_fit(shots)
    print(f"goals {shots.goal.sum()} vs text xG {shots.xg.sum():.0f} (cross-fitted)")
    tm = (shots[shots.side != "?"].pivot_table(index="event_id", columns="side", values="xg", aggfunc="sum")
          .reindex(columns=["home", "away"]).fillna(0).rename(columns={"home": "txg_h", "away": "txg_a"}))
    g = games.set_index("event_id").join(tm).fillna({"txg_h": 0, "txg_a": 0})
    g = g.rename(columns={"home_goals": "hg", "away_goals": "ag", "home": "home_c"}).reset_index()

    us = pd.read_csv(ROOT / "data/processed/understat_team_match_xg.csv",
                     usecols=["date", "home_team_fd", "away_team_fd", "home_xg", "away_xg"])
    us["home_c"] = us.home_team_fd.map(canonical_name)
    us["date"] = pd.to_datetime(us.date)
    g["date"] = pd.to_datetime(g.date)
    out = []
    for off in (0, -1, 1):
        x = g.assign(date=g.date + pd.Timedelta(days=off)).merge(us, on=["date", "home_c"], how="inner")
        out.append(x)
    j = pd.concat(out).drop_duplicates("event_id")
    print(f"joined to Understat: {len(j)} of {len(g)}")
    a = np.r_[j.txg_h, j.txg_a]
    b = np.r_[j.home_xg, j.away_xg]
    print(f"team-match xG: corr {np.corrcoef(a, b)[0, 1]:.3f}  mean text {a.mean():.3f} vs Understat {b.mean():.3f}  "
          f"MAE {np.abs(a - b).mean():.3f}")
    gd = np.r_[j.hg, j.ag]
    print(f"corr with goals: text {np.corrcoef(a, gd)[0, 1]:.3f}  Understat {np.corrcoef(b, gd)[0, 1]:.3f}")
    for comp, x in j.groupby("competition"):
        aa, bb = np.r_[x.txg_h, x.txg_a], np.r_[x.home_xg, x.away_xg]
        print(f"   {comp:15s} n={len(x):4d} corr {np.corrcoef(aa, bb)[0, 1]:.3f}  bias {aa.mean() - bb.mean():+.3f}")
    miss = games[~games.home.isin(set(us.home_c))].home_raw.unique()
    if len(miss):
        print("ESPN names with no Understat match:", sorted(miss)[:20])
    return 0


if __name__ == "__main__":
    sys.exit(main())
