"""Text xG per match, joined to the foundation's match ids, for the walk-forward test.

Fits TextXG on the shots of --fit seasons and scores the --apply season (cross-fitted in 5
folds by match when the two overlap), rescales it to Understat's level per league (the
history the engine is trained on is Understat's, which runs hotter than goals), and joins
each ESPN match to canonical_matches_with_xg.csv on date (+-1 day) and the most similar
pair of team names.

    python scripts/espn_xg/build_text_xg_matches.py --fit 2025-2026 --apply 2025-2026
"""
import argparse
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts/espn_xg")]

from mundialytics.enrichment.text_xg import TextXG  # noqa: E402
from validate_text_xg import cross_fit, load_shots  # noqa: E402

WORK = ROOT / "data/external/transfermarkt/work"
FOUND = ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv"


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def join(sub: pd.DataFrame, games: pd.DataFrame, found: pd.DataFrame, season: str) -> pd.DataFrame:
    """Per-match text xG of one season joined to the foundation's match ids."""
    tm = (sub.pivot_table(index="event_id", columns="side", values="xg", aggfunc="sum")
             .reindex(columns=["home", "away"]).fillna(0.0))
    g = games[games.season == season].set_index("event_id").join(tm.rename(columns={"home": "txg_h", "away": "txg_a"}))
    g = g.dropna(subset=["txg_h"]).reset_index()
    g["date"] = pd.to_datetime(g.date)
    f = found[found.season == season]
    rows = []
    for r in g.itertuples(index=False):
        c = f[(f.date - r.date).abs() <= pd.Timedelta(days=1)]
        c = c[(c.home_goals == r.home_goals) & (c.away_goals == r.away_goals)]
        if c.empty:
            continue
        best = max((_sim(r.home_raw, h) + _sim(r.away_raw, a), i) for h, a, i in zip(c.home_team, c.away_team, c.index))
        if best[0] < 0.8:
            continue
        x = c.loc[best[1]]
        rows.append({"match_id": x.match_id, "competition": x.competition, "event_id": r.event_id,
                     "txg_h": r.txg_h, "txg_a": r.txg_a, "us_h": x.home_xg, "us_a": x.away_xg})
    out = pd.DataFrame(rows).drop_duplicates("match_id")
    print(f"{season}: joined {len(out)} of {len(g)} ESPN matches ({len(f)} in the foundation season)")
    return out


def level(out: pd.DataFrame) -> pd.Series:
    """Understat's level over text xG, per league (the engine's history is Understat)."""
    ok = out.dropna(subset=["us_h"])
    per = (ok.groupby("competition")[["us_h", "us_a"]].sum().sum(1)
           / ok.groupby("competition")[["txg_h", "txg_a"]].sum().sum(1))
    # a league without Understat that season (Bundesliga 2024/25) takes the overall ratio
    overall = ok[["us_h", "us_a"]].values.sum() / ok[["txg_h", "txg_a"]].values.sum()
    return per.reindex(sorted(out["competition"].unique())).fillna(overall)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", nargs="+", required=True)
    ap.add_argument("--apply", required=True)
    ap.add_argument("--out", default=str(WORK / "text_xg_matches.csv"))
    args = ap.parse_args()

    shots, games = load_shots(sorted(set(args.fit) | {args.apply}))
    ev_season = games.set_index("event_id").season
    shots["season"] = shots.event_id.map(ev_season)
    found = pd.read_csv(FOUND, low_memory=False, usecols=["match_id", "date", "season", "competition", "home_team",
                                                          "away_team", "home_goals", "away_goals", "home_xg", "away_xg"])
    found["date"] = pd.to_datetime(found.date, errors="coerce")
    if args.fit == [args.apply]:
        # cross-fitted inside the season; the level is read off the same season
        sub = shots[shots.season == args.apply].copy()
        sub["xg"] = cross_fit(sub)
        out = join(sub, games, found, args.apply)
        scale = level(out)
    else:
        # honest: model and level both come from the --fit seasons only
        train = shots[shots.season.isin(args.fit) & (shots.season != args.apply)].copy()
        train["xg"] = cross_fit(train)
        scale = level(pd.concat([join(train, games, found, s) for s in args.fit if s != args.apply]))
        m = TextXG().fit(train)
        sub = shots[shots.season == args.apply].copy()
        sub["xg"] = m.predict(sub)
        out = join(sub, games, found, args.apply)
    print("scale to Understat:", scale.round(3).to_dict())
    out["home_xg_text"] = out.txg_h * out.competition.map(scale)
    out["away_xg_text"] = out.txg_a * out.competition.map(scale)
    a, b = np.r_[out.home_xg_text, out.away_xg_text], np.r_[out.us_h, out.us_a]
    m = np.isfinite(b)
    print(f"{args.apply}: corr with Understat {np.corrcoef(a[m], b[m])[0, 1]:.3f}, "
          f"mean text {a[m].mean():.3f} vs Understat {b[m].mean():.3f}")
    out.to_csv(args.out, index=False)
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
