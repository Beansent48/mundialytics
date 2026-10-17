"""Production-faithful walk-forward that dumps the FINAL served lambdas.

Why (2026-10-17): the live 2026/27 log under-predicts totals -- Goles 2.5 says
50.2% over, 62.6% happen (n=187, z=3.44), and the season itself is running at
3.09 goals per match against 2.77-2.89 in each of the five seasons before it.
The deployed chain has no league-season level term for goals: the AD model's
time decay and the xG-rate form carry a league-wide shift only damped.

This is `scripts/squad_value/wf_refit.py` with two changes:
  * `squad_value_asof=cutoff`, so the squad-value shift (deployed since v0.55.0,
    after that harness was written) cannot read the future on a historical fold;
  * it dumps the lambdas AS SERVED -- through `predict_match_or_proxy`, after the
    squad-value tilt and the debutant stand-in -- instead of the raw components.

The dump is what makes the level experiment cheap: a common multiplier m on both
lambdas commutes with both tilts (their s depends on log(lh/la), which m leaves
alone), so `score_level.py` can grid over shrinkage without refitting anything.

The engine is untouched: nothing here is wired into production.

usage: python scripts/goals_level/wf_lambdas.py SEASON OUT.csv [--src foundation]

The default source is the xG-available subset the squad-value harness used, so
the baseline stays comparable with the recorded 0.1995/0.2006. `--src foundation`
reads what production fits on (`load_clubs_data`), which is the same data plus
the rows whose xG is missing -- the only way to reach 2026/27, whose xG comes
from the ESPN commentary connector and never landed in the understat-named file.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mundialytics.serving.debutants import predict_match_or_proxy  # noqa: E402
from mundialytics.statistical_core.prediction_engine import (  # noqa: E402
    DEPLOYED_CLUB_ENGINE_KWARGS, PredictionEngine)
from mundialytics.statistical_core.schemas import canonical_name  # noqa: E402

SRC = ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv"


def main() -> None:
    season, out = sys.argv[1], sys.argv[2]
    foundation = "--src" in sys.argv and sys.argv[sys.argv.index("--src") + 1] == "foundation"
    if foundation:
        from mundialytics.statistical_core.engine_utils import load_clubs_data
        df = load_clubs_data()
        need = ["home_goals", "away_goals", "date"]
    else:
        df = pd.read_csv(SRC, low_memory=False)
        df = df[df["xg_available"] == True].copy()  # noqa: E712
        need = ["home_goals", "away_goals", "home_xg", "away_xg", "date"]
    for c in ("home_goals", "away_goals", "home_xg", "away_xg"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=need)
    df["home_team"] = df["home_team"].map(canonical_name)
    df["away_team"] = df["away_team"].map(canonical_name)
    df = df.sort_values("date")

    test = df[df["season"] == season].sort_values("date")
    if test.empty:
        raise SystemExit(f"no rows for {season}")
    rows, t0 = [], time.time()
    for wk, blk in test.groupby(test["date"].dt.to_period("W-SUN"), sort=True):
        cutoff = blk["date"].min()
        train = df[df["date"] < cutoff]
        eng = PredictionEngine(**DEPLOYED_CLUB_ENGINE_KWARGS).fit(
            train, squad_value_asof=cutoff)
        known = set(train["home_team"]) | set(train["away_team"])
        for r in blk.itertuples(index=False):
            comp = r.competition
            current = set(test.loc[test.competition == comp, "home_team"]) | \
                      set(test.loc[test.competition == comp, "away_team"])
            try:
                p, stand = predict_match_or_proxy(eng, train, r.home_team, r.away_team,
                                                  comp, season, current, known=known)
            except LookupError:
                continue
            rows.append({"season": season, "comp": comp, "date": r.date,
                         "home": r.home_team, "away": r.away_team,
                         "hg": int(r.home_goals), "ag": int(r.away_goals),
                         "lh": float(p.lambda_home), "la": float(p.lambda_away),
                         "proxy": bool(stand)})
            if eng.xg_rate_model_ is not None and pd.notna(r.home_xg) and pd.notna(r.away_xg):
                eng.xg_rate_model_.update_form(r.home_team, r.away_team,
                                               r.home_xg, r.away_xg)
        print(f"{season} {wk} n={len(rows)} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("DONE", season, len(rows), flush=True)


if __name__ == "__main__":
    main()
