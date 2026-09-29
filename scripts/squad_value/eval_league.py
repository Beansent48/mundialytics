"""Does the squad-value shift keep the season forecasts (title / top 4 / relegation) honest?

Same design as scripts/evaluate_league_forecast.py: one engine per season fitted before the
season starts, squad values as of that date, the rest of the season simulated from a cutoff
matchday. The SAME fitted engine is scored with the shift off and on, with the same seeds.

    python scripts/squad_value/eval_league.py --cutoff-matchday 5 19
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from mundialytics.statistical_core.competition.cutoff import load_league_state_from_foundation  # noqa: E402
from mundialytics.statistical_core.competition.engine_provider import fixture_lambdas  # noqa: E402
from mundialytics.statistical_core.competition.resume_simulator import simulate_rest_of_season  # noqa: E402
from mundialytics.statistical_core.competition.standings import compute_standings  # noqa: E402
from mundialytics.statistical_core.engine_utils import load_clubs_data  # noqa: E402
from mundialytics.statistical_core.prediction_engine import (  # noqa: E402
    DEPLOYED_CLUB_ENGINE_KWARGS, PredictionEngine)
from mundialytics.statistical_core.schemas import canonical_name  # noqa: E402

LEAGUES = ["Premier League", "LaLiga", "Serie A", "Bundesliga", "Ligue 1"]
SHIFT = {"beta": 0.1051, "kappa": -0.2148}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff-matchday", type=int, nargs="+", default=[5, 19])
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--seasons", nargs="*",
                    default=["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"])
    args = ap.parse_args()
    found = load_clubs_data()
    found["date"] = pd.to_datetime(found["date"], errors="coerce")
    rows = []
    for season in args.seasons:
        sl = found[found.season == season]
        start = sl["date"].min()
        train = found[found["date"] < start]
        eng = PredictionEngine(**{**DEPLOYED_CLUB_ENGINE_KWARGS, "squad_value_shift": SHIFT})
        eng.fit(train, squad_value_asof=start)
        print(f"{season}: fitted before {start:%Y-%m-%d}, {len(eng.squad_values_)} squad values", flush=True)
        for league in LEAGUES:
            full = sl[sl.competition == league]
            final = compute_standings(full, competition=league).reset_index(drop=True)
            order = [canonical_name(t) for t in final.team]
            n = len(order)
            actual = {t: (int(i == 0), int(i < 4), int(i >= n - 3)) for i, t in enumerate(order)}
            for md in args.cutoff_matchday:
                state = load_league_state_from_foundation(league, season, cutoff_matchday=md, foundation=found)
                for mode in ("off", "on"):
                    eng.squad_value_shift = SHIFT if mode == "on" else None
                    np.random.seed(12345)
                    fc = simulate_rest_of_season(fixture_lambdas(eng, state), state, n_sims=args.sims)
                    for r in fc.team_probs.itertuples():
                        t = canonical_name(r.team)
                        if t in actual:
                            rows.append({"season": season, "league": league, "md": md, "mode": mode,
                                         "p_ch": r.p_champion, "p_t4": r.p_top4, "p_rel": r.p_relegation,
                                         "a_ch": actual[t][0], "a_t4": actual[t][1], "a_rel": actual[t][2]})
    d = pd.DataFrame(rows)
    for md in args.cutoff_matchday:
        print(f"\n== cutoff matchday {md} ==  (Brier, lower is better)")
        for ev in ("ch", "t4", "rel"):
            b = {m: ((d[(d.md == md) & (d["mode"] == m)]["p_" + ev]
                      - d[(d.md == md) & (d["mode"] == m)]["a_" + ev]) ** 2) for m in ("off", "on")}
            per = pd.DataFrame({m: b[m].groupby(d.loc[b[m].index, "season"]).mean() for m in b})
            print(f"  {ev:4s} off {b['off'].mean():.4f}  on {b['on'].mean():.4f}  "
                  f"seasons better {(per['on'] < per['off']).sum()}/{len(per)}")
    d.to_csv("data/external/transfermarkt/work/eval_league.csv", index=False)


if __name__ == "__main__":
    main()
