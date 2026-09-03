from __future__ import annotations

"""Out-of-sample evaluation of the LEAGUE forecasting layer.

For each completed Big 5 league-season: stop at a cutoff matchday, forecast the
rest of the season with the deployed engine, then score those forecasts against
what actually happened.

Leakage control is the whole point here, so it is done twice over:
  - the engine is fitted only on matches played strictly before the cutoff date
    (one fit per season, shared by that season's five leagues);
  - the LeagueState only ever sees results before the cutoff.

Scored with the Brier score on three binary events per team — champion, top 4,
relegated — against the base rate a league table gives you for free (1/20, 4/20,
3/20). Beating that is the bar: it is what you would say knowing nothing except
how many teams get each outcome.

    python scripts/evaluate_league_forecast.py [--cutoff-matchday 19] [--sims 4000]
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.ratings.elo import EloConfig, EloRater  # noqa: E402
from mundialytics.statistical_core.competition.cutoff import (  # noqa: E402
    load_league_state_from_foundation,
)
from mundialytics.statistical_core.competition.engine_provider import fixture_lambdas  # noqa: E402
from mundialytics.statistical_core.competition.resume_simulator import (  # noqa: E402
    simulate_rest_of_season,
)
from mundialytics.statistical_core.competition.standings import compute_standings  # noqa: E402
from mundialytics.statistical_core.engine_utils import load_clubs_data  # noqa: E402
from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402
from mundialytics.statistical_core.schemas import canonical_name  # noqa: E402

LEAGUES = ["Premier League", "LaLiga", "Serie A", "Bundesliga", "Ligue 1"]


def build_engine(train: pd.DataFrame) -> PredictionEngine:
    """The configuration the Streamlit app deploys for clubs."""
    elo = EloRater(EloConfig(season_reset_fraction=0.40))
    elo.fit(train)
    eng = PredictionEngine(blend_weight_gl=0.30, ad_rho=-0.07, sharpen_gamma_1x2=1.3,
                           rescale_lambda_to_goals=True, outcome_rho=-0.17,
                           xg_rate_kwargs={"use_ewma": True})
    eng.fit(train, elo_history=pd.DataFrame(elo.history))
    return eng


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff-matchday", type=int, default=19)
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--seasons", nargs="*",
                    default=["2021-2022", "2022-2023", "2023-2024", "2024-2025"])
    args = ap.parse_args()

    found = load_clubs_data()
    found["date"] = pd.to_datetime(found["date"], errors="coerce")
    rows = []

    for season in args.seasons:
        sl = found[found.season == season]
        if sl.empty:
            print(f"  {season}: not in foundation, skipped")
            continue
        # one leakage-safe fit per season, shared by its five leagues
        season_start = sl["date"].min()
        train = found[found["date"] < season_start]
        print(f"{season}: fitting on {len(train)} matches before {season_start:%Y-%m-%d}")
        engine = build_engine(train)

        for league in LEAGUES:
            full = sl[sl.competition == league]
            if full.empty:
                continue
            try:
                state = load_league_state_from_foundation(
                    league, season, cutoff_matchday=args.cutoff_matchday, foundation=found)
                lam = fixture_lambdas(engine, state)
                fc = simulate_rest_of_season(lam, state, n_sims=args.sims)
            except Exception as e:
                print(f"  {league} {season}: skipped ({type(e).__name__}: {e})")
                continue

            final = compute_standings(full, competition=league).reset_index(drop=True)
            order = [canonical_name(t) for t in final.team]
            n = len(order)
            actual = {t: {"champion": int(i == 0), "top4": int(i < 4),
                          "relegated": int(i >= n - 3)} for i, t in enumerate(order)}

            pdf = fc.team_probs
            for r in pdf.itertuples():
                team = canonical_name(getattr(r, "team", ""))
                if team not in actual:
                    continue
                rows.append({
                    "season": season, "league": league, "team": team, "n_teams": n,
                    "p_champion": float(getattr(r, "p_champion", np.nan)),
                    "p_top4": float(getattr(r, "p_top4", np.nan)),
                    "p_relegated": float(getattr(r, "p_relegation", np.nan)),
                    **{f"a_{k}": v for k, v in actual[team].items()},
                })

    d = pd.DataFrame(rows)
    if d.empty:
        print("nothing scored")
        return

    print(f"\nscored {len(d)} team-seasons across {d.league.nunique()} leagues, "
          f"{d.season.nunique()} seasons (cutoff matchday {args.cutoff_matchday})")
    print(f"\n{'event':12s} {'Brier':>8s} {'base rate':>10s} {'pred mean':>10s} {'actual':>8s}")
    for ev, pcol, acol, base in [("champion", "p_champion", "a_champion", 1 / 20),
                                 ("top 4", "p_top4", "a_top4", 4 / 20),
                                 ("relegated", "p_relegated", "a_relegated", 3 / 20)]:
        sub = d.dropna(subset=[pcol])
        if sub.empty:
            print(f"{ev:12s}   (column not produced by the forecast)")
            continue
        p, a = sub[pcol].to_numpy(float), sub[acol].to_numpy(float)
        brier = float(((p - a) ** 2).mean())
        brier_base = float(((np.full_like(p, base) - a) ** 2).mean())
        print(f"{ev:12s} {brier:8.4f} {brier_base:10.4f} {p.mean():10.3f} {a.mean():8.3f}")


if __name__ == "__main__":
    main()
