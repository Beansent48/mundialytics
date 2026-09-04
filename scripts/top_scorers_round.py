from __future__ import annotations

"""Most likely scorers for the upcoming round, ranked.

WHY RANKED AND NOT YES/NO. The scorer market cannot be judged the way 1X2 is.
Measured over 6,327 settled player-matches, the model says "yes" (p >= 0.5)
**zero times** — no individual player is ever more likely than not to score — so
a yes/no track record just reproduces the base rate: right ~92% of the time by
answering "no", which is 8.6% skill and 91.4% arithmetic.

What the model IS good at is ranking. On the same sample:

    top-1 pick scores 28.8% of the time   (x3.36 the 8.6% base rate)
    top-3            20.2%                (x2.36)
    and in 79.5% of team-matches where anyone scored, a real scorer was in
    our top 3 (49.3% for the top 1)

Calibration holds too: it claims 34.3% for its strongest picks and 35.9% of them
score (ECE 0.0147 across bands). So a ranked shortlist is the honest product,
and "of our top-3, how many scored" is the honest way to score it.

Probabilities ARE tied to the team's expected goals: the model scales each
player by `af = (lambda / league_mean) / (team_xg_base / global_xg)`, i.e. how
this fixture compares to that team's own attacking baseline, clipped to
[0.4, 2.5]. The lambda passed in is the deployed engine's own expected goals for
that fixture, so a side expected to score more lifts all of its players.

Run: .venv/Scripts/python.exe scripts/top_scorers_round.py --top 3 --days 8
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

PRED_LOG = ROOT / "data/processed/logs/predictions_log.csv"


def top_scorers(pp, team: str, lam: float, n: int = 3,
                pool: int = 11) -> pd.DataFrame:
    """The n most likely scorers among the likely XI, given the side's lambda.

    `pool` first restricts to the probable starters by expected minutes — the
    model returns the whole squad, and `exp_min` is minutes-when-featuring
    rather than minutes spread over it, so 22 of a 26-man squad clear 45.
    """
    out = pp.team_players_for_lambda(team, float(lam))
    if out is None or out.empty:
        return pd.DataFrame()
    xi = out.nlargest(pool, "exp_min")
    return xi.nlargest(n, "p_anytime_scorer")[
        ["player", "exp_min", "p_anytime_scorer", "p_assist", "p_shots_over_1_5"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=3, help="jugadores por equipo")
    ap.add_argument("--days", type=int, default=8, help="ventana hacia delante")
    ap.add_argument("--from-log", action="store_true",
                    help="usar las probabilidades YA registradas en vez de recalcular")
    args = ap.parse_args()

    if args.from_log:
        if not PRED_LOG.exists():
            print("No hay log de predicciones.")
            return
        log = pd.read_csv(PRED_LOG)
        g = log[log.mercado == "jug_goleador"]
        if g.empty:
            print("El log no tiene mercado de goleador todavia.")
            return
        print(f"Top {args.top} goleadores por partido — desde el log registrado\n")
        for (fecha, partido), gg in g.groupby(["fecha", "partido"]):
            top = gg.nlargest(args.top, "prob")
            print(f"  {fecha}  {partido}")
            for r in top.itertuples(index=False):
                print(f"      {r.ambito:26s} {r.prob:6.1%}")
        return

    # recompute live from the engine's lambdas for the upcoming fixtures
    import joblib
    from log_upcoming_round import fetch_fixtures
    from mundialytics.ratings.elo import EloConfig, EloRater
    from mundialytics.statistical_core.prediction_engine import PredictionEngine

    files = sorted(glob.glob(str(ROOT / "data/processed/cache/props_models_*.joblib")))
    if not files:
        print("Sin cache de props. Arranca la app una vez.")
        return
    _, pp = joblib.load(files[-1])
    if pp is None:
        print("El cache no trae modelo de jugador.")
        return

    now = pd.Timestamp.now()
    year = now.year if now.month >= 7 else now.year - 1
    fx = fetch_fixtures(year)
    if fx.empty:
        print("Sin fixtures.")
        return
    up = fx[(~fx.played) & (fx.date > now)
            & (fx.date <= now + pd.Timedelta(days=args.days))].sort_values("date")
    print(f"{len(up)} partidos en los proximos {args.days} dias\n")

    df = pd.read_csv(ROOT / "data/processed/foundation_big5_multi_season.csv", low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elo = EloRater(EloConfig(season_reset_fraction=0.40))
    elo.fit(df)
    eng = PredictionEngine(blend_weight_gl=0.30, ad_rho=-0.07, sharpen_gamma_1x2=1.3,
                           rescale_lambda_to_goals=True, outcome_rho=-0.17,
                           xg_rate_kwargs={"use_ewma": True}).fit(
        df, elo_history=pd.DataFrame(elo.history))

    for r in up.itertuples(index=False):
        try:
            p = eng.predict_match(r.home, r.away, competition=r.competition, neutral=False)
        except Exception:
            continue
        print(f"  {r.date:%Y-%m-%d}  {r.home} vs {r.away}   "
              f"(goles esperados {p.lambda_home:.2f} - {p.lambda_away:.2f})")
        for team, lam in ((r.home, p.lambda_home), (r.away, p.lambda_away)):
            t = top_scorers(pp, team, lam, args.top)
            if t.empty:
                print(f"      {team:22s} (sin datos de jugador)")
                continue
            for x in t.itertuples(index=False):
                print(f"      {team:14s} {x.player:24s} gol {x.p_anytime_scorer:5.1%}"
                      f"   asist {x.p_assist:5.1%}   +1.5 tiros {x.p_shots_over_1_5:5.1%}")
        print()


if __name__ == "__main__":
    main()
