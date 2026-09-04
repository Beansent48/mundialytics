from __future__ import annotations

"""Does shrinking per-competition parameters fix the national away-lambda bias?

AttackDefenseModel fits mu and home advantage independently per competition, so
a small one estimates both from very little. Held out on internationals since
2023 that shows up as: World Cup qualification (3,422 training matches) accurate
to -0.13/-0.01 goals, while the AFC Asian Cup (~115) over-predicts both sides by
~0.9. This sweeps the credibility constant k in w = n / (n + k) and reports what
each value does out of sample.

The shrinkage is applied after the per-competition fits, so sweeping it
post-hoc on one fitted engine is identical to refitting with each k — and much
faster. k = 0 is the deployed behaviour.

    python scripts/experiment_competition_shrinkage.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.ratings.elo import EloConfig, EloRater  # noqa: E402
from mundialytics.statistical_core.engine_utils import load_international_data  # noqa: E402
from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402

CUTOFF = "2023-01-01"
KS = [0, 100, 250, 500, 1000, 2000]


def rps3(y: np.ndarray, P: np.ndarray) -> float:
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    return float(((np.cumsum(P, 1) - np.cumsum(Y, 1)) ** 2)[:, :2].sum(1).mean() / 2)


def main() -> None:
    d = load_international_data(min_year=2010).sort_values("date")
    train, test = d[d.date < CUTOFF], d[d.date >= CUTOFF]

    elo = EloRater(EloConfig(season_reset_fraction=0.35, k_base=28.0))
    elo.fit(train)
    eng = PredictionEngine(blend_weight_gl=0.45, ad_rho=-0.06)
    eng.fit(train, elo_history=pd.DataFrame(elo.history))

    ad = eng.ad_model_
    per_league = eng.ad_model_.fit_result_["per_league"]
    base_mu = ad.league_effect_.copy()
    base_ha = ad.league_home_adv_.copy()
    print(f"train {len(train)} | test {len(test)} | competitions {len(per_league)}")
    for comp, info in sorted(per_league.items(), key=lambda kv: -kv[1]["n_matches"]):
        print(f"  {comp:32s} {info['n_matches']:5d} train matches")

    print(f"\n{'k':>6} {'RPS':>8} {'lam home':>10} {'lam away':>10} "
          f"{'away bias':>10} {'AsianCup bias':>14}")
    actual_h, actual_a = test.home_goals.mean(), test.away_goals.mean()
    for k in KS:
        ad.league_effect_ = base_mu.copy()
        ad.league_home_adv_ = base_ha.copy()
        if k > 0:
            ad.competition_shrinkage_k = float(k)
            ad._shrink_competition_params(per_league)

        rows = []
        for r in test.itertuples():
            try:
                p = eng.predict_match(r.home_team, r.away_team,
                                      competition=str(r.competition),
                                      neutral=bool(getattr(r, "neutral", 0) or 0))
            except Exception:
                continue
            rows.append((p.p_home_win, p.p_draw, p.p_away_win, p.lambda_home,
                         p.lambda_away, r.home_goals, r.away_goals, str(r.competition)))
        a = pd.DataFrame(rows, columns=["ph", "pd", "pa", "lh", "la", "hg", "ag", "comp"])
        y = np.where(a.hg > a.ag, 0, np.where(a.hg == a.ag, 1, 2))
        P = a[["ph", "pd", "pa"]].to_numpy(float)
        ac = a[a.comp == "AFC Asian Cup"]
        ac_bias = (ac.lh.mean() - ac.hg.mean()) if len(ac) else float("nan")
        print(f"{k:>6} {rps3(y, P):8.4f} {a.lh.mean():10.2f} {a.la.mean():10.2f} "
              f"{a.la.mean() - a.ag.mean():+10.2f} {ac_bias:+14.2f}")

    print(f"\nactual goals: {actual_h:.2f} home / {actual_a:.2f} away")
    print("k = 0 is the deployed behaviour; away bias and Asian Cup bias should "
          "shrink toward 0 without RPS getting worse.")

    ad.league_effect_, ad.league_home_adv_ = base_mu, base_ha
    ad.competition_shrinkage_k = 0.0

    # One held-out window is suggestive, not sufficient: repeat the k=0 vs k=500
    # comparison across several cutoffs before anyone considers deploying it.
    print("\n=== temporal folds, k=0 vs k=500 ===")
    print(f"{'cutoff':>12} {'n':>6} {'RPS k=0':>9} {'RPS k=500':>10} "
          f"{'away bias 0':>12} {'away bias 500':>14}")
    for cut in ("2019-01-01", "2020-06-01", "2021-06-01", "2022-01-01", "2023-01-01"):
        tr, teF = d[d.date < cut], d[(d.date >= cut) & (d.date < pd.Timestamp(cut) + pd.DateOffset(years=2))]
        if len(teF) < 200:
            continue
        eloF = EloRater(EloConfig(season_reset_fraction=0.35, k_base=28.0)); eloF.fit(tr)
        engF = PredictionEngine(blend_weight_gl=0.45, ad_rho=-0.06)
        engF.fit(tr, elo_history=pd.DataFrame(eloF.history))
        adF = engF.ad_model_
        plF = adF.fit_result_["per_league"]
        b_mu, b_ha = adF.league_effect_.copy(), adF.league_home_adv_.copy()
        out = {}
        for k in (0, 500):
            adF.league_effect_, adF.league_home_adv_ = b_mu.copy(), b_ha.copy()
            if k:
                adF.competition_shrinkage_k = float(k)
                adF._shrink_competition_params(plF)
            rr = []
            for r in teF.itertuples():
                try:
                    pm = engF.predict_match(r.home_team, r.away_team,
                                            competition=str(r.competition),
                                            neutral=bool(getattr(r, "neutral", 0) or 0))
                except Exception:
                    continue
                rr.append((pm.p_home_win, pm.p_draw, pm.p_away_win, pm.lambda_away,
                           r.home_goals, r.away_goals))
            f = pd.DataFrame(rr, columns=["ph", "pd", "pa", "la", "hg", "ag"])
            yy = np.where(f.hg > f.ag, 0, np.where(f.hg == f.ag, 1, 2))
            out[k] = (rps3(yy, f[["ph", "pd", "pa"]].to_numpy(float)),
                      f.la.mean() - f.ag.mean(), len(f))
        print(f"{cut:>12} {out[0][2]:6d} {out[0][0]:9.4f} {out[500][0]:10.4f} "
              f"{out[0][1]:+12.2f} {out[500][1]:+14.2f}")


if __name__ == "__main__":
    main()
