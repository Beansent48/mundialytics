from __future__ import annotations

"""Should every goal market be read from one distribution that agrees with the 1X2?

THE INCONSISTENCY. The deployed 1X2 is sharpened (gamma 1.3, validated on the
trio alone) while the scoreline matrix and every market read from it stay raw.
With lambdas 2.4 / 0.7 the page says 84% home win and its own matrix sums to
75%. The HT/FT paths imply a third full-time 1X2, 3.8 points off on average.

THE CANDIDATE. Reweight the matrix so each outcome region carries the sharpened
1X2 mass (every home-win cell times p_sharp/p_raw, and so on), and reweight the
HT/FT paths so their full-time margins match it too. The 1X2 is unchanged by
construction; the question is only what that does to everything else.

RESULT (2026-09-23, 10,403 matches): no market worse overall; exact score
-0.0032 (5/6 seasons), HT/FT -0.0049 (6/6), O/U 3.5 5/6, BTTS 4/6, O/U 2.5
-0.0002 (3/6), O/U 1.5 unchanged. Deployed as coherent_markets=True.

THE TEST. The deployed walk-forward (six seasons, each predicted by an engine
fitted only on earlier ones) now stores both lambdas, so each market can be
rebuilt both ways and scored on the same matches. Deploy only if it does not
make the goal markets worse out of sample (user rule, see memory:
feedback_protect_deployed_baseline).

    .venv/Scripts/python.exe scripts/experiment_market_coherence.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.props.half_time import HalfTimeModel  # noqa: E402
from mundialytics.statistical_core.prediction_engine import (  # noqa: E402
    DEPLOYED_CLUB_ENGINE_KWARGS as K, markets_from_lambdas)

WF = ROOT / "data/processed/enriched/understat_xg/walkforward_preds_deployed.csv"
FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
EPS = 1e-9


def outcome_masks(n: int):
    i, j = np.indices((n, n))
    return i > j, i == j, i < j


def coherent_matrix(m: np.ndarray, trio_sharp) -> np.ndarray:
    """Scale each outcome region of the matrix to the sharpened 1X2 mass."""
    out = m.copy()
    for mask, target in zip(outcome_masks(m.shape[0]), trio_sharp):
        mass = m[mask].sum()
        if mass > 0:
            out[mask] *= target / mass
    return out / out.sum()


def goal_markets(m: np.ndarray) -> dict:
    n = m.shape[0]
    i, j = np.indices((n, n))
    tot = i + j
    return {
        "o15": m[tot >= 2].sum(), "o25": m[tot >= 3].sum(), "o35": m[tot >= 4].sum(),
        "btts": m[(i > 0) & (j > 0)].sum(),
    }


def coherent_htft(paths: dict, trio_sharp) -> dict:
    ft = {b: sum(v for k, v in paths.items() if k.endswith("/" + b)) for b in "1X2"}
    target = dict(zip("1X2", trio_sharp))
    out = {k: v * target[k[-1]] / ft[k[-1]] for k, v in paths.items()}
    s = sum(out.values())
    return {k: v / s for k, v in out.items()}


def bll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> None:
    wf = pd.read_csv(WF)
    if not {"lh", "la"} <= set(wf.columns):
        raise SystemExit("walk-forward sin lambdas: regenera con generate_deployed_walkforward.py")
    ht = pd.read_csv(FOUND, usecols=["match_id", "home_goals_ht", "away_goals_ht"],
                     low_memory=False)
    wf = wf.merge(ht, on="match_id", how="left")
    htm = HalfTimeModel()

    recs = []
    for r in wf.itertuples(index=False):
        # the configuration as it was before this experiment: raw matrix
        probs, dist = markets_from_lambdas(r.lh, r.la, rho=K["outcome_rho"],
                                           temper=K["goal_temper"],
                                           sharpen_gamma=K["sharpen_gamma_1x2"],
                                           coherent=False)
        m = dist.matrix.to_numpy()
        trio = (r.ph, r.pd, r.pa)
        mc = coherent_matrix(m, trio)
        a, b = goal_markets(m), goal_markets(mc)
        hg, ag = int(r.hg), int(r.ag)
        n = m.shape[0]
        cell = (min(hg, n - 1), min(ag, n - 1))
        rec = {"season": r.season, "hg": hg, "ag": ag,
               "gap_raw": abs(np.tril(m, -1).sum() - r.ph),
               "gap_coh": abs(np.tril(mc, -1).sum() - r.ph),
               "cs_raw": -np.log(max(m[cell], EPS)), "cs_coh": -np.log(max(mc[cell], EPS))}
        for k in a:
            rec[f"{k}_raw"], rec[f"{k}_coh"] = a[k], b[k]
        if pd.notna(r.home_goals_ht):
            paths = htm.predict_ht_ft(r.lh, r.la)
            pc = coherent_htft(paths, trio)
            h1, a1 = int(r.home_goals_ht), int(r.away_goals_ht)
            key = (("1" if h1 > a1 else "X" if h1 == a1 else "2") + "/"
                   + ("1" if hg > ag else "X" if hg == ag else "2"))
            rec["htft_raw"] = -np.log(max(paths[key], EPS))
            rec["htft_coh"] = -np.log(max(pc[key], EPS))
            ft_raw = {x: sum(v for k, v in paths.items() if k.endswith("/" + x)) for x in "1X2"}
            rec["htft_gap"] = abs(ft_raw["1"] - r.ph)
        recs.append(rec)
    d = pd.DataFrame(recs)
    tot = d.hg + d.ag
    ys = {"o15": (tot >= 2).astype(float), "o25": (tot >= 3).astype(float),
          "o35": (tot >= 4).astype(float), "btts": ((d.hg > 0) & (d.ag > 0)).astype(float)}

    print(f"{len(d):,} partidos, {d.season.nunique()} temporadas walk-forward\n")
    print(f"desacuerdo 1X2 vs matriz (P local): medio {d.gap_raw.mean():.4f}, "
          f"max {d.gap_raw.max():.4f}  ->  coherente {d.gap_coh.max():.2e}")
    if "htft_gap" in d:
        print(f"desacuerdo 1X2 vs HT/FT (P local): medio {d.htft_gap.mean():.4f}, "
              f"max {d.htft_gap.max():.4f}\n")

    print(f"{'mercado':10s} {'desplegado':>11s} {'coherente':>10s} {'delta':>9s}  temporadas mejor")
    verdict = {}
    for k in ["o15", "o25", "o35", "btts", "cs", "htft"]:
        if k in ys:
            f = lambda g, v: bll(ys[k].loc[g.index].to_numpy(), g[f"{k}_{v}"].to_numpy())  # noqa: E731
        else:
            col_ok = d[f"{k}_raw"].notna()
            f = lambda g, v: float(g.loc[col_ok.loc[g.index], f"{k}_{v}"].mean())  # noqa: E731
        raw, coh = f(d, "raw"), f(d, "coh")
        wins = sum(f(g, "coh") < f(g, "raw") for _, g in d.groupby("season"))
        verdict[k] = (coh - raw, wins)
        print(f"{k:10s} {raw:11.5f} {coh:10.5f} {coh - raw:+9.5f}  {wins}/{d.season.nunique()}")
    print("\n(log loss; menor es mejor. cs = marcador exacto, htft = descanso/final 9 vias)")


if __name__ == "__main__":
    main()
