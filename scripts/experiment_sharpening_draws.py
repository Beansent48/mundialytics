from __future__ import annotations

"""The published 1X2 under-prices draws by 2.2 points. What does that cost?

THE DEFECT (found by scripts/bivariate_goals, 2026-10-18). The deployed trio
sharpening is `p_i ** 1.3` renormalized -- temperature scaling on the 3-vector.
The draw is almost never the largest of the three, so gamma > 1 shrinks it every
time, structurally: the raw trio predicts a 0.2560 draw rate against 0.2540
observed (z +0.48, calibrated), and the published one predicts 0.2318 (O/E
0.913, z -5.2). Coherent markets then pin each outcome region of the scoreline
matrix to that published trio, so the deficit reaches every derived number.

THE CANDIDATES. Four forms, each fitted to the same objective the deployed
gamma was chosen for (1X2 RPS), so the question is only what each one costs
elsewhere:
  none       gamma = 1, no sharpening at all.
  power      THE DEPLOYED FORM, p_i ** g renormalized.
  pair       sharpen the home/away pair among themselves and leave the draw
             exactly where the model put it.
  anchor     sharpen the deviation from the league base rate instead of the
             probability, so a trio already at the base rate does not move.
  pair_draw  pair, plus a free multiplier on the draw (generalizes both).

RESULT (2026-10-18, 10,403 matches, six walk-forward seasons). TWO FINDINGS,
one structural and one measured.
  1. Under coherent reweighting, exact score and HT/FT CANNOT respond to the
     trio independently: their log loss is exactly the 1X2 log loss plus a
     within-region term no trio reweighting touches (verified below). So the
     only markets that can move are the cross-region ones -- BTTS and the
     over/under lines.
  2. There they do not move. Holding the draw (`pair`) ties the deployed form
     on 1X2 RPS and removes the whole deficit, but BTTS and O/U 1.5, 2.5, 3.5
     all land inside 5e-05 and none clears 5 of 6 in either direction. The
     2.2-point defect is real and nearly free.
  `pair_draw` picks a draw multiplier of exactly 1.000 in all six folds: free
  to rescale draws, the fit declines. Not deployed -- by the score there is
  nothing to win, and the deployed baseline is protected (user rule, memory:
  feedback_protect_deployed_baseline). Recorded so the next person does not
  re-run it. scripts/sharpening_draws/README.md has every number.

    .venv/Scripts/python.exe scripts/experiment_sharpening_draws.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.props.half_time import HalfTimeModel  # noqa: E402
from mundialytics.statistical_core.distributions import scoreline_distribution  # noqa: E402
from mundialytics.statistical_core.prediction_engine import (  # noqa: E402
    DEPLOYED_CLUB_ENGINE_KWARGS as K, markets_from_lambdas)

WF = ROOT / "data/processed/enriched/understat_xg/walkforward_preds_deployed.csv"
FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
OUT = ROOT / "scripts/sharpening_draws"
MAXG = 10
EPS = 1e-12
GAMMA = float(K["sharpen_gamma_1x2"])
LINES = (1.5, 2.5, 3.5)
# the markets the engine actually publishes off the matrix (prediction_engine.py
# L134-139). even/odd is NOT one of them; it is carried only as a diagnostic.
PUBLISHED = ["btts", "o15", "o25", "o35"]
LINE_KEY = {1.5: "o15", 2.5: "o25", 3.5: "o35"}


# -- sharpening forms ---------------------------------------------------------
def f_none(p, _th, _base):
    return p


def f_power(p, th, _base):
    """THE DEPLOYED FORM. p_i ** g renormalized == softmax(g * log p)."""
    q = np.clip(p, EPS, 1.0) ** th[0]
    return q / q.sum(1, keepdims=True)


def f_pair(p, th, _base):
    """Sharpen the side pair; the draw keeps the model's own value."""
    return _pair(p, th[0], p[:, 1])


def f_anchor(p, th, base):
    """Sharpen the deviation from the base rate, not the probability."""
    lb = np.log(np.clip(base, EPS, 1.0))
    q = np.exp(th[0] * (np.log(np.clip(p, EPS, 1.0)) - lb) + lb)
    return q / q.sum(1, keepdims=True)


def f_pair_draw(p, th, _base):
    """pair, with a free multiplier on the draw."""
    return _pair(p, th[0], np.clip(p[:, 1] * th[1], 1e-9, 0.9))


def _pair(p, g, draw):
    h = np.clip(p[:, 0], EPS, 1.0) ** g
    a = np.clip(p[:, 2], EPS, 1.0) ** g
    rest = (1.0 - draw) / (h + a)
    return np.stack([h * rest, draw, a * rest], 1)


FORMS = {"none": f_none, "power": f_power, "pair": f_pair,
         "anchor": f_anchor, "pair_draw": f_pair_draw}
GRIDS = {
    "none": [()],
    "power": [(g,) for g in np.arange(1.00, 1.81, 0.02)],
    "pair": [(g,) for g in np.arange(1.00, 2.61, 0.04)],
    "anchor": [(g,) for g in np.arange(1.00, 2.61, 0.04)],
    "pair_draw": [(g, d) for g in np.arange(1.00, 2.61, 0.08)
                  for d in np.arange(0.90, 1.21, 0.02)],
}


# -- scoring ------------------------------------------------------------------
def rps(pub: np.ndarray, truth: np.ndarray) -> float:
    oh = np.zeros_like(pub)
    oh[np.arange(len(truth)), truth] = 1.0
    return float(np.mean(((pub.cumsum(1) - oh.cumsum(1)) ** 2).sum(1) / 2.0))


def bll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def precompute(wf: pd.DataFrame, htm: HalfTimeModel) -> dict:
    """Per-match tables that do not depend on the trio.

    Coherent reweighting scales each outcome region by target/raw, so every
    market is sum_r target_r * mass(market & r) / mass_r. Cache those masses
    once and any candidate trio is then arithmetic.
    """
    n = len(wf)
    i, j = np.indices((MAXG + 1, MAXG + 1))
    region = np.where(i > j, 0, np.where(i == j, 1, 2))
    tot = i + j
    conds = {"btts": (i > 0) & (j > 0), "even": tot % 2 == 0}
    for line, key in LINE_KEY.items():
        conds[key] = tot > line

    raw = np.zeros((n, 3))
    mass = {k: np.zeros((n, 3)) for k in conds}
    cell = np.zeros(n)
    ht_path = np.full(n, np.nan)
    ht_ft_raw = np.zeros((n, 3))
    clipped = 0

    for t, r in enumerate(wf.itertuples(index=False)):
        m = scoreline_distribution(r.lh, r.la, max_goals=MAXG, normalize=True,
                                   dixon_coles_rho=K["outcome_rho"],
                                   temper=K["goal_temper"]).matrix.to_numpy(float)
        for k in range(3):
            sel = region == k
            raw[t, k] = m[sel].sum()
            for name, cond in conds.items():
                mass[name][t, k] = m[sel & cond].sum()
        hg, ag = int(r.hg), int(r.ag)
        if hg > MAXG or ag > MAXG:
            clipped += 1
            hg, ag = min(hg, MAXG), min(ag, MAXG)
        cell[t] = m[hg, ag]

        if pd.notna(getattr(r, "home_goals_ht", np.nan)):
            paths = htm.predict_ht_ft(r.lh, r.la)
            h1, a1 = int(r.home_goals_ht), int(r.away_goals_ht)
            ht = "1" if h1 > a1 else "X" if h1 == a1 else "2"
            ft = "1" if r.hg > r.ag else "X" if r.hg == r.ag else "2"
            ht_path[t] = paths[ht + "/" + ft]
            for idx, x in enumerate("1X2"):
                ht_ft_raw[t, idx] = sum(
                    v for k2, v in paths.items() if k2.endswith("/" + x))

    obs_region = np.where(wf.hg > wf.ag, 0,
                          np.where(wf.hg == wf.ag, 1, 2)).astype(int)
    tots = (wf.hg + wf.ag).to_numpy()
    y = {"btts": ((wf.hg > 0) & (wf.ag > 0)).to_numpy(float),
         "even": (tots % 2 == 0).astype(float)}
    for line, key in LINE_KEY.items():
        y[key] = (tots > line).astype(float)
    return dict(n=n, raw=raw, mass=mass, cell=cell, obs_region=obs_region, y=y,
                ht_path=ht_path, ht_ft_raw=ht_ft_raw, clipped=clipped,
                season=wf.season.to_numpy())


def score(pub: np.ndarray, m: dict, sel: np.ndarray) -> dict:
    """Every published market for a candidate trio, on the subset `sel`."""
    idx = np.arange(m["n"])
    reg = m["obs_region"]
    ratio = pub / np.clip(m["raw"], 1e-15, None)
    out = {"rps": rps(pub[sel], reg[sel]),
           "ll1x2": float(-np.log(np.clip(pub[idx, reg], EPS, 1.0))[sel].mean())}
    for k, mk in m["mass"].items():
        out[k] = bll(m["y"][k][sel], (mk * ratio).sum(1)[sel])
    # exact score and HT/FT are read off the reweighted matrix / paths
    out["exact"] = float(
        -np.log(np.clip(m["cell"] * ratio[idx, reg], 1e-15, 1.0))[sel].mean())
    hp = m["ht_path"] * pub[idx, reg] / np.clip(m["ht_ft_raw"][idx, reg], 1e-15, None)
    ok = sel & np.isfinite(hp)
    out["htft"] = float(-np.log(np.clip(hp[ok], 1e-15, 1.0)).mean())
    out["draw_pred"] = float(pub[sel, 1].mean())
    out["draw_obs"] = float((reg[sel] == 1).mean())
    return out


def harness_checks(wf: pd.DataFrame, m: dict) -> None:
    """The shortcut must reproduce the engine, or nothing below means anything."""
    pub = wf[["ph", "pd", "pa"]].to_numpy(float)
    worst_trio = worst_mkt = 0.0
    for t, r in enumerate(wf.head(300).itertuples(index=False)):
        # sharpen_contrast_only=False ON PURPOSE: this harness measures the POWER
        # form against alternatives, and `power` is the arm it has to reproduce.
        # Since v0.63.0 that is no longer what the engine serves, so this is a
        # deliberate snapshot, not a read of the deployed config -- which is why
        # it is spelled out instead of left to the default. If the cached trio
        # below stops matching, it is because the cache was regenerated under the
        # deployed (contrast) map; regenerate the arms, do not relax the check.
        probs, _ = markets_from_lambdas(
            r.lh, r.la, rho=K["outcome_rho"], temper=K["goal_temper"],
            sharpen_gamma=GAMMA, sharpen_contrast_only=False,
            coherent=bool(K["coherent_markets"]))
        worst_trio = max(worst_trio, abs(probs["p_home_win"] - r.ph),
                         abs(probs["p_draw"] - r.pd), abs(probs["p_away_win"] - r.pa))
        ratio = pub[t] / m["raw"][t]
        for key, pk in (("btts", probs["p_btts"]), ("o25", probs["p_over_25"]),
                        ("o15", probs["p_over_15"]), ("o35", probs["p_over_35"])):
            worst_mkt = max(worst_mkt, abs((m["mass"][key][t] * ratio).sum() - pk))
    print(f"harness: published trio reproduced to {worst_trio:.2e}; "
          f"region-mass markets match the engine to {worst_mkt:.2e}")

    # the structural identity: exact score == 1X2 + a term no trio can touch
    idx, reg = np.arange(m["n"]), m["obs_region"]
    ll = float(-np.log(np.clip(pub[idx, reg], EPS, 1.0)).mean())
    within = float(-np.log(m["cell"] / m["raw"][idx, reg]).mean())
    ex = float(-np.log(np.clip(m["cell"] * pub[idx, reg] / m["raw"][idx, reg],
                               1e-15, 1.0)).mean())
    print(f"structure: exact {ex:.6f} == 1X2 {ll:.6f} + within-region "
          f"{within:.6f} (diff {abs(ex - (ll + within)):.1e})")

    hp = m["ht_path"] * pub[idx, reg] / m["ht_ft_raw"][idx, reg]
    ok = np.isfinite(hp)
    llh = float(-np.log(np.clip(pub[idx, reg], EPS, 1.0))[ok].mean())
    wh = float(-np.log(m["ht_path"][ok] / m["ht_ft_raw"][idx, reg][ok]).mean())
    tot_h = float(-np.log(hp[ok]).mean())
    print(f"structure: HT/FT {tot_h:.6f} == 1X2 {llh:.6f} + within {wh:.6f} "
          f"(diff {abs(tot_h - (llh + wh)):.1e})")
    print("  -> neither can respond to the trio on its own; only BTTS and the "
          "over/under lines cut across the outcome regions.\n")


def main() -> None:
    wf = pd.read_csv(WF)
    if not {"lh", "la"} <= set(wf.columns):
        raise SystemExit(
            "walk-forward sin lambdas: regenera con generate_deployed_walkforward.py")
    ht = pd.read_csv(FOUND, usecols=["match_id", "home_goals_ht", "away_goals_ht"],
                     low_memory=False)
    wf = wf.merge(ht, on="match_id", how="left")

    m = precompute(wf, HalfTimeModel())
    seasons = sorted(pd.unique(m["season"]))
    print(f"{m['n']:,} partidos, {len(seasons)} temporadas walk-forward "
          f"({m['clipped']} marcadores recortados a {MAXG})\n")
    harness_checks(wf, m)

    cols = ["rps", "ll1x2", "exact", "htft", *PUBLISHED, "even",
            "draw_pred", "draw_obs"]
    rows = []
    for form, grid in GRIDS.items():
        for held in seasons:
            tr, te = m["season"] != held, m["season"] == held
            base = np.array([(m["obs_region"][tr] == k).mean() for k in range(3)])
            best, best_r = None, np.inf
            for th in grid:
                r = rps(FORMS[form](m["raw"], th, base)[tr], m["obs_region"][tr])
                if r < best_r:
                    best_r, best = r, th
            s = score(FORMS[form](m["raw"], best, base), m, te)
            s.update(form=form, held=held,
                     theta=tuple(round(float(x), 3) for x in best))
            rows.append(s)
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "folds.csv", index=False)

    print("medias sobre las seis temporadas dejadas fuera "
          "(parametro ajustado en las otras cinco a RPS del 1X2):")
    print(res.groupby("form")[cols].mean().round(5).to_string(), "\n")

    piv = {c: res.pivot(index="held", columns="form", values=c) for c in cols}
    raw_gap = (piv["draw_pred"]["none"] - piv["draw_obs"]["none"]).abs()
    se = float(np.sqrt(0.254 * 0.746 / m["n"]))
    print("calibracion del empate:")
    for f in FORMS:
        gap = piv["draw_pred"][f] - piv["draw_obs"][f]
        worse = int((gap.abs() > raw_gap).sum())
        print(f"  {f:10s} media {piv['draw_pred'][f].mean():.4f}  "
              f"desvio {gap.mean():+.4f}  z {gap.mean() / se:+.2f}  "
              f"peor que el crudo en {worse}/6")

    print("\npair (empate intacto) menos power (desplegado); negativo = pair mejor:")
    for c in ["rps", "ll1x2", "exact", "htft", *PUBLISHED, "even"]:
        d = piv[c]["pair"] - piv[c]["power"]
        tag = "" if c == "even" else "  [publicado]"
        print(f"  {c:7s} media {d.mean():+.6f}  mejor en {int((d < 0).sum())}/6{tag}")

    print("\ncada forma contra el crudo (sin afilar); negativo = la forma mejora:")
    for f in ["power", "pair", "anchor", "pair_draw"]:
        bits = []
        for c in ["rps", *PUBLISHED]:
            d = piv[c][f] - piv[c]["none"]
            bits.append(f"{c} {d.mean():+.5f} ({int((d < 0).sum())}/6)")
        print(f"  {f:10s} " + "  ".join(bits))

    print("\nparametros elegidos por forma:")
    for f in FORMS:
        print(f"  {f:10s} {sorted(set(res[res.form == f].theta))}")
    print(f"\n{OUT / 'folds.csv'}")


if __name__ == "__main__":
    main()
