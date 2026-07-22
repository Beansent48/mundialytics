from __future__ import annotations

"""End-to-end validation of the SETPIECE features in the deployed engine: same
walk-forward protocol, deployed config (gamma=1.3, outcome_rho=-0.17, rescale),
comparing the engine WITH op/sp columns (auto-SETPIECE) vs WITHOUT (base
features), on identical folds. Deploy stays only if this confirms the batch.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402

TEST_SEASONS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
COVERED = ["2014-2015","2015-2016","2016-2017","2017-2018","2018-2019","2019-2020",
           "2020-2021","2021-2022","2022-2023","2023-2024","2024-2025","2025-2026"]
SP = ["home_xg_op", "away_xg_op", "home_xg_sp", "away_xg_sp"]
CFG = dict(blend_weight_gl=0.30, sharpen_gamma_1x2=1.3, outcome_rho=-0.17, rescale_lambda_to_goals=True)


def metrics(pr):
    o = pr["oc"].to_numpy()
    ph, pdw, pa = pr.ph.to_numpy(float), pr.pd.to_numpy(float), pr.pa.to_numpy(float)
    yh, yd = (o == "home").astype(float), (o == "draw").astype(float)
    rps = (0.5 * ((ph - yh) ** 2 + (ph + pdw - yh - yd) ** 2)).mean()
    over = pr.over.to_numpy(float); po = np.clip(pr.po.to_numpy(float), 1e-9, 1 - 1e-9)
    btts = pr.btts.to_numpy(float); pb = np.clip(pr.pb.to_numpy(float), 1e-9, 1 - 1e-9)
    llo = -(over * np.log(po) + (1 - over) * np.log(1 - po)).mean()
    llb = -(btts * np.log(pb) + (1 - btts) * np.log(1 - pb)).mean()
    return {"rps": float(rps), "llo": float(llo), "llb": float(llb), "n": len(pr)}


def run_arm(train, test, with_sp: bool):
    tr = train if with_sp else train.drop(columns=SP, errors="ignore")
    eng = PredictionEngine(**CFG).fit(tr)
    rows = []
    for _, r in test.iterrows():
        p = eng.predict_match(str(r.home_team), str(r.away_team), competition=str(r.competition),
                              neutral=bool(r.get("neutral", 0)))
        rows.append({"ph": p.p_home_win, "pd": p.p_draw, "pa": p.p_away_win,
                     "po": p.p_over_25, "pb": p.p_btts, "oc": r.oc, "over": r.over, "btts": r.btts})
        if eng.xg_rate_model_ is not None:
            eng.xg_rate_model_.update_form(r.home_team, r.away_team, r.home_xg, r.away_xg,
                                           home_xg_op=(r.home_xg_op if with_sp else None),
                                           away_xg_op=(r.away_xg_op if with_sp else None),
                                           home_xg_sp=(r.home_xg_sp if with_sp else None),
                                           away_xg_sp=(r.away_xg_sp if with_sp else None))
    return metrics(pd.DataFrame(rows))


def main() -> None:
    df = pd.read_csv(ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv", low_memory=False)
    df = df[(df.season.isin(COVERED)) & (df.xg_available == True)].copy()  # noqa: E712
    for c in ["home_goals","away_goals","home_xg","away_xg"] + SP:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df.date, errors="coerce")
    df = df.dropna(subset=["home_goals","away_goals","home_xg","away_xg","date"]).sort_values("date")
    hg, ag = df.home_goals.astype(int), df.away_goals.astype(int)
    df["oc"] = np.where(hg > ag, "home", np.where(hg < ag, "away", "draw"))
    df["over"] = ((hg + ag) > 2.5).astype(int); df["btts"] = ((hg > 0) & (ag > 0)).astype(int)

    base_p, sp_p = [], []
    print(f"{'season':10s} {'base':>8s} {'setpiece':>9s} {'dRPS':>9s} {'dLL_BTTS':>9s}")
    for s in TEST_SEASONS:
        test = df[df.season == s].sort_values("date"); train = df[df.date < test.date.min()]
        if len(test) == 0 or len(train) < 500: continue
        t0 = time.time()
        mb = run_arm(train, test, with_sp=False)
        ms = run_arm(train, test, with_sp=True)
        base_p.append(mb); sp_p.append(ms)
        print(f"{s:10s} {mb['rps']:8.4f} {ms['rps']:9.4f} {ms['rps']-mb['rps']:+9.5f} {ms['llb']-mb['llb']:+9.5f}  ({time.time()-t0:.0f}s)", flush=True)

    pool = lambda a, k: sum(m[k] * m["n"] for m in a) / sum(m["n"] for m in a)
    print("\n===== POOLED =====")
    for k in ["rps", "llo", "llb"]:
        print(f"  {k:4s} base {pool(base_p,k):.4f}  setpiece {pool(sp_p,k):.4f}  delta {pool(sp_p,k)-pool(base_p,k):+.5f}")


if __name__ == "__main__":
    main()
