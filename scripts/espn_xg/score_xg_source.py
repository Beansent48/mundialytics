"""Score wf_xg_source.py runs against each other: 1X2 RPS with the deployed squad-value
shift on top, overall and by month (a stale xG form costs more as the season goes on).

    python scripts/espn_xg/score_xg_source.py understat=wfx_us.csv none=wfx_none.csv text=wfx_text.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mundialytics.statistical_core.prediction_engine import DEPLOYED_CLUB_ENGINE_KWARGS as K  # noqa: E402

WORK = ROOT / "data/external/transfermarkt/work"
RHO, TEMP, GAM = K["outcome_rho"], K["goal_temper"], K["sharpen_gamma_1x2"]
SV = K["squad_value_shift"]
G = np.arange(11)


def trio(lh, la):
    lh, la = np.clip(lh, 0.01, 8.0)[:, None], np.clip(la, 0.01, 8.0)[:, None]
    ph = poisson.pmf(G[None, :], lh) ** TEMP
    ph /= ph.sum(1, keepdims=True)
    pa = poisson.pmf(G[None, :], la) ** TEMP
    pa /= pa.sum(1, keepdims=True)
    M = ph[:, :, None] * pa[:, None, :]
    l1, l2 = lh[:, 0], la[:, 0]
    M[:, 0, 0] *= 1 - l1 * l2 * RHO
    M[:, 0, 1] *= 1 + l1 * RHO
    M[:, 1, 0] *= 1 + l2 * RHO
    M[:, 1, 1] *= 1 - RHO
    M /= M.sum((1, 2), keepdims=True)
    i, j = np.indices((11, 11))
    P = np.stack([(M * (i > j)).sum((1, 2)), (M * (i == j)).sum((1, 2)), (M * (i < j)).sum((1, 2))], 1)
    P = np.clip(P, 1e-9, 1) ** GAM
    return P / P.sum(1, keepdims=True)


tv = pd.read_csv(ROOT / "data/processed/squad_values.csv", parse_dates=["snap"]).sort_values("snap")


def score(path):
    d = pd.read_csv(WORK / path, parse_dates=["date"])
    w, ls = d.w.values, d.ls.values
    LH = np.clip((w * d.xr_h + (1 - w) * d.lh_ad) * ls, 0.05, 6.0).values
    LA = np.clip((w * d.xr_a + (1 - w) * d.la_ad) * ls, 0.05, 6.0).values

    def val(side):
        q = d[["match_id", "date", side]].rename(columns={side: "team"}).sort_values("date")
        m = pd.merge_asof(q, tv.rename(columns={"snap": "date"}), on="date", by="team", direction="backward",
                          allow_exact_matches=False, tolerance=pd.Timedelta(days=10)).set_index("match_id")
        return d.match_id.map(m.v18).values
    dv = np.log(val("home")) - np.log(val("away"))
    s = np.where(np.isfinite(dv), SV["beta"] * np.nan_to_num(dv) + SV["kappa"] * np.log(LH / LA), 0.0)
    P = trio(LH * np.exp(s / 2), LA * np.exp(-s / 2))
    o = np.where(d.hg > d.ag, 0, np.where(d.hg == d.ag, 1, 2))
    Y = np.eye(3)[o]
    d["rps"] = (((np.cumsum(P, 1) - np.cumsum(Y, 1)) ** 2)[:, :2].sum(1)) / 2
    return d.set_index("match_id")[["date", "rps"]]


def main() -> int:
    runs = dict(a.split("=", 1) for a in sys.argv[1:])
    res = {k: score(v) for k, v in runs.items()}
    common = sorted(set.intersection(*(set(r.index) for r in res.values())))
    base = next(iter(res))
    t = pd.DataFrame({k: r.loc[common, "rps"] for k, r in res.items()})
    t["month"] = res[base].loc[common, "date"].dt.to_period("M").values
    print(f"{len(common)} matches")
    print("overall:", {k: round(t[k].mean(), 5) for k in runs})
    print(t.groupby("month")[list(runs)].mean().round(4).to_string())
    for k in runs:
        if k != base:
            dlt = t[k] - t[base]
            print(f"{k} - {base}: {dlt.mean():+.5f} (se {dlt.std() / np.sqrt(len(dlt)):.5f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
