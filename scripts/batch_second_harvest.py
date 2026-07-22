from __future__ import annotations

"""Second-harvest batch: 6 feature experiments for the xG-rate predictor, measured
against the deployed BASE feature set under the standard protocol (walk-forward
rolling, temporal folds, shared per-fold AD blend 0.6/0.4, gamma=1.2 scoring).
No engine changes — winners get deployed separately.

Variants:
  BASE     deployed features: xg_for_r{5,10,19} + opp_xg_against_r{5,10,19} + is_home
  SETPIECE open-play / set-piece xG decomposition (from 534k shots w/ situation)
  SHOTQ    + shot-quality profile (xG-per-shot, big-chance rate, for & against)
  FINISH   + rolling finishing delta (goals - xG)
  REST     + rest days & 21-day congestion
  RECENCY  BASE features, recency-weighted regressor fit (half-life 730d)
  INTERACT + attack x opp-defense product term
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mundialytics.statistical_core.attack_defense_model import AttackDefenseModel  # noqa: E402

COVERED = ["2014-2015","2015-2016","2016-2017","2017-2018","2018-2019","2019-2020",
           "2020-2021","2021-2022","2022-2023","2023-2024","2024-2025","2025-2026"]
TEST_SEASONS = ["2020-2021","2021-2022","2022-2023","2023-2024","2024-2025","2025-2026"]
W = (5, 10, 19)
SP_SITU = {"From Corner", "Set Piece", "Direct Freekick"}
OP_SITU = "Open Play"


def build_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv", low_memory=False)
    df = df[(df.season.isin(COVERED)) & (df.xg_available == True)].copy()  # noqa: E712
    for c in ["home_goals","away_goals","home_xg","away_xg"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df.date, errors="coerce")
    df = df.dropna(subset=["home_goals","away_goals","home_xg","away_xg","date"]).sort_values("date")
    df["gid"] = pd.to_numeric(df["provider_match_id"], errors="coerce")

    # per (game, team) situation-split xG + shot quality from the shots file
    sh = pd.read_csv(ROOT / "data/external/advanced/understat/understat_shots.csv",
                     usecols=["game_id","team","xg","situation"])
    sh["xg"] = pd.to_numeric(sh.xg, errors="coerce").fillna(0.0)
    is_sp = sh.situation.isin(SP_SITU)
    is_op = sh.situation.eq(OP_SITU)
    sh["xg_op_v"] = sh.xg.where(is_op, 0.0)          # penalties/unlabelled excluded from both
    sh["xg_sp_v"] = sh.xg.where(is_sp, 0.0)
    sh["is_big"] = (sh.xg > 0.3) & (is_op | is_sp)
    ag = sh.groupby(["game_id","team"]).agg(
        xg_op=("xg_op_v","sum"), xg_sp=("xg_sp_v","sum"),
        nshots=("xg","size"), nbig=("is_big","sum")).reset_index()
    ag["xgps"] = ag.xg_op / ag.nshots.clip(lower=1)

    # long team-rows
    rows = []
    for r in df.itertuples(index=False):
        rows.append(dict(gid=r.gid, date=r.date, season=r.season, comp=r.competition, match_id=r.match_id,
                         team=r.home_team, opp=r.away_team, is_home=1,
                         xg_for=r.home_xg, xg_against=r.away_xg, g_for=r.home_goals))
        rows.append(dict(gid=r.gid, date=r.date, season=r.season, comp=r.competition, match_id=r.match_id,
                         team=r.away_team, opp=r.home_team, is_home=0,
                         xg_for=r.away_xg, xg_against=r.home_xg, g_for=r.away_goals))
    lr = pd.DataFrame(rows)

    # attach shots aggregates (understat team names -> foundation names)
    from mundialytics.enrichment.understat_team_aliases import to_foundation_name
    ag["team_fd"] = ag.team.map(to_foundation_name)
    lr = lr.merge(ag.rename(columns={"game_id":"gid"})[["gid","team_fd","xg_op","xg_sp","xgps","nbig"]],
                  left_on=["gid","team"], right_on=["gid","team_fd"], how="left")

    lr = lr.sort_values(["team","date","gid"])
    lr["fin"] = lr.g_for - lr.xg_for
    lr["rest"] = lr.groupby("team")["date"].diff().dt.days.clip(upper=14)
    m21 = []
    for _, g in lr.groupby("team"):
        dates = g["date"].values
        m21.extend([(dates[max(0,i-8):i] > d - np.timedelta64(21,"D")).sum() for i, d in enumerate(dates)])
    lr["m21"] = m21

    roll_cols = {"xg_for": W, "xg_against": W, "xg_op": (10,), "xg_sp": (10,),
                 "xgps": (10,), "nbig": (10,), "fin": (10,)}
    for col, wins in roll_cols.items():
        lr[col] = pd.to_numeric(lr[col], errors="coerce")
        for w in wins:
            lr[f"{col}_r{w}"] = lr.groupby("team", group_keys=False)[col].apply(
                lambda s: s.shift(1).rolling(w, min_periods=3).mean())
    # opponent columns for the against-side features
    opp_src = ["xg_against_r5","xg_against_r10","xg_against_r19","xg_op_r10","xg_sp_r10","xgps_r10","nbig_r10"]
    opp = lr[["gid","team"] + opp_src].rename(columns={"team":"opp", **{c: f"opp_{c}" for c in opp_src}})
    lr = lr.merge(opp, on=["gid","opp"], how="left")
    lr["inter"] = lr["xg_for_r10"] * lr["opp_xg_against_r10"]
    return df, lr


VARIANTS: dict[str, list[str]] = {}
def define_variants():
    base = [f"xg_for_r{w}" for w in W] + [f"opp_xg_against_r{w}" for w in W] + ["is_home"]
    VARIANTS["BASE"] = base
    VARIANTS["SETPIECE"] = ([f"xg_for_r{w}" for w in W] + ["xg_op_r10","xg_sp_r10"]
                            + [f"opp_xg_against_r{w}" for w in W] + ["opp_xg_op_r10","opp_xg_sp_r10","is_home"])
    VARIANTS["SHOTQ"] = base + ["xgps_r10","nbig_r10","opp_xgps_r10","opp_nbig_r10"]
    VARIANTS["FINISH"] = base + ["fin_r10"]
    VARIANTS["REST"] = base + ["rest","m21"]
    VARIANTS["RECENCY"] = base          # same feats, weighted fit
    VARIANTS["INTERACT"] = base + ["inter"]


K = 11; ks = np.arange(K); RHO = -0.07
def score(lh, la, sub):
    ph_ = poisson.pmf(ks[:,None], lh[None,:]); pa_ = poisson.pmf(ks[:,None], la[None,:])
    J = ph_[:,None,:]*pa_[None,:,:]
    J[0,0,:] *= (1-RHO*lh*la); J[0,1,:] *= (1+RHO*lh); J[1,0,:] *= (1+RHO*la); J[1,1,:] *= (1-RHO)
    J = np.clip(J,0,None); J /= J.sum(axis=(0,1),keepdims=True)
    hm=(ks[:,None]>ks[None,:]); dm=(ks[:,None]==ks[None,:]); om=(ks[:,None]+ks[None,:]>=3); bm=(ks[:,None]>=1)&(ks[None,:]>=1)
    m=lambda M:(J*M[:,:,None]).sum(axis=(0,1))
    ph, pdd, po, pb = m(hm), m(dm), m(om), m(bm)
    P=np.c_[ph,pdd,1-ph-pdd]; P=np.clip(P,1e-9,1)**1.2; P/=P.sum(axis=1,keepdims=True)
    o=np.where(sub.hg>sub.ag,"home",np.where(sub.hg<sub.ag,"away","draw"))
    y=np.c_[o=="home",o=="draw",o=="away"].astype(float)
    rps=float((0.5*((P[:,0]-y[:,0])**2+(P[:,0]+P[:,1]-y[:,0]-y[:,1])**2)).mean())
    over=((sub.hg+sub.ag)>2.5).to_numpy(float); btts=((sub.hg>0)&(sub.ag>0)).to_numpy(float)
    llo=float(-(over*np.log(np.clip(po,1e-9,1))+(1-over)*np.log(np.clip(1-po,1e-9,1))).mean())
    llb=float(-(btts*np.log(np.clip(pb,1e-9,1))+(1-btts)*np.log(np.clip(1-pb,1e-9,1))).mean())
    return rps, llo, llb


def main() -> None:
    t0=time.time()
    df, lr = build_frame(); define_variants()
    print(f"frame ready: {len(lr)} team-rows ({time.time()-t0:.0f}s)", flush=True)
    df = df.assign(hg=df.home_goals.astype(int), ag=df.away_goals.astype(int))

    results = {v: [] for v in VARIANTS}
    for s in TEST_SEASONS:
        test = df[df.season==s]; s_start = test.date.min(); train = df[df.date < s_start]
        if len(test)==0 or len(train)<500: continue
        t1=time.time()
        ad = AttackDefenseModel(time_decay_half_life=None).fit(train)
        adl = {r.match_id: ad.expected_goals(r.home_team, r.away_team, int(getattr(r,"neutral",0) or 0), r.competition)[:2]
               for r in test.itertuples(index=False)}
        tr_rows = lr[lr.date < s_start]; te_rows = lr[lr.match_id.isin(set(test.match_id))]
        line=f"{s}  "
        for name, feats in VARIANTS.items():
            tr = tr_rows.dropna(subset=feats+["xg_for"]); te = te_rows.dropna(subset=feats)
            kw = {}
            if name=="RECENCY":
                age=(s_start-tr.date).dt.days.clip(lower=0).astype(float)
                kw["sample_weight"]=np.power(0.5, age/730.0)
            m = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr.xg_for.clip(lower=0), **kw)
            pred = pd.Series(np.clip(m.predict(te[feats]),0.05,6.0), index=pd.MultiIndex.from_frame(te[["match_id","is_home"]]))
            lh=[]; la=[]; keep=[]
            for r in test.itertuples(index=False):
                try:
                    xh=float(pred.loc[(r.match_id,1)]); xa=float(pred.loc[(r.match_id,0)])
                except KeyError:
                    continue
                a=adl[r.match_id]
                lh.append(0.6*xh+0.4*a[0]); la.append(0.6*xa+0.4*a[1]); keep.append(r.match_id)
            sub=test[test.match_id.isin(keep)]
            res=score(np.array(lh),np.array(la),sub)
            results[name].append((res,len(sub)))
        print(f"{s} done ({time.time()-t1:.0f}s)", flush=True)

    def pool(a,i): return sum(r[i]*n for r,n in a)/sum(n for _,n in a)
    base_rps, base_o, base_b = pool(results["BASE"],0), pool(results["BASE"],1), pool(results["BASE"],2)
    print(f"\n===== POOLED ({sum(n for _,n in results['BASE'])} matches) =====")
    print(f"{'variant':10s} {'RPS':>8s} {'dRPS':>9s} {'dLL_OU':>9s} {'dLL_BTTS':>9s}")
    for name in VARIANTS:
        r,o,b = pool(results[name],0), pool(results[name],1), pool(results[name],2)
        print(f"{name:10s} {r:8.4f} {r-base_rps:+9.5f} {o-base_o:+9.5f} {b-base_b:+9.5f}", flush=True)
    # per-fold dRPS for any variant beating base
    for name in VARIANTS:
        if name=="BASE": continue
        d=[results[name][i][0][0]-results["BASE"][i][0][0] for i in range(len(results["BASE"]))]
        if sum(x<0 for x in d) >= 4:
            print(f"  {name} per-fold dRPS: " + " ".join(f"{x:+.4f}" for x in d), flush=True)


if __name__ == "__main__":
    main()
