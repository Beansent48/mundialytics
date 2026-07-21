from __future__ import annotations

"""Global backtest: deployed engine vs an xG-rate-predictor-augmented model, with
pure-statistical-value skill scores and reference benchmarks.

Per expanding fold (test season, train = all covered before it), computes on the
SAME held-out matches every lambda component:
  gl  : deployed GoalLambdaModel (goals form)      ad  : goals AttackDefense (MLE)
  xr  : dedicated xG-rate predictor (this session) adx : xG AttackDefense (MLE)
Then evaluates model configs across markets and emits a components cache for the
fast analysis step (analyze_global.py).
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_ix = importlib.util.spec_from_file_location("ix", ROOT / "scripts" / "improve_xg_predictor.py")
ix = importlib.util.module_from_spec(_ix); _ix.loader.exec_module(ix)

from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402
from mundialytics.statistical_core.attack_defense_model import AttackDefenseModel  # noqa: E402

TEST_SEASONS = ["2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025"]
OUT = ROOT / "data/processed/enriched/understat_xg/global_components.csv"


def main() -> None:
    df = pd.read_csv(ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv", low_memory=False)
    df = df[(df["season"].isin(ix.COVERED)) & (df["xg_available"] == True)].copy()  # noqa: E712
    for c in ["home_goals", "away_goals", "home_xg", "away_xg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals", "home_xg", "away_xg", "date"]).sort_values("date")

    lr = ix.add_rolling(ix.long_rows(df))
    xg_feats = [f"xg_for_r{w}" for w in ix.WINDOWS] + [f"opp_xg_against_r{w}" for w in ix.WINDOWS] + ["is_home"]

    parts = []
    for season in TEST_SEASONS:
        test = df[df["season"] == season]
        s_start = test["date"].min()
        train = df[df["date"] < s_start]
        if len(test) == 0 or len(train) < 500:
            continue
        t0 = time.time()
        eng = PredictionEngine(use_xg=False, blend_weight_gl=0.30).fit(train)
        ad_x = AttackDefenseModel(time_decay_half_life=None, target="xg").fit(train)
        # xG-rate predictor
        tr = lr[lr["date"] < s_start].dropna(subset=xg_feats + ["xg_for"])
        m_xr = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[xg_feats], tr["xg_for"].clip(lower=0))
        te_lr = lr[lr["match_id"].isin(test["match_id"])].dropna(subset=xg_feats).copy()
        te_lr["xr"] = np.clip(m_xr.predict(te_lr[xg_feats]), 0.05, 6.0)
        xr_map = te_lr.set_index(["match_id", "is_home"])["xr"]

        for _, r in test.iterrows():
            h, a, comp, neu = r["home_team"], r["away_team"], r["competition"], int(r.get("neutral", 0))
            gl = eng._lambdas_gl(h, a, comp)
            ad = eng._lambdas_ad(h, a, comp, bool(neu))
            adx = ad_x.expected_goals(h, a, neutral=neu, competition=comp)[:2]
            def xr(home):
                try: return float(xr_map.loc[(r["match_id"], home)])
                except KeyError: return np.nan
            parts.append({"season": season, "match_id": r["match_id"],
                          "hg": int(r["home_goals"]), "ag": int(r["away_goals"]),
                          "gl_h": gl[0], "gl_a": gl[1], "ad_h": ad[0], "ad_a": ad[1],
                          "adx_h": adx[0], "adx_a": adx[1], "xr_h": xr(1), "xr_a": xr(0)})
        print(f"  {season}: train={len(train)} test={len(test)} ({time.time()-t0:.0f}s)", flush=True)

    out = pd.DataFrame(parts)
    out.to_csv(OUT, index=False)
    print(f"WROTE {OUT} ({len(out)} matches)", flush=True)


if __name__ == "__main__":
    main()
