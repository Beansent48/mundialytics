"""Out-of-sample: engine EventLambdaModel vs TeamPropsModel on the per-side
expected counts the match page shows.

Walk-forward by season: fit both on everything before the season, predict every
match of it, never update. Reports MAE and RMSE per market per season. The
comparison the commit quotes was in-sample for both models, which flatters the
more flexible one (team props), so this is the number that decides.
"""
import sys, time, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from mundialytics.props import TeamPropsModel
from mundialytics.statistical_core.prediction_engine import (
    DEPLOYED_CLUB_ENGINE_KWARGS, PredictionEngine)

SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
# page key -> (engine attr stem, team-props market, actual column stem)
MARKETS = [("shots", "shots", "shots", "shots"),
           ("sot", "sot", "sot", "sot"),
           ("corners", "corners", "corners", "corners"),
           ("fouls", "fouls", "fouls", "fouls"),
           ("yellows", "yellows", "yellows", "yellow_cards")]

df = pd.read_csv(ROOT / "data/processed/foundation_big5_multi_season.csv", low_memory=False)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
need = ["home_goals", "away_goals", "home_xg", "away_xg", "date"]
df = df.dropna(subset=need).sort_values("date")

rows = []
for s in SEASONS:
    test = df[df["season"] == s].sort_values("date")
    train = df[df["date"] < test["date"].min()]
    if len(test) == 0 or len(train) < 500:
        continue
    t0 = time.time()
    eng = PredictionEngine(**DEPLOYED_CLUB_ENGINE_KWARGS).fit(
        train, squad_value_asof=test["date"].min())
    tp = TeamPropsModel().fit(train, root=ROOT)
    print(f"  {s}: fitted in {time.time()-t0:.0f}s", flush=True)
    for _, r in test.iterrows():
        h, a = str(r.home_team), str(r.away_team)
        try:
            p = eng.predict_match(h, a, competition=str(r.competition),
                                  neutral=bool(r.get("neutral", 0)))
        except Exception:
            continue
        try:
            fx = tp.predict_fixture(h, a, lam_home=p.lambda_home, lam_away=p.lambda_away)
        except Exception:
            fx = {}
        for key, stem, market, actual in MARKETS:
            d = fx.get(market) or {}
            if d.get("lambda_home") is None:
                continue
            for side in ("home", "away"):
                act = r.get(f"{side}_{actual}")
                if pd.isna(act):
                    continue
                rows.append({
                    "season": s, "market": key, "side": side,
                    "eng": float(getattr(p, f"expected_{stem}_{side}")),
                    "tp": float(d[f"lambda_{side}"]), "act": float(act)})
        # walk the engine's xG form forward exactly as production does
        if eng.xg_rate_model_ is not None:
            eng.xg_rate_model_.update_form(r.home_team, r.away_team, r.home_xg, r.away_xg)
    print(f"  {s}: {len(test)} matches, {time.time()-t0:.0f}s total", flush=True)

m = pd.DataFrame(rows)
m.to_csv(ROOT / "data/processed/logs/validate_expected_stats_source.csv", index=False)
print(f"\nrows: {len(m)}  (side-observations)\n")

print("MAE by market, pooled over seasons")
print(f"{'market':10s} {'n':>7s} {'engine':>8s} {'props':>8s} {'delta':>8s} {'%':>7s}")
for k, g in m.groupby("market"):
    e, t = (g.eng - g.act).abs().mean(), (g.tp - g.act).abs().mean()
    print(f"{k:10s} {len(g):7d} {e:8.3f} {t:8.3f} {t-e:+8.3f} {100*(t-e)/e:+6.1f}%")

print("\nfolds won by team props (MAE), per market")
for k, g in m.groupby("market"):
    won = []
    for s, gs in g.groupby("season"):
        e, t = (gs.eng - gs.act).abs().mean(), (gs.tp - gs.act).abs().mean()
        won.append(t < e)
    print(f"  {k:10s} {sum(won)}/{len(won)}")

print("\nRMSE by market (pooled)")
for k, g in m.groupby("market"):
    e = np.sqrt(((g.eng - g.act) ** 2).mean())
    t = np.sqrt(((g.tp - g.act) ** 2).mean())
    print(f"  {k:10s} engine {e:6.3f}  props {t:6.3f}  {100*(t-e)/e:+6.1f}%")

print("\nmean bias (predicted - actual)")
for k, g in m.groupby("market"):
    print(f"  {k:10s} engine {(g.eng-g.act).mean():+6.3f}  props {(g.tp-g.act).mean():+6.3f}")
