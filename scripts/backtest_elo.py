from __future__ import annotations

"""Does a real (xG-)ELO fix the crippled GoalLambdaModel and rebalance the blend?

Backtest question: the fixed-60/40 finding was a symptom of GL running ELO-blind
(team_elo=1500 constant). Here we feed GL a real pre-match team rating and test
three regimes, isolating the ELO effect (use_xg=False, so GL = goals rolling form
+ ELO):
    none   : no elo_history (current production path)
    goals  : classic goals-ELO (EloRater default)
    xg     : xG-ELO (rating updates on chance quality)

Per fold we extract GL and goals-AD lambda components on the held-out test season,
then sweep the GL/AD blend weight to report, per regime: the RPS at the current
0.60 GL weight, the RPS-optimal GL weight (does GL reclaim weight?), and the min
RPS (does a real rating beat both no-ELO and the blend re-weighting alone?).

ELO is fit strictly on training matches (leakage-safe); the engine's cached last
train row carries each team's end-of-training rating into test prediction.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.ratings.elo import EloRater, EloConfig  # noqa: E402
from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402

# reuse vectorized scoring from the component analyzer
_spec = importlib.util.spec_from_file_location("az", ROOT / "scripts" / "analyze_xg_components.py")
az = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(az)

COVERED = ["2014-2015", "2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020",
           "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
DEFAULT_TEST = ["2021-2022", "2022-2023", "2023-2024", "2024-2025"]


def elo_history_for(train: pd.DataFrame, mode: str):
    if mode == "none":
        return None
    cfg = EloConfig(xg_weight=1.0) if mode == "xg" else EloConfig()
    return EloRater(cfg).fit(train)


def components(matches_path: str, test_seasons: list[str], modes: list[str]) -> pd.DataFrame:
    df = pd.read_csv(ROOT / matches_path, low_memory=False)
    df = df[df["season"].isin(COVERED)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["home_goals", "away_goals", "home_xg", "away_xg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "xg_available" in df.columns:
        df = df[df["xg_available"] == True].copy()  # noqa: E712
    df = df.dropna(subset=["home_goals", "away_goals", "date"]).sort_values("date").reset_index(drop=True)
    hg = df["home_goals"].astype(int); ag = df["away_goals"].astype(int)
    df["actual_outcome"] = np.where(hg > ag, "home", np.where(hg < ag, "away", "draw"))
    df["actual_over_25"] = ((hg + ag) > 2.5).astype(int)
    df["actual_btts"] = ((hg > 0) & (ag > 0)).astype(int)
    print(f"Loaded {len(df)} covered matches", flush=True)

    parts = []
    for season in test_seasons:
        test = df[df["season"] == season]
        train = df[df["date"] < test["date"].min()]
        if len(test) == 0 or len(train) < 500:
            continue
        for mode in modes:
            t0 = time.time()
            elo = elo_history_for(train, mode)
            eng = PredictionEngine(use_xg=False).fit(train, elo_history=elo)
            rows = []
            for _, r in test.iterrows():
                h, a = str(r["home_team"]), str(r["away_team"])
                comp = str(r.get("competition", "unknown")); neu = bool(r.get("neutral", 0))
                gl = eng._lambdas_gl(h, a, comp)
                ad = eng._lambdas_ad(h, a, comp, neu)
                rows.append({"season": season, "mode": mode, "match_id": r.get("match_id"),
                             "gl_noxg_h": gl[0], "gl_noxg_a": gl[1], "ad_h": ad[0], "ad_a": ad[1],
                             "adxg_h": ad[0], "adxg_a": ad[1],  # placeholder (no xG-AD here)
                             "actual_outcome": r["actual_outcome"],
                             "actual_over_25": int(r["actual_over_25"]), "actual_btts": int(r["actual_btts"])})
            parts.append(pd.DataFrame(rows))
            print(f"  {season} [{mode}] train={len(train)} test={len(test)} ({time.time()-t0:.0f}s)", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def analyze(comp: pd.DataFrame) -> None:
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 3)
    print("\n===== ELO REGIMES (pooled test seasons) =====", flush=True)
    print(f"{'mode':8s} {'RPS@0.60':>10s} {'best_w_gl':>10s} {'RPS@best':>10s} {'LL@best':>10s} {'bias_away@best':>15s}", flush=True)
    summary = {}
    for mode in ["none", "goals", "xg"]:
        c = comp[comp["mode"] == mode]
        if len(c) == 0:
            continue
        rps60 = az.score_arm(c, "gl_noxg", 0.6, 0.4, 0.0)["rps"]
        best = min(((az.score_arm(c, "gl_noxg", w, 1 - w, 0.0)["rps"], w) for w in grid))
        s_best = az.score_arm(c, "gl_noxg", best[1], 1 - best[1], 0.0)
        summary[mode] = (rps60, best[1], s_best)
        print(f"{mode:8s} {rps60:>10.5f} {best[1]:>10.2f} {s_best['rps']:>10.5f} {s_best['logloss_1x2']:>10.5f} {s_best['bias_away']:>15.5f}", flush=True)
    if "none" in summary and "xg" in summary:
        base = summary["none"][2]["rps"]  # best-weight RPS under no-ELO
        print("\nvs no-ELO (best weight each):", flush=True)
        for mode in ["goals", "xg"]:
            if mode in summary:
                print(f"  {mode:6s} dRPS={summary[mode][2]['rps']-base:+.5f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matches", default="data/processed/enriched/understat_xg/canonical_matches_with_xg.csv")
    ap.add_argument("--test-seasons", nargs="*", default=DEFAULT_TEST)
    ap.add_argument("--cache", default="data/processed/enriched/understat_xg/elo_components.csv")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    if args.reuse and (ROOT / args.cache).exists():
        comp = pd.read_csv(ROOT / args.cache)
    else:
        comp = components(args.matches, args.test_seasons, ["none", "goals", "xg"])
        (ROOT / args.cache).parent.mkdir(parents=True, exist_ok=True)
        comp.to_csv(ROOT / args.cache, index=False)
        print(f"WROTE {ROOT / args.cache}", flush=True)
    if not comp.empty:
        analyze(comp)


if __name__ == "__main__":
    main()
