from __future__ import annotations

"""Comprehensive A/B backtest of the xG integration phases in PredictionEngine.

Efficient two-step design:
  STEP A (heavy): per expanding-window fold, fit the component models once and
    dump per-test-match lambda COMPONENTS to a table:
      - gl_noxg : GoalLambdaModel without xG features (baseline form)
      - gl_xg   : GoalLambdaModel with rolling pre-match xG features (Phase 1)
      - ad      : goals-target AttackDefense (legacy 40% estimator)
      - ad_xg   : xG-target AttackDefense (Phase 2)
  STEP B (light): any blend of components -> goal Poisson matrix -> 1X2/O-U/BTTS,
    so fixed arms AND a learned blend weight (Phase 3) are evaluated with NO
    refitting.

Arms scored on held-out test matches:
  baseline    0.60*gl_noxg + 0.40*ad                    (current engine)
  feature     0.60*gl_xg   + 0.40*ad                    (Phase 1)
  target      0.60*gl_noxg + 0.40*ad_xg                 (Phase 2, xG replaces goals-AD)
  both        0.60*gl_xg   + 0.40*ad_xg                 (Phase 1+2)
  learned     weights grid-searched on EARLIER folds only, applied out-of-sample (Phase 3)

Metrics: 1X2 RPS / log-loss / Brier, Over2.5 & BTTS log-loss, and H/D/A bias.
No elo_history (matches the competition-layer production path).
"""

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402
from mundialytics.statistical_core.distributions import outcome_probabilities  # noqa: E402

DEFAULT_MATCHES = "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv"
COVERED_SEASONS = [
    "2014-2015", "2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020",
    "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026",
]
DEFAULT_TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025"]
AD_RHO = -0.07
MAX_GOALS = 10


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


# ── Step A: fit components per fold, dump prediction table ──────────────────────

def compute_components(matches_path: str, test_seasons: list[str]) -> pd.DataFrame:
    df = pd.read_csv(_resolve(matches_path), low_memory=False)
    df = df[df["season"].isin(COVERED_SEASONS)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["home_goals", "away_goals", "home_xg", "away_xg", "home_npxg", "away_npxg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "xg_available" in df.columns:
        df = df[df["xg_available"] == True].copy()  # noqa: E712
    df = df.dropna(subset=["home_goals", "away_goals", "date"]).sort_values("date").reset_index(drop=True)
    hg = df["home_goals"].astype(int); ag = df["away_goals"].astype(int)
    df["actual_outcome"] = np.where(hg > ag, "home", np.where(hg < ag, "away", "draw"))
    df["actual_over_25"] = ((hg + ag) > 2.5).astype(int)
    df["actual_btts"] = ((hg > 0) & (ag > 0)).astype(int)
    print(f"Loaded {len(df)} covered matches, {df['date'].min().date()}..{df['date'].max().date()}", flush=True)

    parts = []
    for season in test_seasons:
        test = df[df["season"] == season]
        train = df[df["date"] < test["date"].min()]
        if len(test) == 0 or len(train) < 500:
            print(f"skip {season}: test={len(test)} train={len(train)}", flush=True)
            continue
        t0 = time.time()
        eng_xg = PredictionEngine(use_xg=True, blend_weight_ad_xg=0.5, ad_rho=AD_RHO).fit(train)
        eng_no = PredictionEngine(use_xg=False, ad_rho=AD_RHO).fit(train)
        rows = []
        for _, r in test.iterrows():
            h, a = str(r["home_team"]), str(r["away_team"])
            comp = str(r.get("competition", "unknown")); neu = bool(r.get("neutral", 0))
            gl_x = eng_xg._lambdas_gl(h, a, comp)
            gl_n = eng_no._lambdas_gl(h, a, comp)
            ad = eng_xg._lambdas_ad(h, a, comp, neu)
            adx = eng_xg.ad_xg_model_.expected_goals(h, a, neutral=int(neu), competition=comp)[:2]
            rows.append({
                "season": season, "match_id": r.get("match_id"),
                "gl_noxg_h": gl_n[0], "gl_noxg_a": gl_n[1],
                "gl_xg_h": gl_x[0], "gl_xg_a": gl_x[1],
                "ad_h": ad[0], "ad_a": ad[1],
                "adxg_h": adx[0], "adxg_a": adx[1],
                "actual_outcome": r["actual_outcome"],
                "actual_over_25": int(r["actual_over_25"]), "actual_btts": int(r["actual_btts"]),
            })
        parts.append(pd.DataFrame(rows))
        print(f"  {season}: train={len(train)} test={len(test)} ({time.time()-t0:.0f}s)", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ── Step B: blend components -> probabilities -> metrics ────────────────────────

def arm_lambdas(comp: pd.DataFrame, gl_col: str, w_gl: float, w_ad: float, w_adx: float) -> tuple[np.ndarray, np.ndarray]:
    lh = w_gl * comp[f"{gl_col}_h"] + w_ad * comp["ad_h"] + w_adx * comp["adxg_h"]
    la = w_gl * comp[f"{gl_col}_a"] + w_ad * comp["ad_a"] + w_adx * comp["adxg_a"]
    return np.clip(lh.to_numpy(float), 0.05, 6.0), np.clip(la.to_numpy(float), 0.05, 6.0)


def probs_from_lambdas(lh: np.ndarray, la: np.ndarray) -> pd.DataFrame:
    out = {"p_home_win": [], "p_draw": [], "p_away_win": [], "p_over_25": [], "p_btts": []}
    for x, y in zip(lh, la):
        p = outcome_probabilities(float(x), float(y), max_goals=MAX_GOALS, dixon_coles_rho=AD_RHO)
        out["p_home_win"].append(p["p_home_win"]); out["p_draw"].append(p["p_draw"])
        out["p_away_win"].append(p["p_away_win"]); out["p_over_25"].append(p["p_over_25"]); out["p_btts"].append(p["p_btts"])
    return pd.DataFrame(out)


def score(comp: pd.DataFrame, probs: pd.DataFrame) -> dict:
    outcome = comp["actual_outcome"].to_numpy()
    ph, pd_, pa = (probs["p_home_win"].to_numpy(float), probs["p_draw"].to_numpy(float), probs["p_away_win"].to_numpy(float))
    yh = (outcome == "home").astype(float); yd = (outcome == "draw").astype(float); ya = (outcome == "away").astype(float)
    cum_p1, cum_p2 = ph, ph + pd_
    cum_o1, cum_o2 = yh, yh + yd
    rps = 0.5 * ((cum_p1 - cum_o1) ** 2 + (cum_p2 - cum_o2) ** 2)
    actual_p = np.where(outcome == "home", ph, np.where(outcome == "draw", pd_, pa))
    logloss = -np.log(np.clip(actual_p, 1e-9, 1.0))
    brier = (ph - yh) ** 2 + (pd_ - yd) ** 2 + (pa - ya) ** 2
    over = comp["actual_over_25"].to_numpy(float); btts = comp["actual_btts"].to_numpy(float)
    po = np.clip(probs["p_over_25"].to_numpy(float), 1e-9, 1 - 1e-9)
    pb = np.clip(probs["p_btts"].to_numpy(float), 1e-9, 1 - 1e-9)
    return {
        "n": int(len(comp)),
        "rps": float(rps.mean()),
        "logloss_1x2": float(logloss.mean()),
        "brier_1x2": float(brier.mean()),
        "logloss_over25": float(-(over * np.log(po) + (1 - over) * np.log(1 - po)).mean()),
        "logloss_btts": float(-(btts * np.log(pb) + (1 - btts) * np.log(1 - pb)).mean()),
        "bias_home": float(ph.mean() - yh.mean()),
        "bias_draw": float(pd_.mean() - yd.mean()),
        "bias_away": float(pa.mean() - ya.mean()),
    }


def score_arm(comp: pd.DataFrame, gl_col: str, w_gl: float, w_ad: float, w_adx: float) -> dict:
    lh, la = arm_lambdas(comp, gl_col, w_gl, w_ad, w_adx)
    return score(comp, probs_from_lambdas(lh, la))


def learn_weights(comp_train: pd.DataFrame, gl_col: str) -> tuple[float, float, float]:
    """Grid-search the blend minimizing 1X2 RPS on comp_train."""
    best, best_rps = (0.6, 0.4, 0.0), 1e9
    grid = [i / 10 for i in range(0, 11)]
    for w_gl, w_adx in product(grid, grid):
        w_ad = 1.0 - w_gl - w_adx
        if w_ad < -1e-9:
            continue
        s = score_arm(comp_train, gl_col, w_gl, w_ad, w_adx)
        if s["rps"] < best_rps:
            best_rps, best = s["rps"], (w_gl, w_ad, w_adx)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", default=DEFAULT_MATCHES)
    parser.add_argument("--test-seasons", nargs="*", default=DEFAULT_TEST_SEASONS)
    parser.add_argument("--components-cache", default="data/processed/enriched/understat_xg/xg_components.csv")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--out", default="data/processed/enriched/understat_xg/xg_phases_backtest.json")
    args = parser.parse_args()

    cache = _resolve(args.components_cache)
    if args.reuse_cache and cache.exists():
        comp = pd.read_csv(cache)
        print(f"Reusing components cache: {len(comp)} rows", flush=True)
    else:
        comp = compute_components(args.matches, args.test_seasons)
        cache.parent.mkdir(parents=True, exist_ok=True)
        comp.to_csv(cache, index=False)
        print(f"WROTE components cache {cache}", flush=True)
    if comp.empty:
        print("no components", flush=True); return

    seasons = [s for s in args.test_seasons if s in set(comp["season"])]

    # Fixed arms (pooled over all test folds).
    fixed_arms = {
        "baseline": ("gl_noxg", 0.6, 0.4, 0.0),
        "feature": ("gl_xg", 0.6, 0.4, 0.0),
        "target": ("gl_noxg", 0.6, 0.0, 0.4),
        "both": ("gl_xg", 0.6, 0.0, 0.4),
    }
    results = {}
    for name, (gl_col, w_gl, w_ad, w_adx) in fixed_arms.items():
        results[name] = score(comp, probs_from_lambdas(*arm_lambdas(comp, gl_col, w_gl, w_ad, w_adx)))

    # Phase 3: out-of-sample learned weight. For each test season, learn on the
    # EARLIER test seasons only; the first season falls back to 60/40 baseline blend.
    learned_rows = []
    chosen = {}
    for i, season in enumerate(seasons):
        past = comp[comp["season"].isin(seasons[:i])]
        if len(past) < 300:
            w = ("gl_xg", 0.6, 0.4, 0.0)
        else:
            wl = learn_weights(past, "gl_xg")
            w = ("gl_xg", *wl)
        chosen[season] = w
        cur = comp[comp["season"] == season]
        lh, la = arm_lambdas(cur, w[0], w[1], w[2], w[3])
        p = probs_from_lambdas(lh, la)
        cur2 = cur.reset_index(drop=True)
        learned_rows.append((cur2, p))
    learned_comp = pd.concat([c for c, _ in learned_rows], ignore_index=True)
    learned_probs = pd.concat([p for _, p in learned_rows], ignore_index=True)
    results["learned_oos"] = score(learned_comp, learned_probs)

    # In-sample best (oracle upper bound) for reference.
    oracle = learn_weights(comp, "gl_xg")
    results["oracle_insample"] = score(comp, probs_from_lambdas(*arm_lambdas(comp, "gl_xg", *oracle)))

    print("\n===== POOLED ARMS (test seasons: %s) =====" % ", ".join(seasons), flush=True)
    keys = ["rps", "logloss_1x2", "brier_1x2", "logloss_over25", "logloss_btts", "bias_home", "bias_away"]
    hdr = f"{'arm':14s} " + " ".join(f"{k:>13s}" for k in keys)
    print(hdr, flush=True)
    base = results["baseline"]
    for name in ["baseline", "feature", "target", "both", "learned_oos", "oracle_insample"]:
        s = results[name]
        line = f"{name:14s} " + " ".join(f"{s[k]:+13.5f}" for k in keys)
        print(line, flush=True)
    print("\nDeltas vs baseline (negative = better for scores):", flush=True)
    for name in ["feature", "target", "both", "learned_oos", "oracle_insample"]:
        s = results[name]
        d = {k: s[k] - base[k] for k in ["rps", "logloss_1x2", "logloss_over25"]}
        print(f"  {name:14s} dRPS={d['rps']:+.5f}  dLL1x2={d['logloss_1x2']:+.5f}  dLLo25={d['logloss_over25']:+.5f}", flush=True)
    print("\nLearned weights per season (gl, ad_goals, ad_xg):", flush=True)
    for s, w in chosen.items():
        print(f"  {s}: gl={w[1]:.1f} ad={w[2]:.1f} adxg={w[3]:.1f}", flush=True)

    out = _resolve(args.out)
    out.write_text(json.dumps({"arms": results, "learned_weights": {s: list(w[1:]) for s, w in chosen.items()},
                               "oracle_weights": list(oracle)}, indent=2), encoding="utf-8")
    print(f"\nWROTE {out}", flush=True)


if __name__ == "__main__":
    main()
