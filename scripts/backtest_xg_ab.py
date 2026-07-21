from __future__ import annotations

"""A/B backtest of xG integration in PredictionEngine.

Fits the ACTUAL PredictionEngine (not the heuristic MatchOutcomeModel) on the
xG-enriched foundation, PRESERVING home/away (the existing evaluation harnesses
build neutral fixtures and use MatchOutcomeModel, so they can't measure this).

Expanding-window temporal folds over the xG-covered window (2014/15+). For each
test season, train on all covered matches strictly before it, then compare two
arms trained on the identical window:
  - baseline  : use_xg=False (goals-only rolling form)
  - treatment : use_xg=True  (adds rolling pre-match xG-for/against form)

Metrics on the held-out test matches: 1X2 RPS / log-loss / Brier, Over2.5 and
BTTS log-loss/Brier, and the H/D/A calibration bias (predicted mean prob minus
realized outcome rate) — the +3.5% away-win bias is the headline structural
problem to watch.

No elo_history is passed, matching the competition layer's production path
(internal ELO deferred as endogenous). Both arms are therefore ELO-free and the
comparison is apples-to-apples.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402

DEFAULT_MATCHES = "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv"
# Foundation season labels that Understat xG covers.
COVERED_SEASONS = [
    "2014-2015", "2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020",
    "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026",
]
DEFAULT_TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025"]


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rps_1x2(p_home: np.ndarray, p_draw: np.ndarray, p_away: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Ranked Probability Score for the ordered {home, draw, away} outcome."""
    o_home = (outcome == "home").astype(float)
    o_draw = (outcome == "draw").astype(float)
    cum_p1 = p_home
    cum_p2 = p_home + p_draw
    cum_o1 = o_home
    cum_o2 = o_home + o_draw
    return 0.5 * ((cum_p1 - cum_o1) ** 2 + (cum_p2 - cum_o2) ** 2)


def _binary_logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def score_arm(preds: pd.DataFrame) -> dict:
    p_home = preds["p_home_win"].to_numpy(float)
    p_draw = preds["p_draw"].to_numpy(float)
    p_away = preds["p_away_win"].to_numpy(float)
    outcome = preds["actual_outcome"].to_numpy()
    y_home = (outcome == "home").astype(float)
    y_draw = (outcome == "draw").astype(float)
    y_away = (outcome == "away").astype(float)

    actual_prob = np.where(outcome == "home", p_home, np.where(outcome == "draw", p_draw, p_away))
    logloss = -np.log(np.clip(actual_prob, 1e-9, 1.0))
    brier = (p_home - y_home) ** 2 + (p_draw - y_draw) ** 2 + (p_away - y_away) ** 2
    rps = rps_1x2(p_home, p_draw, p_away, outcome)

    over = preds["actual_over_25"].to_numpy(float)
    btts = preds["actual_btts"].to_numpy(float)
    return {
        "n": int(len(preds)),
        "rps": float(rps.mean()),
        "logloss_1x2": float(logloss.mean()),
        "brier_1x2": float(brier.mean()),
        "acc_1x2": float((np.argmax(np.c_[p_home, p_draw, p_away], axis=1)
                          == np.where(outcome == "home", 0, np.where(outcome == "draw", 1, 2))).mean()),
        "logloss_over25": float(_binary_logloss(over, preds["p_over_25"].to_numpy(float)).mean()),
        "brier_over25": float(((preds["p_over_25"].to_numpy(float) - over) ** 2).mean()),
        "logloss_btts": float(_binary_logloss(btts, preds["p_btts"].to_numpy(float)).mean()),
        # Calibration bias: predicted mean prob minus realized rate (per class).
        "bias_home": float(p_home.mean() - y_home.mean()),
        "bias_draw": float(p_draw.mean() - y_draw.mean()),
        "bias_away": float(p_away.mean() - y_away.mean()),
        "pred_lambda_home": float(preds["lambda_home"].mean()),
        "pred_lambda_away": float(preds["lambda_away"].mean()),
    }


def run_fold(train: pd.DataFrame, test: pd.DataFrame, use_xg: bool, **engine_kwargs) -> pd.DataFrame:
    engine = PredictionEngine(use_xg=use_xg, **engine_kwargs)
    engine.fit(train, team_rows=None, elo_history=None)
    rows = []
    for _, r in test.iterrows():
        pred = engine.predict_match(
            str(r["home_team"]), str(r["away_team"]),
            competition=str(r.get("competition", "unknown")),
            neutral=bool(r.get("neutral", 0)),
        )
        rows.append({
            "match_id": r.get("match_id"),
            "p_home_win": pred.p_home_win, "p_draw": pred.p_draw, "p_away_win": pred.p_away_win,
            "lambda_home": pred.lambda_home, "lambda_away": pred.lambda_away,
            "p_over_25": pred.p_over_25, "p_btts": pred.p_btts,
            "actual_outcome": r["actual_outcome"],
            "actual_over_25": r["actual_over_25"], "actual_btts": r["actual_btts"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", default=DEFAULT_MATCHES)
    parser.add_argument("--test-seasons", nargs="*", default=DEFAULT_TEST_SEASONS)
    parser.add_argument("--covered-only", action="store_true", default=True,
                        help="Train and test only on matches that have xG (isolates xG's effect).")
    parser.add_argument("--out", default="data/processed/enriched/understat_xg/xg_ab_backtest.json")
    args = parser.parse_args()

    df = pd.read_csv(_resolve(args.matches), low_memory=False)
    df = df[df["season"].isin(COVERED_SEASONS)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["home_goals", "away_goals", "home_xg", "away_xg", "home_npxg", "away_npxg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if args.covered_only and "xg_available" in df.columns:
        df = df[df["xg_available"] == True].copy()  # noqa: E712
    df = df.dropna(subset=["home_goals", "away_goals", "date"]).sort_values("date").reset_index(drop=True)
    hg = df["home_goals"].astype(int); ag = df["away_goals"].astype(int)
    df["actual_outcome"] = np.where(hg > ag, "home", np.where(hg < ag, "away", "draw"))
    df["actual_over_25"] = ((hg + ag) > 2.5).astype(int)
    df["actual_btts"] = ((hg > 0) & (ag > 0)).astype(int)

    print(f"Loaded {len(df)} covered matches, {df['date'].min().date()}..{df['date'].max().date()}", flush=True)

    fold_results = []
    for season in args.test_seasons:
        test = df[df["season"] == season]
        train = df[df["date"] < test["date"].min()]
        if len(test) == 0 or len(train) < 500:
            print(f"skip {season}: test={len(test)} train={len(train)}", flush=True)
            continue
        t0 = time.time()
        base_preds = run_fold(train, test, use_xg=False)
        base = score_arm(base_preds)
        treat_preds = run_fold(train, test, use_xg=True)
        treat = score_arm(treat_preds)
        dt = time.time() - t0
        fold_results.append({"season": season, "train_n": int(len(train)), "test_n": int(len(test)),
                             "baseline": base, "treatment": treat})
        print(f"\n=== {season}  (train {len(train)}, test {len(test)}, {dt:.0f}s) ===", flush=True)
        for k in ["rps", "logloss_1x2", "brier_1x2", "logloss_over25", "logloss_btts", "bias_home", "bias_draw", "bias_away"]:
            d = treat[k] - base[k]
            print(f"  {k:16s} base={base[k]:+.5f}  xg={treat[k]:+.5f}  delta={d:+.5f}", flush=True)

    # Pooled summary (sample-weighted by test_n).
    if fold_results:
        def pooled(arm: str, key: str) -> float:
            num = sum(f[arm][key] * f["test_n"] for f in fold_results)
            den = sum(f["test_n"] for f in fold_results)
            return num / den if den else float("nan")
        print("\n===== POOLED (weighted by test_n) =====", flush=True)
        pooled_summary = {}
        for k in ["rps", "logloss_1x2", "brier_1x2", "acc_1x2", "logloss_over25", "brier_over25",
                  "logloss_btts", "bias_home", "bias_draw", "bias_away"]:
            b = pooled("baseline", k); t = pooled("treatment", k)
            pooled_summary[k] = {"baseline": b, "treatment": t, "delta": t - b}
            print(f"  {k:16s} base={b:+.5f}  xg={t:+.5f}  delta={t-b:+.5f}", flush=True)

        out = _resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"folds": fold_results, "pooled": pooled_summary}, indent=2), encoding="utf-8")
        print(f"\nWROTE {out}", flush=True)


if __name__ == "__main__":
    main()
