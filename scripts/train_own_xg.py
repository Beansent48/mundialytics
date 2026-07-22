from __future__ import annotations

"""Train OUR OWN shot-xG model on the 534k Understat shots.

Motivation (measured): Understat's xG DRIFTS vs goals (goals/xG ratio ~0.99 in
2021 -> ~0.92 in 2025/26 — they changed their model over time), which injects
nonstationarity into every xG-derived feature (the deployed lambda-rescale is a
patch for it). A home-built xG — one fixed mapping shot->P(goal) — is consistent
across the whole history and fully ours.

Protocol: fit ONLY on shots before 2020-07-01 (pre-dating every test fold), apply
everywhere. Features: distance & goal-mouth angle (from normalized x/y), body
part, situation. Own goals excluded (xG=0). Outputs per-shot our_xg and a
per-(game,team) aggregate for the harness. Diagnostics: AUC, calibration by xG
bin, and the per-season goals/xG ratio for OURS vs Understat's (drift check).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "data/external/advanced/understat/understat_shots.csv"
OUT = ROOT / "data/processed/own_xg_team_match.csv"
FIT_BEFORE = pd.Timestamp("2020-07-01")
PITCH_L, PITCH_W, GOAL_W = 105.0, 68.0, 7.32


def features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.to_numeric(df["location_x"], errors="coerce")
    y = pd.to_numeric(df["location_y"], errors="coerce")
    dx = (1.0 - x) * PITCH_L
    dy = (y - 0.5).abs() * PITCH_W
    dist = np.sqrt(dx ** 2 + dy ** 2)
    # angle subtended by the goal mouth
    num = GOAL_W * dx
    den = dx ** 2 + dy ** 2 - (GOAL_W / 2) ** 2
    angle = np.arctan2(num, den)
    angle = np.where(angle < 0, angle + np.pi, angle)
    F = pd.DataFrame({"dist": dist, "angle": angle,
                      "minute": pd.to_numeric(df["minute"], errors="coerce").fillna(45)})
    for col, cats in [("body_part", ["Right Foot", "Left Foot", "Head"]),
                      ("situation", ["Open Play", "From Corner", "Set Piece", "Direct Freekick", "Penalty"])]:
        v = df[col].astype(str)
        for c in cats:
            F[f"{col}_{c}"] = (v == c).astype(float)
    return F


def main() -> None:
    sh = pd.read_csv(SHOTS)
    sh["date"] = pd.to_datetime(sh["date"], errors="coerce")
    sh = sh.dropna(subset=["date", "location_x", "location_y"])
    own_goal = sh["result"].astype(str).eq("OwnGoal")
    sh = sh[~own_goal].copy()
    sh["goal"] = sh["result"].astype(str).eq("Goal").astype(int)
    sh["u_xg"] = pd.to_numeric(sh["xg"], errors="coerce").fillna(0.0)

    F = features(sh)
    fit_mask = (sh["date"] < FIT_BEFORE).to_numpy()
    m = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.08,
                                       min_samples_leaf=200, random_state=42)
    m.fit(F[fit_mask], sh["goal"].to_numpy()[fit_mask])
    sh["our_xg"] = m.predict_proba(F)[:, 1]

    # ── diagnostics ────────────────────────────────────────────────────────────
    hold = ~fit_mask  # 2020+ (never seen in training)
    print(f"train shots {fit_mask.sum()}, holdout {hold.sum()}")
    print(f"AUC   ours {roc_auc_score(sh.goal[hold], sh.our_xg[hold]):.4f} | "
          f"understat {roc_auc_score(sh.goal[hold], sh.u_xg[hold]):.4f}")
    print(f"holdout: goals {sh.goal[hold].mean():.4f} | ours mean {sh.our_xg[hold].mean():.4f} | "
          f"understat mean {sh.u_xg[hold].mean():.4f}")
    # calibration by bin (holdout)
    bins = [0, .05, .1, .2, .35, .6, 1.01]
    b = pd.cut(sh.our_xg[hold], bins)
    cal = sh[hold].groupby(b, observed=True).agg(n=("goal", "size"), pred=("our_xg", "mean"), emp=("goal", "mean"))
    print("\ncalibration (ours, holdout):")
    print(cal.round(3).to_string())
    # drift check: per-season goals/xG ratio, ours vs understat (team-match level)
    print("\nseason   goals/OURS  goals/UNDERSTAT   (flat ~1.00 = no drift)")
    for sn, g in sh.groupby("season"):
        r_o = g.goal.sum() / max(g.our_xg.sum(), 1e-9)
        r_u = g.goal.sum() / max(g.u_xg.sum(), 1e-9)
        print(f"{sn}     {r_o:6.3f}      {r_u:6.3f}")

    # ── per-(game, team) aggregate for the harness ────────────────────────────
    agg = sh.groupby(["game_id", "team"]).agg(our_xg=("our_xg", "sum"), u_xg=("u_xg", "sum"),
                                              goals=("goal", "sum")).reset_index()
    agg.to_csv(OUT, index=False)
    print(f"\nWROTE {OUT} ({len(agg)} team-match rows)")


if __name__ == "__main__":
    main()
