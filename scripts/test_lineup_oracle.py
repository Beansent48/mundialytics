from __future__ import annotations

"""ORACLE TEST: would knowing the starting XI actually improve our 1X2?

diagnose_disagreement.py concluded the Bet365 gap is information we lack —
lineups, injuries, rotation. Before building any acquisition pipeline, size the
prize with an upper bound: use the XI that ACTUALLY started (post-match
knowledge we would not have pre-match) and ask whether it explains our errors.

If even this oracle does not help, no lineup feed will, and the honest move is
to stop chasing 1X2. If it helps a lot, the pipeline is worth building and this
number is the ceiling to aim at.

Method: reconstruct each match's starting XI from Understat player-match rows
(position != "Sub"), score it with our own PlayerStrengthModel, compare it to
that team's typical recent XI (prior matches only), and test out-of-fold whether
those deltas add anything on top of the deployed model's own probabilities.

EVALUATION ONLY — nothing here is deployed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mundialytics.statistical_core.player_strength import PlayerStrengthModel  # noqa: E402
from diagnose_market_gap import load_odds, rps3  # noqa: E402

PM = ROOT / "data/external/advanced/understat/understat_player_match.csv"
NAMEMAP = ROOT / "data/processed/understat_player_name_map.json"
TEAMMATCH = ROOT / "data/processed/understat_team_match_xg.csv"
FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
PREDS = ROOT / "data/processed/enriched/understat_xg/walkforward_preds_deployed.csv"
MIN_MATCHED = 8          # of 11 starters, else the XI estimate is too noisy
TYPICAL_WINDOW = 10      # prior matches defining a team's "usual" XI strength


def build_xi_strength() -> pd.DataFrame:
    """(game_id, team) -> attack/defense index of the XI that actually started."""
    model = PlayerStrengthModel().fit()
    prof = model.profiles_
    namemap = json.loads(NAMEMAP.read_text(encoding="utf-8"))

    pm = pd.read_csv(PM, usecols=["game_id", "team", "player", "position"], low_memory=False)
    start = pm[pm["position"] != "Sub"].copy()
    start["key"] = start["player"].map(namemap)
    start = start.dropna(subset=["key"])

    rows = []
    for (gid, team), g in start.groupby(["game_id", "team"], sort=False):
        squad = [prof[k] for k in g["key"] if k in prof]
        if len(squad) < MIN_MATCHED:
            continue
        st = model.team_strength(squad)
        rows.append({"game_id": gid, "team": team, "n_matched": len(squad),
                     "att": st["attack_index"], "dfn": st["defense_index"]})
    return pd.DataFrame(rows)


def main() -> None:
    print("reconstructing starting XIs and scoring them...", flush=True)
    xi = build_xi_strength()
    print(f"  XIs scored: {len(xi):,} team-matches "
          f"(median {xi.n_matched.median():.0f}/11 players matched)")

    # attach dates so "typical" can look strictly backwards
    tm = pd.read_csv(TEAMMATCH)[["provider_match_id", "date", "home_team", "away_team",
                                 "home_team_fd", "away_team_fd"]]
    tm = tm.rename(columns={"provider_match_id": "game_id"}).drop_duplicates("game_id")
    tm["date"] = pd.to_datetime(tm["date"], errors="coerce")
    xi = xi.merge(tm, on="game_id", how="inner").dropna(subset=["date"])

    # a team's TYPICAL recent XI strength, from prior matches only
    xi = xi.sort_values("date")
    for col in ("att", "dfn"):
        g = xi.groupby("team")[col]
        xi[f"typ_{col}"] = g.transform(lambda s: s.shift(1).rolling(TYPICAL_WINDOW, min_periods=3).mean())
        xi[f"d_{col}"] = xi[col] - xi[f"typ_{col}"]
    xi = xi.dropna(subset=["d_att", "d_dfn"])
    print(f"  with a usable 'typical' baseline: {len(xi):,}")
    print(f"  delta spread: att std={xi.d_att.std():.2f}, def std={xi.d_dfn.std():.2f}")

    # map understat team -> football-data name, per side
    home = xi[xi["team"] == xi["home_team"]][["game_id", "date", "home_team_fd",
                                              "away_team_fd", "d_att", "d_dfn"]]
    home = home.rename(columns={"d_att": "h_datt", "d_dfn": "h_ddfn"})
    away = xi[xi["team"] == xi["away_team"]][["game_id", "d_att", "d_dfn"]]
    away = away.rename(columns={"d_att": "a_datt", "d_dfn": "a_ddfn"})
    lu = home.merge(away, on="game_id", how="inner")
    lu["home_team"] = lu["home_team_fd"].astype(str).str.lower().str.strip()
    lu["away_team"] = lu["away_team_fd"].astype(str).str.lower().str.strip()
    lu["date"] = pd.to_datetime(lu["date"]).dt.normalize()
    print(f"  matches with BOTH XIs: {len(lu):,}")

    # join our deployed walk-forward predictions + the market
    preds = pd.read_csv(PREDS)
    found = pd.read_csv(FOUND, low_memory=False)
    found["date"] = pd.to_datetime(found["date"], errors="coerce")
    m = preds.merge(found[["match_id", "date", "competition", "home_team", "away_team"]],
                    on="match_id", how="left").dropna(subset=["date"])
    m["date"] = m["date"].dt.normalize()
    m = m.merge(lu[["date", "home_team", "away_team", "h_datt", "h_ddfn",
                    "a_datt", "a_ddfn"]], on=["date", "home_team", "away_team"], how="inner")
    m = m.merge(load_odds(), on=["date", "home_team", "away_team"], how="left")
    print(f"\nevaluation set: {len(m):,} matches "
          f"({m.date.min():%Y-%m-%d} -> {m.date.max():%Y-%m-%d})")

    P = m[["ph", "pd", "pa"]].to_numpy()
    P = P / P.sum(axis=1, keepdims=True)
    y = np.where(m.hg > m.ag, 0, np.where(m.hg == m.ag, 1, 2))

    have_odds = m[["oh", "od", "oa"]].notna().all(axis=1).to_numpy()
    if have_odds.any():
        inv = np.c_[1 / m.oh, 1 / m.od, 1 / m.oa]
        M = inv / inv.sum(axis=1, keepdims=True)
        print(f"  modelo   RPS {rps3(y, P):.4f}")
        print(f"  Bet365   RPS {rps3(y[have_odds], M[have_odds]):.4f} (n={have_odds.sum():,})")

    # ── the test: do the lineup deltas add anything, out-of-fold? ────────────
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    base_X = np.log(np.clip(P, 1e-9, 1))
    lin = m[["h_datt", "h_ddfn", "a_datt", "a_ddfn"]].to_numpy()
    lin = np.c_[lin, lin[:, 0] - lin[:, 2], lin[:, 1] - lin[:, 3]]   # explicit differentials
    full_X = np.c_[base_X, lin]

    def oof(X):
        out = np.zeros_like(P)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            lr = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], y[tr])
            out[te] = lr.predict_proba(X[te])
        return out

    r_base = rps3(y, oof(base_X))
    r_full = rps3(y, oof(full_X))
    print("\n=== ¿APORTA CONOCER EL ONCE? (OOF, oraculo) ===")
    print(f"  solo modelo            RPS {r_base:.4f}")
    print(f"  modelo + once real     RPS {r_full:.4f}   ({r_full - r_base:+.4f})")
    if have_odds.any():
        gap = r_base - rps3(y[have_odds], M[have_odds])
        print(f"  cierra {(r_base - r_full) / gap * 100:.0f}% de la brecha con Bet365")

    # ── is the signal REAL but unexploited, or already captured? ─────────────
    # Our model has no lineup input at all. So if realised home-win rate moves
    # across lineup-delta quintiles while our predicted p_home stays flat, the
    # signal is real and unexploited. If our p_home moves with it, the form
    # features already carry it indirectly and there is nothing left to buy.
    print("\n=== ¿SEÑAL REAL SIN EXPLOTAR, O YA CAPTURADA? ===")
    m["supr"] = (m.h_datt - m.a_datt)
    m["real_home"] = (y == 0).astype(float)
    qs = m["supr"].quantile([0, .2, .4, .6, .8, 1.0]).to_numpy()
    print(f"  {'quintil':9s} {'n':>5s} {'delta once':>18s} {'p_home MODELO':>14s} {'gana local REAL':>16s}")
    mod, real = [], []
    for i in range(5):
        s = m[(m.supr >= qs[i]) & (m.supr <= qs[i + 1])]
        mod.append(s.ph.mean())
        real.append(s.real_home.mean())
        print(f"  Q{i+1:<8d} {len(s):5d} {qs[i]:+8.2f}..{qs[i+1]:+6.2f} "
              f"{s.ph.mean():14.3f} {s.real_home.mean():16.3f}")
    print(f"\n  recorrido del MODELO : {max(mod)-min(mod):.3f}")
    print(f"  recorrido REAL       : {max(real)-min(real):.3f}")
    if max(real) - min(real) > 2 * (max(mod) - min(mod)):
        print("  -> la realidad se mueve MUCHO mas que el modelo: señal real sin explotar")
    else:
        print("  -> el modelo ya sigue el movimiento: la señal esta capturada indirectamente")


if __name__ == "__main__":
    main()
