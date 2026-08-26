from __future__ import annotations

"""Sharper follow-up to test_lineup_oracle.py: does a STAR's absence predict?

The oracle test measured lineup quality as team_strength() over the XI — an
average, which dilutes exactly the event that should matter most: the best
player not starting. Averaging an 88-rated star down to a 75-rated deputy moves
an 11-man mean by ~1 point. This script tests the sharp version instead:

  for each team-match, was the team's top regular starter absent from the XI?

and asks whether THAT moves outcomes, and whether it adds anything on top of
the deployed model out-of-fold. Same oracle framing (post-match knowledge), so
it is again an upper bound on what a pre-match lineup feed could buy.

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
from test_lineup_oracle import FOUND, NAMEMAP, PM, PREDS, TEAMMATCH  # noqa: E402

ROLL = 10          # prior matches used to decide who a team's regulars are
TOP_N = 3          # how many "stars" we track per team


def main() -> None:
    model = PlayerStrengthModel().fit()
    prof = model.profiles_
    namemap = json.loads(NAMEMAP.read_text(encoding="utf-8"))

    pm = pd.read_csv(PM, usecols=["game_id", "team", "player", "position"], low_memory=False)
    start = pm[pm["position"] != "Sub"].copy()
    start["key"] = start["player"].map(namemap)
    start = start.dropna(subset=["key"])
    start["ovr"] = start["key"].map(lambda k: prof[k].overall if k in prof else np.nan)
    start = start.dropna(subset=["ovr"])

    tm = pd.read_csv(TEAMMATCH)[["provider_match_id", "date", "home_team", "away_team",
                                 "home_team_fd", "away_team_fd"]]
    tm = tm.rename(columns={"provider_match_id": "game_id"}).drop_duplicates("game_id")
    tm["date"] = pd.to_datetime(tm["date"], errors="coerce")
    start = start.merge(tm[["game_id", "date"]], on="game_id", how="inner").dropna(subset=["date"])

    # who are each team's stars, judged ONLY on prior matches?
    print("identifying each team's regular stars from prior matches...", flush=True)
    recs = []
    for team, g in start.groupby("team", sort=False):
        games = g.sort_values("date").groupby("game_id", sort=False)
        hist: list[set] = []          # recent XIs
        for gid, xi in games:
            date = xi["date"].iloc[0]
            present = set(xi["key"])
            if len(hist) >= 5:
                # a "regular" = started often recently; rank those by rating
                counts: dict[str, int] = {}
                for h in hist[-ROLL:]:
                    for k in h:
                        counts[k] = counts.get(k, 0) + 1
                regs = [k for k, c in counts.items() if c >= max(3, len(hist[-ROLL:]) * 0.5)]
                stars = sorted(regs, key=lambda k: -prof[k].overall)[:TOP_N]
                if stars:
                    missing = [s for s in stars if s not in present]
                    recs.append({
                        "game_id": gid, "team": team, "date": date,
                        "n_star_out": len(missing),
                        "top_star_out": int(stars[0] not in present),
                        "star_ovr_lost": float(sum(prof[s].overall for s in missing)),
                    })
            hist.append(present)
    sa = pd.DataFrame(recs)
    print(f"  team-matches with a star baseline: {len(sa):,}")
    print(f"  top star absent in {sa.top_star_out.mean():.1%} of them; "
          f"mean stars out {sa.n_star_out.mean():.2f}")

    sa = sa.drop(columns=["date"]).merge(tm, on="game_id", how="inner")
    h = sa[sa.team == sa.home_team][["game_id", "date", "home_team_fd", "away_team_fd",
                                     "n_star_out", "top_star_out", "star_ovr_lost"]]
    h = h.rename(columns={"n_star_out": "h_nout", "top_star_out": "h_top",
                          "star_ovr_lost": "h_lost"})
    a = sa[sa.team == sa.away_team][["game_id", "n_star_out", "top_star_out", "star_ovr_lost"]]
    a = a.rename(columns={"n_star_out": "a_nout", "top_star_out": "a_top",
                          "star_ovr_lost": "a_lost"})
    lu = h.merge(a, on="game_id", how="inner")
    lu["home_team"] = lu["home_team_fd"].astype(str).str.lower().str.strip()
    lu["away_team"] = lu["away_team_fd"].astype(str).str.lower().str.strip()
    lu["date"] = pd.to_datetime(lu["date"]).dt.normalize()

    preds = pd.read_csv(PREDS)
    found = pd.read_csv(FOUND, low_memory=False)
    found["date"] = pd.to_datetime(found["date"], errors="coerce")
    m = preds.merge(found[["match_id", "date", "home_team", "away_team"]],
                    on="match_id", how="left").dropna(subset=["date"])
    m["date"] = m["date"].dt.normalize()
    cols = ["date", "home_team", "away_team", "h_nout", "h_top", "h_lost",
            "a_nout", "a_top", "a_lost"]
    m = m.merge(lu[cols], on=["date", "home_team", "away_team"], how="inner")
    m = m.merge(load_odds(), on=["date", "home_team", "away_team"], how="left")
    print(f"\nevaluation set: {len(m):,} matches")

    P = m[["ph", "pd", "pa"]].to_numpy()
    P = P / P.sum(axis=1, keepdims=True)
    y = np.where(m.hg > m.ag, 0, np.where(m.hg == m.ag, 1, 2))
    m = m.reset_index(drop=True)
    m["real_home"] = (y == 0).astype(float)

    # ── does a star absence move the actual result? ──────────────────────────
    print("\n=== ¿MUEVE EL RESULTADO QUE FALTE UNA ESTRELLA? ===")
    print(f"  {'situacion':38s} {'n':>5s} {'p_home MODELO':>14s} {'gana local REAL':>16s}")
    scen = [
        ("ninguna estrella fuera (ambos)", (m.h_nout == 0) & (m.a_nout == 0)),
        ("local pierde su MEJOR jugador",   (m.h_top == 1) & (m.a_top == 0)),
        ("visitante pierde su MEJOR",       (m.a_top == 1) & (m.h_top == 0)),
        ("local pierde 2+ estrellas",       (m.h_nout >= 2) & (m.a_nout == 0)),
        ("visitante pierde 2+ estrellas",   (m.a_nout >= 2) & (m.h_nout == 0)),
    ]
    for lbl, sel in scen:
        s = m[sel]
        if len(s) < 60:
            print(f"  {lbl:38s} {len(s):5d}  (muestra insuficiente)")
            continue
        print(f"  {lbl:38s} {len(s):5d} {s.ph.mean():14.3f} {s.real_home.mean():16.3f}")

    # ── does it add anything out-of-fold? ────────────────────────────────────
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    base_X = np.log(np.clip(P, 1e-9, 1))
    extra = m[["h_nout", "h_top", "h_lost", "a_nout", "a_top", "a_lost"]].to_numpy()
    extra = np.c_[extra, extra[:, 2] - extra[:, 5], extra[:, 0] - extra[:, 3]]
    full_X = np.c_[base_X, extra]

    def oof(X):
        out = np.zeros_like(P)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            out[te] = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], y[tr]).predict_proba(X[te])
        return out

    r_base, r_full = rps3(y, oof(base_X)), rps3(y, oof(full_X))
    print("\n=== ¿APORTA AL MODELO? (OOF, oraculo) ===")
    print(f"  solo modelo              RPS {r_base:.4f}")
    print(f"  modelo + ausencias       RPS {r_full:.4f}   ({r_full - r_base:+.4f})")
    ok = m[["oh", "od", "oa"]].notna().all(axis=1).to_numpy()
    if ok.any():
        inv = np.c_[1 / m.oh, 1 / m.od, 1 / m.oa]
        M = inv / inv.sum(axis=1, keepdims=True)
        r_mkt = rps3(y[ok], M[ok])
        print(f"  Bet365                   RPS {r_mkt:.4f}")
        print(f"  cierra {(r_base - r_full) / (r_base - r_mkt) * 100:.0f}% de la brecha")


if __name__ == "__main__":
    main()
