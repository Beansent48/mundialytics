from __future__ import annotations

"""Measure the value of ACTUAL LINEUPS (XI-strength delta) for match prediction.

Feature per team-match (all walk-forward / leakage-safe):
  player quality q  = trailing per-90 xg_chain BEFORE the match, shrunk toward the
                      global mean with a minutes-credibility prior (K=900').
  actual XI quality = mean q of the 11 starters (position != 'Sub').
  baseline quality  = team's trailing-10-match minutes-weighted played-XI quality
                      (shifted — never includes the current match).
  xi_delta          = actual − baseline   (negative = rotation / key absences)

Value test: adjust the DEPLOYED model's cached walk-forward lambdas
  lambda' = lambda * exp(beta * z(xi_delta))
with beta learned leave-one-fold-out, and measure 1X2 RPS / log-loss deltas.
This is the UPPER BOUND of lineup value (real XIs, known ~1h pre-kickoff).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
PM = ROOT / "data/external/advanced/understat/understat_player_match.csv"
TM = ROOT / "data/processed/understat_team_match_xg.csv"          # game_id -> date
PREDS = ROOT / "data/processed/enriched/understat_xg/walkforward_preds.csv"
FOUND = ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv"  # match_id -> provider_match_id
K_MIN = 900.0      # shrinkage prior in minutes (~10 full matches)
BASE_W = 10        # trailing window (team matches) for the baseline XI quality


def build_xi_deltas() -> pd.DataFrame:
    pm = pd.read_csv(PM)
    tm = (pd.read_csv(TM)[["provider_match_id", "date"]]
          .rename(columns={"provider_match_id": "game_id"}).drop_duplicates("game_id"))
    pm = pm.merge(tm, on="game_id", how="left")
    pm["date"] = pd.to_datetime(pm["date"], errors="coerce")
    pm = pm.dropna(subset=["date", "minutes"]).sort_values(["player_id", "date", "game_id"])
    for c in ["minutes", "xg_chain"]:
        pm[c] = pd.to_numeric(pm[c], errors="coerce").fillna(0.0)

    # walk-forward player quality (per-90 xg_chain, shrunk)
    g = pm.groupby("player_id", group_keys=False)
    pm["cmin"] = g["minutes"].apply(lambda s: s.shift(1).cumsum()).fillna(0.0)
    pm["cxgc"] = g["xg_chain"].apply(lambda s: s.shift(1).cumsum()).fillna(0.0)
    global_q = float(pm["xg_chain"].sum() / pm["minutes"].sum() * 90.0)
    raw_q = np.where(pm["cmin"] > 0, pm["cxgc"] / pm["cmin"].clip(lower=1e-9) * 90.0, global_q)
    cred = pm["cmin"] / (pm["cmin"] + K_MIN)
    pm["q"] = cred * raw_q + (1 - cred) * global_q

    pm["started"] = (pm["position"].astype(str) != "Sub").astype(int)
    # diagnostics: starters per team-match should be ~11
    spm = pm.groupby(["game_id", "team"])["started"].sum()
    print(f"starters per team-match: mean {spm.mean():.2f} (should be ~11), pct==11: {(spm == 11).mean():.1%}")

    # per team-match: actual XI quality + played-minutes-weighted quality
    def agg(gr):
        st = gr[gr["started"] == 1]
        wq = float((gr["minutes"] * gr["q"]).sum() / max(gr["minutes"].sum(), 1e-9))
        return pd.Series({"xi_q": float(st["q"].mean()) if len(st) else np.nan, "wq": wq})
    team_match = pm.groupby(["game_id", "team", "date"]).apply(agg, include_groups=False).reset_index()

    team_match = team_match.sort_values(["team", "date", "game_id"])
    team_match["baseline"] = (team_match.groupby("team", group_keys=False)["wq"]
                              .apply(lambda s: s.shift(1).rolling(BASE_W, min_periods=5).mean()))
    team_match["xi_delta"] = team_match["xi_q"] - team_match["baseline"]
    return team_match[["game_id", "team", "date", "xi_q", "baseline", "xi_delta"]]


def main() -> None:
    xi = build_xi_deltas()
    print(f"xi rows: {len(xi)}  (delta std {xi['xi_delta'].std():.4f})")

    # map cached walk-forward predictions -> game_id via provider_match_id
    found = pd.read_csv(FOUND, low_memory=False)[["match_id", "provider_match_id", "home_team", "away_team"]]
    preds = pd.read_csv(PREDS).merge(found, on="match_id", how="left")
    preds["game_id"] = pd.to_numeric(preds["provider_match_id"], errors="coerce")
    # foundation team names differ from Understat names in xi; join via team side using
    # the canonical builder's foundation-named canonical file? xi 'team' is Understat-named.
    # Use the alias map to convert xi team names to foundation names.
    sys.path.insert(0, str(ROOT / "src"))
    from mundialytics.enrichment.understat_team_aliases import to_foundation_name
    xi["team_fd"] = xi["team"].map(to_foundation_name)
    h = xi.rename(columns={"xi_delta": "xi_delta_h"})[["game_id", "team_fd", "xi_delta_h"]]
    a = xi.rename(columns={"xi_delta": "xi_delta_a"})[["game_id", "team_fd", "xi_delta_a"]]
    d = (preds.merge(h, left_on=["game_id", "home_team"], right_on=["game_id", "team_fd"], how="inner")
              .merge(a, left_on=["game_id", "away_team"], right_on=["game_id", "team_fd"], how="inner"))
    d = d.dropna(subset=["xi_delta_h", "xi_delta_a", "lh", "la"])
    print(f"joined matches with lineups + cached preds: {len(d)} / {len(preds)}")

    # standardize deltas on the whole joined set per-fold-safe (folds use train-only z below)
    K = 11; ks = np.arange(K); RHO = -0.07
    def rps_ll(lh, la, sub):
        ph_ = poisson.pmf(ks[:, None], lh[None, :]); pa_ = poisson.pmf(ks[:, None], la[None, :])
        J = ph_[:, None, :] * pa_[None, :, :]
        J[0, 0, :] *= (1 - RHO * lh * la); J[0, 1, :] *= (1 + RHO * lh)
        J[1, 0, :] *= (1 + RHO * la); J[1, 1, :] *= (1 - RHO)
        J = np.clip(J, 0, None); J /= J.sum(axis=(0, 1), keepdims=True)
        hm = (ks[:, None] > ks[None, :]); dm = (ks[:, None] == ks[None, :])
        ph = (J * hm[:, :, None]).sum(axis=(0, 1)); pd_ = (J * dm[:, :, None]).sum(axis=(0, 1))
        P = np.c_[ph, pd_, 1 - ph - pd_]; P = np.clip(P, 1e-9, 1) ** 1.2; P /= P.sum(axis=1, keepdims=True)
        o = np.where(sub.hg > sub.ag, "home", np.where(sub.hg < sub.ag, "away", "draw"))
        y = np.c_[o == "home", o == "draw", o == "away"].astype(float)
        rps = float((0.5 * ((P[:, 0] - y[:, 0]) ** 2 + (P[:, 0] + P[:, 1] - y[:, 0] - y[:, 1]) ** 2)).mean())
        ll = float(-np.log(np.clip((P * y).sum(axis=1), 1e-9, 1)).mean())
        return rps, ll

    seasons = sorted(d["season"].unique())
    grid = np.round(np.arange(0.0, 0.201, 0.02), 3)
    raw_pool, adj_pool = [], []
    print(f"\n{'fold':10s} {'beta':>6s} {'dRPS':>9s} {'dLL':>9s}")
    for s in seasons:
        tr = d[d.season != s]; te = d[d.season == s]
        mu, sd = tr[["xi_delta_h", "xi_delta_a"]].stack().mean(), tr[["xi_delta_h", "xi_delta_a"]].stack().std()
        zh_tr = ((tr.xi_delta_h - mu) / sd).to_numpy(); za_tr = ((tr.xi_delta_a - mu) / sd).to_numpy()
        best_b, best = 0.0, np.inf
        for b in grid:
            r, _ = rps_ll(tr.lh.to_numpy() * np.exp(b * zh_tr), tr.la.to_numpy() * np.exp(b * za_tr), tr)
            if r < best:
                best, best_b = r, b
        zh = ((te.xi_delta_h - mu) / sd).to_numpy(); za = ((te.xi_delta_a - mu) / sd).to_numpy()
        r0 = rps_ll(te.lh.to_numpy(), te.la.to_numpy(), te)
        r1 = rps_ll(te.lh.to_numpy() * np.exp(best_b * zh), te.la.to_numpy() * np.exp(best_b * za), te)
        raw_pool.append((r0, len(te))); adj_pool.append((r1, len(te)))
        print(f"{s:10s} {best_b:6.2f} {r1[0]-r0[0]:+9.5f} {r1[1]-r0[1]:+9.5f}")

    def pool(a, i): return sum(r[i] * n for r, n in a) / sum(n for _, n in a)
    print(f"\nPOOLED: raw RPS {pool(raw_pool,0):.4f} LL {pool(raw_pool,1):.4f}")
    print(f"        adj RPS {pool(adj_pool,0):.4f} LL {pool(adj_pool,1):.4f}")
    print(f"        delta   {pool(adj_pool,0)-pool(raw_pool,0):+.5f}   {pool(adj_pool,1)-pool(raw_pool,1):+.5f}")


if __name__ == "__main__":
    main()
