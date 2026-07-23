from __future__ import annotations

"""Elo -> EVENT lambdas mapping (corners/yellows/fouls/shots/sot per side),
the cross-league backbone for full European match analysis.

    log lam_side = c_m + hfa_m * is_home + b_m * (eloTeam - eloOpp)/400

Fitted on the big-5 foundation with pre-match ClubElo (17k matches, cached
histories). Walk-forward folds measure how much of the team-props edge an
Elo-only model keeps vs the league-mean baseline — the honest number for
matches where we lack team event history (any non-big5 European side).
Outputs data/processed/elo_event_calibration.json (constants + NB dispersion
per market, total and side).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from calibrate_elo_lambda import FD_TO_CLUBELO, clubelo_name, fetch_history  # noqa: E402

MARKETS = {
    "corners": ("home_corners", "away_corners", [8.5, 9.5, 10.5]),
    "yellows": ("home_yellow_cards", "away_yellow_cards", [3.5, 4.5, 5.5]),
    "fouls":   ("home_fouls", "away_fouls", [21.5, 23.5]),
    "shots":   ("home_shots", "away_shots", [22.5, 24.5]),
    "sot":     ("home_sot", "away_sot", [7.5, 8.5]),
}
TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]


def prob_over(lam, line, disp):
    k = int(np.floor(line))
    lam = np.clip(lam, 0.2, 60.0)
    if disp <= 1.05:
        return 1.0 - poisson.cdf(k, lam)
    r = lam / (disp - 1.0)
    return 1.0 - nbinom.cdf(k, r, 1.0 / disp)


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> None:
    found = pd.read_csv(ROOT / "data/processed/foundation_big5_multi_season.csv", low_memory=False)
    found = found[found["season"] >= "2016-2017"].copy()
    found["date"] = pd.to_datetime(found["date"], errors="coerce")

    teams = sorted(set(found.home_team) | set(found.away_team))
    hist = {}
    for t in teams:
        h = fetch_history(clubelo_name(t))
        if h is not None:
            h = h.copy()
            h["From"] = pd.to_datetime(h["From"], errors="coerce")
            h["To"] = pd.to_datetime(h["To"], errors="coerce")
            hist[t] = h.dropna(subset=["From", "To"])

    def elo_at(team, when):
        h = hist.get(team)
        if h is None:
            return None
        r = h[(h.From <= when) & (h.To >= when)]
        return float(r.Elo.iloc[0]) if len(r) else None

    rows = []
    for r in found.itertuples(index=False):
        eh, ea = elo_at(r.home_team, r.date), elo_at(r.away_team, r.date)
        if eh is None or ea is None:
            continue
        rows.append((r.season, r.competition, r.date, eh, ea,
                     *(getattr(r, c) for m in MARKETS.values() for c in m[:2])))
    cols = ["season", "competition", "date", "eh", "ea"] + \
           [c for m in MARKETS.values() for c in m[:2]]
    d = pd.DataFrame(rows, columns=cols)
    d["d400"] = (d.eh - d.ea) / 400.0
    print(f"matches with Elo: {len(d)}", flush=True)

    out = {"model": "log lam_side = c + hfa*is_home + b*(eloT-eloO)/400", "markets": {}}
    for mk, (hc, ac, lines) in MARKETS.items():
        m = d.dropna(subset=[hc, ac]).copy()
        m[hc] = pd.to_numeric(m[hc], errors="coerce")
        m[ac] = pd.to_numeric(m[ac], errors="coerce")
        m = m.dropna(subset=[hc, ac])

        def fit_c(df):
            X = np.concatenate([np.column_stack([np.ones(len(df)), df.d400]),
                                np.column_stack([np.zeros(len(df)), -df.d400])])
            y = np.concatenate([df[hc], df[ac]])
            reg = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(X, y)
            return float(reg.intercept_), float(reg.coef_[0]), float(reg.coef_[1])

        res = {ln: {"m": [], "b": []} for ln in lines}
        for s in TEST_SEASONS:
            tr = m[m.season < s]
            te = m[m.season == s]
            if len(tr) < 3000 or len(te) == 0:
                continue
            c, hfa, b = fit_c(tr)
            tt = (tr[hc] + tr[ac]).astype(float)
            disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 1.02, 3.0))
            lg = tr.assign(t=tt).groupby("competition")["t"].mean()
            lam_tot = (np.exp(c + hfa + b * te.d400) + np.exp(c - b * te.d400)).to_numpy()
            base = te["competition"].map(lg).fillna(float(tt.mean())).to_numpy()
            act = (te[hc] + te[ac]).astype(float).to_numpy()
            for ln in lines:
                y = (act > ln).astype(float)
                res[ln]["m"].append((bll(y, prob_over(lam_tot, ln, disp)), len(te), s))
                res[ln]["b"].append((bll(y, prob_over(base, ln, disp)), len(te), s))
        pool = lambda a: sum(x * n for x, n, _ in a) / max(sum(n for _, n, _ in a), 1)
        avg_delta = float(np.mean([pool(res[ln]["m"]) - pool(res[ln]["b"]) for ln in lines]))
        deltas = " | ".join(f"O{ln} {pool(res[ln]['m'])-pool(res[ln]['b']):+.4f}" for ln in lines)
        print(f"{mk:8s}: {deltas}  (avg {avg_delta:+.4f})", flush=True)

        c, hfa, b = fit_c(m)
        # cards/fouls are STYLE, not strength: when the Elo slope HURT vs the
        # league-mean baseline, zero it — the honest cross-league fallback is
        # the global level + home effect only
        if avg_delta > 0:
            b = 0.0
        tt = (m[hc] + m[ac]).astype(float)
        sv = pd.concat([m[hc], m[ac]]).astype(float)
        out["markets"][mk] = {
            "c": c, "hfa": hfa, "b": b, "elo_slope_validated": bool(avg_delta <= 0),
            "disp_total": float(np.clip(tt.var() / max(tt.mean(), 1e-9), 1.0, 3.0)),
            "disp_side": float(np.clip(sv.var() / max(sv.mean(), 1e-9), 0.9, 3.0)),
            "lines": lines,
        }

    (ROOT / "data/processed/elo_event_calibration.json").write_text(json.dumps(out, indent=2))
    print("\nWROTE elo_event_calibration.json")
    for mk, v in out["markets"].items():
        print(f"  {mk}: c={v['c']:.3f} hfa={v['hfa']:.3f} b={v['b']:.3f} "
              f"disp={v['disp_total']:.2f}/{v['disp_side']:.2f}")


if __name__ == "__main__":
    main()
