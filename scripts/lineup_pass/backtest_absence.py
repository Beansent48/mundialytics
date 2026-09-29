"""Backtest of the lineup-pass signal with the exact production definition.

Regulars = the 11 players with the most STARTS in the team's last 10 league matches of the
SAME season (ESPN only has the current season; at least 5 played, else no signal).
absent = share of those 11 not in today's starting XI. The tilt keeps total goals:
lh*e^{s/2}, la*e^{-s/2}, s = theta*(absent_home - absent_away).

Base: weekly-refit walk-forward (scripts/squad_value/wf_refit.py components) with the
deployed squad-value shift on the FRESH squad-value history. LOSO over six seasons, then
theta fitted on all six (the deployed constant).

    python scripts/lineup_pass/backtest_absence.py
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, "src")
from mundialytics.features.lineup_absence import absent_share, regular_xi  # noqa: E402
from mundialytics.statistical_core.schemas import canonical_name  # noqa: E402

BETA, KAPPA = float(sys.argv[1]), float(sys.argv[2])   # squad-value shift in force
sys.argv = ["x"]
exec(open("scripts/squad_value/fit_shift.py").read().split("r0 = rps_of")[0])  # d, trio, LH, LA, dv, has, Y, S

# ── the signal, from Understat lineups (starts only, same season) ──────────────
um = pd.read_csv("data/processed/understat_team_match_xg.csv",
                 usecols=["home_team", "home_team_fd", "away_team", "away_team_fd"])
u2c = {**dict(zip(um.home_team, um.home_team_fd)), **dict(zip(um.away_team, um.away_team_fd))}
u = pd.read_csv("data/external/advanced/understat/understat_player_match.csv",
                usecols=["season", "game", "team", "player_id", "position"])
u["date"] = pd.to_datetime(u.game.str[:10])
u["team"] = u.team.map(u2c).map(canonical_name)
u = u.dropna(subset=["team"])
u = u[u.date >= "2020-07-01"]
u["starter"] = u.position != "Sub"
u = u.rename(columns={"player_id": "player"})
absent = {}
for (season, team), g in u.groupby(["season", "team"]):
    for dt in sorted(g.date.unique()):
        regs = regular_xi(g, before=dt)
        if regs is None:
            continue
        starters = set(g.loc[(g.date == dt) & g.starter, "player"])
        absent[(pd.Timestamp(dt), team)] = absent_share(regs, starters)
d["date"] = pd.to_datetime(d.date)
ah = np.array([absent.get((dt, t), 0.0) for dt, t in zip(d.date, d.home)])
aa = np.array([absent.get((dt, t), 0.0) for dt, t in zip(d.date, d.away)])
print(f"signal present: home {np.mean([(dt, t) in absent for dt, t in zip(d.date, d.home)]):.1%}, "
      f"mean absent share {ah[ah > 0].mean():.3f}")

sh = np.where(has, BETA * dv + KAPPA * stretch, 0.0)
L_H, L_A = LH * np.exp(sh / 2), LA * np.exp(-sh / 2)
P0, O0 = trio(L_H, L_A)


def rps(P, m):
    return (((np.cumsum(P, 1) - np.cumsum(Y[m], 1)) ** 2)[:, :2].sum(1)) / 2


def tilt(th, m):
    s = th[0] * (ah[m] - aa[m])
    return trio(L_H[m] * np.exp(s / 2), L_A[m] * np.exp(-s / 2))


over = (d.hg + d.ag > 2.5).values
dr, dl, ths = [], [], []
for s in seasons:
    tr, te = S != s, S == s
    th = minimize(lambda t: rps(tilt(t, tr)[0], tr).mean(), np.zeros(1), method="Nelder-Mead",
                  options={"xatol": 1e-4, "fatol": 1e-9}).x
    P1, O1 = tilt(th, te)
    dr.append(rps(P1, te).mean() - rps(P0[te], te).mean())
    y = over[te]
    ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    dl.append(ll(O1) - ll(O0[te]))
    ths.append(th[0])
    print(f"  hold {s}: theta {th[0]:+.3f}  dRPS {dr[-1]:+.5f}", flush=True)
dr = np.array(dr)
print(f"LOSO dRPS {dr.mean():+.5f}  better {(dr < 0).sum()}/6  OU dLL {np.mean(dl):+.5f}  "
      f"RPS {rps(P0, np.ones(len(d), bool)).mean():.5f} -> {rps(P0, np.ones(len(d), bool)).mean() + dr.mean():.5f}")
allm = np.ones(len(d), bool)
th = minimize(lambda t: rps(tilt(t, allm)[0], allm).mean(), np.zeros(1), method="Nelder-Mead").x
print(f"ALL-SEASON theta = {th[0]:.4f}")
