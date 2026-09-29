"""LOSO comparison of lineup / absence signals on top of the deployed chain.

Base: weekly-refit walk-forward components (scripts/squad_value/wf_refit.py) with the
DEPLOYED squad-value shift. Every signal enters as the same total-goals-preserving tilt as the
deployed lineup pass: s = sum_k theta_k * (x_home,k - x_away,k), lambdas lh*e^{s/2}, la*e^{-s/2}.
Leave-one-season-out over six seasons; the bar is 5/6 seasons better and O/U 2.5 not worse.

Signals:
    us_cnt   deployed definition on Understat lineups (share of regulars not starting)
    tm_*     scripts/lineup_pass/build_tm_absence.py (Transfermarkt lineups + values)
    susp_*   scripts/lineup_pass/build_suspensions.py (bans predicted from cards, morning-known)
    elo_*    scripts/elo_blend/build_clubelo_pre.py (ClubElo pre-match ratings)

    python scripts/lineup_pass/eval_signals.py [SPEC ...] [--fit-all]   # --fit-all: the deployable constant
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, "src")
from mundialytics.features.lineup_absence import absent_share, regular_xi  # noqa: E402
from mundialytics.statistical_core.prediction_engine import DEPLOYED_CLUB_ENGINE_KWARGS as K  # noqa: E402
from mundialytics.statistical_core.schemas import canonical_name  # noqa: E402

FIT_ALL = "--fit-all" in sys.argv
WANT = [a for a in sys.argv[1:] if not a.startswith("--")]
sys.argv = ["x"]
exec(open("scripts/squad_value/fit_shift.py").read().split("r0 = rps_of")[0])  # d, trio, LH, LA, dv, has, Y, S
WORK = "data/external/transfermarkt/work/"
SV = K["squad_value_shift"]
sh = np.where(has, SV["beta"] * dv + SV["kappa"] * stretch, 0.0)
L_H, L_A = LH * np.exp(sh / 2), LA * np.exp(-sh / 2)
P0, O0 = trio(L_H, L_A)
over = (d.hg + d.ag > 2.5).values
d["date"] = pd.to_datetime(d.date)
ALL = np.ones(len(d), bool)


def rps(P, m):
    return (((np.cumsum(P, 1) - np.cumsum(Y[m], 1)) ** 2)[:, :2].sum(1)) / 2


print(f"base RPS (deployed squad value, no lineup) {rps(P0, ALL).mean():.5f}  n={len(d)}", flush=True)


# ── signals, keyed (date, team) ────────────────────────────────────────────────
def understat_absence():
    cache = WORK + "us_absence.pkl"
    if os.path.exists(cache):
        return pd.read_pickle(cache)
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
    rows = []
    for (_, team), g in u.groupby(["season", "team"]):
        for dt in sorted(g.date.unique()):
            regs = regular_xi(g, before=dt)
            if regs is not None:
                rows.append({"date": pd.Timestamp(dt), "team": team,
                             "us_cnt": absent_share(regs, set(g.loc[(g.date == dt) & g.starter, "player"]))})
    out = pd.DataFrame(rows)
    out.to_pickle(cache)
    return out


def attach(tab, cols, prefix=""):
    """Map a (date, team) table onto d's home/away sides; +-1 day tolerance for TZ offsets."""
    tab = tab.copy()
    tab["date"] = pd.to_datetime(tab.date)
    key = {}
    for off in (1, -1, 0):  # exact date written last so it wins
        for r in tab.itertuples(index=False):
            key[(r.date + pd.Timedelta(days=off), r.team)] = r
    out = {}
    for side in ("home", "away"):
        recs = [key.get((dt, t)) for dt, t in zip(d.date, d[side])]
        pres = np.array([r is not None for r in recs])
        for c in cols:
            out[(side, prefix + c)] = np.array([getattr(r, c) if r is not None else 0.0 for r in recs], float)
        out[(side, prefix + "_present")] = pres
    return out


F = {}
F.update(attach(understat_absence(), ["us_cnt"]))
if os.path.exists(WORK + "tm_absence.csv"):
    tm = pd.read_csv(WORK + "tm_absence.csv")
    F.update(attach(tm, [c for c in tm.columns if c not in ("date", "team")], "tm_"))
if os.path.exists(WORK + "suspensions.csv"):
    su = pd.read_csv(WORK + "suspensions.csv")
    F.update(attach(su, [c for c in su.columns if c not in ("date", "team")], "susp_"))
if os.path.exists(WORK + "clubelo_pre.csv"):
    el = pd.read_csv(WORK + "clubelo_pre.csv")
    F.update(attach(el.fillna(0.0), ["elo_live", "elo_frozen"], "cl_"))
    # a side with no rating: drop the Elo term for the match rather than compare to 0
    ok = (el.set_index(["date", "team"]).elo_live.notna())
    for side in ("home", "away"):
        k = pd.MultiIndex.from_arrays([d.date.dt.strftime("%Y-%m-%d"), d[side]])
        okk = ok.copy()
        okk.index = pd.MultiIndex.from_arrays([pd.to_datetime(el.date).dt.strftime("%Y-%m-%d"), el.team])
        F[(side, "elo_ok")] = okk.reindex(k).fillna(False).values.astype(bool)
    elo_ok = F[("home", "elo_ok")] & F[("away", "elo_ok")]
    for c in ("cl_elo_live", "cl_elo_frozen"):
        for side in ("home", "away"):
            F[(side, c)] = np.where(elo_ok, F[(side, c)], 0.0)
    print(f"ClubElo on both sides: {elo_ok.mean():.1%}")
for side in ("home", "away"):
    print(f"{side}: understat {F[(side, '_present')].mean():.1%}"
          + (f"  tm {F[(side, 'tm__present')].mean():.1%}" if (side, "tm__present") in F else "")
          + (f"  susp {F[(side, 'susp__present')].mean():.1%}" if (side, "susp__present") in F else ""))


def diff(name):
    if name == "stretch":      # the base's own log(lh/la): lets a strength signal replace, not add
        return np.log(L_H / L_A) * (elo_ok if "elo_ok" in globals() else 1.0)
    return F[("home", name)] - F[("away", name)]


def tilt(th, X, m):
    s = X[m] @ th
    return trio(L_H[m] * np.exp(s / 2), L_A[m] * np.exp(-s / 2))


def loso(cols, mask=ALL, label=None):
    X = np.column_stack([diff(c) for c in cols])
    delta, dll, ths = np.zeros(len(d)), [], []
    for s in seasons:
        tr, te = mask & (S != s), mask & (S == s)
        th = minimize(lambda t: rps(tilt(t, X, tr)[0], tr).mean(), np.zeros(len(cols)), method="Nelder-Mead",
                      options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 4000}).x
        P1, O1 = tilt(th, X, te)
        delta[te] = rps(P1, te) - rps(P0[te], te)
        y = over[te]
        ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        dll.append(ll(O1) - ll(O0[te]))
        ths.append(th)
    per = pd.Series(delta[mask]).groupby(S[mask]).mean()
    print(f"{label or '+'.join(cols):34s} dRPS/match {delta[mask].mean():+.5f}  better {(per < 0).sum()}/6  "
          f"OU dLL {np.mean(dll):+.5f}  theta {np.round(np.mean(ths, 0), 3)}  n={mask.sum()}", flush=True)
    if FIT_ALL:
        th = minimize(lambda t: rps(tilt(t, X, mask)[0], mask).mean(), np.zeros(len(cols)), method="Nelder-Mead",
                      options={"xatol": 1e-5, "fatol": 1e-11, "maxiter": 4000}).x
        print(f"{'':34s} ALL-SEASON theta = {np.round(th, 4)}", flush=True)
    return delta


SPECS = {
    # 1: count vs value weighting
    "count_us": ["us_cnt"],
    "count_tm": ["tm_abs_cnt"],
    "valshare_tm": ["tm_abs_val"],
    "dlogxi_tm": ["tm_dlog_xi"],
    "count+dlogxi_tm": ["tm_abs_cnt", "tm_dlog_xi"],
    # 2: what a morning injury/suspension feed could know vs rotation
    "unav_cnt": ["tm_unav_cnt"],
    "unav_val": ["tm_unav_val"],
    "bench_cnt": ["tm_bench_cnt"],
    "unav+bench_cnt": ["tm_unav_cnt", "tm_bench_cnt"],
    "unav+bench_val": ["tm_unav_val", "tm_bench_val"],
    "susp_cnt": ["susp_susp_cnt"],
    "susp_val": ["susp_susp_val"],
    "inj_cnt": ["susp_inj_cnt"],
    "inj_val": ["susp_inj_val"],
    "morning_val": ["susp_morning_val"],
    "inj+susp_val": ["susp_inj_val", "susp_susp_val"],
    "inj_nb_cnt": ["susp_inj_nb_cnt"],
    "inj2_cnt": ["susp_inj2_cnt"],
    "unav_val+inj_cnt": ["tm_unav_val", "susp_inj_cnt"],
    # 4: ClubElo on top of the engine (Elo difference / 400, with and without the stretch term)
    "elo_live": ["cl_elo_live"],
    "elo_live+stretch": ["cl_elo_live", "stretch"],
    "elo_frozen+stretch": ["cl_elo_frozen", "stretch"],
}
both_tm = F.get(("home", "tm__present"), np.zeros(len(d), bool)) & F.get(("away", "tm__present"), np.zeros(len(d), bool))
both_us = F[("home", "_present")] & F[("away", "_present")]
common = both_tm & both_us
print(f"\ncommon coverage (both sides, both sources): {common.mean():.1%}\n")
for name, cols in SPECS.items():
    if WANT and name not in WANT:
        continue
    if not all(c == "stretch" or ("home", c) in F for c in cols):
        continue
    loso(cols, label=f"{name} [all]")
    if any(c.startswith("tm_") for c in cols) or name == "count_us":
        loso(cols, mask=common, label=f"{name} [common]")
