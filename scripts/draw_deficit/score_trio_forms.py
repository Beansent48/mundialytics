from __future__ import annotations

"""The 1X2 sharpening removes draw mass that reality puts there. What does it cost?

THE FINDING THIS STARTS FROM (scripts/bivariate_goals/README.md, 2026-10-18).
Over the six walk-forward seasons the real draw rate is 0.2540 and the matrix's
RAW trio says 0.2560 -- calibrated. The PUBLISHED trio, after the gamma = 1.3
sharpening, says 0.2318: a gap of +0.0222 with a standard error of 0.0041,
**z = +5.4, and positive in 6 of 6 seasons**. Since `coherent_markets` (2026-09)
reweights every outcome region of the scoreline matrix onto that published trio,
the missing draw mass is now subtracted from every exact score, BTTS and HT/FT
number on the page.

WHY THE SHARPENING EXISTS, AND WHY IT MIGHT BE STALE. The raw engine is
under-confident -- "says 64% when reality is 71%", reliability compressed toward
the middle by regularization and blend averaging -- and p^gamma renormalized
fixes it. But gamma was chosen by a JOINT gamma/rho LOFO scan that picked
gamma = 1.3 with rho = -0.17 (prediction_engine.py). rho was later smoothed to
-0.06 (the "always 1-1" fix) and gamma was never re-scanned. On top of that, one
exponent is being asked to do two jobs: p^gamma scales BOTH log-odds against the
draw, so correcting under-confidence on the home-vs-away contrast necessarily
drains the draw, whether or not the draw needed it.

THE STRUCTURE THAT DECIDES THIS. With the raw matrix M held fixed, coherence
makes the exact-score and HT/FT log losses equal the 1X2 log loss plus a
constant the trio map cannot touch:

    log P(exact score) = log M[cell] - log M[region] + log t[region]
    log P(HT/FT path)  = log path    - log ft[res]   + log t[res]

so only the last term moves. The script asserts it (`identity`). Three
consequences: the right objective for the trio map is the trio's own LOG LOSS,
not RPS; whatever helps the 1X2 helps exact score and HT/FT by exactly as much;
and only over/under and BTTS can disagree, because they are unions of cells that
cross outcome regions.

THE FORMS (all keep coherence except `decouple`):
    gamma      the deployed family, 1 parameter, re-scanned
    two_axis   2 parameters: alpha on the draw-vs-rest log-odds, beta on the
               home-vs-away contrast. alpha = beta recovers `gamma`;
               alpha = 1 sharpens the contrast and leaves the draw axis alone
    vector     textbook vector scaling, 3 parameters: T on log p plus a free
               intercept per outcome. Aggregate calibration per class comes out
               for free, so this is the arm that fully closes the deficit
    decouple   publish the deployed trio for the 1X2 but reweight the MATRIX to
               a trio of its own. Breaks the coherence invariant on purpose, so
               the reintroduced disagreement is measured and reported

RESULT (2026-10-18): the deficit is real and the fix is a BETTER MAP, not
decoupling -- `decouple` is strictly dominated. See
scripts/draw_deficit/README.md. Nothing here is wired into production: the
engine's defaults are untouched and every form lives in this file.

    python scripts/draw_deficit/score_trio_forms.py all
    python scripts/draw_deficit/score_trio_forms.py gap --data-root PATH
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_bivariate_goals as X  # noqa: E402  the bivariate harness, reused

K = X.K
MARKETS = ("1x2_rps", "1x2_ll", "o25", "btts", "cs", "htft")
LOG = "data/processed/logs/predictions_log.csv"
EPS = 1e-12


# ── the forms: raw trio -> published trio ─────────────────────────────────────

def f_gamma(t: np.ndarray, p) -> np.ndarray:
    """The deployed family: p_i^gamma, renormalized."""
    return X.sharpen(t, float(np.atleast_1d(p)[0]))


def _axes(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(s, d): the draw-vs-rest and home-vs-away halves of the log-odds.

    u = log(p_h/p_d), v = log(p_a/p_d), s = (u+v)/2, d = (u-v)/2. Then p^gamma
    is exactly (s, d) -> (gamma*s, gamma*d), which is the whole point: one
    exponent cannot move the contrast without moving the draw.
    """
    q = np.clip(t, EPS, 1.0)
    u, v = np.log(q[:, 0] / q[:, 1]), np.log(q[:, 2] / q[:, 1])
    return (u + v) / 2.0, (u - v) / 2.0


def _from_axes(s: np.ndarray, d: np.ndarray) -> np.ndarray:
    e = np.stack([np.exp(s + d), np.ones_like(s), np.exp(s - d)], axis=1)
    return e / e.sum(axis=1, keepdims=True)


def f_two_axis(t: np.ndarray, p) -> np.ndarray:
    """alpha on the draw axis, beta on the home-vs-away contrast."""
    alpha, beta = float(p[0]), float(p[1])
    s, d = _axes(t)
    return _from_axes(alpha * s, beta * d)


def f_vector(t: np.ndarray, p) -> np.ndarray:
    """Vector scaling: T*log p_i + b_i, b_draw pinned to 0 as the reference."""
    T, bh, ba = float(p[0]), float(p[1]), float(p[2])
    z = T * np.log(np.clip(t, EPS, 1.0)) + np.array([bh, 0.0, ba])[None, :]
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


FORMS = {
    "gamma": (f_gamma, np.array([1.3]), [(0.6, 2.0)]),
    "two_axis": (f_two_axis, np.array([1.0, 1.3]), [(0.4, 2.0), (0.6, 2.2)]),
    "vector": (f_vector, np.array([1.3, 0.0, 0.0]), [(0.6, 2.0), (-1.0, 1.0), (-1.0, 1.0)]),
}
DEPLOYED = ("gamma", np.array([K["sharpen_gamma_1x2"]]))


# ── fitting: the trio's own log loss, which the identity says is the objective ─

def fit_form(raw: np.ndarray, y: np.ndarray, form: str) -> np.ndarray:
    fn, x0, bounds = FORMS[form]

    def nll(p):
        return X.mll(y, fn(raw, np.atleast_1d(p)))

    if len(x0) == 1:
        r = minimize_scalar(lambda g: nll([g]), bounds=bounds[0], method="bounded",
                            options={"xatol": 1e-4})
        best = np.array([float(r.x)])
    else:
        r = minimize(nll, x0, method="Nelder-Mead", bounds=bounds,
                     options={"xatol": 1e-4, "fatol": 1e-9, "maxiter": 2000})
        best = np.asarray(r.x, dtype=float)
    return best if nll(best) <= nll(x0) else x0


# ── scoring ───────────────────────────────────────────────────────────────────

def raw_matrix(df: pd.DataFrame, rho: float | None = None) -> np.ndarray:
    """The deployed scoreline matrix BEFORE the coherent reweighting."""
    return X.joint(df.lh.to_numpy(float), df.la.to_numpy(float), "dixon_coles",
                   K["outcome_rho"] if rho is None else rho,
                   temper=K["goal_temper"], fix_marginals=False)


def score_trio(df: pd.DataFrame, M: np.ndarray, t_pub: np.ndarray,
               t_matrix: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Every market from one published trio (and optionally a different one for
    the matrix, which is what `decouple` does)."""
    t_m = t_pub if t_matrix is None else t_matrix
    m = X.reweight_to_trio(M, t_m)
    paths = X.htft_paths(df.lh.to_numpy(float), df.la.to_numpy(float),
                         "independent", 0.0, t_m)
    sc = X.score_block(df, m, paths, t_pub)
    sc["disagree"] = np.abs(X.trio_of(m) - t_pub).max(axis=1)
    sc["pdraw"] = t_pub[:, 1]
    return sc


def fold_table(df: pd.DataFrame, base: dict, arms: dict[str, dict], label: str) -> dict:
    """Per-arm deltas against the baseline, with the per-fold win count."""
    seasons = sorted(df.season.unique())
    b_all = X.aggregate(df, base)
    b_season = {s: X.aggregate(g, X.take(base, g.index.to_numpy()))
                for s, g in df.groupby("season")}
    print(f"\n{label}")
    print(f"  {'arm':30s} " + "  ".join(f"{k:>10s}" for k in MARKETS))
    out = {}
    for name, sc in arms.items():
        keep = ~np.isnan(sc["pdraw"])
        sub = df[keep].reset_index(drop=True)
        idx = np.flatnonzero(keep)
        tot = X.aggregate(sub, X.take(sc, idx))
        bt = X.aggregate(sub, X.take(base, idx))
        wins, n = {k: 0 for k in MARKETS}, 0
        for s, g in sub.groupby("season"):
            n += 1
            a = X.aggregate(g, X.take(sc, idx[g.index.to_numpy()]))
            for k in MARKETS:
                if a[k] < b_season[s][k]:
                    wins[k] += 1
        print(f"  {name:30s} " + "  ".join(f"{tot[k] - bt[k]:+10.5f}" for k in MARKETS))
        print(f"  {'':30s} " + "  ".join(f"{str(wins[k]) + '/' + str(n):>10s}"
                                        for k in MARKETS))
        out[name] = {"delta": {k: tot[k] - bt[k] for k in MARKETS}, "wins": wins,
                     "folds": n, "n": len(sub)}
    print(f"\n  baseline on all {len(df):,}: " +
          "  ".join(f"{k} {b_all[k]:.5f}" for k in MARKETS))
    print(f"  ({len(seasons)} seasons; deltas vs the deployed gamma = "
          f"{K['sharpen_gamma_1x2']}; negative is better)")
    return out


# ── 1. the gap ────────────────────────────────────────────────────────────────

def cmd_gap(df: pd.DataFrame, data_root: Path) -> None:
    print("=" * 78)
    print("1. THE DRAW GAP, PER SEASON AND AGAINST THE LONG RUN")
    print("=" * 78)
    y = df.outcome.to_numpy(int)
    pub = df[["ph", "pd", "pa"]].to_numpy(float)
    raw = X.trio_of(raw_matrix(df))
    print(f"\n{'fold':11s} {'n':>5s} {'obs X':>7s} {'raw X':>7s} {'pub X':>7s} "
          f"{'gap':>8s} {'z':>6s} {'obs H':>7s} {'pub H':>7s} {'obs A':>7s} {'pub A':>7s}")
    rows = [(s, g.index.to_numpy()) for s, g in df.groupby("season")]
    for s, i in rows + [("ALL", np.arange(len(df)))]:
        o = np.array([(y[i] == k).mean() for k in range(3)])
        p, r = pub[i].mean(0), raw[i].mean(0)
        se = np.sqrt(p[1] * (1 - p[1]) / len(i))
        print(f"{s:11s} {len(i):5d} {o[1]:7.4f} {r[1]:7.4f} {p[1]:7.4f} "
              f"{o[1] - p[1]:+8.4f} {(o[1] - p[1]) / se:+6.2f} "
              f"{o[0]:7.4f} {p[0]:7.4f} {o[2]:7.4f} {p[2]:7.4f}")
    o, p = np.array([(y == k).mean() for k in range(3)]), pub.mean(0)
    for k, nm in ((0, "home"), (2, "away")):
        se = np.sqrt(p[k] * (1 - p[k]) / len(df))
        print(f"  {nm} is OVER-priced by {p[k] - o[k]:+.4f} (z {(o[k] - p[k]) / se:+.2f}) "
              "-- the sharpening pushes the draw's mass into both tails, near-equally")

    f = pd.read_csv(data_root / X.FOUND_REL, low_memory=False,
                    usecols=["season", "home_goals", "away_goals"]).dropna()
    d = (f.home_goals == f.away_goals)
    print(f"\n  long-run draw rate: {d.mean():.4f} over {len(f):,} matches and "
          f"{f.season.nunique()} seasons")
    print(f"  the six scored:     {d[f.season.isin(df.season.unique())].mean():.4f}")
    print(f"  published:          {p[1]:.4f}")
    print("  So the published trio is short of the LONG-RUN rate by "
          f"{d.mean() - p[1]:.4f}, not just of a draw-heavy sample.")


# ── 2. can the live log see it? ───────────────────────────────────────────────

def cmd_live(df: pd.DataFrame, data_root: Path) -> None:
    print("\n" + "=" * 78)
    print("2. IS IT VISIBLE GOING FORWARD, IN THE REAL LOG?")
    print("=" * 78)
    path = ROOT / LOG
    if not path.exists():
        print(f"\n  no log at {LOG}")
        return
    log = pd.read_csv(path, low_memory=False)
    one = (log[log.mercado == "1X2"].sort_values("logged_at")
           .drop_duplicates(["home", "away", "fecha"], keep="last"))
    f = pd.read_csv(data_root / X.FOUND_REL, low_memory=False,
                    usecols=["date", "home_team", "away_team", "home_goals", "away_goals"])
    f["fecha"] = f.date.astype(str).str[:10]
    m = (one.merge(f.rename(columns={"home_team": "home", "away_team": "away"}),
                   on=["home", "away", "fecha"], how="inner")
         .dropna(subset=["home_goals", "away_goals", "p_draw"]))
    print(f"\n  {len(one)} logged 1X2 rows, {one.p_draw.notna().sum()} of them carrying the "
          "trio (only since PR #25),")
    print(f"  {len(m)} of those settled.")
    if m.empty:
        return
    yl = np.where(m.home_goals > m.away_goals, 0,
                  np.where(m.home_goals == m.away_goals, 1, 2))
    P = m[["p_home", "p_draw", "p_away"]].to_numpy(float)
    P = P / P.sum(axis=1, keepdims=True)
    print(f"\n  {'outcome':8s} {'priced':>8s} {'happened':>9s} {'gap':>8s} {'z':>6s}")
    for k, nm in enumerate(("home", "draw", "away")):
        pk, ok = P[:, k], (yl == k).astype(float)
        var = float((pk * (1 - pk)).sum())
        print(f"  {nm:8s} {pk.mean():8.4f} {ok.mean():9.4f} {ok.mean() - pk.mean():+8.4f} "
              f"{(ok - pk).sum() / np.sqrt(var):+6.2f}")
    pk = P[:, 1]
    need = (0.0222 / np.sqrt(pk.mean() * (1 - pk.mean()))) ** -2
    print(f"\n  Same sign as the backtest, and nowhere near decidable: detecting a "
          f"+0.0222 gap\n  at |z| >= 3 needs about {need * 9:,.0f} settled matches and there are "
          f"{len(m)}.")
    print("\n  AND THE MONITOR CANNOT SEE IT AT ALL. report_market_drift.py's")
    print("  calibration_drift() opens with `ev[~ev.mercado_id.isin(PICK_ONE)]`, and")
    print("  PICK_ONE = {'1X2', 'ht_1x2', 'ht_ft'} -- a pick-one market logs one")
    print("  selection, not three probabilities, so the 1X2's own per-outcome")
    print("  calibration is excluded by construction. The drift report would never")
    print("  have raised this, however long it ran.")


# ── 3. the identity ──────────────────────────────────────────────────────────

def cmd_identity(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("3. THE IDENTITY THAT DECIDES WHICH OBJECTIVE TO FIT")
    print("=" * 78)
    M = raw_matrix(df)
    raw = X.trio_of(M)
    print("\nWith the raw matrix fixed, does (exact score - 1X2) stay constant as the")
    print("trio map changes? If it does, the trio map is the whole story for exact")
    print("score and HT/FT, and the objective is the trio's log loss.\n")
    print(f"  {'map':26s} {'1X2 ll':>9s} {'cs ll':>10s} {'cs - 1X2':>10s} "
          f"{'htft ll':>10s} {'htft - 1X2':>11s}")
    cs_gap, ht_gap = [], []
    for name, t in (("gamma 1.0 (raw)", f_gamma(raw, [1.0])),
                    ("gamma 1.3 (deployed)", f_gamma(raw, [1.3])),
                    ("gamma 1.6", f_gamma(raw, [1.6])),
                    ("two_axis 1.0 / 1.3", f_two_axis(raw, [1.0, 1.3])),
                    ("vector 1.3 / +.2 / -.2", f_vector(raw, [1.3, 0.2, -0.2]))):
        a = X.aggregate(df, score_trio(df, M, t))
        cs_gap.append(a["cs"] - a["1x2_ll"])
        ht_gap.append(a["htft"] - a["1x2_ll"])
        print(f"  {name:26s} {a['1x2_ll']:9.5f} {a['cs']:10.5f} "
              f"{a['cs'] - a['1x2_ll']:10.5f} {a['htft']:10.5f} "
              f"{a['htft'] - a['1x2_ll']:11.5f}")
    sp = max(np.ptp(cs_gap), np.ptp(ht_gap))
    print(f"\n  spread of those constants: {sp:.2e}  ->  "
          f"{'CONSTANT, as derived' if sp < 1e-6 else 'NOT constant -- the derivation is wrong'}")
    if sp >= 1e-6:
        raise SystemExit("identity failed")
    print("  So: fit the trio map on 1X2 LOG LOSS. RPS is not the objective (it is")
    print("  reported as a guard), and exact score and HT/FT follow for free.")


# ── 4. the scan ───────────────────────────────────────────────────────────────

def cmd_scan(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("4. THE SCAN: WHAT DOES EACH FAMILY WANT?")
    print("=" * 78)
    M = raw_matrix(df)
    raw = X.trio_of(M)
    y = df.outcome.to_numpy(int)
    obs = float((y == 1).mean())
    hg, ag = df.hg.to_numpy(int), df.ag.to_numpy(int)
    tot, both = (hg + ag >= 3).astype(float), ((hg > 0) & (ag > 0)).astype(float)

    def row(label, t):
        m = X.reweight_to_trio(M, t)
        gm = X.goal_markets(m, hg, ag)
        pdw = float(t[:, 1].mean())
        print(f"  {label:22s} {X.rps3(y, t):9.5f} {X.mll(y, t):9.5f} "
              f"{X.bll(tot, gm['o25']):9.5f} {X.bll(both, gm['btts']):9.5f} "
              f"{pdw:8.4f} {obs / pdw:9.4f}")

    print(f"\nin sample, all six seasons. observed draw rate {obs:.4f}\n")
    print(f"  {'':22s} {'RPS':>9s} {'1X2 ll':>9s} {'o25 ll':>9s} {'btts ll':>9s} "
          f"{'P(draw)':>8s} {'O/E draw':>9s}")
    print("  -- gamma (the deployed family) --")
    for g in (1.00, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40):
        row(f"gamma {g:.2f}" + ("  DEPLOYED" if abs(g - 1.3) < 1e-9 else ""),
            f_gamma(raw, [g]))
    print("  -- two_axis: alpha on the draw axis, beta on the contrast --")
    for a in (1.00, 1.10, 1.20):
        for b in (1.10, 1.20, 1.30, 1.40):
            row(f"two_axis {a:.2f} / {b:.2f}", f_two_axis(raw, [a, b]))
    print("  -- fitted on all six (the ceiling for each family) --")
    for form in FORMS:
        p = fit_form(raw, y, form)
        row(f"{form} {np.round(p, 4)}", FORMS[form][0](raw, p))


# ── 5. out of sample, per fold ────────────────────────────────────────────────

def fit_per_fold(df: pd.DataFrame, raw: np.ndarray, form: str,
                 regime: str) -> dict[str, np.ndarray | None]:
    seasons = sorted(df.season.unique())
    y = df.outcome.to_numpy(int)
    out: dict[str, np.ndarray | None] = {}
    for i, s in enumerate(seasons):
        if regime == "expanding":
            sel = df.season.isin(seasons[:i]).to_numpy()
        elif regime == "loso":
            sel = (df.season != s).to_numpy()
        else:
            sel = (df.season == s).to_numpy()
        out[s] = fit_form(raw[sel], y[sel], form) if sel.any() else None
    return out


def cmd_validate(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 78)
    print("5. OUT OF SAMPLE, SEASON BY SEASON")
    print("=" * 78)
    M = raw_matrix(df)
    raw = X.trio_of(M)
    seasons = sorted(df.season.unique())
    base = score_trio(df, M, f_gamma(raw, [K["sharpen_gamma_1x2"]]))

    nan3 = np.full(3, np.nan)
    arms: dict[str, dict] = {}
    params: dict[str, dict] = {}
    # The candidate. NOTHING NEW IS FITTED: it reuses the deployed exponent and
    # changes only which axis it is applied to. alpha = 1 is not a fitted value,
    # it comes from section 1 -- the raw trio's draw rate is already calibrated
    # (O/E 0.992), so the draw axis needs no correction and the contrast does.
    arms["two_axis FIXED (1.0, deployed gamma)"] = score_trio(
        df, M, f_two_axis(raw, [1.0, K["sharpen_gamma_1x2"]]))
    for form in FORMS:
        for regime in ("expanding", "loso", "oracle"):
            fits = fit_per_fold(df, raw, form, regime)
            t = np.empty((len(df), 3))
            for s, g in df.groupby("season"):
                i = g.index.to_numpy()
                t[i] = nan3 if fits[s] is None else FORMS[form][0](raw[i], fits[s])
            arms[f"{form} [{regime}]"] = score_trio(df, M, np.nan_to_num(t, nan=1 / 3),
                                                    t_matrix=None)
            arms[f"{form} [{regime}]"]["pdraw"] = t[:, 1]
            params[f"{form} [{regime}]"] = fits
    res = fold_table(df, base, arms, "coherence kept -- a better trio map:")

    print("\n  draw calibration, per fold (observed / published):")
    y = df.outcome.to_numpy(int)
    cand = arms["two_axis FIXED (1.0, deployed gamma)"]
    print(f"    {'fold':11s} {'deployed':>10s} {'candidate':>10s}")
    for s, g in df.groupby("season"):
        i = g.index.to_numpy()
        o = float((y[i] == 1).mean())
        print(f"    {s:11s} {o / base['pdraw'][i].mean():10.4f} "
              f"{o / cand['pdraw'][i].mean():10.4f}")
    o = float((y == 1).mean())
    print(f"    {'ALL':11s} {o / base['pdraw'].mean():10.4f} "
          f"{o / cand['pdraw'].mean():10.4f}")

    print("\n  fitted parameters per fold:")
    for name, fits in params.items():
        if "expanding" in name or "oracle" in name:
            vals = "  ".join("      n/a" if fits[s] is None
                             else "/".join(f"{v:+.3f}" for v in fits[s]) for s in seasons)
            print(f"    {name:26s} {vals}")

    # decouple: publish the deployed trio, give the matrix a trio of its own
    dec: dict[str, dict] = {}
    t_pub = f_gamma(raw, [K["sharpen_gamma_1x2"]])
    for gm_ in (1.0, 1.2):
        sc = score_trio(df, M, t_pub, t_matrix=f_gamma(raw, [gm_]))
        dec[f"decouple: matrix gamma {gm_:.1f}"] = sc
    fits = fit_per_fold(df, raw, "gamma", "expanding")
    t = np.empty((len(df), 3))
    for s, g in df.groupby("season"):
        i = g.index.to_numpy()
        t[i] = nan3 if fits[s] is None else f_gamma(raw[i], fits[s])
    sc = score_trio(df, M, t_pub, t_matrix=np.nan_to_num(t, nan=1 / 3))
    sc["pdraw"] = t[:, 1]
    dec["decouple: matrix gamma [expanding]"] = sc
    res_d = fold_table(df, base, dec, "coherence BROKEN on purpose -- `decouple`:")
    print("\n  and what that costs in incoherence (the bug coherence fixed):")
    for name, sc in dec.items():
        d = sc["disagree"][~np.isnan(sc["pdraw"])] if "expanding" in name else sc["disagree"]
        print(f"    {name:38s} |matrix 1X2 - published| mean {d.mean():.4f} "
              f"max {d.max():.4f}")
    print(f"    {'deployed (coherent)':38s} |matrix 1X2 - published| mean "
          f"{base['disagree'].mean():.2e} max {base['disagree'].max():.2e}")
    return {"coherent": res, "decouple": res_d}


# ── 6. does rho move with it? ─────────────────────────────────────────────────

def cmd_rho(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("6. GAMMA WAS FITTED JOINTLY WITH RHO = -0.17. RE-SCAN THE PAIR.")
    print("=" * 78)
    y = df.outcome.to_numpy(int)
    hg, ag = df.hg.to_numpy(int), df.ag.to_numpy(int)
    tot, both = (hg + ag >= 3).astype(float), ((hg > 0) & (ag > 0)).astype(float)
    print("\nexact-score log loss of the coherent matrix (the full-information")
    print("likelihood), each cell with its own trio map fitted in sample:\n")
    print(f"  {'rho':>7s} " + "  ".join(f"{f:>26s}" for f in FORMS))
    for rho in (-0.17, -0.12, -0.06, -0.03, 0.0):
        M = raw_matrix(df, rho=rho)
        raw = X.trio_of(M)
        cells = []
        for form in FORMS:
            p = fit_form(raw, y, form)
            a = X.aggregate(df, score_trio(df, M, FORMS[form][0](raw, p)))
            cells.append(f"{a['cs']:.5f} @ {np.round(p, 2)}")
        print(f"  {rho:+7.2f} " + "  ".join(f"{c:>26s}" for c in cells))
    M = raw_matrix(df)
    raw = X.trio_of(M)
    a = X.aggregate(df, score_trio(df, M, f_gamma(raw, [K["sharpen_gamma_1x2"]])))
    print(f"\n  deployed (rho -0.06, gamma 1.3): cs {a['cs']:.5f}, "
          f"o25 {a['o25']:.5f}, btts {a['btts']:.5f}")
    print("  (rho = -0.17 is what gamma 1.3 was scanned with; it is no longer served.)")


# ── 7. coherence, for every form ──────────────────────────────────────────────

def cmd_coherence(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("7. COHERENCE: EVERY FORM STILL PUBLISHES WHAT ITS GRID SUMS TO")
    print("=" * 78)
    sub = df.iloc[::7].reset_index(drop=True)
    M = raw_matrix(sub)
    raw = X.trio_of(M)
    worst = 0.0
    print(f"\nchecked on {len(sub):,} matches (every 7th)")
    for name, t in (("gamma 1.3 (deployed)", f_gamma(raw, [1.3])),
                    ("gamma 1.2", f_gamma(raw, [1.2])),
                    ("two_axis 1.0 / 1.3", f_two_axis(raw, [1.0, 1.3])),
                    ("vector 1.3 / +.2 / -.2", f_vector(raw, [1.3, 0.2, -0.2]))):
        m = X.reweight_to_trio(M, t)
        e_t = float(np.abs(X.trio_of(m) - t).max())
        e_s = float(np.abs(m.sum(axis=(1, 2)) - 1.0).max())
        paths = X.htft_paths(sub.lh.to_numpy(float), sub.la.to_numpy(float),
                             "independent", 0.0, t)
        ft = np.stack([paths[:, [X.PATHS.index(f"{x}/{z}") for x in "1X2"]].sum(1)
                       for z in "1X2"], axis=1)
        e_h = float(np.abs(ft - t).max())
        worst = max(worst, e_t, e_s, e_h)
        print(f"  {name:24s} max|matrix 1X2 - published| {e_t:.2e}   "
              f"max|HT/FT 1X2 - published| {e_h:.2e}   sum-1 {e_s:.2e}")
    ok = worst < 1e-9
    print(f"\n  {'OK' if ok else 'FAIL'}: worst {worst:.2e} (< 1e-9 required). The fix that")
    print("  wins does NOT need the invariant broken.")
    if not ok:
        raise SystemExit("coherence invariant broken")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="all",
                    choices=["gap", "live", "identity", "scan", "validate", "rho",
                             "coherence", "all"])
    ap.add_argument("--data-root", default=str(ROOT),
                    help="where data/ lives (default: the repo root; a worktree's is "
                         "gitignored and therefore empty)")
    a = ap.parse_args()
    root = Path(a.data_root)
    df = X.load(root)
    if a.cmd in ("gap", "all"):
        cmd_gap(df, root)
    if a.cmd in ("live", "all"):
        cmd_live(df, root)
    if a.cmd in ("identity", "all"):
        cmd_identity(df)
    if a.cmd in ("scan", "all"):
        cmd_scan(df)
    if a.cmd in ("validate", "all"):
        cmd_validate(df)
    if a.cmd in ("rho", "all"):
        cmd_rho(df)
    if a.cmd in ("coherence", "all"):
        cmd_coherence(df)


if __name__ == "__main__":
    main()
