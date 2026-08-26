from __future__ import annotations

"""Team-event props: corners, yellow cards, fouls, shots, shots on target.

Per market and side, a PoissonRegressor on the validated feature recipe:
  - team rolling event-for windows 5/10/19 + EWMA hl5
  - opponent event-against rollings, is_home
  - ASYM (validated on ALL 5 markets, 2026-07-22): expected-supremacy features
    delta = (gf_ewm + opp_ga_ewm)/2 - (opp_gf_ewm + ga_ewm)/2 and |delta| —
    game-state theory (dominant sides force corners/shots; mismatches change
    fouls/cards). Improved every market/line, ~5/5 folds.
O/U probabilities use Negative Binomial with train-measured dispersion
(corners 1.18, yellows 1.13, fouls 1.59, shots 1.39, SOT 1.19), then a
walk-forward-fitted Platt recalibration per (market, line) — deployed for all
markets EXCEPT yellows (already calibrated; Platt added noise there).

Optional referee feature (EPL only — the one league football-data carries
Referee for): walk-forward referee tendency (expanding mean of match totals,
shrunk n/(n+20)) as an extra feature for yellows/fouls. A/B on EPL: better on
all 6 lines, 5/5 folds. Used when `referee=` is passed to predict_fixture.

Validation numbers live in scripts/backtest_team_props.py,
experiment_corners_asym.py, experiment_team_props_calibration.py,
experiment_referee_epl.py.
"""

import glob
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor

from mundialytics.statistical_core.attack_defense_model import AttackDefenseModel

MARKETS: dict[str, tuple[str, str, list[float]]] = {
    "corners": ("home_corners", "away_corners", [7.5, 8.5, 9.5, 10.5, 11.5]),
    "yellows": ("home_yellow_cards", "away_yellow_cards", [2.5, 3.5, 4.5, 5.5, 6.5]),
    "fouls":   ("home_fouls", "away_fouls", [19.5, 21.5, 23.5, 25.5]),
    "shots":   ("home_shots", "away_shots", [20.5, 22.5, 24.5, 26.5]),
    "sot":     ("home_sot", "away_sot", [6.5, 7.5, 8.5, 9.5]),
}
# TEAM-side lines (validated 2026-07-22: bigger edge than match totals —
# team shots -0.057/-0.070, team corners -0.030/-0.034, team yellows -0.014, all 5/5)
SIDE_LINES = {"corners": [3.5, 4.5, 5.5], "yellows": [1.5, 2.5], "shots": [9.5, 11.5, 13.5]}
# booking points (10/yellow + 25/red): -0.016..-0.021 vs league base 5/5 folds;
# reds at LEAGUE mean (team red tendency measured = noise)
BOOKING_LINES = [30.5, 40.5, 50.5]
WINDOWS = (5, 10, 19)
REF_MARKETS = {"yellows": "ref_yc", "fouls": "ref_foul"}
NO_PLATT = {"yellows"}          # already calibrated; Platt hurt it
LEAGUE_DISP = {"fouls"}         # per-league NB dispersion (validated for fouls only, 5/5)
PLATT_FROM_SEASON = "2016-2017"
# round-5 upgrades (joint-validated per market, 5/5 folds vs previous config):
EWM_HL = {"corners": 12}        # corners want LONG memory (12 > 8 > 5 > 3); others 5
STAKES_MARKETS = {"yellows", "fouls"}   # walk-forward standings features
# MLE strengths prior: corners/yellows only — for shots/sot the tiny totals gain
# (+0.0003) was offset by a side-lines loss (-0.0005); fouls was a wash
ADM_W = {"corners": 0.3, "yellows": 0.3}
ADM_CAPS = {"corners": 20.0, "yellows": 12.0, "fouls": 35.0, "shots": 40.0, "sot": 20.0}


def _prob_over(total_lam: float | np.ndarray, line: float, disp: float) -> np.ndarray:
    """P(total > line); NB with var = disp*mean when over-dispersed, else Poisson."""
    k = int(np.floor(line))
    lam = np.clip(total_lam, 0.2, 40.0)
    if disp > 1.1:
        r = lam / (disp - 1.0)
        return 1.0 - nbinom.cdf(k, r, 1.0 / disp)
    return 1.0 - poisson.cdf(k, lam)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


_RAW_SCAN_CACHE: dict = {}


def _scan_raw_footballdata(root: str | Path) -> pd.DataFrame | None:
    """ONE pass over the raw football-data files with the union of columns the
    referee and red-card loaders need (they used to scan the ~240 CSVs twice)."""
    key = str(Path(root).resolve())
    if key in _RAW_SCAN_CACHE:
        return _RAW_SCAN_CACHE[key]
    want = {"Date", "HomeTeam", "AwayTeam", "Referee", "HY", "AY", "HF", "AF", "HR", "AR"}
    rows = []
    for p in glob.glob(str(Path(root) / "data/raw/football_data/**/*.csv"), recursive=True):
        mdiv = re.search(r"\d{4}_(E0|SP1|D1|I1|F1)\.csv$", p)
        if not mdiv:
            continue
        try:
            df = pd.read_csv(p, encoding="latin-1", on_bad_lines="skip",
                             usecols=lambda c: c in want)
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed")
        df["div"] = mdiv.group(1)
        rows.append(df)
    if not rows:
        _RAW_SCAN_CACHE[key] = None
        return None
    r = pd.concat(rows, ignore_index=True).dropna(subset=["date"])
    r = r.drop_duplicates(subset=["date", "HomeTeam", "AwayTeam"])
    r["home_team"] = r["HomeTeam"].astype(str).str.lower().str.strip()
    r["away_team"] = r["AwayTeam"].astype(str).str.lower().str.strip()
    _RAW_SCAN_CACHE[key] = r
    return r


def load_red_cards(root: str | Path) -> pd.DataFrame | None:
    """Per-match red cards from raw football-data (HR/AR are 100% present in
    all big-5 files; the foundation CSV never kept them)."""
    r = _scan_raw_footballdata(root)
    if r is None or "HR" not in r.columns:
        return None
    r = r.copy()
    r["reds"] = pd.to_numeric(r["HR"], errors="coerce") + pd.to_numeric(r["AR"], errors="coerce")
    return r.dropna(subset=["reds"])[["date", "home_team", "away_team", "reds"]]


def _p_booking_over(lam_y: np.ndarray, disp_y: float, lam_r: np.ndarray, line: float) -> np.ndarray:
    """P(10Y + 25R > line), Y ~ NB, R ~ Poisson, independent grid convolution."""
    lam_y = np.clip(np.asarray(lam_y, float), 0.2, 25)
    lam_r = np.clip(np.asarray(lam_r, float), 0.01, 3)
    ry = lam_y / (disp_y - 1.0)
    out = np.zeros(np.shape(lam_y))
    for r_cnt in range(0, 7):
        pr = poisson.pmf(r_cnt, lam_r)
        thr = np.floor((line - 25 * r_cnt) / 10.0)
        py_over = np.where(thr < 0, 1.0, 1.0 - nbinom.cdf(thr, ry, 1.0 / disp_y))
        out = out + pr * py_over
    return np.clip(out, 0, 1)


def load_referee_rates(root: str | Path) -> pd.DataFrame | None:
    """Walk-forward referee card/foul tendencies from raw football-data (only
    EPL files carry Referee; other rows drop out on the notna filter)."""
    r = _scan_raw_footballdata(root)
    if r is None or "Referee" not in r.columns:
        return None
    # E0 only — matches the validated experiment population exactly (other
    # leagues have sporadic Referee data that would shift the shrinkage mean)
    r = r[r["div"] == "E0"].dropna(subset=["Referee"]).copy()
    if r.empty:
        return None
    r["ref"] = r["Referee"].astype(str).str.strip()
    r["tot_yc"] = pd.to_numeric(r["HY"], errors="coerce") + pd.to_numeric(r["AY"], errors="coerce")
    r["tot_f"] = pd.to_numeric(r["HF"], errors="coerce") + pd.to_numeric(r["AF"], errors="coerce")
    r = r.sort_values("date")
    for src, out in [("tot_yc", "ref_yc"), ("tot_f", "ref_foul")]:
        g = r.groupby("ref", group_keys=False)[src]
        mean_prev = g.apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
        n_prev = g.cumcount()
        glob_mean = r[src].expanding().mean().shift(1).fillna(r[src].mean())
        cred = n_prev / (n_prev + 20.0)
        r[out] = cred * mean_prev.fillna(glob_mean) + (1 - cred) * glob_mean
    return r[["date", "home_team", "away_team", "ref", "ref_yc", "ref_foul"]]


@dataclass
class TeamPropsModel:
    """Fit on the events foundation; predict O/U probabilities for a fixture."""

    seasons_from: str = "2014-2015"
    calibrate: bool = True
    _models: dict = field(default_factory=dict, init=False, repr=False)
    _models_ref: dict = field(default_factory=dict, init=False, repr=False)
    _disp: dict = field(default_factory=dict, init=False, repr=False)
    _league_lam: dict = field(default_factory=dict, init=False, repr=False)
    _team_feats: dict = field(default_factory=dict, init=False, repr=False)
    _platt: dict = field(default_factory=dict, init=False, repr=False)
    _ref_rates: dict = field(default_factory=dict, init=False, repr=False)
    _epl_teams: set = field(default_factory=set, init=False, repr=False)
    _disp_side: dict = field(default_factory=dict, init=False, repr=False)
    _disp_lg: dict = field(default_factory=dict, init=False, repr=False)
    _team_comp: dict = field(default_factory=dict, init=False, repr=False)
    _red_lam: dict = field(default_factory=dict, init=False, repr=False)
    _red_glob: float = field(default=0.4, init=False, repr=False)
    _use_lam: bool = field(default=False, init=False, repr=False)
    _adm: dict = field(default_factory=dict, init=False, repr=False)
    _team_pos: dict = field(default_factory=dict, init=False, repr=False)

    @staticmethod
    def _add_positions(full: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Walk-forward league position + games played BEFORE each match, plus the
        latest (position, played) per team for predict-time stakes features."""
        full = full.sort_values("date").copy()
        rows, latest = [], {}
        for (_, _), g in full.groupby(["competition", "season"], sort=False):
            pts: dict = {}
            played: dict = {}
            for r in g.itertuples(index=False):
                def rank(t):
                    p = pts.get(t, 0)
                    return 1 + sum(1 for v in pts.values() if v > p)
                rh, ra = rank(r.home_team), rank(r.away_team)
                rows.append((r.match_id, rh, ra, played.get(r.home_team, 0)))
                hgo, ago = r.home_goals, r.away_goals
                pts[r.home_team] = pts.get(r.home_team, 0) + (3 if hgo > ago else (1 if hgo == ago else 0))
                pts[r.away_team] = pts.get(r.away_team, 0) + (3 if ago > hgo else (1 if hgo == ago else 0))
                played[r.home_team] = played.get(r.home_team, 0) + 1
                played[r.away_team] = played.get(r.away_team, 0) + 1
                latest[str(r.home_team).lower()] = (rank(r.home_team), played[r.home_team])
                latest[str(r.away_team).lower()] = (rank(r.away_team), played[r.away_team])
        pos = pd.DataFrame(rows, columns=["match_id", "pos_home", "pos_away", "played_home"])
        return full.merge(pos, on="match_id", how="left"), latest

    # ── fitting ────────────────────────────────────────────────────────────────
    def fit(self, matches: pd.DataFrame, referee_data: pd.DataFrame | None = None,
            root: str | Path | None = None,
            lambdas: pd.DataFrame | None = None) -> "TeamPropsModel":
        """`matches`: foundation rows (date/season/teams + event + goal columns).
        `referee_data`: optional output of load_referee_rates (auto-loaded if
        `root` given). `lambdas`: walk-forward engine lambdas per match_id
        (lh/la) — auto-loaded from the caches when `root` is given; powers the
        delta_lam supremacy features (validated better than goals-delta on all
        5 markets, decisive same-eval A/B 2026-07-23)."""
        df = matches.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["season"] >= self.seasons_from]
        if lambdas is None and root is not None:
            parts = []
            for rel in ["data/processed/enriched/understat_xg/walkforward_preds.csv",
                        "data/processed/enriched/understat_xg/walkforward_preds_hist.csv"]:
                p = Path(root) / rel
                if p.exists():
                    parts.append(pd.read_csv(p)[["match_id", "lh", "la"]])
            if parts:
                lambdas = pd.concat(parts, ignore_index=True).drop_duplicates("match_id")
        if lambdas is not None:
            df = df.merge(lambdas[["match_id", "lh", "la"]], on="match_id", how="left")
        else:
            df["lh"] = np.nan
            df["la"] = np.nan
        # delta_lam features only when lambda coverage is real (else goals-delta recipe)
        self._use_lam = bool(df["lh"].notna().mean() > 0.3)
        # walk-forward standings (stakes features for cards/fouls) + predict-time state
        df = df.dropna(subset=["home_goals", "away_goals"])
        df, self._team_pos = self._add_positions(df)
        if referee_data is None and root is not None:
            referee_data = load_referee_rates(root)
        if root is not None:
            reds = load_red_cards(root)
            if reds is not None and "competition" in df.columns:
                rj = df.merge(reds, on=["date", "home_team", "away_team"], how="inner")
                rj = rj.drop_duplicates(subset=["match_id"])
                if len(rj) > 3000:
                    self._red_lam = rj.groupby("competition")["reds"].mean().to_dict()
                    self._red_glob = float(rj["reds"].mean())
        if referee_data is not None:
            latest = referee_data.sort_values("date").groupby("ref").tail(1)
            self._ref_rates = {r.ref: (float(r.ref_yc), float(r.ref_foul))
                              for r in latest.itertuples(index=False)}

        for market, (hc, ac, lines) in MARKETS.items():
            m = df.dropna(subset=[hc, ac, "home_goals", "away_goals", "date"]).copy()
            for c in [hc, ac, "home_goals", "away_goals"]:
                m[c] = pd.to_numeric(m[c], errors="coerce")
            m = m.dropna(subset=[hc, ac, "home_goals", "away_goals"])
            if len(m) < 3000:
                continue
            lr = self._long_rows(m, hc, ac, hl=EWM_HL.get(market, 5))
            feats = self._feature_names(market)
            tr = lr.dropna(subset=feats + ["ev_for"])
            reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
            self._models[market] = reg
            # MLE attack/defense strengths as a stabilizing prior (blend at predict)
            if ADM_W.get(market, 0) > 0:
                adm_tr = (m.rename(columns={hc: "hg2", ac: "ag2"})
                          .drop(columns=["home_goals", "away_goals"])
                          .rename(columns={"hg2": "home_goals", "ag2": "away_goals"}))
                adm = AttackDefenseModel(dixon_coles_rho=0.0, time_decay_half_life=365.0,
                                         goal_cap=ADM_CAPS.get(market, 30.0), max_goals=5)
                self._adm[market] = adm.fit(adm_tr)
            tt = (m[hc] + m[ac]).astype(float)
            self._disp[market] = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
            sv = lr["ev_for"].dropna().astype(float)
            self._disp_side[market] = float(np.clip(sv.var() / max(sv.mean(), 1e-9), 0.9, 3.0))
            if "competition" in m.columns:
                self._league_lam[market] = m.assign(tot=tt).groupby("competition")["tot"].mean().to_dict()
                if market in LEAGUE_DISP:
                    self._disp_lg[market] = (m.assign(tot=tt).groupby("competition")["tot"]
                                             .apply(lambda s2: float(np.clip(s2.var() / max(s2.mean(), 1e-9),
                                                                             1.02, 3.0)))).to_dict()
                if not self._team_comp:
                    last = m.sort_values("date").groupby("home_team")["competition"].last()
                    self._team_comp = {str(t).lower(): c for t, c in last.items()}
            self._team_feats[market] = self._latest_team_state(lr, hl=EWM_HL.get(market, 5))
            if self.calibrate:
                # NO_PLATT only gates TOTAL lines (yellows totals already calibrated);
                # side-line Platt validated positive for every side market incl. yellows.
                # NOTE: Platt is collected on rate-only walk-forward preds; predict applies
                # it to the ADM-blended probs (w=0.3) — approximation, distortion shape is
                # dominated by the shared rate component.
                self._fit_platt(market, m, lr, feats, hc, ac,
                                lines if market not in NO_PLATT else [])
            # referee-augmented model (EPL subset)
            if market in REF_MARKETS and referee_data is not None and "competition" in m.columns:
                epl = m[m["competition"].str.contains("Premier", case=False, na=False)]
                epl = epl.merge(referee_data[["date", "home_team", "away_team", REF_MARKETS[market]]],
                                on=["date", "home_team", "away_team"], how="left")
                epl = epl.drop_duplicates(subset=["match_id"]).dropna(subset=[REF_MARKETS[market]])
                if len(epl) > 2000:
                    lr_e = self._long_rows(epl, hc, ac, extra={"ref_rate": REF_MARKETS[market]},
                                           hl=EWM_HL.get(market, 5))
                    fr = feats + ["ref_rate"]
                    tre = lr_e.dropna(subset=fr + ["ev_for"])
                    self._models_ref[market] = PoissonRegressor(alpha=0.1, max_iter=1000).fit(
                        tre[fr], tre["ev_for"].clip(lower=0))
                    # ref models are EPL-trained: only ever applied to EPL fixtures
                    self._epl_teams |= set(epl["home_team"].astype(str).str.lower())
                    self._epl_teams |= set(epl["away_team"].astype(str).str.lower())
        return self

    def is_epl_fixture(self, home_team: str, away_team: str) -> bool:
        return home_team.lower() in self._epl_teams and away_team.lower() in self._epl_teams

    def _fit_platt(self, market: str, m: pd.DataFrame, lr: pd.DataFrame,
                   feats: list[str], hc: str, ac: str, lines: list[float]) -> None:
        """Walk-forward OOF predictions over past seasons -> per-line Platt (a, b)."""
        seasons = sorted(s for s in m["season"].unique() if s >= PLATT_FROM_SEASON)
        collected: dict[float, list] = {ln: [] for ln in lines}
        side_lines = SIDE_LINES.get(market, [])
        collected_side: dict[float, list] = {ln: [] for ln in side_lines}
        for s in seasons:
            te_m = m[m.season == s]
            s_start = te_m.date.min()
            tr = lr[lr.date < s_start].dropna(subset=feats + ["ev_for"])
            if len(tr) < 2000:
                continue
            reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
            te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
            # A just-started season contributes no usable rows: its opening
            # matchdays have no prior-form features yet, so dropna empties the
            # fold and PoissonRegressor.predict raises on 0 samples. Skipping
            # costs nothing (the fold had no OOF pairs to collect) and keeps the
            # whole fit from failing soft to None every August.
            if te.empty:
                continue
            te["pred"] = np.clip(reg.predict(te[feats]), 0.1, 25)
            pv = te.pivot_table(index="match_id", columns="is_home", values="pred").dropna()
            if pv.empty:
                continue
            tot = pv[1] + pv[0]
            tr_tot = m[m.date < s_start]
            tt = (tr_tot[hc] + tr_tot[ac]).astype(float)
            disp = float(np.clip(tt.var() / max(tt.mean(), 1e-9), 0.8, 3.0))
            act = te_m.set_index("match_id").loc[tot.index, [hc, ac]].sum(axis=1).astype(float)
            for ln in lines:
                p = _prob_over(tot.to_numpy(), ln, disp)
                collected[ln].append((p, (act > ln).astype(int).to_numpy()))
            if side_lines:
                trs = lr[lr.date < s_start]["ev_for"].dropna().astype(float)
                disp_s = float(np.clip(trs.var() / max(trs.mean(), 1e-9), 0.9, 3.0))
                for ln in side_lines:
                    ps = _prob_over(te["pred"].to_numpy(), ln, disp_s)
                    collected_side[ln].append((ps, (te["ev_for"].astype(float) > ln).astype(int).to_numpy()))

        def _fit_pairs(pairs):
            x = _logit(np.concatenate([p for p, _ in pairs]))
            y = np.concatenate([y for _, y in pairs])
            if y.min() == y.max():
                return None
            pl = LogisticRegression(C=1e6, max_iter=1000).fit(x.reshape(-1, 1), y)
            return (float(pl.coef_[0][0]), float(pl.intercept_[0]))

        for ln in lines:
            if collected[ln]:
                ab = _fit_pairs(collected[ln])
                if ab:
                    self._platt[(market, ln)] = ab
        for ln in side_lines:
            if collected_side[ln]:
                ab = _fit_pairs(collected_side[ln])
                if ab:
                    self._platt[("side", market, ln)] = ab

    @staticmethod
    def _long_rows(m: pd.DataFrame, hc: str, ac: str,
                   extra: dict[str, str] | None = None, hl: int = 5) -> pd.DataFrame:
        has_lam = "lh" in m.columns
        has_pos = "pos_home" in m.columns
        # vectorized two-sided view (home perspective + away perspective)
        base = ["match_id", "date"]
        h_map = {"home_team": "team", "away_team": "opp", hc: "ev_for", ac: "ev_against",
                 "home_goals": "gf", "away_goals": "ga"}
        a_map = {"away_team": "team", "home_team": "opp", ac: "ev_for", hc: "ev_against",
                 "away_goals": "gf", "home_goals": "ga"}
        if has_lam:
            h_map.update({"lh": "lam_t", "la": "lam_o"})
            a_map.update({"la": "lam_t", "lh": "lam_o"})
        if has_pos:
            h_map.update({"pos_home": "pos_t", "pos_away": "pos_o", "played_home": "played"})
            a_map.update({"pos_away": "pos_t", "pos_home": "pos_o", "played_home": "played"})
        for k, col in (extra or {}).items():
            h_map[col] = k
            a_map[col] = k
        h = m[base + list(h_map)].rename(columns=h_map).assign(is_home=1)
        a = m[base + list(a_map)].rename(columns=a_map).assign(is_home=0)
        lr = pd.concat([h, a], ignore_index=True).sort_values(["team", "date", "match_id"])
        # cython grouped rolling/ewm on the shifted series (same semantics as the
        # old per-group apply, ~4x faster)
        tkey = lr["team"]
        for col in ["ev_for", "ev_against", "gf", "ga"]:
            shifted = lr.groupby("team", sort=False)[col].shift(1)
            g = shifted.groupby(tkey, sort=False)
            for w in WINDOWS:
                lr[f"{col}_r{w}"] = g.rolling(w, min_periods=3).mean().reset_index(level=0, drop=True)
            use_hl = hl if col in ("ev_for", "ev_against") else 5  # goals delta stays hl5
            lr[f"{col}_ewm"] = g.ewm(halflife=use_hl, min_periods=3).mean().reset_index(level=0, drop=True)
        opp_src = [f"ev_against_r{w}" for w in WINDOWS] + ["ev_against_ewm", "gf_ewm", "ga_ewm"]
        opp = lr[["match_id", "team"] + opp_src].rename(
            columns={"team": "opp", **{c: f"opp_{c}" for c in opp_src}})
        lr = lr.merge(opp, on=["match_id", "opp"], how="left")
        lr["delta"] = (lr["gf_ewm"] + lr["opp_ga_ewm"]) / 2 - (lr["opp_gf_ewm"] + lr["ga_ewm"]) / 2
        lr["abs_delta"] = lr["delta"].abs()
        if has_lam:
            lr["delta_lam"] = lr["lam_t"] - lr["lam_o"]
            lr["abs_delta_lam"] = lr["delta_lam"].abs()
        if has_pos:
            lr["round_frac"] = (lr["played"] / 38.0).clip(0, 1)
            lr["pos_diff_abs"] = (lr["pos_t"] - lr["pos_o"]).abs()
            lr["releg_battle"] = (((lr["pos_t"] >= 15) | (lr["pos_o"] >= 15))
                                  & (lr["round_frac"] > 0.6)).astype(float)
        return lr

    def _feature_names(self, market: str | None = None) -> list[str]:
        base = ([f"ev_for_r{w}" for w in WINDOWS] + ["ev_for_ewm"]
                + [f"opp_ev_against_r{w}" for w in WINDOWS] + ["opp_ev_against_ewm"]
                + ["is_home", "delta", "abs_delta"])
        if self._use_lam:
            base += ["delta_lam", "abs_delta_lam"]
        if market in STAKES_MARKETS:
            base += ["pos_diff_abs", "releg_battle", "round_frac"]
        return base

    @staticmethod
    def _latest_team_state(lr: pd.DataFrame, hl: int = 5) -> dict:
        """Per team: rolling stats INCLUDING its last played game (for the next fixture)."""
        out: dict = {}
        for team, g in lr.groupby("team"):
            if len(g) < 3:
                continue
            st = {}
            for w in WINDOWS:
                st[f"for_r{w}"] = float(g["ev_for"].tail(w).mean())
                st[f"against_r{w}"] = float(g["ev_against"].tail(w).mean())
            st["for_ewm"] = float(g["ev_for"].ewm(halflife=hl).mean().iloc[-1])
            st["against_ewm"] = float(g["ev_against"].ewm(halflife=hl).mean().iloc[-1])
            st["gf_ewm"] = float(g["gf"].ewm(halflife=5).mean().iloc[-1])   # goals delta stays hl5
            st["ga_ewm"] = float(g["ga"].ewm(halflife=5).mean().iloc[-1])
            out[team] = st
        return out

    # ── prediction ─────────────────────────────────────────────────────────────
    def _side_lambda(self, market: str, team: str, opp: str, is_home: int,
                     ref_rate: float | None = None,
                     lam_t: float | None = None, lam_o: float | None = None) -> float | None:
        mf = self._team_feats.get(market, {})
        st_t = mf.get(team) or mf.get(team.lower())      # foundation team names are lowercase
        st_o = mf.get(opp) or mf.get(opp.lower())
        if st_t is None or st_o is None:
            return None
        delta = (st_t["gf_ewm"] + st_o["ga_ewm"]) / 2 - (st_o["gf_ewm"] + st_t["ga_ewm"]) / 2
        row = ([st_t[f"for_r{w}"] for w in WINDOWS] + [st_t["for_ewm"]]
               + [st_o[f"against_r{w}"] for w in WINDOWS] + [st_o["against_ewm"]]
               + [float(is_home), delta, abs(delta)])
        if self._use_lam:
            # engine lambdas from the caller; goals-delta as proxy when absent
            dl = (lam_t - lam_o) if (lam_t is not None and lam_o is not None) else delta
            row = row + [dl, abs(dl)]
        if market in STAKES_MARKETS:
            pos_t, played_t = self._team_pos.get(team.lower(), (10, 19))
            pos_o, _ = self._team_pos.get(opp.lower(), (10, 19))
            round_frac = min(played_t / 38.0, 1.0)
            releg = float((pos_t >= 15 or pos_o >= 15) and round_frac > 0.6)
            row = row + [abs(pos_t - pos_o), releg, round_frac]
        cols = self._feature_names(market)
        model = self._models[market]
        if ref_rate is not None and market in self._models_ref:
            row = row + [ref_rate]
            cols = cols + ["ref_rate"]
            model = self._models_ref[market]
        if any(pd.isna(v) for v in row):
            return None
        x = pd.DataFrame([row], columns=cols)
        return float(np.clip(model.predict(x)[0], 0.1, 25))

    def _apply_platt(self, market: str, ln: float, p: float) -> float:
        ab = self._platt.get((market, ln))
        if ab is None:
            return p
        a, b = ab
        z = a * _logit(p) + b
        return float(1.0 / (1.0 + np.exp(-z)))

    def predict_fixture(self, home_team: str, away_team: str,
                        referee: str | None = None,
                        lam_home: float | None = None,
                        lam_away: float | None = None) -> dict:
        """{market: {"lambda_home", "lambda_away", "lambda_total", "dispersion",
        "over": {line: p}}}. `referee` (optional, EPL): uses the ref-augmented
        cards/fouls models when the referee is known to the model. `lam_home`/
        `lam_away`: the engine's goal lambdas for THIS fixture — power the
        delta_lam supremacy features (goals-delta proxy used when absent)."""
        out: dict = {}
        ref_vals = (self._ref_rates.get(referee)
                    if referee and self.is_epl_fixture(home_team, away_team) else None)
        for market in MARKETS:
            if market not in self._models:
                continue
            rr = None
            if ref_vals is not None and market in REF_MARKETS:
                rr = ref_vals[0] if market == "yellows" else ref_vals[1]
            lh = self._side_lambda(market, home_team, away_team, 1, rr, lam_home, lam_away)
            la = self._side_lambda(market, away_team, home_team, 0, rr, lam_away, lam_home)
            if lh is None or la is None:
                continue
            # MLE strengths prior blend (round-5, joint-validated incl. side lines)
            if market in self._adm:
                wb = ADM_W[market]
                comp_a = self._team_comp.get(home_team.lower())
                ah, aa, _ = self._adm[market].expected_goals(home_team, away_team, 0, comp_a)
                lh = wb * ah + (1 - wb) * lh
                la = wb * aa + (1 - wb) * la
            disp = self._disp[market]
            if market in self._disp_lg:   # per-league dispersion (fouls, validated 5/5)
                comp = self._team_comp.get(home_team.lower())
                disp = self._disp_lg[market].get(comp, disp)
            lines = MARKETS[market][2]
            over = {ln: self._apply_platt(market, ln, float(_prob_over(lh + la, ln, disp)))
                    for ln in lines}
            entry = {
                "lambda_home": round(lh, 2), "lambda_away": round(la, 2),
                "lambda_total": round(lh + la, 2), "dispersion": round(disp, 2),
                "referee_used": bool(rr is not None and market in self._models_ref),
                "over": {ln: round(p, 4) for ln, p in over.items()},
            }
            if market in SIDE_LINES:      # team-side lines (validated: more edge than totals)
                ds = self._disp_side[market]

                def _p_side(lam_s: float, ln: float) -> float:
                    p = float(_prob_over(lam_s, ln, ds))
                    ab = self._platt.get(("side", market, ln))
                    if ab is None:
                        return p
                    a, b = ab
                    return float(1.0 / (1.0 + np.exp(-(a * _logit(p) + b))))

                entry["over_home"] = {ln: round(_p_side(lh, ln), 4) for ln in SIDE_LINES[market]}
                entry["over_away"] = {ln: round(_p_side(la, ln), 4) for ln in SIDE_LINES[market]}
            out[market] = entry

        # booking points: yellows model + league-mean reds (team red tendency = noise)
        if "yellows" in out and self._red_lam:
            comp = self._team_comp.get(home_team.lower())
            lam_r = self._red_lam.get(comp, self._red_glob)
            lam_y = out["yellows"]["lambda_total"]
            disp_y = max(self._disp["yellows"], 1.05)
            out["booking_pts"] = {
                "lambda_yellows": lam_y, "lambda_reds": round(lam_r, 2),
                "over": {ln: round(float(_p_booking_over(np.array([lam_y]), disp_y,
                                                         np.array([lam_r]), ln)[0]), 4)
                         for ln in BOOKING_LINES},
            }
        return out

    @property
    def known_referees(self) -> list[str]:
        return sorted(self._ref_rates)

    def team_profile(self, team: str) -> dict:
        """Per-market recent event form for one team: {market: {"for_r10",
        "against_r10", "league_avg_side"}} — feeds the web analysis view."""
        out: dict = {}
        comp = self._team_comp.get(team.lower())
        for market in MARKETS:
            mf = self._team_feats.get(market, {})
            st = mf.get(team) or mf.get(team.lower())
            if st is None:
                continue
            lg_tot = self._league_lam.get(market, {}).get(comp)
            out[market] = {
                "for_r10": round(st["for_r10"], 2),
                "against_r10": round(st["against_r10"], 2),
                "league_avg_side": round(lg_tot / 2, 2) if lg_tot else None,
            }
        return out
