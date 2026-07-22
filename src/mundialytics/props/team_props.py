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


def load_red_cards(root: str | Path) -> pd.DataFrame | None:
    """Per-match red cards from raw football-data (HR/AR are 100% present in
    all big-5 files; the foundation CSV never kept them)."""
    rows = []
    for p in glob.glob(str(Path(root) / "data/raw/football_data/**/*.csv"), recursive=True):
        if not re.search(r"\d{4}_(E0|SP1|D1|I1|F1)\.csv$", p):
            continue
        try:
            df = pd.read_csv(p, encoding="latin-1", on_bad_lines="skip",
                             usecols=lambda c: c in {"Date", "HomeTeam", "AwayTeam", "HR", "AR"})
        except Exception:
            continue
        if "HR" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed")
        rows.append(df)
    if not rows:
        return None
    r = pd.concat(rows, ignore_index=True).dropna(subset=["date"])
    r = r.drop_duplicates(subset=["date", "HomeTeam", "AwayTeam"])
    r["home_team"] = r["HomeTeam"].astype(str).str.lower().str.strip()
    r["away_team"] = r["AwayTeam"].astype(str).str.lower().str.strip()
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
    """Walk-forward referee card/foul tendencies from raw football-data E0 files."""
    rows = []
    for p in glob.glob(str(Path(root) / "data/raw/football_data/**/*E0.csv"), recursive=True):
        if not re.search(r"\d{4}_E0\.csv$", p):
            continue
        try:
            df = pd.read_csv(p, encoding="latin-1", on_bad_lines="skip",
                             usecols=lambda c: c in {"Date", "HomeTeam", "AwayTeam", "Referee",
                                                     "HY", "AY", "HF", "AF"})
        except Exception:
            continue
        if "Referee" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        rows.append(df)
    if not rows:
        return None
    r = pd.concat(rows, ignore_index=True).dropna(subset=["date", "Referee"])
    r = r.drop_duplicates(subset=["date", "HomeTeam", "AwayTeam"])
    r["home_team"] = r["HomeTeam"].astype(str).str.lower().str.strip()
    r["away_team"] = r["AwayTeam"].astype(str).str.lower().str.strip()
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

    # ── fitting ────────────────────────────────────────────────────────────────
    def fit(self, matches: pd.DataFrame, referee_data: pd.DataFrame | None = None,
            root: str | Path | None = None) -> "TeamPropsModel":
        """`matches`: foundation rows (date/season/teams + event + goal columns).
        `referee_data`: optional output of load_referee_rates (auto-loaded if
        `root` given)."""
        df = matches.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["season"] >= self.seasons_from]
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
            lr = self._long_rows(m, hc, ac)
            feats = self._feature_names()
            tr = lr.dropna(subset=feats + ["ev_for"])
            reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
            self._models[market] = reg
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
            self._team_feats[market] = self._latest_team_state(lr)
            if self.calibrate and market not in NO_PLATT:
                self._fit_platt(market, m, lr, feats, hc, ac, lines)
            # referee-augmented model (EPL subset)
            if market in REF_MARKETS and referee_data is not None and "competition" in m.columns:
                epl = m[m["competition"].str.contains("Premier", case=False, na=False)]
                epl = epl.merge(referee_data[["date", "home_team", "away_team", REF_MARKETS[market]]],
                                on=["date", "home_team", "away_team"], how="left")
                epl = epl.drop_duplicates(subset=["match_id"]).dropna(subset=[REF_MARKETS[market]])
                if len(epl) > 2000:
                    lr_e = self._long_rows(epl, hc, ac, extra={"ref_rate": REF_MARKETS[market]})
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
        for s in seasons:
            te_m = m[m.season == s]
            s_start = te_m.date.min()
            tr = lr[lr.date < s_start].dropna(subset=feats + ["ev_for"])
            if len(tr) < 2000:
                continue
            reg = PoissonRegressor(alpha=0.1, max_iter=1000).fit(tr[feats], tr["ev_for"].clip(lower=0))
            te = lr[lr.match_id.isin(set(te_m.match_id))].dropna(subset=feats).copy()
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
        for ln in lines:
            if not collected[ln]:
                continue
            x = _logit(np.concatenate([p for p, _ in collected[ln]]))
            y = np.concatenate([y for _, y in collected[ln]])
            if y.min() == y.max():
                continue
            pl = LogisticRegression(C=1e6, max_iter=1000).fit(x.reshape(-1, 1), y)
            self._platt[(market, ln)] = (float(pl.coef_[0][0]), float(pl.intercept_[0]))

    @staticmethod
    def _long_rows(m: pd.DataFrame, hc: str, ac: str,
                   extra: dict[str, str] | None = None) -> pd.DataFrame:
        rows = []
        for r in m.itertuples(index=False):
            ex = {k: getattr(r, col) for k, col in (extra or {}).items()}
            rows.append(dict(match_id=r.match_id, date=r.date, team=r.home_team, opp=r.away_team,
                             is_home=1, ev_for=getattr(r, hc), ev_against=getattr(r, ac),
                             gf=r.home_goals, ga=r.away_goals, **ex))
            rows.append(dict(match_id=r.match_id, date=r.date, team=r.away_team, opp=r.home_team,
                             is_home=0, ev_for=getattr(r, ac), ev_against=getattr(r, hc),
                             gf=r.away_goals, ga=r.home_goals, **ex))
        lr = pd.DataFrame(rows).sort_values(["team", "date", "match_id"])
        for col in ["ev_for", "ev_against", "gf", "ga"]:
            for w in WINDOWS:
                lr[f"{col}_r{w}"] = (lr.groupby("team", group_keys=False)[col]
                                     .apply(lambda s: s.shift(1).rolling(w, min_periods=3).mean()))
            lr[f"{col}_ewm"] = (lr.groupby("team", group_keys=False)[col]
                                .apply(lambda s: s.shift(1).ewm(halflife=5, min_periods=3).mean()))
        opp_src = [f"ev_against_r{w}" for w in WINDOWS] + ["ev_against_ewm", "gf_ewm", "ga_ewm"]
        opp = lr[["match_id", "team"] + opp_src].rename(
            columns={"team": "opp", **{c: f"opp_{c}" for c in opp_src}})
        lr = lr.merge(opp, on=["match_id", "opp"], how="left")
        lr["delta"] = (lr["gf_ewm"] + lr["opp_ga_ewm"]) / 2 - (lr["opp_gf_ewm"] + lr["ga_ewm"]) / 2
        lr["abs_delta"] = lr["delta"].abs()
        return lr

    @staticmethod
    def _feature_names() -> list[str]:
        return ([f"ev_for_r{w}" for w in WINDOWS] + ["ev_for_ewm"]
                + [f"opp_ev_against_r{w}" for w in WINDOWS] + ["opp_ev_against_ewm"]
                + ["is_home", "delta", "abs_delta"])

    @staticmethod
    def _latest_team_state(lr: pd.DataFrame) -> dict:
        """Per team: rolling stats INCLUDING its last played game (for the next fixture)."""
        out: dict = {}
        for team, g in lr.groupby("team"):
            if len(g) < 3:
                continue
            st = {}
            for w in WINDOWS:
                st[f"for_r{w}"] = float(g["ev_for"].tail(w).mean())
                st[f"against_r{w}"] = float(g["ev_against"].tail(w).mean())
            st["for_ewm"] = float(g["ev_for"].ewm(halflife=5).mean().iloc[-1])
            st["against_ewm"] = float(g["ev_against"].ewm(halflife=5).mean().iloc[-1])
            st["gf_ewm"] = float(g["gf"].ewm(halflife=5).mean().iloc[-1])
            st["ga_ewm"] = float(g["ga"].ewm(halflife=5).mean().iloc[-1])
            out[team] = st
        return out

    # ── prediction ─────────────────────────────────────────────────────────────
    def _side_lambda(self, market: str, team: str, opp: str, is_home: int,
                     ref_rate: float | None = None) -> float | None:
        mf = self._team_feats.get(market, {})
        st_t = mf.get(team) or mf.get(team.lower())      # foundation team names are lowercase
        st_o = mf.get(opp) or mf.get(opp.lower())
        if st_t is None or st_o is None:
            return None
        delta = (st_t["gf_ewm"] + st_o["ga_ewm"]) / 2 - (st_o["gf_ewm"] + st_t["ga_ewm"]) / 2
        row = ([st_t[f"for_r{w}"] for w in WINDOWS] + [st_t["for_ewm"]]
               + [st_o[f"against_r{w}"] for w in WINDOWS] + [st_o["against_ewm"]]
               + [float(is_home), delta, abs(delta)])
        cols = self._feature_names()
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
                        referee: str | None = None) -> dict:
        """{market: {"lambda_home", "lambda_away", "lambda_total", "dispersion",
        "over": {line: p}}}. `referee` (optional, EPL): uses the ref-augmented
        cards/fouls models when the referee is known to the model."""
        out: dict = {}
        ref_vals = (self._ref_rates.get(referee)
                    if referee and self.is_epl_fixture(home_team, away_team) else None)
        for market in MARKETS:
            if market not in self._models:
                continue
            rr = None
            if ref_vals is not None and market in REF_MARKETS:
                rr = ref_vals[0] if market == "yellows" else ref_vals[1]
            lh = self._side_lambda(market, home_team, away_team, 1, rr)
            la = self._side_lambda(market, away_team, home_team, 0, rr)
            if lh is None or la is None:
                continue
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
                entry["over_home"] = {ln: round(float(_prob_over(lh, ln, ds)), 4)
                                      for ln in SIDE_LINES[market]}
                entry["over_away"] = {ln: round(float(_prob_over(la, ln, ds)), 4)
                                      for ln in SIDE_LINES[market]}
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
