from __future__ import annotations

"""European tournament simulator: Champions / Europa / Conference (2024+ format).

Swiss league phase (36 teams; 8 games UCL/UEL, 6 UECL) -> top 8 straight to
R16, 9-24 to a seeded playoff round, fixed seeded bracket, two-legged ties
(extra time + penalties), single-leg neutral final. Monte Carlo from any
point of the season.

Strengths: ClubElo ratings (single cross-league scale, all UEFA clubs) mapped
to goal expectations with constants CALIBRATED on our own big-5 foundation
(scripts/calibrate_elo_lambda.py):

    log lam_h = c + hfa + b*(eloH - eloA)/400     (hfa dropped when neutral)

Two modes:
  - fixtures known (post-draw): pass the league-phase fixture list, played
    results included -> simulates only what remains.
  - PRE-DRAW: no fixtures yet -> each Monte Carlo iteration samples a valid
    pot-based draw (pots by Elo quartiles, 2 opponents per pot, 4 home /
    4 away), so probabilities integrate over the draw itself.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

FORMATS = {
    "champions": {"n": 36, "games": 8, "label": "Champions League"},
    "europa": {"n": 36, "games": 8, "label": "Europa League"},
    "conference": {"n": 36, "games": 6, "label": "Conference League"},
}
ROUND_LABELS = ["top24", "top8", "playoff", "r16", "qf", "sf", "final", "champion"]


def load_calibration(root: str | Path) -> dict:
    """European constants preferred (validated on 1,000 real UCL/UEL/UECL
    matches — European home advantage measured HIGHER than domestic, hfa 0.28
    vs 0.21, and slightly higher scoring); big-5 domestic fit as fallback."""
    euro = Path(root) / "data/processed/elo_lambda_calibration_euro.json"
    if euro.exists():
        return json.loads(euro.read_text())
    p = Path(root) / "data/processed/elo_lambda_calibration.json"
    return json.loads(p.read_text())


def fetch_current_elo(root: str | Path, date_str: str | None = None) -> dict[str, float]:
    """Full ClubElo snapshot for one day ({Club: Elo}, all UEFA clubs), disk-cached."""
    import datetime

    from io import StringIO

    import requests

    d = date_str or datetime.date.today().isoformat()
    cache = Path(root) / f"data/external/clubelo/daily/{d}.csv"
    if cache.exists() and cache.stat().st_size > 5000:
        df = pd.read_csv(cache)
    else:
        r = requests.get(f"http://api.clubelo.com/{d}", timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    df = df.dropna(subset=["Club", "Elo"])
    return dict(zip(df["Club"].astype(str), df["Elo"].astype(float)))


KO_ROUNDS = ["playoff", "r16", "qf", "sf", "final"]
KO_LABELS = {"Play-off": "playoff", "R16": "r16", "QF": "qf", "SF": "sf", "Final": "final"}
FD_SLUG = {"champions": "champions-league", "europa": "europa-league",
           "conference": "conference-league"}


def normalize_club(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


def make_resolver(elo_names) -> "callable":
    """fixturedownload club name -> ClubElo name (aliases + normalized + substring)."""
    from mundialytics.statistical_core.competition.club_aliases import CLUB_ALIASES
    by_norm = {normalize_club(n): n for n in elo_names}

    def resolver(name: str):
        low = str(name).lower().strip()
        if low in CLUB_ALIASES:
            return CLUB_ALIASES[low]
        n = normalize_club(name)
        if n in by_norm:
            return by_norm[n]
        cands = [v for k, v in by_norm.items() if n in k or k in n]
        return cands[0] if len(cands) == 1 else None

    return resolver


def fetch_season_fixtures(root: str | Path, competition: str, year: int) -> pd.DataFrame | None:
    """Current-season fixture/result CSV from fixturedownload (cached; tolerant)."""
    import requests

    from io import StringIO

    slug = FD_SLUG[competition]
    cache = Path(root) / f"data/external/uefa/raw_{slug}_{year}.csv"
    if cache.exists() and cache.stat().st_size > 500:
        return pd.read_csv(cache)
    try:
        r = requests.get(f"https://fixturedownload.com/download/{slug}-{year}-UTC.csv",
                         timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.encoding = "utf-8"
        if r.status_code != 200 or len(r.text) < 500:
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(r.text, encoding="utf-8")
        return pd.read_csv(StringIO(r.text))
    except Exception:
        return None


def parse_fixturedownload(raw: pd.DataFrame, resolver) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a fixturedownload frame into (league_fixtures, knockout_legs).

    league:   home, away, home_goals, away_goals (NaN = pending)
    knockout: round, leg, home, away, hg, ag (NaN = pending) — real pairings.
    """
    df = raw.copy()
    res = df["Result"].astype(str).str.extract(r"(\d+)\s*-\s*(\d+)")
    df["hg"] = pd.to_numeric(res[0], errors="coerce")
    df["ag"] = pd.to_numeric(res[1], errors="coerce")
    df["home"] = df["Home Team"].astype(str).map(resolver)
    df["away"] = df["Away Team"].astype(str).map(resolver)
    df = df.dropna(subset=["home", "away"])
    rn = df["Round Number"].astype(str)
    is_league = pd.to_numeric(rn, errors="coerce").notna()
    league = df[is_league].rename(columns={"hg": "home_goals", "ag": "away_goals"})[
        ["home", "away", "home_goals", "away_goals"]]
    ko = df[~is_league].copy()
    if len(ko):
        ko["round"] = (ko["Round Number"].astype(str).str.split(" Game").str[0]
                       .map(KO_LABELS).fillna("final"))
        ko["leg"] = pd.to_numeric(ko["Round Number"].astype(str).str.extract(r"Game (\d)")[0],
                                  errors="coerce").fillna(1).astype(int)
        ko = ko[["round", "leg", "home", "away", "hg", "ag"]]
    else:
        ko = pd.DataFrame(columns=["round", "leg", "home", "away", "hg", "ag"])
    return league, ko


@dataclass
class EuropeanTournament:
    competition: str                   # champions | europa | conference
    elo: dict[str, float]              # team -> Elo (ClubElo names)
    calib: dict                        # c, hfa, b
    fixtures: pd.DataFrame | None = None   # cols: home, away, home_goals, away_goals (NaN=pending)
    knockout: pd.DataFrame | None = None   # cols: round, leg, home, away, hg, ag — real bracket state
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(7))

    def __post_init__(self) -> None:
        self.teams = sorted(self.elo, key=self.elo.get, reverse=True)[:FORMATS[self.competition]["n"]]
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.n = len(self.teams)
        e = np.array([self.elo[t] for t in self.teams])
        self.d400 = (e[:, None] - e[None, :]) / 400.0
        c, hfa, b = self.calib["c"], self.calib["hfa"], self.calib["b"]
        self.lam_h = np.exp(c + hfa + b * self.d400)     # [home, away] grids
        self.lam_a = np.exp(c - b * self.d400)
        self.lam_h_neutral = np.exp(c + b * self.d400)
        self.lam_a_neutral = np.exp(c - b * self.d400)

    # ── match simulation ───────────────────────────────────────────────────────
    def _sim_goals(self, lam: np.ndarray) -> np.ndarray:
        return self.rng.poisson(np.clip(lam, 0.05, 6.0))

    def _pots(self) -> list[list[int]]:
        order = list(range(self.n))     # already Elo-sorted
        q = self.n // 4
        return [order[:q], order[q:2 * q], order[2 * q:3 * q], order[3 * q:]]

    def _sample_draw(self) -> list[tuple[int, int]]:
        """Random valid-ish league-phase draw: per team, 2 opponents per pot,
        half at home — relaxed constraints (no country checks), enough for MC."""
        games = FORMATS[self.competition]["games"]
        per_pot = games // 4
        pots = self._pots()
        pairs: set[tuple[int, int]] = set()
        for pot in pots:
            for team in range(self.n):
                have = sum(1 for a, b in pairs if (a == team and b in pot) or (b == team and a in pot))
                cands = [o for o in pot if o != team]
                self.rng.shuffle(cands)
                for o in cands:
                    if have >= per_pot:
                        break
                    key = (min(team, o), max(team, o))
                    if key in pairs:
                        continue
                    opp_have = sum(1 for a, b in pairs
                                   if (a == o and self._pot_of(b) == self._pot_of(team) and b != o)
                                   or (b == o and self._pot_of(a) == self._pot_of(team) and a != o))
                    if opp_have >= per_pot:
                        continue
                    pairs.add(key)
                    have += 1
        out = []
        for a, b in pairs:
            if self.rng.random() < 0.5:
                out.append((a, b))
            else:
                out.append((b, a))
        return out

    def _pot_of(self, i: int) -> int:
        return min(3, i // (self.n // 4))

    # ── league phase ───────────────────────────────────────────────────────────
    def _league_tables(self, n_sims: int) -> np.ndarray:
        """(n_sims, n) final league-phase points+gd/1000 score for ranking."""
        pts0 = np.zeros(self.n)
        gd0 = np.zeros(self.n)
        fixed: list[tuple[int, int]] = []
        if self.fixtures is not None and len(self.fixtures):
            for r in self.fixtures.itertuples(index=False):
                h, a = self.idx.get(r.home), self.idx.get(r.away)
                if h is None or a is None:
                    continue
                if pd.notna(r.home_goals):
                    hg, ag = int(r.home_goals), int(r.away_goals)
                    pts0[h] += 3 if hg > ag else (1 if hg == ag else 0)
                    pts0[a] += 3 if ag > hg else (1 if hg == ag else 0)
                    gd0[h] += hg - ag
                    gd0[a] += ag - hg
                else:
                    fixed.append((h, a))
        scores = np.zeros((n_sims, self.n))
        for s in range(n_sims):
            pts = pts0.copy()
            gd = gd0.copy()
            pend = fixed if (self.fixtures is not None and len(self.fixtures)) else self._sample_draw()
            for h, a in pend:
                hg = self._sim_goals(self.lam_h[h, a])
                ag = self._sim_goals(self.lam_a[h, a])
                pts[h] += 3 if hg > ag else (1 if hg == ag else 0)
                pts[a] += 3 if ag > hg else (1 if hg == ag else 0)
                gd[h] += hg - ag
                gd[a] += ag - hg
            scores[s] = pts + gd / 1000.0 + self.rng.random(self.n) / 1e6
        return scores

    # ── knockout ───────────────────────────────────────────────────────────────
    def _two_leg(self, i: int, j: int) -> int:
        """Aggregate two-legged tie, first leg at j (lower seed hosts leg 1)."""
        g1a = self._sim_goals(self.lam_h[j, i])
        g1b = self._sim_goals(self.lam_a[j, i])
        g2a = self._sim_goals(self.lam_h[i, j])
        g2b = self._sim_goals(self.lam_a[i, j])
        agg_i = g1b + g2a
        agg_j = g1a + g2b
        if agg_i != agg_j:
            return i if agg_i > agg_j else j
        # extra time at the second leg (~1/3 of a match), then penalties 50/50
        et_i = self._sim_goals(self.lam_h[i, j] / 3.0)
        et_j = self._sim_goals(self.lam_a[i, j] / 3.0)
        if et_i != et_j:
            return i if et_i > et_j else j
        return i if self.rng.random() < 0.5 else j

    def _single_neutral(self, i: int, j: int) -> int:
        gi = self._sim_goals(self.lam_h_neutral[i, j])
        gj = self._sim_goals(self.lam_a_neutral[i, j])
        if gi != gj:
            return i if gi > gj else j
        et_i = self._sim_goals(self.lam_h_neutral[i, j] / 3.0)
        et_j = self._sim_goals(self.lam_a_neutral[i, j] / 3.0)
        if et_i != et_j:
            return i if et_i > et_j else j
        return i if self.rng.random() < 0.5 else j

    # ── knockout-state resume ──────────────────────────────────────────────────
    def _parse_ties(self, ko: pd.DataFrame) -> dict[str, list[dict]]:
        """Per round: [{a, b, agg_a, agg_b, leg2_home, done, winner}] from real rows."""
        out: dict[str, list[dict]] = {}
        for rnd in KO_ROUNDS:
            rows = ko[ko["round"] == rnd]
            if rows.empty:
                continue
            ties: dict[frozenset, dict] = {}
            for r in rows.itertuples(index=False):
                a, b = self.idx.get(r.home), self.idx.get(r.away)
                if a is None or b is None:
                    continue
                key = frozenset((a, b))
                t = ties.setdefault(key, {"a": a, "b": b, "agg_a": 0, "agg_b": 0,
                                          "legs_played": 0, "legs_seen": 0, "leg2_home": None})
                t["legs_seen"] += 1
                if r.leg == 1:
                    t["a"], t["b"] = a, b            # leg-1 home defines orientation
                    t["leg2_home"] = b
                else:
                    t["leg2_home"] = a
                if pd.notna(r.hg):
                    t["legs_played"] += 1
                    ha, hb = (a, b)
                    t["agg_a" if ha == t["a"] else "agg_b"] += int(r.hg)
                    t["agg_b" if ha == t["a"] else "agg_a"] += int(r.ag)
            expected_legs = 1 if rnd == "final" else 2
            for t in ties.values():
                t["done"] = t["legs_played"] >= expected_legs
                t["winner"] = None
                if t["done"]:
                    if t["agg_a"] != t["agg_b"]:
                        t["winner"] = t["a"] if t["agg_a"] > t["agg_b"] else t["b"]
                    else:
                        nxt_rounds = KO_ROUNDS[KO_ROUNDS.index(rnd) + 1:]
                        nxt = ko[ko["round"].isin(nxt_rounds)]
                        names = set(nxt["home"]) | set(nxt["away"])
                        for cand in (t["a"], t["b"]):
                            if self.teams[cand] in names:
                                t["winner"] = cand
                                break
            out[rnd] = list(ties.values())
        return out

    def _resume_tie(self, t: dict) -> int:
        """Winner of a tie from its real partial state (simulating what's left)."""
        if t["winner"] is not None:
            return t["winner"]
        a, b = t["a"], t["b"]
        agg_a, agg_b = t["agg_a"], t["agg_b"]
        if t["done"]:   # real tie ended level and next round can't tell us -> pens
            return a if self.rng.random() < 0.5 else b
        if t["legs_played"] == 0:
            return self._two_leg(b, a)   # _two_leg: first arg plays leg1 AWAY
        # leg 1 played -> simulate leg 2 at its real venue
        h = t["leg2_home"] if t["leg2_home"] is not None else b
        o = a if h == b else b
        hg = self._sim_goals(self.lam_h[h, o])
        og = self._sim_goals(self.lam_a[h, o])
        agg_a += hg if h == a else og
        agg_b += hg if h == b else og
        if agg_a != agg_b:
            return a if agg_a > agg_b else b
        et_h = self._sim_goals(self.lam_h[h, o] / 3.0)
        et_o = self._sim_goals(self.lam_a[h, o] / 3.0)
        if et_h != et_o:
            return h if et_h > et_o else o
        return a if self.rng.random() < 0.5 else b

    def _simulate_from_knockout(self, n_sims: int, ko: pd.DataFrame) -> pd.DataFrame:
        counts = {k: np.zeros(self.n) for k in ROUND_LABELS}
        ties_by_round = self._parse_ties(ko)
        # league phase is over: real standings give deterministic top24/top8/playoff
        if self.fixtures is not None and len(self.fixtures):
            pts = np.zeros(self.n)
            gd = np.zeros(self.n)
            for r in self.fixtures.itertuples(index=False):
                h, a = self.idx.get(r.home), self.idx.get(r.away)
                if h is None or a is None or pd.isna(r.home_goals):
                    continue
                hg, ag = int(r.home_goals), int(r.away_goals)
                pts[h] += 3 if hg > ag else (1 if hg == ag else 0)
                pts[a] += 3 if ag > hg else (1 if hg == ag else 0)
                gd[h] += hg - ag
                gd[a] += ag - hg
            rank = np.argsort(-(pts + gd / 1000.0))
            counts["top24"][rank[:24]] = n_sims
            counts["top8"][rank[:8]] = n_sims
            counts["playoff"][rank[8:24]] = n_sims
        # real pairings are only usable up to the FIRST round with pending ties;
        # later rounds' rows name teams that leak the real outcomes we're simulating
        cursor = next((r for r in KO_ROUNDS if r in ties_by_round
                       and any(not t["done"] for t in ties_by_round[r])), None)
        cutoff = KO_ROUNDS.index(cursor) if cursor is not None else len(KO_ROUNDS)
        for s in range(n_sims):
            alive: list[int] = []
            champ = None
            for ri, rnd in enumerate(KO_ROUNDS):
                if rnd in ties_by_round and ri <= cutoff:
                    ties = ties_by_round[rnd]
                    parts = [x for t in ties for x in (t["a"], t["b"])]
                    if rnd == "final":
                        counts["final"][parts] += 1
                        champ = self._resume_tie(ties[0]) if len(ties) else None
                        break
                    if rnd != "playoff":
                        counts[rnd][parts] += 1
                    alive = [self._resume_tie(t) for t in ties]
                else:
                    if not alive:
                        continue
                    if rnd == "final":
                        counts["final"][alive] += 1
                        champ = self._single_neutral(alive[0], alive[1])
                        break
                    # future round, pairings unknown -> random pairing of winners
                    self.rng.shuffle(alive)
                    if rnd != "playoff":
                        counts[rnd][alive] += 1
                    alive = [self._two_leg(alive[2 * k], alive[2 * k + 1])
                             for k in range(len(alive) // 2)]
            if champ is None and len(alive) == 1:
                champ = alive[0]
            if champ is not None:
                counts["champion"][champ] += 1
        out = pd.DataFrame({"team": self.teams,
                            "elo": [round(self.elo[t], 0) for t in self.teams]})
        for k in ROUND_LABELS:
            out[f"p_{k}"] = counts[k] / n_sims
        return out.sort_values("p_champion", ascending=False).reset_index(drop=True)

    def simulate(self, n_sims: int = 5000) -> pd.DataFrame:
        if self.knockout is not None and len(self.knockout):
            return self._simulate_from_knockout(n_sims, self.knockout)
        scores = self._league_tables(n_sims)
        counts = {k: np.zeros(self.n) for k in ROUND_LABELS}
        for s in range(n_sims):
            rank = np.argsort(-scores[s])          # rank[0] = 1st
            top8 = list(rank[:8])
            po_seeded = list(rank[8:16])
            po_unseeded = list(rank[16:24])
            counts["top24"][rank[:24]] += 1     # pasar la fase liga
            counts["top8"][top8] += 1
            counts["playoff"][rank[8:24]] += 1
            # playoff: seeded (9-16) vs reversed unseeded band (17-24)
            po_winners = []
            for k in range(8):
                a, b = po_seeded[k], po_unseeded[7 - k]
                po_winners.append(self._two_leg(a, b))   # seeded hosts leg 2
            # R16: seed k vs playoff-winner (official tree pairs 1-8 with bands; approx)
            r16 = []
            for k in range(8):
                r16.append((top8[k], po_winners[7 - k]))
            alive = []
            for a, b in r16:
                w = self._two_leg(a, b)
                counts["r16"][[a, b]] += 1
                alive.append(w)
            qf = [self._two_leg(alive[0], alive[7]), self._two_leg(alive[3], alive[4]),
                  self._two_leg(alive[1], alive[6]), self._two_leg(alive[2], alive[5])]
            counts["qf"][alive] += 1
            sf = [self._two_leg(qf[0], qf[1]), self._two_leg(qf[2], qf[3])]
            counts["sf"][qf] += 1
            champ = self._single_neutral(sf[0], sf[1])
            counts["final"][sf] += 1
            counts["champion"][champ] += 1
        out = pd.DataFrame({"team": self.teams,
                            "elo": [round(self.elo[t], 0) for t in self.teams]})
        for k in ROUND_LABELS:
            out[f"p_{k}"] = counts[k] / n_sims
        return out.sort_values("p_champion", ascending=False).reset_index(drop=True)
