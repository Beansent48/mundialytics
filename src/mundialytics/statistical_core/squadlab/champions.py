"""SquadLab's Champions League: your drafted eleven, in the real 36-team field.

WHY NOT THE LEAGUE SIMULATOR. The domestic SquadLab season ran on the club
engine's AttackDefenseModel, which knows the big five and nothing else. Two
thirds of a Champions League field — Bodø/Glimt, Slavia, Sabah, Viking, Club
Brugge, Fenerbahçe — have no row in it, and inventing one for each would be a
worse answer than the one the European layer already uses: Elo, calibrated on a
thousand real UCL/UEL/UECL matches (see competition/european.py).

So the tournament runs on Elo, and the only new problem is what Elo a squad that
has never played a match should have. That is answered by measurement rather
than by taste: build every real club's best eleven out of the same card
catalogue, push it through the same squad-to-lambda bridge, and fit the club's
actual Elo against what the bridge says about it. A drafted squad then reads off
that line. It is a fitted relationship with a reported R², not a guess.

THE SLOT. Your club takes the place of the weakest side in the real draw, which
keeps the fixture list, the pots and the schedule exactly as they are — you play
the eight opponents that team was actually drawn against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mundialytics.statistical_core.competition.european import (
    EuropeanTournament, load_calibration, load_event_calibration, make_resolver,
    parse_fixturedownload, predict_euro_events,
)
from mundialytics.statistical_core.player_strength import PlayerStrengthProfile
from mundialytics.statistical_core.squadlab.player_rating import (
    attribute_cards, attribute_goals, compute_match_ratings,
)

ROOT = Path(__file__).resolve().parents[4]
RAW_CL = ROOT / "data/external/uefa/raw_champions-league_2026.csv"
CLUBELO = ROOT / "data/processed/clubelo_local.csv"

SQUAD_TEAM_NAME = "Tu Equipo"
LEAGUE_PHASE_GAMES = 8

# ClubElo rates neither Slavia Praha nor Torreense — a real gap the forecasting
# page reports honestly as "sin Elo", because a made-up rating there would be a
# made-up probability. A GAME cannot do that: a 35-team Champions League has no
# league phase and no bracket. So the game layer, and only the game layer,
# stands an unrated participant next to the best-rated club from its own league
# — a country's entrant is its champion, and Sparta is the Czech side ClubElo
# does carry. Nothing outside SquadLab reads this.
GAME_ELO_STAND_IN = {"Slavia Prague": "Sparta Praha", "Slavia Praha": "Sparta Praha"}

ROUND_LABELS = {
    "playoff": "Playoff", "r16": "Octavos", "qf": "Cuartos",
    "sf": "Semifinales", "final": "Final",
}


# ── the field ──────────────────────────────────────────────────────────────────
def load_field(raw_path: Path | None = None, elo_path: Path | None = None) -> dict:
    """The real Champions League field: {club: elo} plus the drawn fixtures.

    Read off disk on purpose. The live fetch belongs to the forecasting page,
    which has to be current; a game wants the same draw every time it is opened.
    """
    raw = pd.read_csv(raw_path or RAW_CL)
    elo_df = pd.read_csv(elo_path or CLUBELO).dropna(subset=["club", "elo"])
    elo_all = dict(zip(elo_df["club"].astype(str), elo_df["elo"].astype(float)))
    resolver = make_resolver(list(elo_all))
    league, _ = parse_fixturedownload(raw, resolver)

    # keep the clubs the resolver could not place, under their own name
    raw_names = sorted(set(raw["Home Team"].astype(str)) | set(raw["Away Team"].astype(str)))
    unresolved = {n: resolver(n) for n in raw_names}
    league2 = raw.copy()
    res = league2["Result"].astype(str).str.extract(r"(\d+)\s*-\s*(\d+)")
    league2["home_goals"] = pd.to_numeric(res[0], errors="coerce")
    league2["away_goals"] = pd.to_numeric(res[1], errors="coerce")
    league2["home"] = [unresolved.get(str(x)) or str(x) for x in league2["Home Team"]]
    league2["away"] = [unresolved.get(str(x)) or str(x) for x in league2["Away Team"]]
    league2 = league2[pd.to_numeric(league2["Round Number"], errors="coerce").notna()]
    league = league2[["home", "away", "home_goals", "away_goals"]].reset_index(drop=True)

    teams = sorted(set(league["home"]) | set(league["away"]))
    elo, stand_in = {}, {}
    for t in teams:
        if t in elo_all:
            elo[t] = float(elo_all[t])
            continue
        peer = GAME_ELO_STAND_IN.get(t)
        if peer and peer in elo_all:
            elo[t] = float(elo_all[peer])
            stand_in[t] = peer
        else:
            elo[t] = float(min(elo_all[x] for x in teams if x in elo_all))
            stand_in[t] = "el más débil del cuadro"
    league = league[league["home"].isin(elo) & league["away"].isin(elo)]
    return {"elo": elo, "fixtures": league.reset_index(drop=True), "stand_in": stand_in}


def weakest_team(elo: dict[str, float]) -> str:
    return min(elo, key=elo.get)


# ── squad -> Elo ───────────────────────────────────────────────────────────────
@dataclass
class SquadEloScale:
    """elo ≈ a + b·(mean rating of the eleven), fitted on the real clubs.

    The first attempt ran the eleven through the squad-to-lambda bridge and
    fitted Elo on the strength parameters it produced. That works (R² 0.53) but
    inherits the bridge's known compression at the top — Arsenal's real 2071
    came back as 1790 — which in a game means a squad of icons could never
    actually become the best team in Europe. Going straight from the card
    ratings fits better (R² 0.56) and is the honest tool for the job: in
    Champions mode the match odds come from Elo and nothing else, so what is
    wanted here is a predictor of Elo, not a detour through a bridge calibrated
    for something else.

    Adding the mean of the best five nudged the fit to R² 0.60 and was dropped
    anyway: collinear with the mean, it came back with a NEGATIVE coefficient,
    which says a squad gets worse when its stars get better. Fine inside the
    range it was fitted on and nonsense on a drafted squad, which is exactly the
    shape this is asked to extrapolate to.
    """
    intercept: float
    slope_mean: float
    r2: float
    n: int

    def elo_for(self, overalls) -> float:
        vals = [float(x) for x in overalls]
        if not vals:
            return 1500.0
        return float(np.clip(self.intercept + self.slope_mean * float(np.mean(vals)),
                             1200.0, 2250.0))


def fit_squad_elo_scale(club_overalls: dict[str, list[float]],
                        club_elo: dict[str, float]) -> SquadEloScale:
    """Fit the line on real clubs' own best elevens, drawn from the same cards."""
    xs, ys = [], []
    for club, ovrs in club_overalls.items():
        if club not in club_elo or len(ovrs) < 8:
            continue
        xs.append(float(np.mean(ovrs)))
        ys.append(club_elo[club])
    if len(xs) < 12:
        # nothing to fit on: place any squad at a mid-table European level
        return SquadEloScale(1650.0, 0.0, float("nan"), len(xs))
    x, y = np.asarray(xs), np.asarray(ys)
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return SquadEloScale(float(intercept), float(slope),
                         1 - ss_res / ss_tot if ss_tot else float("nan"), len(xs))


def squad_elo_scale(strength_model, cards_df, club_elo: dict[str, float]) -> SquadEloScale:
    """The fitted squad-rating -> Elo line, anchored on the real clubs.

    Uses the same card catalogue the draft deals from, so the eleven a player
    assembles is measured on exactly the scale its opponents were measured on.
    `strength_model` is accepted and unused; kept so callers do not have to know
    that the fit stopped needing it.
    """
    from mundialytics.statistical_core.squadlab.cards import best_eleven

    # the catalogue names clubs the way the match data does ("man city") and the
    # Elo table names them its own way ("Man City"); a lowercase compare matched
    # 19 of 96 and fitted the line on a fifth of the evidence
    resolver = make_resolver(list(club_elo))
    overalls, elos = {}, {}
    for club in cards_df.loc[cards_df["kind"] == "actual", "club"].unique():
        xi = best_eleven(cards_df, club)
        hit = resolver(str(club))
        if len(xi) >= 10 and hit is not None:
            overalls[club] = [c.overall for c in xi]
            elos[club] = float(club_elo[hit])
    return fit_squad_elo_scale(overalls, elos)


# ── the run ────────────────────────────────────────────────────────────────────
@dataclass
class ChampionsMatch:
    stage: str                 # "liga" | playoff | r16 | qf | sf | final
    matchday: int
    home: str
    away: str
    home_goals: int
    away_goals: int
    squad_is_home: bool | None = None
    goal_events: list = field(default_factory=list)     # [(scorer, assister)]
    card_players: list = field(default_factory=list)
    ratings: dict = field(default_factory=dict)
    note: str = ""
    # Only drawn for the squad's own matches, and only so the live replay has
    # something to count up to. Same Elo->events mapping the European
    # forecasting page uses (elo_event_calibration.json), so a match against
    # Bayern really does produce fewer shots than one against Sabah.
    shots: tuple[int, int] = (0, 0)
    sot: tuple[int, int] = (0, 0)
    corners: tuple[int, int] = (0, 0)
    yellows: tuple[int, int] = (0, 0)
    xg: tuple[float, float] = (0.0, 0.0)


@dataclass
class ChampionsResult:
    table: pd.DataFrame
    matches: list[ChampionsMatch]
    rounds: dict
    champion: str
    runner_up: str
    squad_stage: str
    squad_position: int
    scorers: pd.DataFrame
    squad_elo: float


class ChampionsRun:
    """One playthrough: eight league-phase games, then the bracket.

    Your matches are played in full — scorers, assists, cards, ratings — while
    the other 35 clubs' games resolve as scorelines, exactly the split the
    domestic season already used. Attributing events for 144 matches nobody
    reads would cost far more than it tells anyone.
    """

    def __init__(self, squad_name: str, squad: list[PlayerStrengthProfile],
                 squad_elo: float, field_: dict, calib: dict | None = None,
                 rng: np.random.Generator | None = None):
        self.squad_name = squad_name
        self.squad = squad
        self.rng = rng or np.random.default_rng(7)
        self.calib = calib or load_calibration(ROOT)

        elo = dict(field_["elo"])
        self.replaced = weakest_team(elo)
        del elo[self.replaced]
        elo[squad_name] = float(squad_elo)
        self.elo = elo
        self.squad_elo = float(squad_elo)

        fx = field_["fixtures"].copy()
        fx["home"] = fx["home"].replace(self.replaced, squad_name)
        fx["away"] = fx["away"].replace(self.replaced, squad_name)
        self.fixtures = fx

        self.tour = EuropeanTournament("champions", self.elo, self.calib, rng=self.rng)
        self.idx = self.tour.idx
        self.events_calib = load_event_calibration(ROOT)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _goals(self, lam: float) -> int:
        return int(self.rng.poisson(float(np.clip(lam, 0.05, 6.0))))

    def _play(self, home: str, away: str, stage: str, matchday: int,
              neutral: bool = False) -> ChampionsMatch:
        i, j = self.idx[home], self.idx[away]
        lam_h = self.tour.lam_h_neutral[i, j] if neutral else self.tour.lam_h[i, j]
        lam_a = self.tour.lam_a_neutral[i, j] if neutral else self.tour.lam_a[i, j]
        hg, ag = self._goals(lam_h), self._goals(lam_a)
        m = ChampionsMatch(stage=stage, matchday=matchday, home=home, away=away,
                           home_goals=hg, away_goals=ag)
        if self.squad_name in (home, away):
            m.squad_is_home = home == self.squad_name
            gf, ga = (hg, ag) if m.squad_is_home else (ag, hg)
            m.goal_events = attribute_goals(self.squad, gf, self.rng)
            self._draw_events(m, home, away, neutral)
            n_cards = int(m.yellows[0] if m.squad_is_home else m.yellows[1])
            m.card_players = attribute_cards(self.squad, n_cards, self.rng)
            m.ratings = compute_match_ratings(self.squad, m.goal_events, m.card_players,
                                              goals_conceded=ga, rng=self.rng)
        return m

    def _draw_events(self, m: ChampionsMatch, home: str, away: str, neutral: bool) -> None:
        """Shots, corners and cards for one match, from the Elo event mapping."""
        if not self.events_calib:
            return
        eh, ea = self.elo[home], self.elo[away]
        pred = predict_euro_events(eh if not neutral else eh, ea, self.events_calib)
        for market, attr in (("shots", "shots"), ("sot", "sot"),
                             ("corners", "corners"), ("yellows", "yellows")):
            v = pred.get(market)
            if not v:
                continue
            setattr(m, attr, (int(self.rng.poisson(v["lambda_home"])),
                              int(self.rng.poisson(v["lambda_away"]))))
        i, j = self.idx[home], self.idx[away]
        lam_h = self.tour.lam_h_neutral[i, j] if neutral else self.tour.lam_h[i, j]
        lam_a = self.tour.lam_a_neutral[i, j] if neutral else self.tour.lam_a[i, j]
        # xG anchored on the model's expected goals and nudged by the shots this
        # match actually produced, with Gamma dispersion — the same shape the
        # domestic season simulator uses, so the two read alike.
        m.xg = (round(float(self.rng.gamma(4.0, max(0.5 * float(lam_h) + 0.055 * m.shots[0], .05) / 4.0)), 2),
                round(float(self.rng.gamma(4.0, max(0.5 * float(lam_a) + 0.055 * m.shots[1], .05) / 4.0)), 2))

    # ── league phase ──────────────────────────────────────────────────────────
    def _league_phase(self) -> tuple[pd.DataFrame, list[ChampionsMatch]]:
        state = {t: {"pts": 0, "gf": 0, "ga": 0, "played": 0} for t in self.elo}
        matches: list[ChampionsMatch] = []
        own = 0
        for r in self.fixtures.itertuples(index=False):
            if r.home not in state or r.away not in state:
                continue
            involves = self.squad_name in (r.home, r.away)
            if involves:
                own += 1
            m = self._play(r.home, r.away, "liga", own if involves else 0)
            matches.append(m)
            for team, gf, ga in ((m.home, m.home_goals, m.away_goals),
                                 (m.away, m.away_goals, m.home_goals)):
                s = state[team]
                s["played"] += 1
                s["gf"] += gf
                s["ga"] += ga
                s["pts"] += 3 if gf > ga else (1 if gf == ga else 0)
        rows = [{"team": t, **s, "gd": s["gf"] - s["ga"]} for t, s in state.items()]
        table = (pd.DataFrame(rows)
                 .sort_values(["pts", "gd", "gf"], ascending=False)
                 .reset_index(drop=True))
        table.insert(0, "pos", range(1, len(table) + 1))
        return table, matches

    # ── knockout ──────────────────────────────────────────────────────────────
    def _tie(self, a: str, b: str, stage: str) -> dict:
        """Two legs, a hosting the second — the seeding convention the European
        layer already uses. Extra time is a third of a match, then coin-flip
        penalties, same as the forecasting model."""
        leg1 = self._play(b, a, stage, 1)
        leg2 = self._play(a, b, stage, 2)
        agg_a = leg1.away_goals + leg2.home_goals
        agg_b = leg1.home_goals + leg2.away_goals
        note = ""
        if agg_a != agg_b:
            win = a if agg_a > agg_b else b
        else:
            i, j = self.idx[a], self.idx[b]
            et_a = self._goals(self.tour.lam_h[i, j] / 3.0)
            et_b = self._goals(self.tour.lam_a[i, j] / 3.0)
            if et_a != et_b:
                win, note = (a, "pró.") if et_a > et_b else (b, "pró.")
            else:
                win, note = (a if self.rng.random() < 0.5 else b), "pen."
        return {"team_a": a, "team_b": b,
                "leg1": f"{leg1.away_goals}-{leg1.home_goals}",
                "leg2": f"{leg2.home_goals}-{leg2.away_goals}",
                "agg": f"{agg_a}-{agg_b}", "winner": win, "note": note,
                "matches": [leg1, leg2]}

    def _final(self, a: str, b: str) -> dict:
        m = self._play(a, b, "final", 1, neutral=True)
        note = ""
        if m.home_goals == m.away_goals:
            i, j = self.idx[a], self.idx[b]
            ea = self._goals(self.tour.lam_h_neutral[i, j] / 3.0)
            eb = self._goals(self.tour.lam_a_neutral[i, j] / 3.0)
            win, note = ((a, "pró.") if ea > eb else (b, "pró.")) if ea != eb else \
                ((a if self.rng.random() < 0.5 else b), "pen.")
        else:
            win = a if m.home_goals > m.away_goals else b
        m.note = note
        return {"team_a": a, "team_b": b, "leg1": f"{m.home_goals}-{m.away_goals}",
                "leg2": "", "agg": f"{m.home_goals}-{m.away_goals}", "winner": win,
                "note": note, "matches": [m]}

    # ── the whole thing ───────────────────────────────────────────────────────
    def play(self) -> ChampionsResult:
        table, matches = self._league_phase()
        order = list(table["team"])
        top8, seeded, unseeded = order[:8], order[8:16], order[16:24]

        rounds: dict[str, list] = {}
        rounds["playoff"] = [self._tie(seeded[k], unseeded[7 - k], "playoff") for k in range(8)]
        po = [t["winner"] for t in rounds["playoff"]]
        rounds["r16"] = [self._tie(top8[k], po[7 - k], "r16") for k in range(8)]
        alive = [t["winner"] for t in rounds["r16"]]
        rounds["qf"] = [self._tie(alive[a], alive[b], "qf")
                        for a, b in ((0, 7), (3, 4), (1, 6), (2, 5))]
        q = [t["winner"] for t in rounds["qf"]]
        rounds["sf"] = [self._tie(q[0], q[1], "sf"), self._tie(q[2], q[3], "sf")]
        s = [t["winner"] for t in rounds["sf"]]
        rounds["final"] = [self._final(s[0], s[1])]

        for key in ("playoff", "r16", "qf", "sf", "final"):
            for tie in rounds[key]:
                matches.extend(tie["matches"])

        champ = rounds["final"][0]["winner"]
        runner = s[1] if champ == s[0] else s[0]
        pos = int(table.loc[table["team"] == self.squad_name, "pos"].iloc[0])
        return ChampionsResult(
            table=table, matches=matches, rounds=rounds, champion=champ,
            runner_up=runner, squad_stage=self._squad_stage(rounds, order, champ),
            squad_position=pos, scorers=self._scorers(matches), squad_elo=self.squad_elo,
        )

    def _squad_stage(self, rounds: dict, order: list[str], champ: str) -> str:
        if champ == self.squad_name:
            return "Campeón"
        if self.squad_name in [t["team_a"] for t in rounds["final"]] + \
                [t["team_b"] for t in rounds["final"]]:
            return "Finalista"
        for key in ("sf", "qf", "r16", "playoff"):
            in_round = any(self.squad_name in (t["team_a"], t["team_b"]) for t in rounds[key])
            if in_round:
                won = any(t["winner"] == self.squad_name for t in rounds[key])
                if not won:
                    return f"Eliminado en {ROUND_LABELS[key].lower()}"
        if self.squad_name in order[24:]:
            return "Eliminado en la fase liga"
        return "Eliminado"

    def _scorers(self, matches: list[ChampionsMatch]) -> pd.DataFrame:
        tally: dict[str, dict[str, int]] = {}
        for m in matches:
            for scorer, assister in m.goal_events or []:
                tally.setdefault(scorer, {"goles": 0, "asistencias": 0})["goles"] += 1
                if assister:
                    tally.setdefault(assister, {"goles": 0, "asistencias": 0})["asistencias"] += 1
        if not tally:
            return pd.DataFrame(columns=["jugador", "goles", "asistencias"])
        return (pd.DataFrame([{"jugador": k, **v} for k, v in tally.items()])
                .sort_values(["goles", "asistencias"], ascending=False)
                .reset_index(drop=True))
