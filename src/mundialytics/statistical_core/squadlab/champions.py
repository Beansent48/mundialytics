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

THE SLOT. Your club takes a RANDOM club's place in the real draw (it used to be
always the weakest, Sabah), which keeps the fixture list, the pots and the
schedule exactly as they are — you play the eight opponents that club was drawn
against, and the draw changes every time you play. Pass a fresh rng for a fresh
draw.
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
from mundialytics.statistical_core.squadlab.rival_squads import load_rival_squads
from mundialytics.statistical_core.squadlab.player_rating import attribute_goals

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
    # the matchday's date, so injuries can be counted in days on the real calendar
    league2["date"] = pd.to_datetime(league2["Date"], format="%d/%m/%Y %H:%M", errors="coerce")
    league = league2[["home", "away", "home_goals", "away_goals", "date"]].reset_index(drop=True)

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


# ── the squad's shape ──────────────────────────────────────────────────────────
# What an eleven's SHAPE adds to its level, per standard deviation of each axis
# beyond what its mean overall already implies. Fitted on real clubs' own best
# elevens against 2024/25-2025/26 big-five goals (Poisson, Elo from the squad
# line as the baseline) and judged on 2026/27, which no card has seen:
# scripts/experiment_squadlab_attack_defense_split.py. The signs held in 5-6 of
# 6 season fits; the out-of-sample gain was −0.0046 NLL/match on 209 matches
# (90% CI −0.011..+0.001), small and not yet conclusive — re-check as the
# season grows. It replaces the old balance dock (a harmonic-mean goal dock), a
# bounded guess that was never measured.
SHAPE_COEF = {"att": 0.017, "def": -0.025, "gk": -0.018}
SHAPE_CLIP = 2.5


def shape_features(xi) -> dict[str, float] | None:
    """The three axes the fit reads, from eleven cards (engine scale)."""
    out = [c for c in xi if c.position != "Goalkeeper"]
    gk = [c for c in xi if c.position == "Goalkeeper"]
    dfn = [c.raw_defense for c in out if c.position in ("Defender", "Midfielder")]
    if not out or not gk or not dfn:
        return None
    att = sorted((c.raw_attack for c in out), reverse=True)[:5]
    return {"ovr": float(np.mean([c.overall for c in xi])), "att": float(np.mean(att)),
            "def": float(np.mean(dfn)), "gk": float(gk[0].raw_gk)}


@dataclass
class SquadShape:
    """Residual lines (axis ~ mean overall across real clubs) and their spread."""
    lines: dict[str, tuple[float, float, float]]      # axis -> (intercept, slope, sd)

    def z(self, xi) -> dict[str, float]:
        f = shape_features(xi)
        if f is None:
            return {k: 0.0 for k in SHAPE_COEF}
        out = {}
        for k, (a, b, sd) in self.lines.items():
            r = (f[k] - (a + b * f["ovr"])) / sd if sd > 0 else 0.0
            out[k] = float(np.clip(r, -SHAPE_CLIP, SHAPE_CLIP))
        return out

    def multipliers(self, xi) -> tuple[float, float]:
        """(own goals ×, goals conceded ×) for an eleven."""
        z = self.z(xi)
        return (float(np.exp(SHAPE_COEF["att"] * z["att"])),
                float(np.exp(SHAPE_COEF["def"] * z["def"] + SHAPE_COEF["gk"] * z["gk"])))


def fit_squad_shape(cards_df) -> SquadShape:
    from mundialytics.statistical_core.squadlab.cards import best_eleven

    rows = []
    for club in cards_df.loc[cards_df["kind"] == "actual", "club"].unique():
        xi = best_eleven(cards_df, club)
        if len(xi) >= 10 and (f := shape_features(xi)):
            rows.append(f)
    f = pd.DataFrame(rows)
    lines = {}
    for k in SHAPE_COEF:
        if len(f) < 12:
            lines[k] = (0.0, 0.0, 0.0)
            continue
        b, a = np.polyfit(f["ovr"], f[k], 1)
        sd = float((f[k] - (a + b * f["ovr"])).std())
        lines[k] = (float(a), float(b), sd)
    return SquadShape(lines)


# ── the calendar ───────────────────────────────────────────────────────────────
# The knockout dates are UEFA's usual rhythm for a season starting in 2026 (the
# fixture file only carries the league phase): a week between legs, three or
# four weeks between rounds, the final at the end of May. Only injuries read
# them, as days out, so a day either way changes nothing.
KO_DATES = {
    ("playoff", 1): "2027-02-17", ("playoff", 2): "2027-02-24",
    ("r16", 1): "2027-03-10", ("r16", 2): "2027-03-17",
    ("qf", 1): "2027-04-07", ("qf", 2): "2027-04-14",
    ("sf", 1): "2027-04-28", ("sf", 2): "2027-05-05",
    ("final", 1): "2027-05-29",
}

# UEFA's disciplinary rules, as a competition runs them: a one-match ban on the
# third booking and on every second one after it, cautions wiped after the
# quarter-finals, a one-match ban for a sending-off (two bookings in one match
# count as the red, not towards the tally).
YELLOW_BAN_AT = (3, 5, 7, 9)
YELLOWS_RESET_BEFORE = "sf"


def _person(name: str) -> str:
    """One man's key, so a drafted player is struck off his real club too."""
    from mundialytics.identity.current_squads import short_key
    return short_key(str(name))


@dataclass
class Absence:
    player: str
    reason: str          # "sancion" | "lesion"
    detail: str = ""


class SquadState:
    """Your eleven and bench across the whole competition: bans and injuries.

    `xi` is the eleven the user picked, in slot order; `bench` the reserves.
    Before each match an unavailable starter is replaced by the best available
    reserve in his position, else the best available outfielder — what any
    manager would do. The eleven the user chose is restored as soon as its
    players are back.
    """

    def __init__(self, xi, bench, dyn):
        from mundialytics.statistical_core.squadlab.match_players import from_card

        self.cards = {c.card_id: c for c in list(xi) + list(bench)}
        self.mp = {c.card_id: from_card(c) for c in self.cards.values()}
        self.xi = [c.card_id for c in xi]
        self.bench = [c.card_id for c in bench]
        self.yellows = {k: 0 for k in self.cards}
        self.banned: dict[str, int] = {}                 # key -> matches left
        self.injured_until: dict[str, pd.Timestamp] = {}
        self.dyn = dyn
        # tournament totals, for the result screen
        self.totals = {k: {"apps": 0, "minutes": 0, "goals": 0, "assists": 0, "yellows": 0,
                           "reds": 0, "injuries": 0, "rating_sum": 0.0} for k in self.cards}

    def available(self, key: str, when: pd.Timestamp | None) -> str | None:
        if self.banned.get(key, 0) > 0:
            return "sancion"
        until = self.injured_until.get(key)
        if until is not None and (when is None or when < until):
            return "lesion"
        return None

    def lineup(self, when: pd.Timestamp | None):
        """(starter cards, bench cards, absences) for a match on `when`."""
        out_reason = {k: self.available(k, when) for k in self.cards}
        free_bench = [k for k in self.bench if out_reason[k] is None]
        starters, absences = [], []
        for k in self.xi:
            if out_reason[k] is None:
                starters.append(k)
                continue
            absences.append(Absence(self.cards[k].player, out_reason[k], self._detail(k, when)))
            pos = self.cards[k].position
            same = [b for b in free_bench if self.cards[b].position == pos]
            pool = same or [b for b in free_bench if self.cards[b].position != "Goalkeeper"]
            if pool:
                pick = max(pool, key=lambda b: self.cards[b].overall)
                free_bench.remove(pick)
                starters.append(pick)
        for k in self.bench:
            if out_reason[k] is not None:
                absences.append(Absence(self.cards[k].player, out_reason[k], self._detail(k, when)))
        return ([self.cards[k] for k in starters], [self.cards[k] for k in free_bench], absences)

    def _detail(self, key: str, when) -> str:
        if self.banned.get(key, 0) > 0:
            return "1 partido"
        until = self.injured_until.get(key)
        if until is not None and when is not None:
            return f"{max((until - when).days, 1)} días"
        return ""

    def after_match(self, log, side: str, stage: str, when: pd.Timestamp | None, rng) -> None:
        """Serve bans, book new ones, start injuries."""
        played = set(log.lines[side])
        for k in list(self.banned):
            if k in self.cards and k not in played and self.banned[k] > 0:
                self.banned[k] -= 1
        for key, line in log.lines[side].items():
            t = self.totals.get(key)
            if t is None:
                continue
            t["apps"] += 1
            t["minutes"] += line.minutes
            t["goals"] += line.goals
            t["assists"] += line.assists
            t["rating_sum"] += line.rating
            sent_off_2y = line.red and line.yellows >= 2
            if line.red:
                t["reds"] += 1
                self.banned[key] = self.banned.get(key, 0) + 1
            # a second booking in one match counts as the red, not the tally
            new_y = 0 if sent_off_2y else line.yellows
            t["yellows"] += line.yellows
            before = self.yellows[key]
            self.yellows[key] = before + new_y
            if any(before < n <= self.yellows[key] for n in YELLOW_BAN_AT):
                self.banned[key] = self.banned.get(key, 0) + 1
            if line.injured:
                t["injuries"] += 1
                days = self.dyn.injury_days(rng)
                if days > 0 and when is not None:
                    self.injured_until[key] = when + pd.Timedelta(days=days)

    def reset_yellows(self) -> None:
        self.yellows = {k: 0 for k in self.yellows}


@dataclass
class ChampionsMatch:
    stage: str                 # "liga" | playoff | r16 | qf | sf | final
    matchday: int
    home: str
    away: str
    home_goals: int
    away_goals: int
    squad_is_home: bool | None = None
    goal_events: list = field(default_factory=list)     # [(scorer, assister)] — your players
    # The same, for the clubs you are not managing, keyed by club.
    rival_events: dict = field(default_factory=dict)
    card_players: list = field(default_factory=list)
    ratings: dict = field(default_factory=dict)
    note: str = ""
    shots: tuple[int, int] = (0, 0)
    sot: tuple[int, int] = (0, 0)
    corners: tuple[int, int] = (0, 0)
    yellows: tuple[int, int] = (0, 0)
    xg: tuple[float, float] = (0.0, 0.0)
    # the live log of your own matches: every event, minute by minute
    log: object = None
    date: object = None
    absences: list = field(default_factory=list)
    lineup: list = field(default_factory=list)        # the eleven that started
    own_goals_for: int = 0
    own_goals_against: int = 0


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
    squad_totals: dict = field(default_factory=dict)


class ChampionsRun:
    """One playthrough: eight league-phase games, then the bracket.

    YOUR matches are played minute by minute by the live engine
    (match_engine.py): your eleven and bench against the opponent's real one,
    with bans and injuries carried from match to match and the eleven's Elo
    recomputed from whoever actually starts. The other clubs' games resolve as
    scorelines, named by the same attribution as before — playing 150 matches
    nobody watches minute by minute would cost far more than it tells anyone.
    """

    def __init__(self, squad_name: str, squad, squad_elo: float, field_: dict,
                 calib: dict | None = None, rng: np.random.Generator | None = None,
                 *, bench=(), scale: SquadEloScale | None = None,
                 shape: SquadShape | None = None, cards_df: pd.DataFrame | None = None):
        from mundialytics.statistical_core.squadlab.match_engine import load_dynamics

        self.squad_name = squad_name
        self.squad = list(squad)                     # eleven Cards
        self.bench = list(bench)
        self.rng = rng or np.random.default_rng(7)
        self.calib = calib or load_calibration(ROOT)
        self.scale = scale
        self.shape = shape
        self.dyn = load_dynamics()

        elo = dict(field_["elo"])
        # a RANDOM draw: your eleven takes a random club's slot in the real
        # field, so the eight opponents and the path change every time you play
        self.replaced = str(self.rng.choice(sorted(elo)))
        del elo[self.replaced]
        elo[squad_name] = float(squad_elo)
        self.elo = elo
        self.squad_elo = float(squad_elo)

        fx = field_["fixtures"].copy()
        fx["home"] = fx["home"].replace(self.replaced, squad_name)
        fx["away"] = fx["away"].replace(self.replaced, squad_name)
        self.fixtures = fx

        drafted = {_person(c.player) for c in self.squad + self.bench}
        self.drafted = drafted
        self.rival_squads = {
            team: [pl for pl in players if _person(pl.player) not in drafted]
            for team, players in load_rival_squads(
                [t for t in self.elo if t != squad_name]).items()
        }
        self.cards_df = cards_df
        self._sides_cache: dict[str, tuple] = {}
        self.state = SquadState(self.squad, self.bench, self.dyn)
        self.gap = typical_gap_cached(cards_df) if cards_df is not None else {}

        self.tour = EuropeanTournament("champions", self.elo, self.calib, rng=self.rng)
        self.idx = self.tour.idx
        self.events_calib = load_event_calibration(ROOT)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _goals(self, lam: float) -> int:
        return int(self.rng.poisson(float(np.clip(lam, 0.05, 6.0))))

    def _rival_side(self, team: str):
        """The opponent's real eleven and bench (drafted players struck off)."""
        if team in self._sides_cache:
            return self._sides_cache[team]
        from mundialytics.statistical_core.squadlab.match_players import (
            club_side_from_cards, club_side_from_rows,
        )
        xi, bench = [], []
        club = _card_club_for(self.cards_df, team) if self.cards_df is not None else None
        if club:
            xi, bench = club_side_from_cards(self.cards_df, club, self.drafted)
        if len(xi) < 11:
            rows = _current_rows_for(team)
            if rows is not None and len(rows):
                xi, bench = club_side_from_rows(rows, self.drafted)
        self._sides_cache[team] = (xi, bench)
        return xi, bench

    def _squad_lams(self, home: str, away: str, neutral: bool, xi) -> tuple[float, float, float]:
        """(λ home, λ away, squad Elo) with your fielded eleven's own Elo and shape."""
        squad_elo = (self.scale.elo_for([c.overall for c in xi]) if self.scale is not None
                     else self.squad_elo)
        eh = squad_elo if home == self.squad_name else self.elo[home]
        ea = squad_elo if away == self.squad_name else self.elo[away]
        c, hfa, b = self.calib["c"], self.calib["hfa"], self.calib["b"]
        d = (eh - ea) / 400.0
        lh = float(np.exp(c + (0.0 if neutral else hfa) + b * d))
        la = float(np.exp(c - b * d))
        if self.shape is not None:
            own, conceded = self.shape.multipliers(xi)
            if home == self.squad_name:
                lh, la = lh * own, la * conceded
            else:
                la, lh = la * own, lh * conceded
        return lh, la, squad_elo

    def _play(self, home: str, away: str, stage: str, matchday: int,
              neutral: bool = False, when=None, start_score=None,
              decide: bool = False) -> ChampionsMatch:
        if self.squad_name in (home, away):
            return self._play_live(home, away, stage, matchday, neutral, when, start_score, decide)
        i, j = self.idx[home], self.idx[away]
        lam_h = self.tour.lam_h_neutral[i, j] if neutral else self.tour.lam_h[i, j]
        lam_a = self.tour.lam_a_neutral[i, j] if neutral else self.tour.lam_a[i, j]
        hg, ag = self._goals(lam_h), self._goals(lam_a)
        m = ChampionsMatch(stage=stage, matchday=matchday, home=home, away=away,
                           home_goals=hg, away_goals=ag, date=when)
        for team, scored in ((home, hg), (away, ag)):
            if scored <= 0:
                continue
            players = self.rival_squads.get(team)
            if players:
                m.rival_events[team] = attribute_goals(players, scored, self.rng)
        return m

    def _play_live(self, home, away, stage, matchday, neutral, when, start_score, decide):
        from mundialytics.statistical_core.squadlab.match_engine import Side, play_match
        from mundialytics.statistical_core.squadlab.match_players import from_card

        xi_cards, bench_cards, absences = self.state.lineup(when)
        lh, la, squad_elo = self._squad_lams(home, away, neutral, xi_cards)
        eh = squad_elo if home == self.squad_name else self.elo[home]
        ea = squad_elo if away == self.squad_name else self.elo[away]
        ev = predict_euro_events(eh, ea, self.events_calib) if self.events_calib else {}

        def lam_of(mk, side):
            v = ev.get(mk)
            return float(v[f"lambda_{side}"]) if v else {"yellows": 2.0, "shots": 12.0, "corners": 5.0}[mk]

        slope = self.scale.slope_mean if self.scale is not None else 0.0
        mine = Side(self.squad_name, [from_card(c) for c in xi_cards], [from_card(c) for c in bench_cards],
                    elo_per_ovr=slope)
        opp_name = away if home == self.squad_name else home
        oxi, obench = self._rival_side(opp_name)
        theirs = Side(opp_name, list(oxi) or _anonymous_eleven(opp_name), list(obench))
        H, A = (mine, theirs) if home == self.squad_name else (theirs, mine)
        for side, tag in ((H, "home"), (A, "away")):
            side.lam = lh if tag == "home" else la
            side.yellows_lam = lam_of("yellows", tag)
            side.shots_lam = lam_of("shots", tag)
            side.corners_lam = lam_of("corners", tag)

        log = play_match(H, A, self.rng, dyn=self.dyn, neutral=neutral,
                         extra_time_if_level=decide, shootout_if_level=decide,
                         typical_gap=self.gap, start_score=start_score)
        from mundialytics.statistical_core.squadlab.match_engine import rate_lines
        rate_lines(log, self.rng)

        squad_tag = "home" if home == self.squad_name else "away"
        m = ChampionsMatch(stage=stage, matchday=matchday, home=home, away=away,
                           home_goals=log.home_goals, away_goals=log.away_goals,
                           squad_is_home=home == self.squad_name, log=log, date=when,
                           absences=absences, lineup=[c.player for c in xi_cards])
        for e in log.events:
            if e.kind in ("goal", "pen_goal"):
                pair = (e.player, e.assist if e.kind == "goal" else None)
                if e.side == squad_tag:
                    m.goal_events.append(pair)
                else:
                    m.rival_events.setdefault(opp_name, []).append(pair)
            elif e.kind == "own_goal":
                if e.side == squad_tag:
                    m.own_goals_for += 1
                else:
                    m.own_goals_against += 1
            elif e.kind in ("yellow",) and e.side == squad_tag:
                m.card_players.append(e.player)
        m.ratings = log.lines[squad_tag]
        m.shots, m.sot = log.stats["shots"], log.stats["sot"]
        m.corners, m.yellows, m.xg = log.stats["corners"], log.stats["yellows"], log.stats["xg"]
        if log.shootout:
            m.note = "pen."
        elif log.extra_time:
            m.note = "pró."
        self.state.after_match(log, squad_tag, stage, when, self.rng)
        return m

    # ── league phase ──────────────────────────────────────────────────────────
    def _league_phase(self) -> tuple[pd.DataFrame, list[ChampionsMatch]]:
        state = {t: {"pts": 0, "gf": 0, "ga": 0, "played": 0} for t in self.elo}
        matches: list[ChampionsMatch] = []
        own = 0
        fx = self.fixtures
        if "date" in fx.columns:
            fx = fx.sort_values("date", kind="stable")
        for r in fx.itertuples(index=False):
            if r.home not in state or r.away not in state:
                continue
            involves = self.squad_name in (r.home, r.away)
            if involves:
                own += 1
            when = getattr(r, "date", None)
            when = None if when is None or pd.isna(when) else pd.Timestamp(when)
            m = self._play(r.home, r.away, "liga", own if involves else 0, when=when)
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
        """Two legs, a hosting the second. Your ties are decided on the pitch —
        extra time and a shoot-out played kick by kick in the second leg; the
        rest keep the forecasting layer's rule (a third of a match, then a coin
        toss, which is what equal conversion makes a shoot-out)."""
        if stage == YELLOWS_RESET_BEFORE and self.squad_name in (a, b):
            self.state.reset_yellows()
        d1 = pd.Timestamp(KO_DATES[(stage, 1)])
        d2 = pd.Timestamp(KO_DATES[(stage, 2)])
        leg1 = self._play(b, a, stage, 1, when=d1)
        mine = self.squad_name in (a, b)
        # the second leg, from the home side's point of view: a is at home
        start = (leg1.away_goals, leg1.home_goals)
        leg2 = self._play(a, b, stage, 2, when=d2, start_score=start, decide=mine)
        agg_a = leg1.away_goals + leg2.home_goals
        agg_b = leg1.home_goals + leg2.away_goals
        note = ""
        if mine:
            log = leg2.log
            win = log.winner if log.shootout else (a if agg_a > agg_b else b)
            if log.shootout:
                note = "pen."
                win = a if log.shootout[0] > log.shootout[1] else b
            elif log.extra_time:
                note = "pró."
        elif agg_a != agg_b:
            win = a if agg_a > agg_b else b
        else:
            i, j = self.idx[a], self.idx[b]
            et_a = self._goals(self.tour.lam_h[i, j] / 3.0)
            et_b = self._goals(self.tour.lam_a[i, j] / 3.0)
            if et_a != et_b:
                win, note = (a, "pró.") if et_a > et_b else (b, "pró.")
                agg_a, agg_b = agg_a + et_a, agg_b + et_b
            else:
                win, note = (a if self.rng.random() < 0.5 else b), "pen."
        return {"team_a": a, "team_b": b,
                "leg1": f"{leg1.away_goals}-{leg1.home_goals}",
                "leg2": f"{leg2.home_goals}-{leg2.away_goals}",
                "agg": f"{agg_a}-{agg_b}", "winner": win, "note": note,
                "matches": [leg1, leg2]}

    def _final(self, a: str, b: str) -> dict:
        when = pd.Timestamp(KO_DATES[("final", 1)])
        mine = self.squad_name in (a, b)
        m = self._play(a, b, "final", 1, neutral=True, when=when, decide=mine)
        note = ""
        if mine:
            log = m.log
            if log.shootout:
                note = "pen."
                win = a if log.shootout[0] > log.shootout[1] else b
            else:
                win = a if m.home_goals > m.away_goals else b
                note = "pró." if log.extra_time else ""
        elif m.home_goals == m.away_goals:
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
            squad_totals={self.state.cards[k].player: v for k, v in self.state.totals.items()},
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
        tally: dict[str, dict] = {}

        def add(who: str, team: str, key: str) -> None:
            row = tally.setdefault(who, {"equipo": team, "goles": 0, "asistencias": 0})
            row[key] += 1

        for m in matches:
            for scorer, assister in m.goal_events or []:
                add(scorer, self.squad_name, "goles")
                if assister:
                    add(assister, self.squad_name, "asistencias")
            for team, events in (m.rival_events or {}).items():
                for scorer, assister in events:
                    add(scorer, team, "goles")
                    if assister:
                        add(assister, team, "asistencias")
        if not tally:
            return pd.DataFrame(columns=["jugador", "equipo", "goles", "asistencias"])
        return (pd.DataFrame([{"jugador": k, **v} for k, v in tally.items()])
                .sort_values(["goles", "asistencias"], ascending=False)
                .reset_index(drop=True))


# ── lookups ────────────────────────────────────────────────────────────────────
_GAP_CACHE: dict[int, dict] = {}


def typical_gap_cached(cards_df: pd.DataFrame) -> dict[str, float]:
    from mundialytics.statistical_core.squadlab.match_players import typical_bench_gap

    key = id(cards_df)
    if key not in _GAP_CACHE:
        _GAP_CACHE[key] = typical_bench_gap(cards_df)
    return _GAP_CACHE[key]


_CLUB_MAP: dict[int, dict] = {}


def _card_club_for(cards_df: pd.DataFrame, team: str) -> str | None:
    """Field club (ClubElo's spelling) -> the card catalogue's club, if any."""
    key = id(cards_df)
    if key not in _CLUB_MAP:
        from mundialytics.statistical_core.squadlab.rival_squads import FIELD_TO_SQUAD

        clubs = sorted(cards_df.loc[cards_df["kind"] == "actual", "club"].astype(str).unique())
        resolve = make_resolver(clubs)
        _CLUB_MAP[key] = {"_resolve": resolve, "_alias": FIELD_TO_SQUAD, "_clubs": set(clubs)}
    m = _CLUB_MAP[key]
    alias = m["_alias"].get(str(team))
    if alias and alias in m["_clubs"]:
        return alias
    return m["_resolve"](str(team))


def _current_rows_for(team: str) -> pd.DataFrame | None:
    from mundialytics.statistical_core.squadlab.rival_squads import FIELD_TO_SQUAD

    cs = _current_squads()
    if cs is None or cs.empty:
        return None
    teams = sorted(cs["team"].astype(str).unique())
    key = FIELD_TO_SQUAD.get(str(team)) or make_resolver(teams)(str(team))
    if not key:
        return None
    return cs[cs["team"] == key]


_CS: list = []


def _current_squads():
    if not _CS:
        from mundialytics.identity.current_squads import load_current_squads
        _CS.append(load_current_squads())
    return _CS[0]


def _anonymous_eleven(team: str):
    """A club nobody can name still has to field eleven men."""
    from mundialytics.statistical_core.squadlab.match_engine import MatchPlayer

    shape = ["Goalkeeper"] + ["Defender"] * 4 + ["Midfielder"] * 3 + ["Forward"] * 3
    return [MatchPlayer(f"{team}:{i}", team, pos, 70.0,
                        g=0.3 if pos == "Forward" else 0.1 if pos == "Midfielder" else 0.05)
            for i, pos in enumerate(shape)]
