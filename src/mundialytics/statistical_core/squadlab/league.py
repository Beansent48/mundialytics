"""SquadLab domestic league: your drafted eleven in a real 20-team season.

The squad replaces one real club and plays 38 matchdays with full minute-by-
minute simulation (bans, injuries, lineup management via SquadState).  The
other 342 fixtures resolve as Poisson scorelines through SeasonLambdaSource,
exactly as SeasonOrchestrator does — only the squad's own matches run through
the live match engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from mundialytics.statistical_core.player_strength import PlayerStrengthProfile
from mundialytics.statistical_core.squadlab.calendar import (
    Fixture,
    generate_double_round_robin,
)
from mundialytics.statistical_core.squadlab.champions import (
    SQUAD_TEAM_NAME,
    Absence,
    ChampionsMatch,
    SquadState,
)
from mundialytics.statistical_core.squadlab.lambda_source import SeasonLambdaSource
from mundialytics.statistical_core.squadlab.player_rating import attribute_goals
from mundialytics.statistical_core.squadlab.rival_squads import load_rival_squads


# ── competition discipline ────────────────────────────────────────────────────

@dataclass(frozen=True)
class CompetitionDiscipline:
    yellow_ban_at: tuple[int, ...]
    yellows_reset_before_md: int | None = None

DISCIPLINE: dict[str, CompetitionDiscipline] = {
    "La Liga":        CompetitionDiscipline((5,), 19),
    "Premier League": CompetitionDiscipline((5, 10), None),
    "Serie A":        CompetitionDiscipline((5,), 19),
    "Bundesliga":     CompetitionDiscipline((5,), None),
    "Ligue 1":        CompetitionDiscipline((3,), None),
}

# Stage labels from final league position.
STAGE_LABELS = [
    (1, 1, "Campeón"),
    (2, 4, "Clasificación Champions"),
    (5, 6, "Europa League"),
    (7, 13, "Mitad de tabla"),
    (14, 17, "Salvación"),
    (18, 20, "Descenso"),
]


def stage_for_position(pos: int) -> str:
    for lo, hi, label in STAGE_LABELS:
        if lo <= pos <= hi:
            return label
    return "Mitad de tabla"


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class LeagueResult:
    competition: str
    competition_name: str
    replaced: str
    table: pd.DataFrame
    matches: list[ChampionsMatch]       # the squad's 38 matches (live detail)
    all_results: list[dict]             # all 380 scorelines (matchday, home, away, hg, ag)
    scorers: pd.DataFrame
    squad_totals: dict
    squad_position: int
    stage: str


# ── orchestrator ──────────────────────────────────────────────────────────────

class LeagueRun:
    """One playthrough of a 20-team domestic season."""

    def __init__(
        self,
        squad_name: str,
        squad: Sequence,            # eleven Cards
        bench: Sequence,            # up to seven Cards
        competition: str,           # AD-model competition name
        teams: list[str],           # the 20 real teams
        replaced: str,              # which team the user replaces
        lambda_source: SeasonLambdaSource,
        cards_df: pd.DataFrame | None = None,
        rng: np.random.Generator | None = None,
        discipline: CompetitionDiscipline | None = None,
    ):
        from mundialytics.statistical_core.squadlab.match_engine import load_dynamics
        from mundialytics.statistical_core.squadlab.match_players import (
            club_side_from_cards,
        )

        self.squad_name = squad_name
        self.squad = list(squad)
        self.bench = list(bench)
        self.competition = competition
        self.replaced = replaced
        self.lambda_source = lambda_source
        self.cards_df = cards_df
        self.rng = rng or np.random.default_rng(42)
        self.discipline = discipline or DISCIPLINE.get(
            competition, CompetitionDiscipline((5,), None)
        )
        self.dyn = load_dynamics()

        # Build the field: replace one team with the squad.
        self.teams = [
            squad_name if t == replaced else t for t in teams
        ]

        # Rival squads for the live engine (drafted players excluded).
        drafted = {_person(c.player) for c in self.squad + self.bench}
        self.drafted = drafted
        real_teams = [t for t in self.teams if t != squad_name]
        self.rival_squads = {
            team: [pl for pl in players if _person(pl.player) not in drafted]
            for team, players in load_rival_squads(real_teams).items()
        }

        self.state = SquadState(self.squad, self.bench, self.dyn)
        self._sides_cache: dict[str, tuple] = {}

        # Typical gap for rating calibration.
        from mundialytics.statistical_core.squadlab.champions import typical_gap_cached
        self.gap = typical_gap_cached(cards_df) if cards_df is not None else {}

        # Calendar.
        seed_int = int(self.rng.integers(0, 2**31))
        self.fixtures = generate_double_round_robin(self.teams, shuffle_seed=seed_int)

    # ── play the season ───────────────────────────────────────────────────────

    def play(self) -> LeagueResult:
        n_matchdays = max(f.matchday for f in self.fixtures)
        all_results: list[dict] = []
        own_matches: list[ChampionsMatch] = []

        # Precompute lambdas for all 380 fixtures (cached inside the source).
        lam_cache: dict[tuple[str, str], tuple[float, float]] = {}
        for f in self.fixtures:
            key = (f.home, f.away)
            if key not in lam_cache:
                lam_cache[key] = self.lambda_source.match_lambdas(
                    f.home, f.away, competition=self.competition, neutral=False,
                )

        # Table state for progressive standings.
        table_state = {t: {"played": 0, "pts": 0, "gf": 0, "ga": 0} for t in self.teams}

        # Group fixtures by matchday.
        by_md: dict[int, list[Fixture]] = {}
        for f in self.fixtures:
            by_md.setdefault(f.matchday, []).append(f)

        squad_own_idx = 0

        for md in range(1, n_matchdays + 1):
            # Discipline: reset yellows before specified matchday.
            if (self.discipline.yellows_reset_before_md is not None
                    and md == self.discipline.yellows_reset_before_md):
                self.state.reset_yellows()

            # Synthetic date: 7 days apart, starting from a fixed base.
            when = pd.Timestamp("2026-08-18") + pd.Timedelta(weeks=md - 1)

            for fix in by_md.get(md, []):
                involves_squad = self.squad_name in (fix.home, fix.away)

                if involves_squad:
                    squad_own_idx += 1
                    m = self._play_live(
                        fix.home, fix.away, f"J{md}", squad_own_idx, when,
                    )
                    own_matches.append(m)
                    hg, ag = m.home_goals, m.away_goals
                else:
                    lh, la = lam_cache[(fix.home, fix.away)]
                    hg = int(self.rng.poisson(lh))
                    ag = int(self.rng.poisson(la))

                all_results.append({
                    "matchday": md,
                    "home": fix.home,
                    "away": fix.away,
                    "home_goals": hg,
                    "away_goals": ag,
                })

                # Update table.
                for team, gf, ga in ((fix.home, hg, ag), (fix.away, ag, hg)):
                    s = table_state[team]
                    s["played"] += 1
                    s["gf"] += gf
                    s["ga"] += ga
                    s["pts"] += 3 if gf > ga else (1 if gf == ga else 0)

            # Tick bans down for players who did NOT play this matchday.
            for k in list(self.state.banned):
                if self.state.banned[k] > 0:
                    pass  # after_match already handles ban countdown

        # Build final standings.
        from mundialytics.statistical_core.competition.standings import (
            compute_standings,
        )
        results_df = pd.DataFrame(all_results).rename(
            columns={"home": "home_team", "away": "away_team",
                      "home_goals": "home_goals", "away_goals": "away_goals"},
        )
        table = compute_standings(results_df, teams=self.teams,
                                  competition=self.competition)

        from mundialytics.statistical_core.schemas import canonical_name
        squad_canon = canonical_name(self.squad_name)
        squad_row = table[table["team"] == squad_canon]
        squad_pos = int(squad_row["rank"].iloc[0]) if len(squad_row) else 20

        # Scorers table (all teams).
        scorers = self._build_scorers(all_results, own_matches)

        return LeagueResult(
            competition=self.competition,
            competition_name=self.competition,
            replaced=self.replaced,
            table=table,
            matches=own_matches,
            all_results=all_results,
            scorers=scorers,
            squad_totals=_totals_from_state(self.state),
            squad_position=squad_pos,
            stage=stage_for_position(squad_pos),
        )

    # ── live match (reuses ChampionsRun._play_live pattern) ───────────────────

    def _play_live(self, home: str, away: str, stage: str,
                   matchday: int, when: pd.Timestamp) -> ChampionsMatch:
        from mundialytics.statistical_core.squadlab.match_engine import (
            Side, play_match, rate_lines,
        )
        from mundialytics.statistical_core.squadlab.match_players import from_card

        xi_cards, bench_cards, absences = self.state.lineup(when)

        lh, la = self.lambda_source.match_lambdas(
            home, away, competition=self.competition, neutral=False,
        )
        ev = self.lambda_source.event_lambdas(
            home, away, competition=self.competition,
        )

        def lam_of(mk: str, side: str) -> float:
            v = ev.get(mk)
            if v:
                return float(v[0]) if side == "home" else float(v[1])
            return {"yellow_cards_for": 2.0, "shots_for": 12.0, "corners_for": 5.0}.get(mk, 2.0)

        mine = Side(
            self.squad_name,
            [from_card(c) for c in xi_cards],
            [from_card(c) for c in bench_cards],
        )
        opp_name = away if home == self.squad_name else home
        oxi, obench = self._rival_side(opp_name)
        theirs = Side(opp_name, list(oxi) or _anonymous_eleven(opp_name), list(obench))

        H, A = (mine, theirs) if home == self.squad_name else (theirs, mine)
        for side, tag in ((H, "home"), (A, "away")):
            side.lam = lh if tag == "home" else la
            side.yellows_lam = lam_of("yellow_cards_for", tag)
            side.shots_lam = lam_of("shots_for", tag)
            side.corners_lam = lam_of("corners_for", tag)

        log = play_match(
            H, A, self.rng, dyn=self.dyn, neutral=False,
            extra_time_if_level=False, shootout_if_level=False,
            typical_gap=self.gap,
        )
        rate_lines(log, self.rng)

        squad_tag = "home" if home == self.squad_name else "away"
        m = ChampionsMatch(
            stage=stage, matchday=matchday, home=home, away=away,
            home_goals=log.home_goals, away_goals=log.away_goals,
            squad_is_home=home == self.squad_name, log=log, date=when,
            absences=absences, lineup=[c.player for c in xi_cards],
        )

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
        m.shots = log.stats["shots"]
        m.sot = log.stats["sot"]
        m.corners = log.stats["corners"]
        m.yellows = log.stats["yellows"]
        m.xg = log.stats["xg"]

        # SquadState: update bans/injuries with league-specific discipline.
        self._after_match_league(log, squad_tag, when)
        return m

    def _after_match_league(self, log, squad_tag: str, when: pd.Timestamp) -> None:
        """Like SquadState.after_match but with per-competition yellow rules."""
        played = set(log.lines[squad_tag])
        state = self.state
        disc = self.discipline

        # Tick down existing bans for players who sat out.
        for k in list(state.banned):
            if k in state.cards and k not in played and state.banned[k] > 0:
                state.banned[k] -= 1

        for key, line in log.lines[squad_tag].items():
            t = state.totals.get(key)
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
                state.banned[key] = state.banned.get(key, 0) + 1

            new_y = 0 if sent_off_2y else line.yellows
            t["yellows"] += line.yellows
            before = state.yellows[key]
            state.yellows[key] = before + new_y

            # League-specific yellow accumulation ban.
            if any(before < n <= state.yellows[key] for n in disc.yellow_ban_at):
                state.banned[key] = state.banned.get(key, 0) + 1

            if line.injured:
                t["injuries"] += 1
                days = state.dyn.injury_days(self.rng)
                if days > 0:
                    state.injured_until[key] = when + pd.Timedelta(days=days)

    # ── rival side ────────────────────────────────────────────────────────────

    def _rival_side(self, team: str):
        if team in self._sides_cache:
            return self._sides_cache[team]
        from mundialytics.statistical_core.squadlab.match_players import (
            club_side_from_cards,
        )
        xi, bench = [], []
        if self.cards_df is not None:
            club = _card_club_for(self.cards_df, team)
            if club:
                xi, bench = club_side_from_cards(self.cards_df, club, self.drafted)
        if len(xi) < 11:
            profiles = self.rival_squads.get(team, [])
            if profiles:
                from mundialytics.statistical_core.squadlab.match_players import (
                    from_profile,
                )
                xi = [from_profile(p) for p in profiles[:11]]
                bench = [from_profile(p) for p in profiles[11:18]]
        self._sides_cache[team] = (xi, bench)
        return xi, bench

    # ── scorers ───────────────────────────────────────────────────────────────

    def _build_scorers(self, all_results: list[dict],
                       own_matches: list[ChampionsMatch]) -> pd.DataFrame:
        tally: dict[str, dict] = {}

        # Squad goals from live matches.
        for m in own_matches:
            for scorer, assister in m.goal_events:
                key = f"{self.squad_name}|{scorer}"
                if key not in tally:
                    tally[key] = {"jugador": scorer, "equipo": self.squad_name,
                                  "goles": 0, "asistencias": 0}
                tally[key]["goles"] += 1
                if assister:
                    akey = f"{self.squad_name}|{assister}"
                    if akey not in tally:
                        tally[akey] = {"jugador": assister, "equipo": self.squad_name,
                                       "goles": 0, "asistencias": 0}
                    tally[akey]["asistencias"] += 1

            for team, events in m.rival_events.items():
                for scorer, assister in events:
                    key = f"{team}|{scorer}"
                    if key not in tally:
                        tally[key] = {"jugador": scorer, "equipo": team,
                                      "goles": 0, "asistencias": 0}
                    tally[key]["goles"] += 1
                    if assister:
                        akey = f"{team}|{assister}"
                        if akey not in tally:
                            tally[akey] = {"jugador": assister, "equipo": team,
                                           "goles": 0, "asistencias": 0}
                        tally[akey]["asistencias"] += 1

        # Rival goals from Poisson-resolved fixtures: attribute goals.
        for r in all_results:
            if self.squad_name in (r["home"], r["away"]):
                continue  # already counted from the live match
            for team, goals in ((r["home"], r["home_goals"]),
                                (r["away"], r["away_goals"])):
                profiles = self.rival_squads.get(team, [])
                if profiles and goals > 0:
                    ge = attribute_goals(profiles, goals, self.rng)
                    for scorer, assister in ge:
                        key = f"{team}|{scorer}"
                        if key not in tally:
                            tally[key] = {"jugador": scorer, "equipo": team,
                                          "goles": 0, "asistencias": 0}
                        tally[key]["goles"] += 1
                        if assister:
                            akey = f"{team}|{assister}"
                            if akey not in tally:
                                tally[akey] = {"jugador": assister, "equipo": team,
                                               "goles": 0, "asistencias": 0}
                            tally[akey]["asistencias"] += 1

        if not tally:
            return pd.DataFrame(columns=["jugador", "equipo", "goles", "asistencias"])
        df = pd.DataFrame(tally.values())
        return df.sort_values(["goles", "asistencias"], ascending=False).reset_index(drop=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _totals_from_state(state: SquadState) -> dict:
    out = {}
    for key, t in state.totals.items():
        card = state.cards[key]
        out[card.player] = dict(t)
    return out


def _person(name: str) -> str:
    return str(name).split("(")[0].strip()


def _card_club_for(cards_df: pd.DataFrame, team: str) -> str | None:
    """The card catalogue's spelling of a team, matched fuzzy."""
    from mundialytics.statistical_core.schemas import canonical_name
    canon = canonical_name(team)
    for col in ("club",):
        if col not in cards_df.columns:
            continue
        for v in cards_df[col].dropna().unique():
            if canonical_name(str(v)) == canon:
                return str(v)
    return None


def _anonymous_eleven(team: str):
    from mundialytics.statistical_core.squadlab.match_engine import MatchPlayer
    return [MatchPlayer(name=f"{team} #{i+1}", position="Defender" if i < 4 else "Midfielder",
                        overall=60, attack=55, defense=60, creation=55, gk=40, minutes_share=0.9)
            for i in range(11)]
