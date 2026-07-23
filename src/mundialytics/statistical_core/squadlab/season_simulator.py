"""SeasonOrchestrator — plays a calendar of fixtures through a LambdaSource,
either ONCE with full narrative detail (results, player events, ratings) or
N times for instant Monte Carlo percentages.

Both modes draw scorelines through the same `_lambda_arrays` + vectorised
`rng.poisson(...)` path, so there is exactly one simulation code path with
identical RNG consumption in both modes (verified: same seed produces
bit-identical fixture-by-fixture scorelines whether drawn via play_once or
the first iteration of run_monte_carlo) — this is the literal mechanism
behind "Sandbox's 1M sims should fall out almost for free" once the
narrative engine exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mundialytics.statistical_core.player_strength import PlayerStrengthProfile
from mundialytics.statistical_core.squadlab.calendar import Fixture
from mundialytics.statistical_core.squadlab.lambda_source import LambdaSource
from mundialytics.statistical_core.squadlab.player_rating import (
    PlayerMatchEvent,
    attribute_cards,
    attribute_goals,
    compute_match_ratings,
)

POINTS_WIN, POINTS_DRAW = 3, 1


@dataclass
class MatchResult:
    matchday: int
    home: str
    away: str
    home_goals: int
    away_goals: int
    home_events: dict[str, PlayerMatchEvent] | None = None   # None when side is a real team
    away_events: dict[str, PlayerMatchEvent] | None = None
    # Granular per-goal/per-card detail for squad-involved matches, kept
    # around (not just the aggregated *_events totals above) so a UI can
    # replay "who scored, who assisted, who got booked" as a sequence of
    # moments — e.g. a live-playback animation — rather than just a final
    # tally. None when the corresponding side isn't a squad.
    home_goal_events: list[tuple[str, str | None]] | None = None   # [(scorer, assister), ...]
    away_goal_events: list[tuple[str, str | None]] | None = None
    home_card_players: list[str] | None = None
    away_card_players: list[str] | None = None
    # Match-level advanced stats (only drawn for squad-involved fixtures —
    # real-vs-real matches don't need them for anything currently built).
    home_shots: int = 0
    away_shots: int = 0
    home_sot: int = 0
    away_sot: int = 0
    home_corners: int = 0
    away_corners: int = 0
    home_fouls: int = 0
    away_fouls: int = 0
    home_yellow_cards: int = 0
    away_yellow_cards: int = 0
    home_xg: float = 0.0
    away_xg: float = 0.0


@dataclass
class SeasonResult:
    matches: list[MatchResult]
    table: pd.DataFrame
    player_season_tallies: pd.DataFrame
    metadata: dict = field(default_factory=dict)


def _build_table(team_results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for team, r in team_results.items():
        rows.append({
            "team": team, "played": r["played"], "pts": r["pts"],
            "gf": r["gf"], "ga": r["ga"], "gd": r["gf"] - r["ga"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["pts", "gd", "gf"], ascending=False).reset_index(drop=True)


def table_through_matchday(matches: list[MatchResult], up_to_matchday: int) -> pd.DataFrame:
    """Standings using only matches played on or before `up_to_matchday` —
    for a live-playback UI that reveals the table progressively, one
    matchday at a time, instead of only at the end of the season."""
    subset = [m for m in matches if m.matchday <= up_to_matchday]
    teams = sorted({m.home for m in subset} | {m.away for m in subset})
    state = {t: {"played": 0, "pts": 0, "gf": 0, "ga": 0} for t in teams}
    for m in subset:
        state[m.home]["played"] += 1
        state[m.away]["played"] += 1
        state[m.home]["gf"] += m.home_goals
        state[m.home]["ga"] += m.away_goals
        state[m.away]["gf"] += m.away_goals
        state[m.away]["ga"] += m.home_goals
        if m.home_goals > m.away_goals:
            state[m.home]["pts"] += POINTS_WIN
        elif m.away_goals > m.home_goals:
            state[m.away]["pts"] += POINTS_WIN
        else:
            state[m.home]["pts"] += POINTS_DRAW
            state[m.away]["pts"] += POINTS_DRAW
    return _build_table(state)


class SeasonOrchestrator:
    """Note: per-match goal draws are plain independent Poisson on the
    (already Dixon-Coles-aware) lambdas returned by the LambdaSource — the
    DC low-score correction itself is not reapplied at the sampling step,
    matching the existing convention in
    PredictionEngine.simulate_league/simulate_tournament (their bulk
    Monte Carlo loops also draw plain rng.poisson(lambda), DC only shapes
    the analytical probability matrix used for single-match display)."""

    def __init__(
        self,
        lambda_source: LambdaSource,
        fixtures: list[Fixture],
        squad_roster: dict[str, list[PlayerStrengthProfile]],
        competition: str,
        random_seed: int = 42,
    ):
        self.lambda_source = lambda_source
        self.fixtures = fixtures
        self.squad_roster = squad_roster
        self.competition = competition
        self.random_seed = random_seed
        self._lambda_arrays_cache: tuple[np.ndarray, np.ndarray] | None = None

    def _lambda_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Match lambdas depend only on (home, away, competition) — NOT on
        which simulation draw we're on — so they're computed once per
        unique fixture and reused across every Monte Carlo iteration,
        exactly like PredictionEngine.simulate_league/simulate_tournament
        already precompute match_cache before their n_sims loops. Without
        this, a real-vs-real fixture's lambda would recompute a full
        GoalLambdaModel + AttackDefenseModel + 5 EventLambdaModel prediction
        (and build a scoreline matrix) on every single simulated match —
        the difference between ~1 second and ~36 minutes for just 2000 sims
        of a 6-team league, measured directly during this build.

        Returns arrays aligned with self.fixtures, so both play_once and
        run_monte_carlo can draw hg_all/ag_all via ONE vectorised
        rng.poisson() call per side, instead of one call per fixture —
        that second change is what makes n_sims in the hundreds of
        thousands actually tractable (fewer, larger calls beats many tiny
        Python-level calls).

        Memoised on the instance: for a full 20-team league (380 fixtures)
        this loop alone costs ~12s (380 PredictionEngine.predict_match
        calls) — a fixed cost that must be paid once per orchestrator, not
        once per play_once()/run_monte_carlo() call. A typical UI flow
        calls both methods on the same orchestrator (narrative season, then
        the Monte Carlo button) — recomputing here would silently double
        that 12s for no reason.
        """
        if self._lambda_arrays_cache is not None:
            return self._lambda_arrays_cache

        cache: dict[tuple[str, str], tuple[float, float]] = {}
        for fixture in self.fixtures:
            key = (fixture.home, fixture.away)
            if key not in cache:
                cache[key] = self.lambda_source.match_lambdas(
                    fixture.home, fixture.away, competition=self.competition, neutral=False
                )
        lams_h = np.array([cache[(f.home, f.away)][0] for f in self.fixtures])
        lams_a = np.array([cache[(f.home, f.away)][1] for f in self.fixtures])
        self._lambda_arrays_cache = (lams_h, lams_a)
        return self._lambda_arrays_cache

    def _squad_match_events(
        self, team: str, goals_for: int, goals_against: int,
        n_cards: int, rng: np.random.Generator,
    ) -> tuple[dict[str, PlayerMatchEvent], list[tuple[str, str | None]], list[str]]:
        squad = self.squad_roster[team]
        goal_events = attribute_goals(squad, goals_for, rng)
        card_players = attribute_cards(squad, n_cards, rng)
        ratings = compute_match_ratings(squad, goal_events, card_players, goals_conceded=goals_against, rng=rng)
        return ratings, goal_events, card_players

    def play_once(self, *, narrative: bool = True) -> SeasonResult:
        """narrative=True: full per-match player events/ratings, for the
        Draft 'Simular temporada' view. narrative=False: scoreline-only,
        used internally by run_monte_carlo."""
        rng = np.random.default_rng(self.random_seed)
        lams_h, lams_a = self._lambda_arrays()
        hg_all, ag_all = rng.poisson(lams_h), rng.poisson(lams_a)
        teams = sorted({f.home for f in self.fixtures} | {f.away for f in self.fixtures})
        table_state = {t: {"played": 0, "pts": 0, "gf": 0, "ga": 0} for t in teams}
        matches: list[MatchResult] = []

        for fixture, hg, ag, lam_h, lam_a in zip(self.fixtures, hg_all, ag_all, lams_h, lams_a):
            hg, ag = int(hg), int(ag)

            table_state[fixture.home]["played"] += 1
            table_state[fixture.away]["played"] += 1
            table_state[fixture.home]["gf"] += hg
            table_state[fixture.home]["ga"] += ag
            table_state[fixture.away]["gf"] += ag
            table_state[fixture.away]["ga"] += hg
            if hg > ag:
                table_state[fixture.home]["pts"] += POINTS_WIN
            elif ag > hg:
                table_state[fixture.away]["pts"] += POINTS_WIN
            else:
                table_state[fixture.home]["pts"] += POINTS_DRAW
                table_state[fixture.away]["pts"] += POINTS_DRAW

            home_events = away_events = None
            home_goal_events = away_goal_events = None
            home_card_players = away_card_players = None
            stats: dict[str, int] = {}
            involves_squad = fixture.home in self.squad_roster or fixture.away in self.squad_roster
            if narrative and involves_squad:
                ev_lams = self.lambda_source.event_lambdas(fixture.home, fixture.away, competition=self.competition)

                def _draw(market: str) -> tuple[int, int]:
                    lh, la = ev_lams.get(market, (0.0, 0.0))
                    return int(rng.poisson(lh)), int(rng.poisson(la))

                sh_h, sh_a = _draw("shots_for")
                sot_h, sot_a = _draw("sot_for")
                cor_h, cor_a = _draw("corners_for")
                fo_h, fo_a = _draw("fouls_for")
                yc_h, yc_a = _draw("yellow_cards_for")

                def _xg(lam: float, shots: int) -> float:
                    # xG anchored to the model's expected goals (lam), nudged by
                    # the shots actually taken this match, with natural Gamma
                    # dispersion — so it correlates with shots & averages ~lam.
                    base = max(0.5 * float(lam) + 0.5 * shots * 0.11, 0.05)
                    return round(float(rng.gamma(4.0, base / 4.0)), 2)

                xg_h, xg_a = _xg(lam_h, sh_h), _xg(lam_a, sh_a)
                stats = dict(
                    home_shots=sh_h, away_shots=sh_a, home_sot=sot_h, away_sot=sot_a,
                    home_corners=cor_h, away_corners=cor_a, home_fouls=fo_h, away_fouls=fo_a,
                    home_yellow_cards=yc_h, away_yellow_cards=yc_a, home_xg=xg_h, away_xg=xg_a,
                )
                if fixture.home in self.squad_roster:
                    home_events, home_goal_events, home_card_players = self._squad_match_events(
                        fixture.home, hg, ag, yc_h, rng)
                if fixture.away in self.squad_roster:
                    away_events, away_goal_events, away_card_players = self._squad_match_events(
                        fixture.away, ag, hg, yc_a, rng)

            matches.append(MatchResult(
                matchday=fixture.matchday, home=fixture.home, away=fixture.away,
                home_goals=hg, away_goals=ag, home_events=home_events, away_events=away_events,
                home_goal_events=home_goal_events, away_goal_events=away_goal_events,
                home_card_players=home_card_players, away_card_players=away_card_players,
                **stats,
            ))

        table = _build_table(table_state)
        tallies = self._aggregate_player_tallies(matches) if narrative else pd.DataFrame()
        return SeasonResult(
            matches=matches, table=table, player_season_tallies=tallies,
            metadata={"competition": self.competition, "narrative": narrative, "n_fixtures": len(self.fixtures)},
        )

    def _aggregate_player_tallies(self, matches: list[MatchResult]) -> pd.DataFrame:
        agg: dict[str, dict] = {}
        for team, players in self.squad_roster.items():
            for p in players:
                agg[p.player] = {"player": p.player, "team": team, "position": p.position,
                                  "goals": 0, "assists": 0, "yellow_cards": 0, "matches": 0, "rating_sum": 0.0}

        for m in matches:
            for events in (m.home_events, m.away_events):
                if not events:
                    continue
                for player, ev in events.items():
                    if player not in agg:
                        continue
                    agg[player]["goals"] += ev.goals
                    agg[player]["assists"] += ev.assists
                    agg[player]["yellow_cards"] += ev.yellow_cards
                    agg[player]["matches"] += 1
                    agg[player]["rating_sum"] += ev.rating

        rows = []
        for r in agg.values():
            avg_rating = r["rating_sum"] / r["matches"] if r["matches"] > 0 else float("nan")
            rows.append({
                "player": r["player"], "team": r["team"], "position": r["position"],
                "goals": r["goals"], "assists": r["assists"], "yellow_cards": r["yellow_cards"],
                "matches": r["matches"], "avg_rating": round(avg_rating, 2) if r["matches"] else None,
            })
        return pd.DataFrame(rows).sort_values("goals", ascending=False).reset_index(drop=True)

    def run_monte_carlo(self, n_sims: int = 100_000) -> pd.DataFrame:
        """Same lambda source and fixture list as play_once, just without
        player-event attribution and without storing per-match detail —
        only standings-position tallies accumulate. Draws are vectorised
        across all fixtures per simulation (one rng.poisson() call per side
        per simulation, not one per fixture) — the same pattern
        PredictionEngine.simulate_league already uses for its own bulk
        Monte Carlo loop, needed to make n_sims in the hundreds of
        thousands actually tractable."""
        rng = np.random.default_rng(self.random_seed)
        lams_h, lams_a = self._lambda_arrays()
        teams = sorted({f.home for f in self.fixtures} | {f.away for f in self.fixtures})
        n_teams = len(teams)
        relegation_zone = 3 if n_teams > 6 else max(1, n_teams // 4)

        counts = {t: {"win": 0, "top2": 0, "top4": 0, "relegated": 0, "pts_total": 0.0, "goals_total": 0.0} for t in teams}

        for _ in range(n_sims):
            hg_all = rng.poisson(lams_h)
            ag_all = rng.poisson(lams_a)
            table_state = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}
            for fixture, hg, ag in zip(self.fixtures, hg_all, ag_all):
                table_state[fixture.home]["gf"] += int(hg)
                table_state[fixture.home]["ga"] += int(ag)
                table_state[fixture.away]["gf"] += int(ag)
                table_state[fixture.away]["ga"] += int(hg)
                if hg > ag:
                    table_state[fixture.home]["pts"] += POINTS_WIN
                elif ag > hg:
                    table_state[fixture.away]["pts"] += POINTS_WIN
                else:
                    table_state[fixture.home]["pts"] += POINTS_DRAW
                    table_state[fixture.away]["pts"] += POINTS_DRAW

            ranked = sorted(
                teams,
                key=lambda t: (table_state[t]["pts"], table_state[t]["gf"] - table_state[t]["ga"], rng.uniform()),
                reverse=True,
            )
            for rank, t in enumerate(ranked, start=1):
                counts[t]["pts_total"] += table_state[t]["pts"]
                counts[t]["goals_total"] += table_state[t]["gf"]
                if rank == 1:
                    counts[t]["win"] += 1
                if rank <= 2:
                    counts[t]["top2"] += 1
                if rank <= 4:
                    counts[t]["top4"] += 1
                if rank > n_teams - relegation_zone:
                    counts[t]["relegated"] += 1

        rows = []
        for t in teams:
            c = counts[t]
            rows.append({
                "team": t,
                "p_champion": c["win"] / n_sims,
                "p_top2": c["top2"] / n_sims,
                "p_top4": c["top4"] / n_sims,
                "p_relegation": c["relegated"] / n_sims,
                "avg_pts": c["pts_total"] / n_sims,
                "avg_goals": c["goals_total"] / n_sims,
            })
        return pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)
