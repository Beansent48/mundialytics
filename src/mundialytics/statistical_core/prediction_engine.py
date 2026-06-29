"""
PredictionEngine — unified match and tournament prediction interface.

Wraps GoalLambdaModel + AttackDefenseModel + EventLambdaModel into a single
object with clean methods for:
  - Single match: all probabilities, goal matrix, team event expectations
  - Full tournament: group stage + knockout + Golden Boot (1M simulations)
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mundialytics.statistical_core.attack_defense_model import AttackDefenseModel
from mundialytics.statistical_core.distributions import (
    outcome_probabilities,
    scoreline_distribution,
)
from mundialytics.statistical_core.event_model import EventLambdaModel, MARKET_FEATURES
from mundialytics.models.goal_model import GoalLambdaModel, GoalModelConfig
from mundialytics.features.team_features import build_goal_training_frame
from mundialytics.statistical_core.schemas import canonical_name


# ── Match prediction dataclass ─────────────────────────────────────────────────

@dataclass
class MatchPrediction:
    home_team: str
    away_team: str
    competition: str
    neutral: bool

    # 1X2
    p_home_win: float = 0.0
    p_draw: float = 0.0
    p_away_win: float = 0.0

    # Goals
    lambda_home: float = 0.0
    lambda_away: float = 0.0
    p_btts: float = 0.0
    p_over_15: float = 0.0
    p_over_25: float = 0.0
    p_over_35: float = 0.0
    p_under_25: float = 0.0

    # Scoreline matrix (top rows)
    top_scorelines: list[dict] = field(default_factory=list)
    score_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Team events
    expected_shots_home: float = 0.0
    expected_shots_away: float = 0.0
    expected_sot_home: float = 0.0
    expected_sot_away: float = 0.0
    expected_corners_home: float = 0.0
    expected_corners_away: float = 0.0
    expected_fouls_home: float = 0.0
    expected_fouls_away: float = 0.0
    expected_yellows_home: float = 0.0
    expected_yellows_away: float = 0.0

    # Model metadata
    model_source: str = "ensemble"
    warnings: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k not in {"score_matrix", "top_scorelines"}}

    def summary(self) -> str:
        fav = self.home_team if self.p_home_win > max(self.p_draw, self.p_away_win) else \
              (self.away_team if self.p_away_win > self.p_draw else "Draw")
        fav_p = max(self.p_home_win, self.p_draw, self.p_away_win)
        return (
            f"{self.home_team} vs {self.away_team} "
            f"[H:{self.p_home_win:.1%} D:{self.p_draw:.1%} A:{self.p_away_win:.1%}] "
            f"λ={self.lambda_home:.2f}-{self.lambda_away:.2f} "
            f"xGoals={self.lambda_home+self.lambda_away:.2f} "
            f"BTTS:{self.p_btts:.1%} O2.5:{self.p_over_25:.1%} "
            f"→ Fav: {fav} ({fav_p:.1%})"
        )


# ── Tournament simulation result ───────────────────────────────────────────────

@dataclass
class TournamentResult:
    n_sims: int
    team_stats: pd.DataFrame          # team, p_win, p_final, p_semis, p_quarters, p_advance_groups
    golden_boot: pd.DataFrame         # player, team, expected_goals, p_golden_boot (if player model)
    group_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        top = self.team_stats.head(5)
        lines = [f"Tournament simulation ({self.n_sims:,} sims):"]
        for _, r in top.iterrows():
            lines.append(f"  {r['team']:<20} win:{r['p_win']:.1%}  final:{r['p_final']:.1%}")
        return "\n".join(lines)


# ── Main engine ────────────────────────────────────────────────────────────────

class PredictionEngine:
    """Unified prediction interface combining all statistical models.

    Usage:
        engine = PredictionEngine()
        engine.fit(matches_df, team_rows=None, elo_history=None)

        pred = engine.predict_match("Spain", "Germany", competition="World Cup")
        print(pred.summary())

        result = engine.simulate_world_cup(groups, n_sims=100_000)
        print(result.summary())
    """

    def __init__(
        self,
        goal_model_type: str = "poisson",
        event_model_type: str = "poisson",
        ad_rho: float = -0.07,
        ad_time_decay: float | None = None,
        blend_weight_gl: float = 0.60,   # weight for GoalLambdaModel vs AttackDefense
        max_goals: int = 10,
    ):
        self.goal_model_type = goal_model_type
        self.event_model_type = event_model_type
        self.ad_rho = ad_rho
        self.ad_time_decay = ad_time_decay
        self.blend_gl = float(np.clip(blend_weight_gl, 0.0, 1.0))
        self.max_goals = max_goals

        self.goal_model_: GoalLambdaModel | None = None
        self.ad_model_: AttackDefenseModel | None = None
        self.event_models_: dict[str, EventLambdaModel] = {}
        self._train_frame: pd.DataFrame | None = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        matches: pd.DataFrame,
        team_rows: pd.DataFrame | None = None,
        elo_history: pd.DataFrame | None = None,
    ) -> "PredictionEngine":
        """Fit all internal models.

        Parameters
        ----------
        matches : match-level DataFrame (home_team, away_team, home_goals, away_goals, date, competition)
        team_rows : optional pre-built per-team rows with rolling stats
        elo_history : optional pre-computed ELO history DataFrame
        """
        # AttackDefenseModel (per-league MLE)
        self.ad_model_ = AttackDefenseModel(
            dixon_coles_rho=self.ad_rho,
            time_decay_half_life=self.ad_time_decay,
        )
        self.ad_model_.fit(matches)

        # GoalLambdaModel (Poisson GLM with rolling features + ELO)
        if team_rows is not None and not team_rows.empty:
            frame = build_goal_training_frame(team_rows, elo_history=elo_history)
        else:
            frame = self._build_team_rows(matches, elo_history)
        self._train_frame = frame

        cfg = GoalModelConfig(model_type=self.goal_model_type, time_decay_half_life_days=None)
        self.goal_model_ = GoalLambdaModel(config=cfg)
        self.goal_model_.fit(frame, target="goals_for")

        # EventLambdaModels (one per market)
        for market in MARKET_FEATURES:
            if market in frame.columns and frame[market].notna().mean() >= 0.5:
                em = EventLambdaModel(market=market, model_type=self.event_model_type)
                em.fit(frame)
                self.event_models_[market] = em

        return self

    def _build_team_rows(
        self, matches: pd.DataFrame, elo_history: pd.DataFrame | None
    ) -> pd.DataFrame:
        rows = []
        stat_cols = {
            "home": ["home_goals","away_goals","home_shots","away_shots","home_sot","away_sot",
                     "home_corners","away_corners","home_fouls","away_fouls",
                     "home_yellow_cards","away_yellow_cards"],
            "away": ["away_goals","home_goals","away_shots","home_shots","away_sot","home_sot",
                     "away_corners","home_corners","away_fouls","home_fouls",
                     "away_yellow_cards","home_yellow_cards"],
        }
        target_cols = ["goals_for","goals_against","shots_for","shots_against",
                       "sot_for","sot_against","corners_for","corners_against",
                       "fouls_for","fouls_against","yellow_cards_for","yellow_cards_against"]
        for _, r in matches.iterrows():
            for side, src_cols in [("home", stat_cols["home"]), ("away", stat_cols["away"])]:
                is_h = 1 if side == "home" else 0
                team = r["home_team"] if is_h else r["away_team"]
                opp  = r["away_team"] if is_h else r["home_team"]
                row = {"team": team, "opponent": opp, "is_home": is_h,
                       "date": r.get("date"), "match_id": r.get("match_id"),
                       "season": r.get("season", ""), "competition": r.get("competition", ""),
                       "neutral": r.get("neutral", 0)}
                for tc, sc in zip(target_cols, src_cols):
                    row[tc] = r.get(sc, np.nan)
                rows.append(row)
        team_rows = pd.DataFrame(rows)
        return build_goal_training_frame(team_rows, elo_history=elo_history)

    # ── Predict single match ──────────────────────────────────────────────────

    def _lambdas_ad(self, home: str, away: str, competition: str, neutral: bool) -> tuple[float, float]:
        lh, la, _ = self.ad_model_.expected_goals(home, away, neutral=int(neutral), competition=competition)
        return lh, la

    def _lambdas_gl(self, home: str, away: str, competition: str) -> tuple[float, float]:
        """Predict via GoalLambdaModel using last training features of each team."""
        if self.goal_model_ is None or self._train_frame is None:
            return 1.5, 1.2
        frame = self._train_frame
        h, a = canonical_name(home), canonical_name(away)

        def last_row(team: str, opp: str, is_home: int) -> pd.DataFrame:
            mask = (frame["team"] == team) & (frame["is_home"] == is_home)
            hist = frame[mask].sort_values("date") if "date" in frame.columns else frame[mask]
            if hist.empty:
                mask2 = frame["team"] == team
                hist = frame[mask2].sort_values("date") if "date" in frame.columns else frame[mask2]
            if hist.empty:
                return pd.DataFrame()
            row = hist.iloc[[-1]].copy()
            row["competition"] = competition
            row["is_home"] = is_home
            return row

        home_row = last_row(h, a, 1)
        away_row = last_row(a, h, 0)
        if home_row.empty or away_row.empty:
            return 1.5, 1.2
        lh = float(self.goal_model_.predict_lambda(home_row)[0])
        la = float(self.goal_model_.predict_lambda(away_row)[0])
        return lh, la

    def _event_lambda(self, market: str, home: str, away: str, competition: str) -> tuple[float, float]:
        em = self.event_models_.get(market)
        if em is None or self._train_frame is None:
            return em.mean_ if em else 0.0, em.mean_ if em else 0.0
        frame = self._train_frame
        h, a = canonical_name(home), canonical_name(away)

        def get_row(team: str, is_home: int) -> pd.DataFrame:
            mask = (frame["team"] == team) & (frame["is_home"] == is_home)
            hist = frame[mask].sort_values("date") if "date" in frame.columns else frame[mask]
            if hist.empty:
                hist = frame[frame["team"] == team]
            return hist.iloc[[-1]].copy() if not hist.empty else pd.DataFrame()

        home_row = get_row(h, 1)
        away_row = get_row(a, 0)
        if home_row.empty or away_row.empty:
            return em.mean_, em.mean_
        home_row["competition"] = competition
        away_row["competition"] = competition
        lh = float(em.predict_lambda(home_row)[0])
        la = float(em.predict_lambda(away_row)[0])
        return lh, la

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        competition: str = "unknown",
        neutral: bool = False,
    ) -> MatchPrediction:
        """Full match prediction: 1X2, goals, events."""
        h, a = canonical_name(home_team), canonical_name(away_team)

        # Goal lambdas — blend GoalLambda and AttackDefense
        lh_ad, la_ad = self._lambdas_ad(h, a, competition, neutral)
        lh_gl, la_gl = self._lambdas_gl(h, a, competition)
        w = self.blend_gl
        lh = w * lh_gl + (1 - w) * lh_ad
        la = w * la_gl + (1 - w) * la_ad
        lh = float(np.clip(lh, 0.05, 6.0))
        la = float(np.clip(la, 0.05, 6.0))

        # Probabilities from score matrix
        probs = outcome_probabilities(lh, la, max_goals=self.max_goals, dixon_coles_rho=self.ad_rho)
        dist = scoreline_distribution(lh, la, max_goals=self.max_goals, normalize=True, dixon_coles_rho=self.ad_rho)

        # Event lambdas
        sh, sa = self._event_lambda("shots_for", h, a, competition)
        sch, sca = self._event_lambda("sot_for", h, a, competition)
        ch, ca = self._event_lambda("corners_for", h, a, competition)
        fh, fa = self._event_lambda("fouls_for", h, a, competition)
        yh, ya = self._event_lambda("yellow_cards_for", h, a, competition)

        return MatchPrediction(
            home_team=h, away_team=a, competition=competition, neutral=neutral,
            p_home_win=probs["p_home_win"], p_draw=probs["p_draw"], p_away_win=probs["p_away_win"],
            lambda_home=lh, lambda_away=la,
            p_btts=probs["p_btts"],
            p_over_15=probs["p_over_15"], p_over_25=probs["p_over_25"],
            p_over_35=probs["p_over_35"], p_under_25=probs["p_under_25"],
            top_scorelines=dist.top_scorelines(8),
            score_matrix=dist.matrix,
            expected_shots_home=sh, expected_shots_away=sa,
            expected_sot_home=sch, expected_sot_away=sca,
            expected_corners_home=ch, expected_corners_away=ca,
            expected_fouls_home=fh, expected_fouls_away=fa,
            expected_yellows_home=yh, expected_yellows_away=ya,
            model_source=f"blend_gl{w:.0%}_ad{1-w:.0%}",
        )

    # ── Full tournament simulation ────────────────────────────────────────────

    def simulate_tournament(
        self,
        groups: dict[str, list[str]],
        knockout_slots: int = 2,
        n_sims: int = 100_000,
        competition: str = "unknown",
        neutral: bool = True,
        player_goals: dict[str, dict[str, float]] | None = None,
        random_seed: int = 42,
    ) -> TournamentResult:
        """Monte Carlo tournament simulation with group stage + knockout + Golden Boot.

        Parameters
        ----------
        groups : {"A": ["Spain","Germany","Japan","Costa Rica"], "B": [...], ...}
        knockout_slots : teams advancing from each group (default 2)
        n_sims : Monte Carlo iterations
        competition : competition name for model context
        neutral : treat all matches as neutral venue
        player_goals : {player: {team: goals_per_match_rate}} for Golden Boot tracking
        random_seed : for reproducibility
        """
        rng = np.random.default_rng(random_seed)
        all_teams = [t for teams in groups.values() for t in teams]

        # Pre-compute match lambdas once
        match_cache: dict[tuple[str, str], tuple[float, float]] = {}

        def get_lam(h: str, a: str) -> tuple[float, float]:
            key = (h, a)
            if key not in match_cache:
                pred = self.predict_match(h, a, competition=competition, neutral=neutral)
                match_cache[key] = (pred.lambda_home, pred.lambda_away)
            return match_cache[key]

        # Accumulators
        team_counts = defaultdict(lambda: {"win": 0, "final": 0, "semis": 0,
                                            "quarters": 0, "advance": 0, "goals": 0.0})
        player_goal_counts: dict[str, float] = defaultdict(float)

        for _ in range(n_sims):
            sim_goals: dict[str, int] = defaultdict(int)   # team goals in this sim (for GD/GF)

            # ── Group stage ──────────────────────────────────────────────────
            group_winners: list[str] = []  # 1st in each group
            group_runnersup: list[str] = []  # 2nd in each group

            for group_name, teams in groups.items():
                table = defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0})
                fixtures = list(itertools.combinations(teams, 2))
                for ht, at in fixtures:
                    lh, la = get_lam(ht, at)
                    hg = int(rng.poisson(lh))
                    ag = int(rng.poisson(la))
                    sim_goals[ht] += hg
                    sim_goals[at] += ag
                    table[ht]["gf"] += hg; table[ht]["ga"] += ag
                    table[at]["gf"] += ag; table[at]["ga"] += hg
                    if hg > ag:
                        table[ht]["pts"] += 3
                    elif ag > hg:
                        table[at]["pts"] += 3
                    else:
                        table[ht]["pts"] += 1; table[at]["pts"] += 1

                ranked = sorted(
                    teams,
                    key=lambda t: (table[t]["pts"], table[t]["gf"] - table[t]["ga"], table[t]["gf"],
                                   rng.uniform()),  # tiebreak with RNG
                    reverse=True,
                )
                for rank, t in enumerate(ranked):
                    if rank < knockout_slots:
                        team_counts[t]["advance"] += 1
                group_winners.append(ranked[0])
                if len(ranked) > 1:
                    group_runnersup.append(ranked[1])

            # ── Knockout stage ───────────────────────────────────────────────
            def sim_ko(h: str, a: str) -> str:
                lh, la = get_lam(h, a)
                hg = int(rng.poisson(lh))
                ag = int(rng.poisson(la))
                sim_goals[h] += hg; sim_goals[a] += ag
                if hg > ag:
                    return h
                if ag > hg:
                    return a
                # Penalties (toss weighted by pre-match strength)
                lh_tot, la_tot = lh, la
                p_hp = lh_tot / max(lh_tot + la_tot, 1e-6)
                return h if rng.random() < p_hp else a

            # Build knockout bracket: winners face runners-up from paired groups
            n_groups = len(groups)
            ko_round = []
            group_list = list(groups.keys())
            # Pair groups: A vs B, C vs D, etc.
            for i in range(0, n_groups - 1, 2):
                if i < len(group_winners) and i < len(group_runnersup):
                    ko_round.append((group_winners[i], group_runnersup[i + 1] if i + 1 < len(group_runnersup) else group_winners[i + 1]))
                    ko_round.append((group_winners[i + 1] if i + 1 < len(group_winners) else group_runnersup[i], group_runnersup[i]))

            # If odd number of groups, add extra
            if n_groups % 2 != 0 and n_groups > 0:
                ko_round.append((group_winners[-1], group_runnersup[-1] if group_runnersup else group_winners[-1]))

            survivors = [sim_ko(h, a) for h, a in ko_round if h != a]
            # Add any unpaired teams
            for i in range(len(ko_round) * 2, len(group_winners) + len(group_runnersup)):
                pass

            for t in survivors:
                team_counts[t]["quarters"] += 1

            # Semis
            semi_winners = []
            for i in range(0, len(survivors) - 1, 2):
                w = sim_ko(survivors[i], survivors[i + 1])
                semi_winners.append(w)
                team_counts[w]["semis"] += 1

            # Final
            if len(semi_winners) >= 2:
                champion = sim_ko(semi_winners[0], semi_winners[1])
                team_counts[champion]["final"] += 1
                team_counts[champion]["win"] += 1
            elif len(semi_winners) == 1:
                team_counts[semi_winners[0]]["win"] += 1

            # Goals accumulation for Golden Boot
            for t, g in sim_goals.items():
                team_counts[t]["goals"] += g

            # Player Golden Boot (optional)
            if player_goals:
                for player, teams_map in player_goals.items():
                    for team, rate in teams_map.items():
                        # Only for teams that advanced
                        if team in sim_goals and sim_goals[team] > 0:
                            # Approximate: player scores rate * (team_goals / expected_team_goals)
                            exp_tg = 2 * len(groups.get(list(groups.keys())[0], [4 * 3])) / 2
                            actual_ratio = sim_goals[team] / max(exp_tg, 0.1)
                            player_goals_sim = rng.poisson(rate * actual_ratio)
                            player_goal_counts[player] += player_goals_sim

        # ── Aggregate results ─────────────────────────────────────────────────
        rows = []
        for team in all_teams:
            c = team_counts[team]
            rows.append({
                "team": team,
                "p_win": c["win"] / n_sims,
                "p_final": c["final"] / n_sims,
                "p_semis": c["semis"] / n_sims,
                "p_quarters": c["quarters"] / n_sims,
                "p_advance_groups": c["advance"] / n_sims,
                "avg_goals_per_sim": c["goals"] / n_sims,
            })
        team_df = pd.DataFrame(rows).sort_values("p_win", ascending=False).reset_index(drop=True)

        # Golden Boot
        if player_goal_counts:
            pb_rows = [{"player": p, "avg_goals": v / n_sims} for p, v in player_goal_counts.items()]
            golden_boot = pd.DataFrame(pb_rows).sort_values("avg_goals", ascending=False).reset_index(drop=True)
            total = golden_boot["avg_goals"].sum()
            golden_boot["p_golden_boot"] = golden_boot["avg_goals"] / max(total, 1e-6)
        else:
            # Estimate from team goal rates
            gb_rows = []
            for team in all_teams:
                avg_goals = team_counts[team]["goals"] / n_sims
                gb_rows.append({"player": f"{team} top scorer", "team": team,
                                  "avg_goals": avg_goals, "p_golden_boot": 0.0})
            golden_boot = pd.DataFrame(gb_rows).sort_values("avg_goals", ascending=False).reset_index(drop=True)

        return TournamentResult(
            n_sims=n_sims,
            team_stats=team_df,
            golden_boot=golden_boot,
            metadata={"competition": competition, "n_groups": len(groups),
                      "teams_per_group": len(next(iter(groups.values()))) if groups else 0,
                      "knockout_slots": knockout_slots},
        )

    def simulate_league(
        self,
        teams: list[str],
        n_sims: int = 100_000,
        competition: str = "unknown",
        home_away: bool = True,
        random_seed: int = 42,
    ) -> TournamentResult:
        """Simulate a full round-robin league season N times."""
        rng = np.random.default_rng(random_seed)
        team_counts = defaultdict(lambda: {"win": 0, "top4": 0, "top2": 0, "pts_total": 0.0, "goals": 0.0})

        fixtures = list(itertools.combinations(teams, 2))
        if home_away:
            fixtures = fixtures + [(a, h) for h, a in fixtures]

        # Pre-compute lambdas
        cache: dict[tuple[str, str], tuple[float, float]] = {}
        for h, a in fixtures:
            key = (h, a)
            if key not in cache:
                pred = self.predict_match(h, a, competition=competition, neutral=not home_away)
                cache[key] = (pred.lambda_home, pred.lambda_away)

        lams_h = np.array([cache[(h, a)][0] for h, a in fixtures])
        lams_a = np.array([cache[(h, a)][1] for h, a in fixtures])

        for _ in range(n_sims):
            hg_all = rng.poisson(lams_h)
            ag_all = rng.poisson(lams_a)
            pts = defaultdict(int)
            goals = defaultdict(int)
            for (h, a), hg, ag in zip(fixtures, hg_all, ag_all):
                goals[h] += int(hg); goals[a] += int(ag)
                if hg > ag:
                    pts[h] += 3
                elif ag > hg:
                    pts[a] += 3
                else:
                    pts[h] += 1; pts[a] += 1
            ranked = sorted(teams, key=lambda t: (pts[t], goals[t], rng.uniform()), reverse=True)
            for rank, t in enumerate(ranked, 1):
                team_counts[t]["pts_total"] += pts[t]
                team_counts[t]["goals"] += goals[t]
                if rank == 1:
                    team_counts[t]["win"] += 1
                if rank <= 2:
                    team_counts[t]["top2"] += 1
                if rank <= 4:
                    team_counts[t]["top4"] += 1

        rows = []
        for team in teams:
            c = team_counts[team]
            rows.append({
                "team": team,
                "p_win": c["win"] / n_sims,
                "p_top2": c["top2"] / n_sims,
                "p_top4": c["top4"] / n_sims,
                "avg_pts": c["pts_total"] / n_sims,
                "avg_goals": c["goals"] / n_sims,
            })
        team_df = pd.DataFrame(rows).sort_values("p_win", ascending=False).reset_index(drop=True)
        return TournamentResult(
            n_sims=n_sims,
            team_stats=team_df,
            golden_boot=pd.DataFrame(),
            metadata={"competition": competition, "home_away": home_away},
        )
