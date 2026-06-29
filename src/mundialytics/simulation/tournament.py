from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mundialytics.models.result_model import match_probabilities


@dataclass
class SimulatedMatch:
    home_goals: int
    away_goals: int
    winner: str | None


class TournamentSimulator:
    """Lightweight Monte Carlo tournament simulator.

    This supports group-stage points and simple knockout winner simulation.
    It is deliberately generic; for World Cup 2026 exact third-place rules can
    be added by customizing `advance_from_groups`.
    """

    def __init__(self, random_state: int = 42, max_goals: int = 8):
        self.rng = np.random.default_rng(random_state)
        self.max_goals = max_goals

    def simulate_match(self, home: str, away: str, lambda_home: float, lambda_away: float, knockout: bool = False) -> SimulatedMatch:
        hg = int(self.rng.poisson(lambda_home))
        ag = int(self.rng.poisson(lambda_away))
        winner = None
        if hg > ag:
            winner = home
        elif ag > hg:
            winner = away
        elif knockout:
            probs = match_probabilities(lambda_home, lambda_away)
            # Penalties: if draw after regulation, bias slightly by pre-match strength.
            p_home_pen = probs.p_home_win / max(probs.p_home_win + probs.p_away_win, 1e-9)
            winner = home if self.rng.random() < p_home_pen else away
        return SimulatedMatch(hg, ag, winner)

    def simulate_group(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        table = defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0})
        for _, row in fixtures.iterrows():
            res = self.simulate_match(row["home_team"], row["away_team"], row["lambda_home"], row["lambda_away"])
            h, a = row["home_team"], row["away_team"]
            table[h]["gf"] += res.home_goals
            table[h]["ga"] += res.away_goals
            table[a]["gf"] += res.away_goals
            table[a]["ga"] += res.home_goals
            if res.home_goals > res.away_goals:
                table[h]["pts"] += 3; table[h]["w"] += 1; table[a]["l"] += 1
            elif res.home_goals < res.away_goals:
                table[a]["pts"] += 3; table[a]["w"] += 1; table[h]["l"] += 1
            else:
                table[h]["pts"] += 1; table[a]["pts"] += 1; table[h]["d"] += 1; table[a]["d"] += 1
        rows = []
        for team, stats in table.items():
            rows.append({"team": team, **stats, "gd": stats["gf"] - stats["ga"]})
        return pd.DataFrame(rows).sort_values(["pts", "gd", "gf"], ascending=False).reset_index(drop=True)

    def monte_carlo_group(self, fixtures: pd.DataFrame, n_sims: int = 10000, top_n: int = 2) -> pd.DataFrame:
        counts = defaultdict(lambda: {"top_n": 0, "first": 0})
        for _ in range(n_sims):
            table = self.simulate_group(fixtures)
            for pos, team in enumerate(table["team"], start=1):
                if pos == 1:
                    counts[team]["first"] += 1
                if pos <= top_n:
                    counts[team]["top_n"] += 1
        return pd.DataFrame([
            {"team": t, "p_first": c["first"] / n_sims, f"p_top{top_n}": c["top_n"] / n_sims}
            for t, c in counts.items()
        ]).sort_values("p_first", ascending=False)
