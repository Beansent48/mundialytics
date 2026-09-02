from __future__ import annotations

"""Half-time markets: HT result, HT over/under, and half-time/full-time.

Built on `home_goals_ht`/`away_goals_ht`, carried into the foundation on
2026-09-03 (100% coverage across all 26 seasons, 45,934 matches).

WHY THIS IS A SCALING AND NOT A MODEL. The obvious design is a dedicated
attack/defence fit on first-half goals, capturing sides that start fast or slow.
Measured first, and the signal is not there: splitting each team's history in
two and correlating its first-half goal SHARE across the halves gives **-0.118**
over 155 teams with 120+ matches — Spearman-Brown reliability 0.000. The spread
between teams (std 0.037 around a mean of 0.440) is what pure sampling noise
produces. A per-team half-time model would be fitting noise, so the first-half
share is applied as a global constant instead.

What IS stable:
  - the share itself: 0.4412 mean, std 0.0098 across 26 seasons;
  - home advantage barely differs by half (home/away goal ratio 1.336 in the
    first, 1.299 in the second), so the split is near-identical per side.

HALVES ARE NEARLY INDEPENDENT (correlation of first- and second-half goals:
+0.054), which is what makes HT/FT tractable. The small dependence that does
exist is a game-state effect and is modelled explicitly: 1.431 second-half goals
when the score is level at the break versus 1.538/1.598 when either side leads —
the trailing team opens up.

Nothing here touches the deployed 1X2 chain; it consumes its lambdas.
"""

import numpy as np

# Share of a side's goals scored in the first half (measured over 45,934 matches).
HT_SHARE_HOME: float = 0.4439
HT_SHARE_AWAY: float = 0.4370

# Second-half intensity relative to its unconditional mean (1.507 goals),
# conditioned on the half-time state. Measured, small, but real.
H2_ADJ_LEVEL: float = 0.950      # 1.431 / 1.507
H2_ADJ_HOME_LEAD: float = 1.021  # 1.538 / 1.507
H2_ADJ_AWAY_LEAD: float = 1.060  # 1.598 / 1.507

HT_LINES: tuple[float, ...] = (0.5, 1.5, 2.5)
MAX_GOALS = 8


def _pmf(lam: float, n: int = MAX_GOALS) -> np.ndarray:
    """Poisson pmf over 0..n with the tail folded into the last cell."""
    lam = max(float(lam), 1e-9)
    k = np.arange(n + 1, dtype=float)
    logp = -lam + k * np.log(lam) - np.array([np.sum(np.log(np.arange(1, i + 1))) if i else 0.0
                                              for i in range(n + 1)])
    p = np.exp(logp)
    p[-1] += max(0.0, 1.0 - p.sum())
    return p / p.sum()


class HalfTimeModel:
    """Derives half-time markets from full-time lambdas.

    Stateless by design — there is nothing to fit (see the module docstring),
    so it is a thin, testable transformation of the engine's own output.
    """

    def __init__(self, share_home: float = HT_SHARE_HOME,
                 share_away: float = HT_SHARE_AWAY) -> None:
        self.share_home = float(share_home)
        self.share_away = float(share_away)

    # ── half-time only ────────────────────────────────────────────────────────
    def ht_matrix(self, lam_home: float, lam_away: float) -> np.ndarray:
        """Joint pmf of the half-time scoreline."""
        return np.outer(_pmf(lam_home * self.share_home),
                        _pmf(lam_away * self.share_away))

    def predict_half_time(self, lam_home: float, lam_away: float) -> dict:
        m = self.ht_matrix(lam_home, lam_away)
        idx = np.arange(m.shape[0])
        home = float(np.tril(m, -1).sum())          # home_goals > away_goals
        draw = float(np.trace(m))
        away = float(np.triu(m, 1).sum())
        totals = np.zeros(2 * MAX_GOALS + 1)
        for h in idx:
            for a in idx:
                totals[h + a] += m[h, a]
        over = {ln: float(totals[int(np.ceil(ln)):].sum()) for ln in HT_LINES}
        return {"p_home": home, "p_draw": draw, "p_away": away,
                "lambda_home": lam_home * self.share_home,
                "lambda_away": lam_away * self.share_away,
                "over": over}

    # ── half-time / full-time ────────────────────────────────────────────────
    def predict_ht_ft(self, lam_home: float, lam_away: float) -> dict:
        """The nine HT/FT combinations, e.g. "X/1" = level at the break, home win.

        Halves are treated as independent (measured correlation +0.054) except
        for a game-state adjustment on second-half intensity.
        """
        ht = self.ht_matrix(lam_home, lam_away)
        l2h = lam_home * (1.0 - self.share_home)
        l2a = lam_away * (1.0 - self.share_away)

        out = {f"{a}/{b}": 0.0 for a in "1X2" for b in "1X2"}
        for h1 in range(ht.shape[0]):
            for a1 in range(ht.shape[1]):
                p1 = ht[h1, a1]
                if p1 < 1e-12:
                    continue
                if h1 > a1:
                    ht_res, adj = "1", H2_ADJ_HOME_LEAD
                elif h1 == a1:
                    ht_res, adj = "X", H2_ADJ_LEVEL
                else:
                    ht_res, adj = "2", H2_ADJ_AWAY_LEAD
                m2 = np.outer(_pmf(l2h * adj), _pmf(l2a * adj))
                for h2 in range(m2.shape[0]):
                    for a2 in range(m2.shape[1]):
                        p2 = m2[h2, a2]
                        if p2 < 1e-12:
                            continue
                        fh, fa = h1 + h2, a1 + a2
                        ft_res = "1" if fh > fa else ("X" if fh == fa else "2")
                        out[f"{ht_res}/{ft_res}"] += p1 * p2
        tot = sum(out.values())
        return {k: v / tot for k, v in out.items()}

    def predict_fixture(self, lam_home: float, lam_away: float) -> dict:
        ht = self.predict_half_time(lam_home, lam_away)
        return {"half_time": ht, "ht_ft": self.predict_ht_ft(lam_home, lam_away)}
