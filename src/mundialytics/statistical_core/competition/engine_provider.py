"""
Leakage-free lambda provider for the competition layer (option A).

Trains a PredictionEngine on foundation matches **strictly before** a state's
cutoff, then produces expected goals (lambda_home, lambda_away) for every
remaining fixture. Training only on the past is what makes a "who wins the league
from here?" answer honest — and what makes the completed-season split a valid
backtest.

The engine itself is untouched: this module only *consumes* PredictionEngine
(`fit` + `predict_match`). SquadLab and pre-season sims keep using the same engine
class the stateless way.
"""
from __future__ import annotations

import pandas as pd

from mundialytics.statistical_core.competition.state import LeagueState
from mundialytics.statistical_core.prediction_engine import PredictionEngine

# Default rolling history window for team strength. Older matches are already
# down-weighted by AttackDefenseModel's time decay; this bound mainly keeps the
# fit fast and relevant (a team's 2005 form says little about 2025).
DEFAULT_LOOKBACK_DAYS = 365 * 3


def train_engine_before_cutoff(
    state: LeagueState,
    foundation: pd.DataFrame,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    same_competition_only: bool = True,
    blend_weight_gl: float = 0.30,
) -> PredictionEngine:
    """Fit a PredictionEngine on matches before ``state.cutoff_date``.

    Parameters
    ----------
    state
        The LeagueState whose remaining fixtures we will predict.
    foundation
        Full foundation match DataFrame (all seasons/competitions).
    lookback_days
        Only train on matches within this many days before the cutoff.
    same_competition_only
        If True, train only on the state's competition (correct for league play,
        and faster). If False, train on all Big5 (AttackDefenseModel fits per
        league internally either way).

    Returns
    -------
    A fitted PredictionEngine (trained strictly on pre-cutoff data).
    """
    df = foundation.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    cutoff = state.cutoff_date
    if cutoff is None:
        # No cutoff => season complete => nothing to predict. Train on everything
        # before the last played date so callers still get a usable engine.
        cutoff = pd.to_datetime(state.played["date"]).max()

    lower = cutoff - pd.Timedelta(days=lookback_days)
    mask = (df["date"] < cutoff) & (df["date"] >= lower)
    if same_competition_only:
        mask &= df["competition"] == state.competition
    train = df[mask].dropna(subset=["home_goals", "away_goals"])

    if train.empty:
        raise ValueError(
            f"No pre-cutoff training matches for {state.competition} {state.season} "
            f"before {cutoff.date() if hasattr(cutoff, 'date') else cutoff}."
        )

    # blend_weight_gl default 0.30 (was 0.60): an 8-fold temporal backtest of the
    # ELO-free engine (scripts/backtest_elo.py) found the goals-AttackDefense
    # estimator deserves ~70% and GL ~30% — 0.30 beat 0.60 in 8/8 folds
    # (pooled RPS -0.0022). See [[project_xg_modeling_findings]].
    engine = PredictionEngine(blend_weight_gl=blend_weight_gl)
    engine.fit(train)
    return engine


def fixture_lambdas(engine: PredictionEngine, state: LeagueState) -> pd.DataFrame:
    """Expected goals for every remaining fixture.

    Returns ``state.remaining`` with two added columns: ``lambda_home`` and
    ``lambda_away``. Neutral flag is honoured when present (always False for
    league play, but kept general for reuse by the tournament phase).
    """
    rows = []
    comp = state.competition
    for f in state.remaining.itertuples(index=False):
        neutral = bool(getattr(f, "neutral", 0) or 0)
        pred = engine.predict_match(f.home_team, f.away_team, competition=comp, neutral=neutral)
        rows.append({
            "match_id": getattr(f, "match_id", None),
            "date": getattr(f, "date", None),
            "home_team": f.home_team,
            "away_team": f.away_team,
            "lambda_home": pred.lambda_home,
            "lambda_away": pred.lambda_away,
        })
    return pd.DataFrame(rows)
