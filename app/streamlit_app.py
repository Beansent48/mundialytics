from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st
import plotly.express as px

from mundialytics.config import load_config
from mundialytics.data.loaders import load_matches, load_player_events, load_odds, load_lineups, to_long_team_rows
from mundialytics.features.team_features import build_goal_training_frame, fixture_feature_row
from mundialytics.models.goal_model import GoalLambdaModel, GoalModelConfig
from mundialytics.models.minutes_model import MinutesModel
from mundialytics.models.player_event_model import PlayerEventModel
from mundialytics.models.result_model import match_probabilities, summarize_score_matrix
from mundialytics.models.team_event_model import TeamEventModel
from mundialytics.ratings.elo import EloConfig, EloRater
from mundialytics.reports.daily_picks import build_daily_player_props

st.set_page_config(page_title="Mundialytics Betting Engine", layout="wide")
st.title("Mundialytics Betting Engine")
st.caption("ELO + Poisson/Skellam + player props + value detection. Paper mode only.")

@st.cache_data
def load_all():
    cfg = load_config(ROOT / "config" / "default.yaml")
    matches = load_matches(ROOT / cfg["data"]["sample_matches_path"])
    events = load_player_events(ROOT / cfg["data"]["sample_player_events_path"])
    odds = load_odds(ROOT / cfg["data"]["sample_odds_path"])
    lineups = load_lineups(ROOT / cfg["data"]["sample_lineups_path"])
    return cfg, matches, events, odds, lineups

cfg, matches, events, odds, lineups = load_all()
completed = matches.dropna(subset=["home_goals", "away_goals"])
rater = EloRater(EloConfig(**cfg["elo"]))
elo_hist = rater.fit(completed)
team_rows = to_long_team_rows(completed)
training_frame = build_goal_training_frame(team_rows, elo_hist)
goal_model = GoalLambdaModel(GoalModelConfig(model_type="poisson")).fit(training_frame)
team_event_model = TeamEventModel().fit(training_frame)
player_model = PlayerEventModel().fit(events)
minutes_model = MinutesModel().fit(events, projected_lineups=lineups)

teams = sorted(set(matches["home_team"].dropna()).union(matches["away_team"].dropna()))
col1, col2, col3 = st.columns(3)
with col1:
    home = st.selectbox("Team A", teams, index=teams.index("uruguay") if "uruguay" in teams else 0)
with col2:
    away = st.selectbox("Team B", teams, index=teams.index("spain") if "spain" in teams else min(1, len(teams)-1))
with col3:
    neutral = st.checkbox("Neutral venue", value=True)

fixture = rater.transform_fixture(home, away, neutral=int(neutral))
ctx = {**fixture, "neutral": int(neutral), "competition": "World Cup", "stage": "Group"}
X = fixture_feature_row(home, away, ctx, training_frame)
lambdas = goal_model.predict_lambda(X)
probs = match_probabilities(lambdas[0], lambdas[1])

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"P {home}", f"{probs.p_home_win:.1%}")
m2.metric("P Draw", f"{probs.p_draw:.1%}")
m3.metric(f"P {away}", f"{probs.p_away_win:.1%}")
m4.metric("Most likely", probs.most_likely_score)

st.subheader("Score probabilities")
score_df = summarize_score_matrix(probs.score_matrix, top_n=10)
st.dataframe(score_df, hide_index=True)
st.plotly_chart(px.bar(score_df, x="score", y="probability", title="Top scorelines"), use_container_width=True)

st.subheader("Team event estimates")
st.dataframe(team_event_model.predict(X), hide_index=True)

st.subheader("Player props from sample odds")
ctx_by_match_team = {
    (17, "uruguay"): {"elo_diff": fixture["elo_diff"], "expected_possession": 47},
    (17, "spain"): {"elo_diff": -fixture["elo_diff"], "expected_possession": 53},
}
picks = build_daily_player_props(odds, player_model, minutes_model, lineups=lineups, team_context_by_match_team=ctx_by_match_team)
st.dataframe(picks, hide_index=True)

st.warning("No uses esto para apostar dinero real todavía. Primero backtesting, calibración y paper trading.")
