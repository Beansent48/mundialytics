import pandas as pd

from mundialytics.data.identity import add_team_identity_columns
from mundialytics.data.loaders import load_lineups, load_player_events
from mundialytics.models.minutes_model import MinutesModel
from mundialytics.models.player_event_model import PlayerEventModel
from mundialytics.models.substitute_plus import SubstitutePlusModel
from mundialytics.reports.paper_tracker import prepare_picks_for_tracking


def test_add_team_identity_without_scope_keeps_all_rows():
    df = pd.DataFrame({"team": ["Spain", "Real Madrid", "USA", "Barcelona", "Uruguay", "Valencia", "France", "Italy"]})
    out = add_team_identity_columns(df)
    assert len(out) == len(df)
    assert out["team_scope"].eq("unknown").all()
    assert out["team_id"].notna().all()


def test_lineup_normalization_enables_minutes_and_substitute_plus(tmp_path):
    events_path = tmp_path / "events.csv"
    pd.DataFrame({
        "match_id": [1, 2],
        "date": ["2024-01-01", "2024-01-05"],
        "team": ["Spain", "Spain"],
        "opponent": ["Italy", "France"],
        "player": ["Lamine Yamal", "Ferran Torres"],
        "position": ["RW", "RW"],
        "minutes": [90, 90],
        "shots": [2, 1],
        "shots_on_target": [1, 1],
        "fouls_committed": [0, 1],
        "fouls_drawn": [2, 1],
        "yellow_cards": [0, 0],
        "goals": [0, 0],
        "assists": [1, 0],
    }).to_csv(events_path, index=False)
    lineups_path = tmp_path / "lineups.csv"
    pd.DataFrame({
        "match_id": [99],
        "team": ["Spain"],
        "player": ["Lamine Yamal"],
        "position": ["RW"],
        "started": [1],
        "minutes": [75],
        "replaced_by": ["Ferran Torres"],
        "replacement_minute": [75],
    }).to_csv(lineups_path, index=False)

    events = load_player_events(events_path)
    lineups = load_lineups(lineups_path)
    minutes = MinutesModel().fit(events, projected_lineups=lineups)
    estimate = minutes.estimate("lamine yamal", match_id=99)
    assert estimate["expected_minutes"] == 75
    assert estimate["replaced_by"] == "ferran torres"

    player_model = PlayerEventModel(min_minutes_for_rate=0).fit(events)
    pred = player_model.predict_market("lamine yamal", "player_shots_on_target", "1+", 75)
    adjusted = SubstitutePlusModel(player_model, lineups=lineups).apply(pred, match_id=99)
    assert adjusted["replacement"] == "ferran torres"
    assert adjusted["probability_substitute_plus"] >= pred.probability


def test_paper_tracker_uses_team_when_player_missing():
    picks = pd.DataFrame({
        "match_id": [1],
        "market_type": ["match_winner"],
        "team": ["spain"],
        "player": [pd.NA],
        "line": ["win"],
        "odds": [1.8],
        "model_probability": [0.60],
        "implied_probability": [0.55],
        "edge": [0.05],
        "expected_return": [0.08],
        "value_flag": [True],
    })
    tracked = prepare_picks_for_tracking(picks, "2026-01-01T00:00:00Z")
    assert tracked.loc[0, "selection"] == "spain"

from mundialytics.reports.paper_tracker import ledger_summary


def test_ledger_summary_counts_open_stake():
    ledger = pd.DataFrame({"status": ["open", "open"], "stake": [1.0, 0.5], "profit": [pd.NA, pd.NA]})
    summary = ledger_summary(ledger)
    assert summary["open_stake"] == 1.5
    assert summary["total_stake"] == 1.5
    assert summary["settled_stake"] == 0.0
