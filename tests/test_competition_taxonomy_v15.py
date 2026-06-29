from __future__ import annotations

import pandas as pd

from mundialytics.data.competition_taxonomy import classify_competition, enrich_competition_metadata
from mundialytics.evaluation.player_props import PlayerPropBacktestConfig, backtest_player_props


def test_competition_taxonomy_objective_labels():
    assert classify_competition("La Liga").team_type == "club"
    assert classify_competition("La Liga").competition_context == "domestic_league"
    assert classify_competition("Champions League").team_scope == "club"
    assert classify_competition("Champions League").competition_context == "continental_club"
    assert classify_competition("FIFA World Cup").team_type == "national_team"
    assert classify_competition("FIFA World Cup").competition_context == "international_national_tournament"
    assert classify_competition("Friendly").competition_context == "friendly"
    assert classify_competition("UEFA Euro qualification").competition_context == "qualifier"
    # No subjective match_importance label is introduced.
    labels = enrich_competition_metadata(pd.DataFrame({"competition": ["La Liga"]}))
    assert "match_importance" not in labels.columns


def test_enrich_competition_metadata_overwrites_bad_team_scope():
    df = pd.DataFrame({"competition": ["La Liga", "FIFA World Cup", "UEFA Europa League"], "team_scope": ["national", "club", "national"]})
    out = enrich_competition_metadata(df, overwrite=True)
    assert out["team_scope"].tolist() == ["club", "national", "club"]
    assert out["team_type"].tolist() == ["club", "national_team", "club"]
    assert out["competition_context"].tolist() == ["domestic_league", "international_national_tournament", "continental_club"]


def test_player_prop_predictions_preserve_domain_labels():
    rows = []
    dates = pd.date_range("2024-01-01", periods=6, freq="7D")
    for i, d in enumerate(dates):
        for team, opp, player, comp in [
            ("Barcelona", "Real Madrid", "Player Club", "La Liga"),
            ("Spain", "Uruguay", "Player Nat", "FIFA World Cup"),
        ]:
            rows.append({
                "match_id": f"m{i}_{team}",
                "date": d.date().isoformat(),
                "competition": comp,
                "team_scope": "national",  # intentionally wrong for La Liga, taxonomy should correct it upstream
                "team": team,
                "opponent": opp,
                "player": player,
                "player_id_global": f"pid_{player}",
                "player_context_id": f"ctx_{player}",
                "position": "FW",
                "started": 1,
                "minutes": 80,
                "shots": 1,
                "shots_on_target": 0,
                "fouls_committed": 1,
                "fouls_drawn": 0,
                "yellow_cards": 0,
                "goals": 0,
                "assists": 0,
            })
    events = enrich_competition_metadata(pd.DataFrame(rows), overwrite=True)
    pred, _ = backtest_player_props(events, PlayerPropBacktestConfig(min_train_matches=4, test_matches=2, markets=("player_shots",)))
    assert {"team_type", "competition_context", "gender"}.issubset(pred.columns)
    assert set(pred.loc[pred["competition"] == "La Liga", "team_scope"]) == {"club"}
    assert set(pred.loc[pred["competition"] == "FIFA World Cup", "team_scope"]) == {"national"}


def test_audit_table_catches_bad_domain_labels():
    from scripts.audit_props_pipeline import audit_table

    df = pd.DataFrame({
        "match_id": ["m1"],
        "date": ["2024-01-01"],
        "competition": ["La Liga"],
        "team_scope": ["national"],
        "team": ["barcelona"],
        "opponent": ["real madrid"],
        "player": ["player"],
        "minutes": [90],
        "shots": [1],
        "shots_on_target": [0],
        "fouls_committed": [1],
        "fouls_drawn": [0],
        "yellow_cards": [0],
    })
    audit = audit_table(
        df,
        kind="player_events",
        require_valid_date=True,
        max_date_null_rate=0.01,
        forbid_placeholders={"statsbomb open data"},
        include_competitions=set(),
        exclude_competitions=set(),
        expected_domain=None,
    )
    assert any("team_scope_mismatch_from_competition" in e for e in audit["errors"])
