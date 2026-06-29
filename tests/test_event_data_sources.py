from pathlib import Path

from mundialytics.data.adapters import (
    statsbomb_events_to_lineups,
    statsbomb_events_to_player_events,
    statsbomb_events_to_tactical_shifts,
    wyscout_events_to_player_events,
    wyscout_matches_to_lineups,
)

ROOT = Path(__file__).resolve().parents[1]


def test_statsbomb_events_lineups_and_tactical_shifts():
    fp = ROOT / "data/sample/event_json/statsbomb_sample_events.json"
    pe = statsbomb_events_to_player_events(fp, match_id="sb_1", team_scope="national", competition="sample")
    lamine = pe.loc[pe["player"] == "lamine yamal"].iloc[0]
    valverde = pe.loc[pe["player"] == "federico valverde"].iloc[0]
    assert lamine["shots"] == 1
    assert lamine["shots_on_target"] == 1
    assert lamine["fouls_drawn"] == 1
    assert valverde["fouls_committed"] == 1
    assert valverde["yellow_cards"] == 1

    lu = statsbomb_events_to_lineups(fp, match_id="sb_1", team_scope="national", competition="sample")
    lamine_lu = lu.loc[lu["player"] == "lamine yamal"].iloc[0]
    ferran_lu = lu.loc[lu["player"] == "ferran torres"].iloc[0]
    assert lamine_lu["minutes"] == 70
    assert lamine_lu["replaced_by"] == "Ferran Torres"
    assert ferran_lu["started"] == 0
    assert ferran_lu["minutes"] == 20

    ts = statsbomb_events_to_tactical_shifts(fp, match_id="sb_1", team_scope="national", competition="sample")
    assert set(ts["event_type"]) == {"Starting XI", "Tactical Shift"}
    assert 4231 in set(ts["formation"])


def test_wyscout_events_and_lineups():
    base = ROOT / "data/sample/event_json"
    pe = wyscout_events_to_player_events(
        base / "wyscout_events_sample.json",
        matches_json=base / "wyscout_matches_sample.json",
        players_json=base / "wyscout_players_sample.json",
        teams_json=base / "wyscout_teams_sample.json",
        competition="sample",
        team_scope="club",
    )
    lamine = pe.loc[pe["player"] == "lamine yamal"].iloc[0]
    valverde = pe.loc[pe["player"] == "federico valverde"].iloc[0]
    assert lamine["shots"] == 1
    assert lamine["shots_on_target"] == 1
    assert lamine["key_passes"] == 1
    assert valverde["fouls_committed"] == 1
    assert valverde["yellow_cards"] == 1

    lu = wyscout_matches_to_lineups(
        base / "wyscout_matches_sample.json",
        players_json=base / "wyscout_players_sample.json",
        teams_json=base / "wyscout_teams_sample.json",
        competition="sample",
        team_scope="club",
    )
    lamine_lu = lu.loc[lu["player"] == "lamine yamal"].iloc[0]
    ferran_lu = lu.loc[lu["player"] == "ferran torres"].iloc[0]
    assert lamine_lu["minutes"] == 65
    assert ferran_lu["started"] == 0
    assert ferran_lu["minutes"] == 25


def test_lineup_minutes_override_wyscout_default_when_merging(tmp_path):
    from mundialytics.data.events import merge_player_events_with_lineups, add_basic_event_metrics
    base = ROOT / "data/sample/event_json"
    pe = wyscout_events_to_player_events(
        base / "wyscout_events_sample.json",
        matches_json=base / "wyscout_matches_sample.json",
        players_json=base / "wyscout_players_sample.json",
        teams_json=base / "wyscout_teams_sample.json",
        competition="sample",
        team_scope="club",
    )
    lu = wyscout_matches_to_lineups(
        base / "wyscout_matches_sample.json",
        players_json=base / "wyscout_players_sample.json",
        teams_json=base / "wyscout_teams_sample.json",
        competition="sample",
        team_scope="club",
    )
    merged = add_basic_event_metrics(merge_player_events_with_lineups(pe, lu))
    lamine = merged.loc[merged["player"] == "lamine yamal"].iloc[0]
    assert lamine["minutes"] == 65
    assert round(float(lamine["shots_per90"]), 3) == round(1 / 65 * 90, 3)
