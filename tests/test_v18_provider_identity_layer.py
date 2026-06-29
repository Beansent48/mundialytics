from __future__ import annotations

import pandas as pd

from mundialytics.data.provider_identity import (
    canonical_provider_player_id,
    standardize_provider_players,
    attach_identity_map_to_lineups,
)
from mundialytics.inference.safe_props import predict_props_for_lineups
from scripts.build_provider_identity_map import build_identity_map


def _events() -> pd.DataFrame:
    rows = []
    for i in range(4):
        rows.append({
            "match_id": f"club{i}", "date": f"2024-0{i+1}-01", "competition": "La Liga", "team_scope": "club",
            "team": "real madrid", "opponent": "barcelona", "player": "Federico Santiago Valverde Dipetta", "position": "CM",
            "minutes": 90, "shots": 2, "shots_on_target": 1, "fouls_committed": 2, "fouls_drawn": 1,
            "yellow_cards": 0, "goals": 0, "assists": 0,
        })
    rows.append({
        "match_id": "nat1", "date": "2024-06-01", "competition": "Copa America", "team_scope": "national",
        "team": "uruguay", "opponent": "brazil", "player": "Federico Santiago Valverde Dipetta", "position": "CM",
        "minutes": 90, "shots": 1, "shots_on_target": 0, "fouls_committed": 1, "fouls_drawn": 1,
        "yellow_cards": 1, "goals": 0, "assists": 0,
    })
    return pd.DataFrame(rows)


def test_provider_canonical_id_is_stable():
    assert canonical_provider_player_id("api-football", "123.0") == "api_football:123"
    assert canonical_provider_player_id("API Sports", 456) == "api_football:456"


def test_build_provider_identity_map_links_short_provider_name_to_history():
    provider = pd.DataFrame({
        "provider": ["api_football"],
        "provider_player_id": [1002],
        "player": ["Federico Valverde"],
        "team": ["Uruguay"],
        "position": ["CM"],
    })
    identity_map, summary = build_identity_map(provider, _events())
    assert summary["match_status_counts"].get("matched", 0) == 1
    row = identity_map.iloc[0]
    assert row["canonical_player_id"] == "api_football:1002"
    assert row["historical_player_id_global"]
    assert "valverde" in row["historical_player_name"]


def test_attach_identity_map_sets_historical_lookup_id():
    provider = pd.DataFrame({"provider": ["api_football"], "provider_player_id": [1002], "player": ["Federico Valverde"]})
    identity_map, _ = build_identity_map(provider, _events())
    lineups = pd.DataFrame({
        "match_id": ["m1"], "date": ["2026-06-26"], "competition": ["FIFA World Cup"],
        "team": ["Uruguay"], "opponent": ["Spain"], "player": ["Federico Valverde"],
        "provider": ["api_football"], "provider_player_id": [1002], "position": ["CM"],
        "expected_minutes": [90], "started": [1],
    })
    out = attach_identity_map_to_lineups(lineups, identity_map)
    assert out.loc[0, "identity_map_status"] == "matched"
    assert out.loc[0, "player_id_global"] == identity_map.loc[0, "historical_player_id_global"]


def test_safe_props_uses_provider_identity_map_not_generic_prior():
    provider = pd.DataFrame({"provider": ["api_football"], "provider_player_id": [1002], "player": ["Federico Valverde"]})
    identity_map, _ = build_identity_map(provider, _events())
    lineups = pd.DataFrame({
        "match_id": ["m1"], "date": ["2026-06-26"], "competition": ["FIFA World Cup"],
        "team": ["Uruguay"], "opponent": ["Spain"], "player": ["Federico Valverde"],
        "provider": ["api_football"], "provider_player_id": [1002], "position": ["CM"],
        "expected_minutes": [90], "started": [1],
    })
    preds = predict_props_for_lineups(_events(), lineups, markets=["player_shots"], identity_map=identity_map)
    r = preds.iloc[0]
    assert r["identity_map_status"] == "matched"
    assert r["sample_size"] > 0
    assert "generic_prior" not in r["explanation"]
