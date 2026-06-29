import pandas as pd
import pytest

from mundialytics.artifacts.model_bundle import create_model_bundle
from mundialytics.data.adapters.football_data_uk import football_data_uk_to_matches
from mundialytics.data.identity import canonical_team_name, player_context_id
from mundialytics.data.schema import infer_single_scope, normalize_fixtures, validate_fixture_scope
from mundialytics.evaluation.metrics import safe_log_loss


def test_canonical_team_aliases():
    assert canonical_team_name("Real Madrid CF") == "real madrid"
    assert canonical_team_name("España") == "spain"


def test_player_context_separates_club_and_national():
    club = player_context_id("Federico Valverde", "Real Madrid", "club", "LaLiga")
    national = player_context_id("Federico Valverde", "Uruguay", "national", "World Cup")
    assert club != national
    assert "player_federico_valverde" in club


def test_infer_single_scope_rejects_mixed():
    df = pd.DataFrame({"team_scope": ["club", "national"]})
    with pytest.raises(ValueError):
        infer_single_scope(df)


def test_validate_fixture_scope_blocks_wrong_scope():
    fx = normalize_fixtures(pd.DataFrame([{
        "fixture_id": "f1", "date": "2026-06-01", "home_team": "Spain", "away_team": "Uruguay", "neutral": 1, "team_scope": "national"
    }]))
    with pytest.raises(ValueError):
        validate_fixture_scope(fx, "club")


def test_safe_log_loss_preserves_hda_order():
    y = ["H", "A"]
    P = [[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]]  # columns H,D,A
    assert safe_log_loss(y, P, labels=["H", "D", "A"]) < 0.3
