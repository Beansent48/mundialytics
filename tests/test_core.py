from mundialytics.models.result_model import match_probabilities
from mundialytics.betting.value import expected_return
from mundialytics.models.player_event_model import PlayerEventModel


def test_match_probabilities_sum_close():
    p = match_probabilities(1.5, 1.0)
    assert abs((p.p_home_win + p.p_draw + p.p_away_win) - 1) < 1e-8


def test_expected_return_positive():
    assert expected_return(0.60, 2.0) > 0


def test_parse_line():
    assert PlayerEventModel.parse_line("2+") == 2
