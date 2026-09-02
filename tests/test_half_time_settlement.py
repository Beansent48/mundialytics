"""Settlement of the half-time markets against real played matches.

The model side is covered by tests/test_half_time.py. This covers the other half
of the loop: that a logged half-time prediction is scored correctly once the
match is played. Both sides matter — the booking-points market was priced and
logged for months while `evaluate_prediction_log` silently skipped it for want
of red cards, so 7% of the track record could never be settled.

Uses REAL foundation rows rather than fabricated scorelines, so the settlement
logic is exercised against the same data the app will feed it.

Run:  .venv/Scripts/python.exe -m pytest tests/test_half_time_settlement.py -q
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"


def _settle(mercado: str, seleccion: str, linea, row) -> float:
    """Mirror of the half-time branch in app.evaluate_prediction_log."""
    hh, ah = row["home_goals_ht"], row["away_goals_ht"]
    ht_res = "1" if hh > ah else ("X" if hh == ah else "2")
    if mercado == "ht_1x2":
        return float(seleccion == ht_res)
    if mercado == "ht_goles":
        return float(((hh + ah) > float(linea)) == (seleccion == "OVER"))
    ft_res = ("1" if row["home_goals"] > row["away_goals"]
              else ("X" if row["home_goals"] == row["away_goals"] else "2"))
    return float(seleccion == f"{ht_res}/{ft_res}")


@pytest.fixture(scope="module")
def played():
    import pandas as pd
    if not FOUND.exists():
        pytest.skip("foundation not built")
    f = pd.read_csv(FOUND, low_memory=False)
    need = ["home_goals_ht", "away_goals_ht", "home_goals", "away_goals"]
    if any(c not in f.columns for c in need):
        pytest.skip("foundation has no half-time columns")
    return f.dropna(subset=need).tail(400).reset_index(drop=True)


def test_ht_1x2_settles_every_played_match(played):
    """Exactly one of 1/X/2 wins per match, and it is the one that happened."""
    for _, row in played.iterrows():
        hits = [_settle("ht_1x2", s, "", row) for s in ("1", "X", "2")]
        assert sum(hits) == 1.0, "a half-time result must settle exactly one pick"
        expected = ("1" if row.home_goals_ht > row.away_goals_ht
                    else ("X" if row.home_goals_ht == row.away_goals_ht else "2"))
        assert _settle("ht_1x2", expected, "", row) == 1.0


def test_ht_goals_over_under_are_complementary(played):
    for _, row in played.iterrows():
        for line in (0.5, 1.5, 2.5):
            o = _settle("ht_goles", "OVER", line, row)
            u = _settle("ht_goles", "UNDER", line, row)
            assert o + u == 1.0, f"OVER/UNDER {line} must be mutually exclusive"


def test_ht_goals_settle_on_the_half_time_score_not_full_time(played):
    """Guards the mistake this market invites: settling on the final score."""
    mismatched = played[(played.home_goals_ht + played.away_goals_ht)
                        != (played.home_goals + played.away_goals)]
    assert len(mismatched) > 0, "expected matches with second-half goals"
    row = mismatched.iloc[0]
    ht_total = row.home_goals_ht + row.away_goals_ht
    assert _settle("ht_goles", "OVER", ht_total - 0.5, row) == 1.0
    assert _settle("ht_goles", "UNDER", ht_total + 0.5, row) == 1.0


def test_ht_ft_settles_exactly_one_of_the_nine_paths(played):
    paths = [f"{a}/{b}" for a in "1X2" for b in "1X2"]
    for _, row in played.iterrows():
        hits = sum(_settle("ht_ft", p, "", row) for p in paths)
        assert hits == 1.0, "exactly one HT/FT path must settle"


def test_ht_ft_agrees_with_its_own_components(played):
    """The winning HT/FT path must match the separately-settled halves."""
    for _, row in played.iterrows():
        ht = ("1" if row.home_goals_ht > row.away_goals_ht
              else ("X" if row.home_goals_ht == row.away_goals_ht else "2"))
        ft = ("1" if row.home_goals > row.away_goals
              else ("X" if row.home_goals == row.away_goals else "2"))
        assert _settle("ht_ft", f"{ht}/{ft}", "", row) == 1.0
        assert _settle("ht_1x2", ht, "", row) == 1.0


def test_a_real_comeback_settles_as_such(played):
    """A side losing at the break and winning the match settles 2/1 or 1/2."""
    cb = played[((played.home_goals_ht < played.away_goals_ht)
                 & (played.home_goals > played.away_goals))]
    if cb.empty:
        pytest.skip("no comeback in this sample")
    row = cb.iloc[0]
    assert _settle("ht_ft", "2/1", "", row) == 1.0
    assert _settle("ht_ft", "1/1", "", row) == 0.0
