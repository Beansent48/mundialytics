"""Squad market value as an engine input (src/mundialytics/features/squad_value.py).

The shift tilts the final lambdas toward the side with the more valuable squad. It was
validated on the weekly-refit walk-forward (scripts/squad_value/README.md). What these
tests pin down is what makes it safe to serve: dated values never leak the future into a
past fit, total goals are untouched, a missing value leaves the prediction alone, and the
engine is byte-identical while the flag is off.

Run:  .venv/Scripts/python.exe -m pytest tests/test_squad_value.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.features.squad_value import (  # noqa: E402
    _norm, squad_values_asof, upsert_snapshots)
from mundialytics.statistical_core.prediction_engine import (  # noqa: E402
    DEPLOYED_CLUB_ENGINE_KWARGS, PredictionEngine)

SHIFT = {"beta": 0.105, "kappa": -0.215}


def _snaps():
    return pd.DataFrame({
        "snap": pd.to_datetime(["2024-08-05", "2024-08-05", "2025-08-04", "2025-08-04", "2025-08-04"]),
        "team": ["arsenal", "ipswich", "arsenal", "ipswich", "como"],
        "v18": [1.1e9, 1.5e8, 1.2e9, 2.6e8, 2.0e8],
    })


def test_asof_never_reads_a_later_snapshot():
    v = squad_values_asof(_snaps(), "2024-10-15")
    assert v == {"arsenal": 1.1e9, "ipswich": 1.5e8}   # Como's 2025 row is in the future


def test_asof_none_reads_the_latest():
    v = squad_values_asof(_snaps(), None)
    assert v["ipswich"] == 2.6e8 and v["como"] == 2.0e8


def test_stale_snapshots_are_dropped():
    # a club last seen more than MAX_AGE_DAYS before the as-of date (relegated,
    # out of the data) must not keep an old value
    assert squad_values_asof(_snaps(), "2025-06-30") == {}


def test_upsert_replaces_a_snapshot_date_and_keeps_the_rest():
    old = _snaps()
    new = pd.DataFrame({"snap": pd.to_datetime(["2025-08-04"]), "team": ["arsenal"], "v18": [9.9e8]})
    out = upsert_snapshots(old, new)
    assert len(out) == 3
    assert out.loc[out.snap == "2025-08-04", "v18"].tolist() == [9.9e8]


def test_names_normalise_across_sources():
    assert _norm("Pascal Groß") == _norm("Pascal Gross")
    assert _norm("Óscar Zambrano") == "oscar zambrano"
    assert _norm("Aaron Wan-Bissaka") == "aaron wan bissaka"


def _engine(shift=SHIFT):
    eng = PredictionEngine(**{**DEPLOYED_CLUB_ENGINE_KWARGS, "squad_value_shift": shift})
    eng.set_squad_values({"Arsenal": 1.2e9, "Ipswich": 2.6e8, "Como": 2.0e8})
    return eng


def test_tilt_keeps_total_goals_and_favours_the_richer_side():
    lh, la, applied = _engine()._squad_value_tilt("ipswich", "arsenal", 1.2, 1.4)
    assert applied
    assert lh * la == pytest.approx(1.2 * 1.4)
    assert la / lh > 1.4 / 1.2          # Arsenal's squad is worth ~4.6x Ipswich's


def test_tilt_matches_the_validated_formula():
    lh, la, _ = _engine()._squad_value_tilt("arsenal", "como", 2.0, 0.8)
    s = SHIFT["beta"] * np.log(1.2e9 / 2.0e8) + SHIFT["kappa"] * np.log(2.0 / 0.8)
    assert lh == pytest.approx(2.0 * np.exp(s / 2))
    assert la == pytest.approx(0.8 * np.exp(-s / 2))


def test_missing_value_leaves_the_lambdas_alone():
    assert _engine()._squad_value_tilt("arsenal", "elversberg", 1.5, 1.1) == (1.5, 1.1, False)


def test_off_by_default():
    eng = PredictionEngine()
    eng.set_squad_values({"arsenal": 1.2e9, "ipswich": 2.6e8})
    assert eng._squad_value_tilt("arsenal", "ipswich", 1.5, 1.1) == (1.5, 1.1, False)


def test_historical_fit_without_asof_is_refused(tmp_path):
    f = tmp_path / "sv.csv"
    _snaps().to_csv(f, index=False)
    eng = PredictionEngine(squad_value_shift={**SHIFT, "path": str(f)})
    old = pd.DataFrame({"date": pd.to_datetime(["2021-05-01", "2021-05-08"])})
    with pytest.raises(ValueError, match="squad_value_asof"):
        eng._load_squad_values(old, None)
    eng._load_squad_values(old, "2024-09-01")
    assert eng.squad_values_ == {"arsenal": 1.1e9, "ipswich": 1.5e8}


def test_deployed_config_carries_the_validated_constants():
    assert DEPLOYED_CLUB_ENGINE_KWARGS["squad_value_shift"] == {"beta": 0.1051, "kappa": -0.2148}
