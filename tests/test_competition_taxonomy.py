"""enrich_competition_metadata: the `overwrite` flag eleven call sites depend on.

The function only ever accepted `df`, while callers across adapters, features
and scripts have always passed `overwrite=` -- so every one of those paths
raised TypeError from the initial commit onward. These tests pin both the
restored signature and the non-destructive semantics chosen for it.

Run:  .venv/Scripts/python.exe -m pytest tests/test_competition_taxonomy.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.competition_taxonomy import enrich_competition_metadata  # noqa: E402


def test_accepts_overwrite_keyword_both_ways():
    df = pd.DataFrame({"x": [1, 2]})
    for flag in (True, False):
        out = enrich_competition_metadata(df, overwrite=flag)
        assert {"team_scope", "competition"} <= set(out.columns)


def test_missing_columns_get_defaults():
    out = enrich_competition_metadata(pd.DataFrame({"x": [1]}))
    assert out["team_scope"].iloc[0] == "club"
    assert out["competition"].iloc[0] == "unknown"


def test_existing_values_are_never_replaced():
    df = pd.DataFrame({"team_scope": ["international"], "competition": ["World Cup"]})
    for flag in (True, False):
        out = enrich_competition_metadata(df, overwrite=flag)
        assert out["team_scope"].iloc[0] == "international"
        assert out["competition"].iloc[0] == "World Cup"


def test_overwrite_fills_blanks_only_when_asked():
    df = pd.DataFrame({"team_scope": ["club", np.nan, ""],
                       "competition": ["LaLiga", np.nan, "  "]})

    kept = enrich_competition_metadata(df, overwrite=False)
    assert kept["competition"].isna().sum() == 1, "overwrite=False must not fill blanks"

    filled = enrich_competition_metadata(df, overwrite=True)
    assert filled["competition"].tolist() == ["LaLiga", "unknown", "unknown"]
    assert filled["team_scope"].tolist() == ["club", "club", "club"]


def test_input_frame_is_not_mutated():
    df = pd.DataFrame({"x": [1]})
    enrich_competition_metadata(df, overwrite=True)
    assert list(df.columns) == ["x"]
