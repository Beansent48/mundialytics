"""npxG must actually exclude penalties.

It did not, for twelve seasons: `build_understat_xg_matches` looked for
`situation == "penalty"` and Understat leaves `situation` EMPTY on a penalty, so
`astype(str)` made it the string "nan", `isin({"penalty"})` was never true, and
home_npxg came out byte-for-byte equal to home_xg -- a column whose name said
non-penalty xG and whose contents were xG. See scripts/npxg/README.md.

Run:  .venv/Scripts/python.exe -m pytest tests/test_npxg_penalties.py -q
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "bux", ROOT / "scripts/build_understat_xg_matches.py")
bux = importlib.util.module_from_spec(_spec)
sys.modules["bux"] = bux
_spec.loader.exec_module(bux)


def _df(situations, xgs):
    return pd.DataFrame({"situation": situations, "xg": xgs})


def test_a_penalty_with_an_empty_situation_is_still_a_penalty():
    """The whole bug. Understat prices every penalty at 0.761169."""
    flags = bux.flag_penalties(_df(["Open Play", None, "From Corner"],
                                   [0.10, 0.761169, 0.05]), log=lambda m: None)
    assert list(flags) == [False, True, False]


def test_the_literal_penalty_situation_still_works():
    flags = bux.flag_penalties(_df(["Open Play", "Penalty"], [0.10, 0.76]),
                               log=lambda m: None)
    assert list(flags) == [False, True]


def test_an_empty_situation_outside_the_penalty_band_is_left_alone():
    """Checked, not trusted: if a future scrape leaves `situation` empty for
    something else, those shots keep their xG in npxg and the build says so."""
    said = []
    flags = bux.flag_penalties(_df(["Open Play", None, None], [0.10, 0.04, 0.06]),
                               log=said.append)
    assert not flags.any()
    assert any("WARNING" in m for m in said)


def test_the_string_nan_counts_as_empty():
    """astype(str) on a NaN gives "nan" -- the exact shape of the original bug."""
    flags = bux.flag_penalties(_df(["nan", "Open Play"], [0.761169, 0.10]),
                               log=lambda m: None)
    assert list(flags) == [True, False]
