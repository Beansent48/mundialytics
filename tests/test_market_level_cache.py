"""The market-level memo must never cache a failure.

`api/engine._market_levels` writes its result to disk under a key that folds in
a stamp of the football-data directory. When those files cannot be read the
stamp is the hash of nothing, `market_level_detail` returns `{}`, and writing
that empty answer would cache it *under the very key that describes the files
being missing* -- so every later start with the same problem reads it back as a
hit and serves every match with no market shift, in under a second, without an
exception and with a perfectly plausible set of probabilities on the page.

Found on 2026-11-03 while building the A2 containers: a bind mount that could
not expose `data/raw/football_data` produced
`market_level_..._e3b0c442.json` (`e3b0c442` is the hash of the empty string),
and three later containers served without the level.

Why this is worth a test rather than only a comment: the market level has been
part of the published model since 2026-10-17, so the failure is a silent
divergence from the deployed baseline, and the memo it lives in exists purely
as a cold-start optimisation -- exactly the kind of code someone tightens later
without knowing what the guard is for.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def engine(monkeypatch, tmp_path):
    """`api.engine` with its cache pointed at a temporary directory."""
    api_engine = pytest.importorskip("api.engine")
    monkeypatch.setattr(api_engine, "CACHE_DIR", tmp_path / "cache")
    return api_engine


def _frame() -> pd.DataFrame:
    """The smallest thing `_market_levels` keys on: a size and a last date."""
    return pd.DataFrame({"date": pd.to_datetime(["2026-08-01", "2026-08-08"])})


def test_an_empty_level_is_not_written_to_disk(engine, monkeypatch, tmp_path):
    """The whole defect in one assertion: no file, so no later hit."""
    monkeypatch.setattr(engine, "market_level_detail", lambda *a, **k: {},
                        raising=False)
    monkeypatch.setitem(sys.modules, "mundialytics.features.market_level",
                        _stub({}))

    assert engine._market_levels(_frame(), 70.0) == {}

    cache = tmp_path / "cache"
    written = list(cache.glob("market_level_*.json")) if cache.exists() else []
    assert written == [], (
        "an empty market level was memoised; every later start with the same "
        f"unreadable football-data directory would read it back: {written}"
    )


def test_a_real_level_is_still_written(engine, tmp_path, monkeypatch):
    """The guard must not cost the optimisation it protects."""
    level = {"Bundesliga": (72, 1.0294302067286532)}
    monkeypatch.setitem(sys.modules, "mundialytics.features.market_level",
                        _stub(level))

    assert engine._market_levels(_frame(), 70.0) == level

    written = list((tmp_path / "cache").glob("market_level_*.json"))
    assert len(written) == 1, "a usable level should still be cached"
    assert json.loads(written[0].read_text(encoding="utf-8")) == {
        "Bundesliga": [72, 1.0294302067286532]
    }


def test_a_cached_empty_file_is_not_trusted_as_a_level(engine, tmp_path,
                                                       monkeypatch):
    """Belt and braces for the files the defect already wrote.

    A cache directory carrying one of the bad files from before the guard must
    recompute rather than serve it, so an installation that already has one
    heals on the next start instead of needing it deleted by hand.
    """
    level = {"Bundesliga": (72, 1.03)}
    monkeypatch.setitem(sys.modules, "mundialytics.features.market_level",
                        _stub(level))

    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    # The shape the defect left behind: valid JSON, and empty.
    (cache / "market_level_stale_e3b0c442.json").write_text("{}",
                                                            encoding="utf-8")

    assert engine._market_levels(_frame(), 70.0) == level, (
        "an empty cache file from before the guard was served as a real level"
    )


def _stub(detail):
    """A stand-in for `mundialytics.features.market_level`.

    `_market_levels` imports it inside the function, so patching the attribute
    on `api.engine` is not enough -- the import has to be intercepted.
    """
    import types

    module = types.ModuleType("mundialytics.features.market_level")
    module.market_level_detail = lambda *a, **k: detail
    return module
