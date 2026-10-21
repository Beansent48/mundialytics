"""Startup warm-up: who pays for the model load, and what may be memoised.

The API used to bind its port holding nothing, so the first request after a
start or a daily refresh paid the whole load: measured 25-55 s with the joblib
fit cache warm and 119 s without it, against a web client that gives up after
20 s and renders the timeout as a 404. Two changes fixed that — a lifespan that
loads the models before uvicorn reports ready, and a disk memo for the market
level, which is 28 s of the 35 and deliberately cannot live in the joblib.

What is pinned here is the part that could serve wrong numbers: the memo's key
has to move when anything it reads moves, and warming must not turn the models
into process-lifetime state, because the daily refresh reaches the site through
`data_version()`.

Run:  .venv/Scripts/python.exe -m pytest tests/test_cold_start_warmup.py -q
"""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402

from api import engine as eng  # noqa: E402

DETAIL = {"LaLiga": (87, 1.0516), "Premier League": (69, 1.0394)}


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(["2026-10-18", "2026-10-19"])})


@pytest.fixture
def memo(tmp_path, monkeypatch):
    """The level memo in a temporary directory, with a counted estimator."""
    from mundialytics.features import market_level as ml

    calls = []
    monkeypatch.setattr(eng, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ml, "market_level_detail",
                        lambda df, asof, k=70.0: calls.append(k) or dict(DETAIL))
    lam = tmp_path / "served_lambdas_current.csv"
    lam.write_text("fecha,home,away,lh,la\n", encoding="utf-8")
    monkeypatch.setattr(eng, "LEVEL_INPUTS", (lam,))
    monkeypatch.setattr(eng, "FOOTBALL_DATA", tmp_path / "football_data")
    return calls, lam


def test_the_memo_returns_the_same_numbers_it_computed(memo):
    calls, _ = memo
    fresh = eng._market_levels(_frame(), 70.0)
    cached = eng._market_levels(_frame(), 70.0)
    assert fresh == DETAIL
    assert cached == DETAIL, "the memo must not round-trip the multiplier away"
    assert len(calls) == 1, "the second call recomputed"


def test_a_new_logged_lambda_recomputes_the_level(memo):
    """The level is measured from the lambdas we served; a new row changes it."""
    calls, lam = memo
    eng._market_levels(_frame(), 70.0)
    lam.write_text("fecha,home,away,lh,la\n2026-10-19,valencia,betis,1.4,1.1\n",
                   encoding="utf-8")
    eng._market_levels(_frame(), 70.0)
    assert len(calls) == 2


def test_new_data_recomputes_the_level(memo):
    calls, _ = memo
    eng._market_levels(_frame(), 70.0)
    more = pd.DataFrame({"date": pd.to_datetime(["2026-10-18", "2026-10-19", "2026-10-26"])})
    eng._market_levels(more, 70.0)
    assert len(calls) == 2


def test_a_different_k_recomputes_the_level(memo):
    calls, _ = memo
    eng._market_levels(_frame(), 70.0)
    eng._market_levels(_frame(), 120.0)
    assert len(calls) == 2


def test_an_odds_file_landing_recomputes_the_level(memo):
    """`book_over25` reads data/raw/football_data; a new download changes the
    market side of the comparison without touching the foundation."""
    calls, _ = memo
    fd = eng.FOOTBALL_DATA
    fd.mkdir(parents=True, exist_ok=True)
    (fd / "2627_SP1.csv").write_text("Date,HomeTeam,AwayTeam\n", encoding="utf-8")
    eng._market_levels(_frame(), 70.0)
    (fd / "2627_E0.csv").write_text("Date,HomeTeam,AwayTeam\n", encoding="utf-8")
    eng._market_levels(_frame(), 70.0)
    assert len(calls) == 2


def test_the_memo_keeps_one_file(memo):
    _, lam = memo
    eng._market_levels(_frame(), 70.0)
    lam.write_text("fecha,home,away,lh,la\n2026-10-19,valencia,betis,1.4,1.1\n",
                   encoding="utf-8")
    eng._market_levels(_frame(), 70.0)
    assert len(list(eng.CACHE_DIR.glob("market_level_*.json"))) == 1


# ── readiness ────────────────────────────────────────────────────────────────
@pytest.fixture
def ready(monkeypatch):
    """api.main's readiness state, reset, with the model loads stubbed out."""
    import api.main as m

    loads = []
    monkeypatch.setattr(m, "_READY", None, raising=False)
    monkeypatch.setattr(m, "_LOAD", threading.Lock())
    monkeypatch.setattr(m, "data_version", lambda: 1.0)
    monkeypatch.setattr(m, "club_engine", lambda: loads.append("engine") or (1, 2, 3))
    monkeypatch.setattr(m, "props_models", lambda: loads.append("props") or (4, 5))
    return m, loads


def test_the_models_load_once_and_then_cost_nothing(ready):
    m, loads = ready
    m._ready()
    m._ready()
    m._ready()
    assert loads == ["engine", "props"]


def test_a_request_arriving_mid_load_gets_a_503_not_a_queue(ready):
    """Four tabs opened on a cold API used to wait four times for the same
    load, every one of them past the client's 20 s abort."""
    m, _ = ready
    m._LOAD.acquire()
    try:
        with pytest.raises(HTTPException) as caught:
            m._ready()
    finally:
        m._LOAD.release()
    assert caught.value.status_code == 503
    assert caught.value.headers["Retry-After"] == "5"


def test_a_refresh_reloads_rather_than_serving_the_old_fit(ready, monkeypatch):
    """Warming at startup must not become process-lifetime state: the daily
    refresh moves the foundation's mtime and that is how it reaches the site."""
    m, loads = ready
    m._ready()
    monkeypatch.setattr(m, "data_version", lambda: 2.0)       # a refresh landed
    m._ready()
    assert loads == ["engine", "props", "engine", "props"]


def test_the_lifespan_loads_the_models_before_it_yields(monkeypatch):
    import api.main as m

    order = []
    monkeypatch.setattr(m, "_ready", lambda: order.append("ready"))
    monkeypatch.setattr(m, "_warm_payloads", lambda: order.append("payloads"))

    async def run():
        async with m.lifespan(m.app):
            order.append("serving")

    asyncio.run(run())
    assert order[0] == "ready", "the port opened before the models were loaded"
    assert "serving" in order


def test_a_failed_warm_up_still_lets_the_server_start(monkeypatch):
    """A start that cannot load the models must still answer /health, or there
    is no way to see what is wrong."""
    import api.main as m

    def boom():
        raise RuntimeError("no foundation on disk")

    monkeypatch.setattr(m, "_ready", boom)
    monkeypatch.setattr(m, "_warm_payloads", lambda: None)

    served = []

    async def run():
        async with m.lifespan(m.app):
            served.append(True)

    asyncio.run(run())
    assert served == [True]
