"""The pre-kickoff lineup pass (scripts/log_lineup_pass.py) and its pieces.

The signal is the share of a team's usual starters missing from the confirmed XI. In
the backtest it improved 1X2 RPS in 6 of 6 seasons on top of the squad value, with O/U
unchanged (scripts/lineup_pass/backtest_absence.py). These tests pin what makes it safe
to serve: the regulars rule matches the backtest, the tilt keeps total goals, nothing
moves without a confirmed XI, and the page reads back what was logged.

Run:  .venv/Scripts/python.exe -m pytest tests/test_lineup_pass.py -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from mundialytics.features.lineup_absence import absent_share, regular_xi  # noqa: E402
from mundialytics.statistical_core.prediction_engine import (  # noqa: E402
    DEPLOYED_CLUB_ENGINE_KWARGS, PredictionEngine)


def _history(n_matches: int) -> pd.DataFrame:
    """Players p0..p10 start every match; p11 starts only the last two; p12 never."""
    rows = []
    for k in range(n_matches):
        date = pd.Timestamp("2026-08-15") + pd.Timedelta(days=7 * k)
        for i in range(11):
            starter = not (i == 10 and k >= n_matches - 2)   # p10 dropped lately
            rows.append({"date": date, "player": f"p{i}", "starter": starter})
        rows.append({"date": date, "player": "p11", "starter": k >= n_matches - 2})
        rows.append({"date": date, "player": "p12", "starter": False})
    return pd.DataFrame(rows)


def test_too_little_history_gives_no_signal():
    assert regular_xi(_history(4), before="2027-01-01") is None


def test_regulars_are_the_eleven_most_frequent_starters():
    regs = regular_xi(_history(8), before="2027-01-01")
    assert len(regs) == 11
    assert "p12" not in regs and "p10" in regs     # 6 starts beat p11's 2


def test_only_matches_before_the_kickoff_count():
    h = _history(8)
    regs = regular_xi(h, before=h["date"].max())    # the last match is today's
    assert regs is not None and "p11" not in regs


def test_absent_share():
    regs = [f"p{i}" for i in range(11)]
    assert absent_share(regs, regs) == 0.0
    assert absent_share(regs, regs[:8] + ["x", "y", "z"]) == pytest.approx(3 / 11)
    assert absent_share([], ["a"]) == 0.0


def test_tilt_keeps_total_goals_and_hurts_the_weakened_side():
    eng = PredictionEngine(lineup_shift={"theta": -0.3647})
    lh, la, applied = eng._lineup_tilt(1.5, 1.1, (3 / 11, 0.0))
    assert applied
    assert lh * la == pytest.approx(1.5 * 1.1)
    assert lh < 1.5 and la > 1.1


def test_no_lineup_no_change():
    eng = PredictionEngine(**DEPLOYED_CLUB_ENGINE_KWARGS)
    assert eng._lineup_tilt(1.5, 1.1, None) == (1.5, 1.1, False)
    assert eng._lineup_tilt(1.5, 1.1, (None, 0.2)) == (1.5, 1.1, False)
    assert PredictionEngine()._lineup_tilt(1.5, 1.1, (0.5, 0.0)) == (1.5, 1.1, False)


def test_deployed_lineup_constant():
    assert DEPLOYED_CLUB_ENGINE_KWARGS["lineup_shift"] == {"theta": -0.3647}


def _summary(n_home: int, n_away: int) -> dict:
    def side(where, name, n):
        return {"homeAway": where, "team": {"displayName": name},
                "roster": [{"starter": i < n, "athlete": {"displayName": f"{name} {i}"}} for i in range(18)]}
    return {"rosters": [side("home", "Arsenal", n_home), side("away", "Chelsea", n_away)]}


def test_xis_confirmed_only_with_eleven_starters_each():
    from log_lineup_pass import confirmed_xis

    canon = str.lower
    xis = confirmed_xis(_summary(11, 11), canon)
    assert xis["home"]["team"] == "arsenal" and len(xis["away"]["starters"]) == 11
    assert confirmed_xis(_summary(11, 0), canon) is None       # away XI not out yet
    assert confirmed_xis({"rosters": []}, canon) is None


def test_page_reads_the_logged_lineup_pass(tmp_path, monkeypatch):
    from mundialytics.serving import logged_prediction as lp

    f = tmp_path / "lineup_pass_log.csv"
    pd.DataFrame([{
        "logged_at_utc": "2026-10-10T13:05:00", "event_id": "1", "kickoff_utc": "2026-10-10T14:00:00",
        "competition": "Premier League", "home": "arsenal", "away": "chelsea",
        "absent_home": 3 / 11, "absent_away": 0.0, "missing_home": "Saka; Rice; Odegaard", "missing_away": "",
        "lambda_home": 1.3, "lambda_away": 1.2, "p_home": 0.41, "p_draw": 0.27, "p_away": 0.32,
        "p_over_25": 0.51, "model_source": "x_sv_xi", "train_cutoff": "2026-10-05", "model_fp": "abc",
    }]).to_csv(f, index=False)
    monkeypatch.setattr(lp, "LINEUP_LOG", f)
    found = lp.find_lineup_pass("Arsenal", "Chelsea", "2026-10-10")
    assert found is not None
    pred, details = found
    assert pred.trio == {"home": 0.41, "draw": 0.27, "away": 0.32} and pred.lambdas == (1.3, 1.2)
    assert details["missingHome"] == ["Saka", "Rice", "Odegaard"] and details["missingAway"] == []
    assert lp.find_lineup_pass("Arsenal", "Chelsea", "2026-11-10") is None   # another fixture


def test_watcher_sleeps_to_the_window_polls_and_exits(tmp_path, monkeypatch):
    """A simulated match day: nothing is fetched early, the pass runs only inside the
    70-minute window, and the watcher exits once the match is logged."""
    import log_lineup_pass as L

    clock = {"now": pd.Timestamp("2026-10-10T10:00:00Z")}
    kickoff = pd.Timestamp("2026-10-10T14:00:00Z")
    logged: set = set()
    passes: list = []

    monkeypatch.setattr(L, "LOCK", tmp_path / "watch.lock")
    monkeypatch.setattr(L, "fixtures_logged_ahead", lambda now, h: 1)   # the morning log has the match
    monkeypatch.setattr(L.pd.Timestamp, "now", classmethod(lambda cls, tz=None: clock["now"]))
    monkeypatch.setattr(L, "upcoming", lambda now, minutes: [{"event_id": "1", "code": "eng.1",
                                                               "competition": "Premier League", "kickoff": kickoff}])
    monkeypatch.setattr(L, "logged_ids", lambda: set(logged))
    monkeypatch.setattr(L.time, "sleep", lambda s: clock.update(now=clock["now"] + pd.Timedelta(seconds=s)))

    def fake_pass(lookahead):
        passes.append(clock["now"])
        if clock["now"] >= kickoff - pd.Timedelta(minutes=55):   # XIs out ~55 min before
            logged.add("1")
        return 0

    monkeypatch.setattr(L, "run_pass", fake_pass)
    assert L.watch() == 0
    assert passes[0] == kickoff - pd.Timedelta(minutes=L.WAKE_BEFORE_MIN)
    assert all(b - a == pd.Timedelta(minutes=L.POLL_MIN) for a, b in zip(passes, passes[1:]))
    assert "1" in logged and len(passes) == 4
    assert not (tmp_path / "watch.lock").exists()


def test_watcher_retries_the_schedule_while_espn_is_down(tmp_path, monkeypatch):
    """ESPN (or DNS) is down when the watcher starts: it retries the schedule every
    POLL_MIN minutes instead of exiting for the day, and still logs the match
    (2026-09-29: getaddrinfo failed for an hour)."""
    import log_lineup_pass as L

    clock = {"now": pd.Timestamp("2026-10-10T06:30:00Z")}
    kickoff = pd.Timestamp("2026-10-10T14:00:00Z")
    logged: set = set()
    fetches: list = []

    monkeypatch.setattr(L, "LOCK", tmp_path / "watch.lock")
    monkeypatch.setattr(L, "fixtures_logged_ahead", lambda now, h: 1)
    monkeypatch.setattr(L, "_kickoffs_logged_ahead", lambda now, h: [kickoff])
    monkeypatch.setattr(L.pd.Timestamp, "now", classmethod(lambda cls, tz=None: clock["now"]))
    monkeypatch.setattr(L, "logged_ids", lambda: set(logged))
    monkeypatch.setattr(L.time, "sleep", lambda s: clock.update(now=clock["now"] + pd.Timedelta(seconds=s)))

    def flaky_upcoming(now, minutes):
        fetches.append(now)
        if len(fetches) <= 3:
            raise OSError("[Errno 11001] getaddrinfo failed")
        return [{"event_id": "1", "code": "eng.1", "competition": "Premier League", "kickoff": kickoff}]

    def fake_pass(lookahead):
        if clock["now"] >= kickoff - pd.Timedelta(minutes=55):
            logged.add("1")
        return 0

    monkeypatch.setattr(L, "upcoming", flaky_upcoming)
    monkeypatch.setattr(L, "run_pass", fake_pass)
    assert L.watch() == 0
    assert len(fetches) == 4
    assert all(b - a == pd.Timedelta(minutes=L.POLL_MIN) for a, b in zip(fetches, fetches[1:]))
    assert "1" in logged
    assert not (tmp_path / "watch.lock").exists()


def test_watcher_gives_up_on_the_schedule_after_the_last_kickoff(tmp_path, monkeypatch):
    """If ESPN stays down past the last kick-off the local log knows about, the watcher
    stops retrying (nothing left to re-price) and releases the lock."""
    import log_lineup_pass as L

    clock = {"now": pd.Timestamp("2026-10-10T06:30:00Z")}
    kickoff = pd.Timestamp("2026-10-10T14:00:00Z")

    monkeypatch.setattr(L, "LOCK", tmp_path / "watch.lock")
    monkeypatch.setattr(L, "fixtures_logged_ahead", lambda now, h: 1)
    monkeypatch.setattr(L, "_kickoffs_logged_ahead", lambda now, h: [kickoff])
    monkeypatch.setattr(L.pd.Timestamp, "now", classmethod(lambda cls, tz=None: clock["now"]))
    monkeypatch.setattr(L.time, "sleep", lambda s: clock.update(now=clock["now"] + pd.Timedelta(seconds=s)))

    def down(now, minutes):
        raise OSError("[Errno 11001] getaddrinfo failed")

    monkeypatch.setattr(L, "upcoming", down)
    monkeypatch.setattr(L, "run_pass", lambda lookahead: pytest.fail("no pass without a schedule"))
    assert L.watch() == 1
    assert kickoff <= clock["now"] < kickoff + pd.Timedelta(minutes=L.POLL_MIN)
    assert not (tmp_path / "watch.lock").exists()


def test_last_logged_kickoff_bounds_the_retries(tmp_path, monkeypatch):
    """The retry deadline comes from the local log: timed rows by their kick-off, a row
    with only a date by the end of that Madrid day."""
    import log_lineup_pass as L

    pd.DataFrame([{"fecha": "2026-10-10", "kickoff_utc": "2026-10-10T14:00Z", "home": "sunderland", "mercado": "1X2"},
                  {"fecha": "2026-10-10", "kickoff_utc": "", "home": "brighton", "mercado": "1X2"},
                  {"fecha": "2026-10-10", "kickoff_utc": "2026-10-10T19:00Z", "home": "braga", "mercado": "1X2"}]
                 ).to_csv(tmp_path / "log.csv", index=False)
    pd.DataFrame({"team": ["sunderland", "brighton"]}).to_csv(tmp_path / "rosters.csv", index=False)
    monkeypatch.setattr(L, "PRED_LOG", tmp_path / "log.csv")
    monkeypatch.setattr(L, "ROSTERS", tmp_path / "rosters.csv")
    kicks = L._kickoffs_logged_ahead(pd.Timestamp("2026-10-10T06:00Z"), 20)
    assert sorted(kicks) == [pd.Timestamp("2026-10-10T14:00Z"), pd.Timestamp("2026-10-10T22:00Z")]
    assert L.fixtures_logged_ahead(pd.Timestamp("2026-10-10T06:00Z"), 20) == 2


def test_no_fixture_logged_means_no_network(tmp_path, monkeypatch):
    """On a day the morning log holds no Big Five fixture the watcher exits without
    asking ESPN anything."""
    import log_lineup_pass as L

    pd.DataFrame([{"fecha": "2026-10-22", "kickoff_utc": "2026-10-22T19:00Z", "home": "braga", "mercado": "1X2"},
                  {"fecha": "2026-10-10", "kickoff_utc": "2026-10-10T14:00Z", "home": "sunderland", "mercado": "1X2"}]
                 ).to_csv(tmp_path / "log.csv", index=False)
    pd.DataFrame({"team": ["sunderland", "brighton"]}).to_csv(tmp_path / "rosters.csv", index=False)
    monkeypatch.setattr(L, "PRED_LOG", tmp_path / "log.csv")
    monkeypatch.setattr(L, "ROSTERS", tmp_path / "rosters.csv")
    assert L.fixtures_logged_ahead(pd.Timestamp("2026-10-10T06:00Z"), 20) == 1
    assert L.fixtures_logged_ahead(pd.Timestamp("2026-10-22T06:00Z"), 20) == 0   # braga is not Big Five

    def boom(*a, **k):
        raise AssertionError("ESPN must not be called")

    monkeypatch.setattr(L, "upcoming", boom)
    monkeypatch.setattr(L.pd.Timestamp, "now", classmethod(lambda cls, tz=None: pd.Timestamp("2026-10-22T06:00Z")))
    assert L.watch() == 0
