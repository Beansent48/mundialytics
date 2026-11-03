"""The daily refresh as a container process (scripts/refresh_loop.py).

On Windows the schedule is Task Scheduler's problem; in the stack it is this
script's, and a clock bug here is invisible until a matchday passes unrefreshed.
So the arithmetic is pinned down: when the next run is, which day is the heavy
one, and that a malformed setting falls back instead of crashing a container
that would then restart into the same crash.

The `last_run.json` shape is tested too, because `scripts/run_update.ps1` writes
the same file on the PC and anything reading it (a person, a future monitor)
must not have to know which of the two wrote it.

Run:  .venv/Scripts/python.exe -m pytest tests/test_refresh_loop.py -q
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import refresh_loop as rl  # noqa: E402


def test_the_next_run_is_today_while_the_hour_is_still_ahead():
    assert rl.next_run(datetime(2026, 11, 3, 9, 0), "10:00") == datetime(2026, 11, 3, 10, 0)


def test_and_tomorrow_once_it_has_passed():
    # Including the minute it fires: a loop that treats "now == target" as
    # still-ahead runs twice.
    assert rl.next_run(datetime(2026, 11, 3, 11, 0), "10:00") == datetime(2026, 11, 4, 10, 0)
    assert rl.next_run(datetime(2026, 11, 3, 10, 0), "10:00") == datetime(2026, 11, 4, 10, 0)


def test_a_malformed_time_falls_back_rather_than_killing_the_container():
    # `restart: unless-stopped` would otherwise turn a typo in .env into a crash
    # loop, which is the one failure mode a scheduled container must not have.
    assert rl.next_run(datetime(2026, 11, 3, 9, 0), "ten o'clock") == datetime(2026, 11, 3, 10, 0)


def test_only_one_weekday_gets_the_heavy_refresh():
    # 2026-11-02 is a Monday; the ~15 min walk-forward regeneration behind the
    # Resultados page is weekly on purpose.
    assert rl.is_full_day(datetime(2026, 11, 2), "Monday")
    assert not rl.is_full_day(datetime(2026, 11, 3), "Monday")
    assert not rl.is_full_day(datetime(2026, 11, 2), "never")
    assert rl.is_full_day(datetime(2026, 11, 3), "daily")
    # An unknown value means no heavy run, never a crash and never "every day".
    assert not rl.is_full_day(datetime(2026, 11, 2), "Lunes")


def test_last_run_json_has_the_same_shape_the_windows_wrapper_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(rl, "LAST_RUN", tmp_path / "last_run.json")
    monkeypatch.setattr(rl, "LAST_STEPS", tmp_path / "last_steps.json")
    started = datetime(2026, 11, 3, 10, 0)
    rl.write_last_run(started, full=True, code=2, log=tmp_path / "update.log", tail="SUMMARY")

    state = json.loads((tmp_path / "last_run.json").read_text(encoding="utf-8"))
    assert set(state) == {
        "finished_at", "started_at", "full", "exit_code", "ok",
        "failed_steps", "failed_optional", "log", "tail",
    }
    assert state["exit_code"] == 2 and state["ok"] is False
    assert state["full"] is True and state["tail"] == "SUMMARY"


def test_a_previous_runs_step_file_is_not_reported_as_this_runs(tmp_path, monkeypatch):
    # update_season.py writes last_steps.json itself. If it died before writing
    # one, the file on disk belongs to an earlier run and its failures are not
    # this run's -- reporting them would send somebody after a fixed problem.
    monkeypatch.setattr(rl, "LAST_STEPS", tmp_path / "last_steps.json")
    (tmp_path / "last_steps.json").write_text(json.dumps({
        "finished_at": "2026-11-01T10:12:00", "failed": ["step 4"], "failed_optional": [],
    }), encoding="utf-8")

    assert rl.read_failed_steps("2026-11-03T10:00:00") == ([], [])
    assert rl.read_failed_steps("2026-11-01T10:00:00") == (["step 4"], [])
