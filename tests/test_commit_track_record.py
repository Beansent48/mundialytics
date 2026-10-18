"""Committing the track record (scripts/commit_track_record.py).

It exists because the prediction log only ever reached git when somebody
remembered: by 2026-10-18 the committed copy was 19 days behind the working
one. The log is 14 MB, so it rides on a round's cadence rather than on days.

The cadence is a clock and not a round read out of the file, and these tests
pin that down because the obvious alternative is wrong: the log pre-registers
whole rounds in bulk, so every (season, jornada) pair is in git already, and
the rows that arrive in between are the lineup pass re-pricing matches logged
weeks ago. A round key reported "nothing new" on a file 19 days out of date.

Run:  .venv/Scripts/python.exe -m pytest tests/test_commit_track_record.py -q
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import commit_track_record as ctr  # noqa: E402


def test_the_log_goes_in_once_the_cadence_has_passed():
    assert ctr.log_is_due(19.1, every_days=5.0)
    assert ctr.log_is_due(5.0, every_days=5.0)


def test_a_days_worth_of_re_pricing_does_not_pay_for_a_14mb_commit():
    # The lineup pass appends thousands of rows on a matchday; that is not a
    # reason to copy the whole file into git again.
    assert not ctr.log_is_due(0.3, every_days=5.0)
    assert not ctr.log_is_due(4.9, every_days=5.0)


def test_saturday_and_sunday_collapse_into_one_commit():
    assert not ctr.log_is_due(1.0, every_days=5.0)


def test_a_log_git_has_never_seen_is_always_due():
    # The whole point is that the track record reaches the remote at all, so
    # "no commit to compare against" can never mean "wait".
    assert ctr.log_is_due(None, every_days=5.0)


def test_the_default_cadence_is_a_round_not_a_day():
    assert ctr.LOG_EVERY_DAYS >= 2.0
    assert ctr.log_is_due(ctr.LOG_EVERY_DAYS)
    assert not ctr.log_is_due(0.0)


def test_an_uncommitted_path_reports_no_age_rather_than_zero():
    if subprocess.run(["git", "rev-parse", "--git-dir"], cwd=ROOT,
                      capture_output=True).returncode != 0:
        pytest.skip("not a git checkout")
    assert ctr.days_since_last_commit("data/processed/logs/no_such_file.csv") is None


def test_the_age_is_measured_in_days():
    now = 1_000_000.0
    stamp = now - 3 * 86400
    done = subprocess.CompletedProcess([], 0, stdout=str(int(stamp)).encode(), stderr=b"")
    real, ctr.git = ctr.git, lambda *a, **k: done
    try:
        assert ctr.days_since_last_commit("whatever", now=now) == pytest.approx(3.0)
    finally:
        ctr.git = real
