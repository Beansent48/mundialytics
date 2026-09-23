"""The daily refresh must say when a step failed.

Until 2026-09-23 run_step's result was thrown away: ESPN answered 403 on three
consecutive days, the player markets went unsettled, and every one of those
runs exited 0 with last_run.json reporting ok.

Run:  .venv/Scripts/python.exe -m pytest tests/test_update_season_steps.py -q
"""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def us(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("update_season", ROOT / "scripts/update_season.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "STEPS_JSON", tmp_path / "last_steps.json")
    mod.STEPS.clear()
    return mod


def test_a_failing_required_step_is_recorded(us):
    ok = us.run_step("x/8 fails", [sys.executable, "-c", "raise SystemExit(3)"])
    assert ok is False
    assert us.failed_steps() == ["x/8 fails"]


def test_a_failing_optional_step_does_not_fail_the_run(us):
    us.run_step("y/8 optional", [sys.executable, "-c", "raise SystemExit(3)"], optional=True)
    us.run_step("z/8 fine", [sys.executable, "-c", "pass"])
    assert us.failed_steps() == []


def test_steps_file_lists_failures(us):
    us.run_step("x/8 fails", [sys.executable, "-c", "raise SystemExit(3)"])
    us.run_step("y/8 optional", [sys.executable, "-c", "raise SystemExit(3)"], optional=True)
    us.write_steps(us.EXIT_STEP_FAILED)
    data = json.loads(us.STEPS_JSON.read_text(encoding="utf-8"))
    assert data["exit_code"] == 2
    assert data["failed"] == ["x/8 fails"]
    assert data["failed_optional"] == ["y/8 optional"]


def test_atomic_csv_write_leaves_no_temp_file(tmp_path):
    sys.path.insert(0, str(ROOT / "src"))
    from mundialytics.utils import atomic_to_csv

    p = tmp_path / "f.csv"
    atomic_to_csv(pd.DataFrame({"a": [1, 2]}), p)
    atomic_to_csv(pd.DataFrame({"a": [3]}), p)
    assert pd.read_csv(p)["a"].tolist() == [3]
    assert [f.name for f in tmp_path.iterdir()] == ["f.csv"]
