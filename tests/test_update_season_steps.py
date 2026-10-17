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


def _xg_rows(dates, comp, xg, teams=("a", "b")):
    cols = {f"{s}_{v}": (xg if v == "xg" else xg * 0.9)
            for s in ("home", "away") for v in ("xg", "npxg", "xg_op", "xg_sp")}
    return pd.DataFrame({"date": dates, "competition": comp,
                         "home_team_fd": teams[0], "away_team_fd": teams[1], **cols})


def test_a_current_season_provider_is_lifted_onto_the_history_level(tmp_path, monkeypatch):
    """Audit finding 1: history is Understat, which runs hotter than Sofascore's
    Opta numbers, and `lambda_scale_` corrects for Understat. Dropping
    Sofascore-level rows in beside them corrected that inflation twice."""
    us = importlib.util.spec_from_file_location("us_h", ROOT / "scripts/update_season.py")
    mod = importlib.util.module_from_spec(us)
    sys.modules["us_h"] = mod
    us.loader.exec_module(mod)

    dates = [f"2026-08-{d:02d}" for d in range(1, 29)] * 4      # 112 shared matches
    text = _xg_rows(dates, "LaLiga", 1.60)      # already on the Understat level
    sofa = _xg_rows(dates, "LaLiga", 1.45)      # ~9% colder
    tp, sp = tmp_path / "text.csv", tmp_path / "sofa.csv"
    text.to_csv(tp, index=False)
    sofa.to_csv(sp, index=False)
    monkeypatch.setattr(mod, "TEXT_TEAM_MATCH", tp)
    monkeypatch.setattr(mod, "SOFA_TEAM_MATCH", sp)

    lifted = mod._on_history_level(sofa.copy(), sp)
    assert lifted["home_xg"].mean() == pytest.approx(1.60, rel=1e-6)   # now on the text level
    # the text xG passes through: it is already the reference
    assert mod._on_history_level(text.copy(), tp)["home_xg"].mean() == pytest.approx(1.60)


def test_too_few_shared_matches_leaves_the_provider_alone(tmp_path, monkeypatch):
    us = importlib.util.spec_from_file_location("us_h2", ROOT / "scripts/update_season.py")
    mod = importlib.util.module_from_spec(us)
    sys.modules["us_h2"] = mod
    us.loader.exec_module(mod)
    text = _xg_rows(["2026-08-01"], "LaLiga", 1.60)
    sofa = _xg_rows(["2026-08-01"], "LaLiga", 1.45)
    tp, sp = tmp_path / "t.csv", tmp_path / "s.csv"
    text.to_csv(tp, index=False)
    sofa.to_csv(sp, index=False)
    monkeypatch.setattr(mod, "TEXT_TEAM_MATCH", tp)
    monkeypatch.setattr(mod, "SOFA_TEAM_MATCH", sp)
    # 1 shared match cannot measure an offset -> untouched, not scaled by noise
    assert mod._on_history_level(sofa.copy(), sp)["home_xg"].mean() == pytest.approx(1.45)
