import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mundialytics.evaluation.prop_calibration import run_market_calibration_search, incoherence_checks


def test_prop_calibration_search_improves_biased_market():
    rng = np.random.default_rng(7)
    n = 400
    # Biased-underconfident probabilities: true rate is systematically higher than raw p.
    raw = rng.uniform(0.05, 0.65, n)
    true_p = np.clip(raw + 0.15, 0.01, 0.95)
    y = rng.binomial(1, true_p)
    df = pd.DataFrame({
        "match_id": [f"m{i//8}" for i in range(n)],
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "market_type": "player_shots",
        "player": [f"p{i%20}" for i in range(n)],
        "probability": raw,
        "actual": y,
        "expected_minutes": 70,
        "sample_size": 500,
        "expected_count": 0.5,
    })
    results, calibrated, report = run_market_calibration_search(df, calibration_fraction=0.5, min_market_rows=50)
    assert not results.empty
    assert not calibrated.empty
    assert "player_shots" in report["markets"]
    identity = results[(results.market_type == "player_shots") & (results.method == "identity")].iloc[0]
    best = results[results.market_type == "player_shots"].sort_values(["log_loss", "brier"]).iloc[0]
    assert best["log_loss"] <= identity["log_loss"]


def test_incoherence_checks_flags_bad_probabilities():
    df = pd.DataFrame({
        "match_id": ["m1", "m1"],
        "player": ["a", "a"],
        "market_type": ["player_shots", "player_shots"],
        "probability": [1.2, -0.1],
        "actual": [1, 0],
        "expected_minutes": [90, -5],
    })
    rep = incoherence_checks(df)
    assert rep["checks"]["invalid_probability_rows"] == 2
    assert any("expected_minutes_negative" in w for w in rep["warnings"])


def test_calibrate_player_props_cli(tmp_path):
    src = Path("outputs/v09_sample/props/player_props_backtest_predictions.csv")
    assert src.exists()
    out_dir = tmp_path / "calibration"
    subprocess.run(
        [sys.executable, "scripts/calibrate_player_props.py", "--predictions", str(src), "--out-dir", str(out_dir), "--min-market-rows", "10"],
        check=True,
    )
    assert (out_dir / "calibration_search_results.csv").exists()
    assert (out_dir / "calibration_report.json").exists()
    report = json.loads((out_dir / "calibration_report.json").read_text(encoding="utf-8"))
    assert "markets" in report
