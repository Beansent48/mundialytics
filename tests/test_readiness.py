from mundialytics.evaluation.readiness import evaluate_readiness, ReadinessThresholds


def test_readiness_fails_small_sample():
    result = evaluate_readiness(
        {"rows": 10, "warnings": [], "unknown_scope_rows": 0},
        {"n_predictions": 5, "log_loss": 0.8, "rps": 0.2},
        ReadinessThresholds(min_matches=100, min_backtest_predictions=50),
    )
    assert result["passed"] is False
    assert result["status"].startswith("NOT_READY")


def test_readiness_passes_clean_large_sample():
    result = evaluate_readiness(
        {"rows": 500, "warnings": [], "unknown_scope_rows": 0},
        {"n_predictions": 150, "log_loss": 0.9, "rps": 0.21},
        ReadinessThresholds(min_matches=100, min_backtest_predictions=50),
    )
    assert result["passed"] is True
