from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReadinessThresholds:
    min_matches: int = 200
    min_backtest_predictions: int = 100
    max_log_loss: float = 1.10
    max_rps: float = 0.24
    max_unknown_scope_rows: int = 0
    require_no_data_warnings: bool = True


def evaluate_readiness(data_report: dict[str, Any] | None = None, backtest_summary: dict[str, Any] | None = None, thresholds: ReadinessThresholds | None = None) -> dict[str, Any]:
    """Return a cautious production-readiness assessment.

    This does not claim profitability. It only checks whether the dataset and
    validation sample are large/clean enough to start trusting probability
    diagnostics. Anything failing should stay in paper mode.
    """
    t = thresholds or ReadinessThresholds()
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    if data_report is not None:
        rows = int(data_report.get("rows", 0) or 0)
        warnings = data_report.get("warnings", []) or []
        unknown_scope_rows = int(data_report.get("unknown_scope_rows", 0) or 0) if "unknown_scope_rows" in data_report else 0
        add("dataset_size", rows >= t.min_matches, f"rows={rows}, required>={t.min_matches}")
        add("known_scope", unknown_scope_rows <= t.max_unknown_scope_rows, f"unknown_scope_rows={unknown_scope_rows}")
        add("data_warnings", (not warnings) if t.require_no_data_warnings else True, f"warnings={warnings}")

    if backtest_summary is not None:
        n = int(backtest_summary.get("n_predictions", 0) or 0)
        log_loss = float(backtest_summary.get("log_loss", 999) or 999)
        rps = float(backtest_summary.get("rps", 999) or 999)
        add("backtest_size", n >= t.min_backtest_predictions, f"n_predictions={n}, required>={t.min_backtest_predictions}")
        add("log_loss", log_loss <= t.max_log_loss, f"log_loss={log_loss:.4f}, max={t.max_log_loss}")
        add("rps", rps <= t.max_rps, f"rps={rps:.4f}, max={t.max_rps}")

    passed = bool(checks) and all(c["passed"] for c in checks)
    if passed:
        status = "READY_FOR_EXTENDED_PAPER_MODE"
        recommendation = "Probabilities are clean enough to run longer paper tracking; still not a guarantee of betting profitability."
    else:
        status = "NOT_READY_KEEP_IN_DEVELOPMENT_OR_PAPER_ONLY"
        recommendation = "Fix failed checks before trusting value flags. Do not use real-money staking."
    return {"status": status, "passed": passed, "checks": checks, "recommendation": recommendation}
