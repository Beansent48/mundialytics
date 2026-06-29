from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    p = argparse.ArgumentParser(description="Validate team_match_stats.csv for team/match props.")
    p.add_argument("--team-match-stats", required=True)
    p.add_argument("--out", default="outputs/team_props_validation_report.json")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()
    df = pd.read_csv(_resolve(args.team_match_stats))
    required = {"match_id", "date", "team", "opponent"}
    errors = []
    warnings = []
    missing = sorted(required - set(df.columns))
    if missing:
        errors.append(f"missing_required_columns={missing}")
    if "date" in df.columns:
        date_null = float(pd.to_datetime(df["date"], errors="coerce").isna().mean()) if len(df) else 0.0
        if date_null > 0.01:
            errors.append(f"date_null_rate_gt_0.01={date_null}")
    else:
        date_null = 1.0
    numeric_cols = [c for c in ["shots_for", "sot_for", "corners_for", "fouls_for", "yellow_cards_for", "red_cards_for", "goals_for"] if c in df.columns]
    numeric_checks = {}
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        numeric_checks[col] = {"missing": int(s.isna().sum()), "negative": int((s < 0).sum()), "min": float(s.min()) if s.notna().any() else None, "max": float(s.max()) if s.notna().any() else None}
        if (s < 0).any():
            errors.append(f"negative_values_in_{col}")
    if "corners_for" not in df.columns or pd.to_numeric(df.get("corners_for", pd.Series(dtype=float)), errors="coerce").notna().sum() == 0:
        warnings.append("corners_unavailable_do_not_offer_corner_markets")
    duplicates = int(df.duplicated(["match_id", "team"]).sum()) if {"match_id", "team"}.issubset(df.columns) else None
    if duplicates:
        errors.append(f"duplicate_match_team_rows={duplicates}")
    report = {
        "status": "TEAM_PROPS_VALIDATION_FAILED" if errors else "TEAM_PROPS_VALIDATION_PASSED",
        "errors": errors,
        "warnings": warnings,
        "rows": int(len(df)),
        "matches": int(df["match_id"].nunique()) if "match_id" in df.columns else 0,
        "teams": int(df["team"].nunique()) if "team" in df.columns else 0,
        "date_null_rate": date_null,
        "numeric_checks": numeric_checks,
        "available_markets": numeric_cols,
    }
    out = _resolve(args.out)
    assert out is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
