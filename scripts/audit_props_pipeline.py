from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.competition_taxonomy import enrich_competition_metadata, find_competition_label_mismatches, competition_domain_report

DEFAULT_PLACEHOLDERS = {"statsbomb open data", "unknown", "<na>", "nan", "none", ""}
REQUIRED_EVENT_COLS = {
    "match_id", "date", "competition", "team", "opponent", "player", "minutes",
    "shots", "shots_on_target", "fouls_committed", "fouls_drawn", "yellow_cards",
}
REQUIRED_PRED_COLS = {
    "match_id", "date", "competition", "team_scope", "team_type", "competition_context", "gender", "team", "opponent",
    "player", "player_id_global", "player_context_id", "position", "started",
    "market_type", "line", "probability", "raw_probability", "actual",
    "expected_minutes", "expected_minutes_source", "actual_minutes", "sample_size",
    "club_minutes_sample", "national_minutes_sample", "cross_context_feature_used",
}
NATIONAL_COMPETITIONS = {
    "fifa world cup", "uefa euro", "african cup of nations", "copa america", "copa américa",
    "gold cup", "afc asian cup", "women's world cup", "uefa women's euro", "fifa u20 world cup",
}
CLUB_COMPETITION_HINTS = {
    "la liga", "premier league", "serie a", "ligue 1", "1. bundesliga", "bundesliga",
    "champions league", "uefa europa league", "copa del rey", "major league soccer",
    "nwsl", "liga f", "fa women's super league", "frauen bundesliga", "serie a women",
    "indian super league", "liga profesional", "north american league",
}


def _resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _parse_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _norm(x: Any) -> str:
    return str(x).strip().casefold()


def _top_counts(df: pd.DataFrame, col: str, limit: int = 30) -> dict[str, int]:
    if col not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df[col].fillna("<NA>").value_counts().head(limit).items()}


def _date_null_rate(df: pd.DataFrame) -> float | None:
    if "date" not in df.columns or len(df) == 0:
        return None
    return float(pd.to_datetime(df["date"], errors="coerce").isna().mean())


def _date_range(df: pd.DataFrame) -> dict[str, str | None]:
    if "date" not in df.columns or len(df) == 0:
        return {"min": None, "max": None}
    d = pd.to_datetime(df["date"], errors="coerce")
    if d.notna().sum() == 0:
        return {"min": None, "max": None}
    return {"min": str(d.min().date()), "max": str(d.max().date())}


def _numeric_checks(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in cols:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        out[col] = {
            "missing": int(x.isna().sum()),
            "negative": int((x < 0).sum()),
            "min": None if x.dropna().empty else float(x.min()),
            "max": None if x.dropna().empty else float(x.max()),
        }
    return out


def audit_table(
    df: pd.DataFrame,
    *,
    kind: str,
    require_valid_date: bool,
    max_date_null_rate: float,
    forbid_placeholders: set[str],
    include_competitions: set[str],
    exclude_competitions: set[str],
    expected_domain: str | None,
) -> dict[str, Any]:
    df = enrich_competition_metadata(df, overwrite=False)
    errors: list[str] = []
    warnings: list[str] = []
    required = REQUIRED_EVENT_COLS if kind == "player_events" else REQUIRED_PRED_COLS
    missing = sorted(required - set(df.columns))
    if missing:
        errors.append(f"{kind}_missing_required_columns={missing}")

    date_null_rate = _date_null_rate(df)
    if require_valid_date:
        if date_null_rate is None:
            errors.append(f"{kind}_missing_date_column")
        elif date_null_rate > max_date_null_rate:
            errors.append(f"{kind}_date_null_rate={date_null_rate:.3f} > {max_date_null_rate:.3f}")
    elif date_null_rate is not None and date_null_rate > 0.05:
        warnings.append(f"{kind}_date_null_rate={date_null_rate:.3f}")

    if "competition" in df.columns:
        comps_norm = set(df["competition"].dropna().map(_norm).unique())
        placeholders_found = sorted(comps_norm & forbid_placeholders)
        if placeholders_found:
            errors.append(f"{kind}_placeholder_competitions_present={placeholders_found}")
        excluded_found = sorted(comps_norm & exclude_competitions)
        if excluded_found:
            errors.append(f"{kind}_excluded_competitions_present={excluded_found}")
        if include_competitions:
            outside = sorted(c for c in comps_norm if c not in include_competitions)
            if outside:
                errors.append(f"{kind}_competitions_outside_include_set={outside[:20]}")
        # Objective domain checks from the competition taxonomy.
        mismatch_errors = find_competition_label_mismatches(df)
        errors.extend([f"{kind}_{e}" for e in mismatch_errors])
        derived = enrich_competition_metadata(df, overwrite=True)
        if expected_domain == "national":
            bad = sorted(derived.loc[derived["team_scope"] != "national", "competition"].dropna().astype(str).unique().tolist())
            if bad:
                errors.append(f"{kind}_non_national_competitions_in_national_domain={bad[:20]}")
        elif expected_domain == "club":
            bad = sorted(derived.loc[derived["team_scope"] != "club", "competition"].dropna().astype(str).unique().tolist())
            if bad:
                errors.append(f"{kind}_non_club_competitions_in_club_domain={bad[:20]}")

    if kind == "predictions" and {"match_id", "player", "market_type"}.issubset(df.columns):
        # Team and line matter: the same player name can appear in different teams or markets/lines.
        dup_key = [c for c in ["match_id", "team", "player", "market_type", "line"] if c in df.columns]
        dup = int(df.duplicated(dup_key).sum())
        if dup:
            errors.append(f"predictions_duplicate_rows_by_{'_'.join(dup_key)}={dup}")
    if kind == "predictions" and "probability" in df.columns:
        p = pd.to_numeric(df["probability"], errors="coerce")
        invalid = int((p.isna() | (p < 0) | (p > 1)).sum())
        if invalid:
            errors.append(f"predictions_invalid_probability_rows={invalid}")
        extreme_hi = int((p > 0.98).sum())
        extreme_lo = int((p < 0.02).sum())
        if extreme_hi:
            warnings.append(f"predictions_extreme_high_probability_rows_gt_0.98={extreme_hi}")
        if extreme_lo:
            warnings.append(f"predictions_extreme_low_probability_rows_lt_0.02={extreme_lo}")

    if kind == "predictions" and "expected_minutes_source" in df.columns:
        leaky = int(df["expected_minutes_source"].astype(str).str.contains("LEAKY|observed_test_minutes", case=False, regex=True).sum())
        if leaky:
            errors.append(f"predictions_use_observed_test_minutes_leakage_rows={leaky}")

    if kind == "predictions" and {"team_type", "club_minutes_sample", "cross_context_feature_used"}.issubset(df.columns):
        national_with_club = (df["team_type"].astype(str) == "national_team") & (pd.to_numeric(df["club_minutes_sample"], errors="coerce").fillna(0) > 0)
        flagged = df["cross_context_feature_used"].astype(str).str.lower().isin({"true", "1", "yes"})
        missing_flags = int((national_with_club & ~flagged).sum())
        if missing_flags:
            errors.append(f"predictions_cross_context_feature_flag_missing_rows={missing_flags}")

    numeric_cols = [
        "minutes", "shots", "shots_on_target", "fouls_committed", "fouls_drawn", "yellow_cards",
        "expected_minutes", "sample_size", "club_minutes_sample", "national_minutes_sample",
        "expected_count", "actual_count", "actual", "probability",
    ]
    numeric = _numeric_checks(df, numeric_cols)
    for col, chk in numeric.items():
        if chk["negative"]:
            errors.append(f"{kind}_{col}_negative_rows={chk['negative']}")
    if "expected_minutes" in numeric and numeric["expected_minutes"]["max"] is not None and numeric["expected_minutes"]["max"] > 130:
        errors.append(f"{kind}_expected_minutes_over_130")
    if "minutes" in numeric and numeric["minutes"]["max"] is not None and numeric["minutes"]["max"] > 130:
        errors.append(f"{kind}_minutes_over_130")

    return {
        "kind": kind,
        "rows": int(len(df)),
        "matches": int(df["match_id"].astype(str).nunique()) if "match_id" in df.columns else None,
        "players": int(df["player"].astype(str).nunique()) if "player" in df.columns else None,
        "date_null_rate": date_null_rate,
        "date_range": _date_range(df),
        "competitions_top": _top_counts(df, "competition"),
        "team_scope_counts": _top_counts(df, "team_scope"),
        "domain_counts": competition_domain_report(df),
        "market_counts": _top_counts(df, "market_type"),
        "numeric_checks": numeric,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Preflight/audit checks for Mundialytics player-prop datasets and predictions.")
    p.add_argument("--player-events", default=None)
    p.add_argument("--lineups", default=None)
    p.add_argument("--predictions", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--require-valid-date", action="store_true")
    p.add_argument("--max-date-null-rate", type=float, default=0.01)
    p.add_argument("--forbid-placeholder-competitions", nargs="*", default=["StatsBomb Open Data"])
    p.add_argument("--exclude-competitions", nargs="*", default=None)
    p.add_argument("--include-competitions", nargs="*", default=None)
    p.add_argument("--expected-domain", choices=["club", "national", "mixed"], default=None)
    p.add_argument("--strict", action="store_true", help="Exit non-zero if any hard error is found.")
    args = p.parse_args()

    forbid = set(_norm(x) for x in _parse_list(args.forbid_placeholder_competitions)) | DEFAULT_PLACEHOLDERS
    excluded = set(_norm(x) for x in _parse_list(args.exclude_competitions))
    included = set(_norm(x) for x in _parse_list(args.include_competitions))
    expected_domain = None if args.expected_domain == "mixed" else args.expected_domain

    audits: dict[str, Any] = {}
    all_errors: list[str] = []
    all_warnings: list[str] = []

    for kind, path_arg in [("player_events", args.player_events), ("predictions", args.predictions)]:
        path = _resolve(path_arg)
        if path is None:
            continue
        if not path.exists():
            all_errors.append(f"{kind}_file_not_found={path}")
            continue
        df = pd.read_csv(path)
        audit = audit_table(
            df,
            kind=kind,
            require_valid_date=args.require_valid_date,
            max_date_null_rate=args.max_date_null_rate,
            forbid_placeholders=forbid,
            include_competitions=included,
            exclude_competitions=excluded,
            expected_domain=expected_domain,
        )
        audits[kind] = audit
        all_errors.extend(audit["errors"])
        all_warnings.extend(audit["warnings"])

    if args.lineups:
        lu_path = _resolve(args.lineups)
        if lu_path is None or not lu_path.exists():
            all_errors.append(f"lineups_file_not_found={lu_path}")
        else:
            lu = pd.read_csv(lu_path)
            lu = enrich_competition_metadata(lu, overwrite=False)
            lu_errors = find_competition_label_mismatches(lu)
            lu_audit = {
                "rows": int(len(lu)),
                "matches": int(lu["match_id"].astype(str).nunique()) if "match_id" in lu.columns else None,
                "players": int(lu["player"].astype(str).nunique()) if "player" in lu.columns else None,
                "date_null_rate": _date_null_rate(lu),
                "competitions_top": _top_counts(lu, "competition"),
                "team_scope_counts": _top_counts(lu, "team_scope"),
                "domain_counts": competition_domain_report(lu),
                "errors": [f"lineups_{e}" for e in lu_errors],
                "warnings": [],
            }
            if args.require_valid_date and lu_audit["date_null_rate"] is not None and lu_audit["date_null_rate"] > args.max_date_null_rate:
                lu_audit["errors"].append(f"lineups_date_null_rate={lu_audit['date_null_rate']:.3f}")
            if "player_events" in audits and "match_id" in lu.columns:
                # A lineup file may contain fewer rows, but it should not contain matches outside the cleaned player-events set.
                pe_path = _resolve(args.player_events)
                if pe_path and pe_path.exists():
                    pe = pd.read_csv(pe_path, usecols=lambda c: c == "match_id")
                    extra = set(lu["match_id"].astype(str)) - set(pe["match_id"].astype(str))
                    if extra:
                        lu_audit["errors"].append(f"lineups_match_ids_not_in_player_events={len(extra)}")
            audits["lineups"] = lu_audit
            all_errors.extend(lu_audit["errors"])
            all_warnings.extend(lu_audit["warnings"])

    status = "AUDIT_PASSED" if not all_errors else "AUDIT_FAILED"
    payload = {
        "status": status,
        "errors": all_errors,
        "warnings": all_warnings,
        "audits": audits,
    }
    out = _resolve(args.out)
    assert out is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.strict and all_errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
