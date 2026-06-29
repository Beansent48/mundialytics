from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.competition_taxonomy import enrich_competition_metadata, competition_domain_report


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


def _norm(s: object) -> str:
    return str(s).strip().casefold()


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Filter processed player-event data before props validation/calibration. "
            "Use this to drop match rows with missing dates, placeholder competitions, "
            "or competitions outside the target domain."
        )
    )
    p.add_argument("--player-events", required=True, help="statsbomb_player_events.csv or another processed player-events CSV")
    p.add_argument("--lineups", default=None, help="Optional statsbomb_lineups.csv to filter consistently by match_id")
    p.add_argument("--out-player-events", required=True)
    p.add_argument("--out-lineups", default=None)
    p.add_argument("--report", required=True)
    p.add_argument("--require-valid-date", action="store_true", help="Drop rows where date is missing/unparseable")
    p.add_argument("--exclude-competitions", nargs="*", default=["StatsBomb Open Data"], help="Competition names to exclude. Accepts comma-separated values too.")
    p.add_argument("--include-competitions", nargs="*", default=None, help="If supplied, keep only these competitions. Accepts comma-separated values too.")
    p.add_argument("--min-minutes", type=float, default=0.0, help="Drop player rows below this minutes value")
    p.add_argument("--set-team-scope", default=None, choices=["club", "national", "mixed"], help="Deprecated/diagnostic override for team_scope in output. Prefer taxonomy-derived labels.")
    p.add_argument("--preserve-existing-domain-labels", action="store_true", help="Do not overwrite team_scope/team_type/competition_context/gender from competition taxonomy.")
    args = p.parse_args()

    pe_path = _resolve(args.player_events)
    out_pe_path = _resolve(args.out_player_events)
    out_lu_path = _resolve(args.out_lineups)
    report_path = _resolve(args.report)
    assert pe_path is not None and out_pe_path is not None and report_path is not None

    pe = pd.read_csv(pe_path)
    before = pe.copy()

    if "match_id" in pe.columns:
        pe["match_id"] = pe["match_id"].astype(str)
    # Always add objective competition labels before filtering/auditing.
    pe = enrich_competition_metadata(pe, overwrite=not args.preserve_existing_domain_labels)
    if "date" in pe.columns:
        pe["date"] = pd.to_datetime(pe["date"], errors="coerce")
    else:
        pe["date"] = pd.NaT

    include = set(_norm(x) for x in _parse_list(args.include_competitions))
    exclude = set(_norm(x) for x in _parse_list(args.exclude_competitions))

    filters_applied: list[str] = []

    if args.require_valid_date:
        n0 = len(pe)
        pe = pe[pe["date"].notna()].copy()
        filters_applied.append(f"require_valid_date: {n0}->{len(pe)}")

    if "competition" in pe.columns and exclude:
        n0 = len(pe)
        pe = pe[~pe["competition"].map(_norm).isin(exclude)].copy()
        filters_applied.append(f"exclude_competitions={sorted(exclude)}: {n0}->{len(pe)}")

    if "competition" in pe.columns and include:
        n0 = len(pe)
        pe = pe[pe["competition"].map(_norm).isin(include)].copy()
        filters_applied.append(f"include_competitions={sorted(include)}: {n0}->{len(pe)}")

    if "minutes" in pe.columns and args.min_minutes > 0:
        n0 = len(pe)
        mins = pd.to_numeric(pe["minutes"], errors="coerce").fillna(0)
        pe = pe[mins >= args.min_minutes].copy()
        filters_applied.append(f"min_minutes>={args.min_minutes}: {n0}->{len(pe)}")

    if args.set_team_scope:
        pe["team_scope"] = args.set_team_scope
        filters_applied.append(f"set_team_scope={args.set_team_scope}")

    # Keep dates as yyyy-mm-dd for stable CSVs.
    if "date" in pe.columns:
        pe = pe.sort_values(["date", "match_id", "team", "player"], na_position="last")
        pe["date"] = pd.to_datetime(pe["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    out_pe_path.parent.mkdir(parents=True, exist_ok=True)
    pe.to_csv(out_pe_path, index=False)

    lineup_report = None
    if args.lineups and args.out_lineups:
        lu_path = _resolve(args.lineups)
        assert lu_path is not None and out_lu_path is not None
        lu = pd.read_csv(lu_path)
        lu = enrich_competition_metadata(lu, overwrite=not args.preserve_existing_domain_labels)
        if "match_id" in lu.columns:
            lu["match_id"] = lu["match_id"].astype(str)
            keep_ids = set(pe["match_id"].astype(str).unique()) if "match_id" in pe.columns else set()
            n0 = len(lu)
            lu = lu[lu["match_id"].isin(keep_ids)].copy()
            lineup_report = {"rows_before": int(n0), "rows_after": int(len(lu)), "matches_after": int(lu["match_id"].nunique())}
        if args.set_team_scope:
            lu["team_scope"] = args.set_team_scope
        if "date" in lu.columns:
            lu["date"] = pd.to_datetime(lu["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        out_lu_path.parent.mkdir(parents=True, exist_ok=True)
        lu.to_csv(out_lu_path, index=False)

    def top_counts(df: pd.DataFrame, col: str, limit: int = 25) -> dict:
        if col not in df.columns:
            return {}
        return {str(k): int(v) for k, v in df[col].fillna("<NA>").value_counts().head(limit).items()}

    report = {
        "status": "PLAYER_EVENTS_FILTERED_FOR_PROPS",
        "input": str(pe_path),
        "out_player_events": str(out_pe_path),
        "out_lineups": str(out_lu_path) if out_lu_path else None,
        "filters_applied": filters_applied,
        "rows_before": int(len(before)),
        "rows_after": int(len(pe)),
        "matches_before": int(before["match_id"].astype(str).nunique()) if "match_id" in before.columns else None,
        "matches_after": int(pe["match_id"].astype(str).nunique()) if "match_id" in pe.columns else None,
        "date_null_rate_before": float(pd.to_datetime(before.get("date", pd.Series([pd.NaT] * len(before))), errors="coerce").isna().mean()) if len(before) else 0.0,
        "date_null_rate_after": float(pd.to_datetime(pe.get("date", pd.Series([pd.NaT] * len(pe))), errors="coerce").isna().mean()) if len(pe) else 0.0,
        "competitions_before_top": top_counts(before, "competition"),
        "competitions_after_top": top_counts(pe, "competition"),
        "team_scope_after": top_counts(pe, "team_scope"),
        "domain_after": competition_domain_report(pe),
        "lineups": lineup_report,
        "warnings": [],
    }
    if len(pe) == 0:
        report["warnings"].append("filtered_output_is_empty")
    if report["date_null_rate_after"] and report["date_null_rate_after"] > 0.05:
        report["warnings"].append(f"date_null_rate_after={report['date_null_rate_after']:.3f}")
    if "StatsBomb Open Data" in report["competitions_after_top"]:
        report["warnings"].append("placeholder_competition_still_present")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
