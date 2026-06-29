from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read(path: str | None) -> pd.DataFrame:
    p = _resolve(path)
    if not p or not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _table(df: pd.DataFrame, cols: list[str] | None = None, max_rows: int = 50) -> str:
    if df.empty:
        return "<p class='muted'>No data available.</p>"
    shown = df.copy()
    if cols:
        shown = shown[[c for c in cols if c in shown.columns]]
    shown = shown.head(max_rows)
    return shown.to_html(index=False, escape=True, classes="data-table")


def main() -> None:
    p = argparse.ArgumentParser(description="Build a simple HTML daily MVP report from Mundialytics outputs.")
    p.add_argument("--fixtures", default=None)
    p.add_argument("--match-predictions", default=None)
    p.add_argument("--team-props", default=None)
    p.add_argument("--player-props", default=None)
    p.add_argument("--picks", default=None)
    p.add_argument("--out", default="outputs/daily_report.html")
    args = p.parse_args()

    fixtures = _read(args.fixtures)
    match_preds = _read(args.match_predictions)
    team_props = _read(args.team_props)
    player_props = _read(args.player_props)
    picks = _read(args.picks)

    warning_bits = []
    for name, df in [("fixtures", fixtures), ("match_predictions", match_preds), ("team_props", team_props), ("player_props", player_props), ("picks", picks)]:
        if not df.empty and "warnings" in df.columns:
            n = int(df["warnings"].notna().sum())
            if n:
                warning_bits.append(f"{name}: {n} warnings")
        if not df.empty and "confidence_flag" in df.columns:
            counts = df["confidence_flag"].value_counts(dropna=False).to_dict()
            warning_bits.append(f"{name} confidence: {counts}")

    css = """
    body { font-family: Arial, sans-serif; margin: 32px; color: #18202a; }
    h1, h2 { margin-bottom: 8px; }
    .muted { color: #667085; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 16px 0; }
    .warn { background: #fff7e6; border: 1px solid #ffd591; padding: 12px; border-radius: 8px; }
    table.data-table { border-collapse: collapse; width: 100%; font-size: 13px; }
    table.data-table th, table.data-table td { border: 1px solid #ddd; padding: 6px; text-align: left; }
    table.data-table th { background: #f6f8fa; }
    """
    parts = [f"<html><head><meta charset='utf-8'><title>Mundialytics Daily Report</title><style>{css}</style></head><body>"]
    parts.append("<h1>Mundialytics Daily MVP Report</h1>")
    parts.append("<p class='muted'>Paper mode only. Warnings and confidence flags are part of the product, not noise.</p>")
    if warning_bits:
        parts.append("<div class='warn'><b>Warnings / confidence summary</b><ul>" + "".join(f"<li>{html.escape(str(x))}</li>" for x in warning_bits) + "</ul></div>")
    sections = [
        ("Fixtures", fixtures, ["fixture_id", "provider", "kickoff_local", "date", "competition", "home_team", "away_team", "status_short", "status_long"]),
        ("Match Result Predictions", match_preds, ["fixture_id", "date", "home_team", "away_team", "lambda_home", "lambda_away", "p_home_win", "p_draw", "p_away_win", "p_over_25", "most_likely_score"]),
        ("Team / Match Props", team_props, ["match_id", "team", "opponent", "expected_shots_for", "expected_sot_for", "expected_corners_for", "expected_fouls_for", "expected_yellow_cards_for", "expected_match_total_shots_for", "expected_match_total_corners_for"]),
        ("Player Props", player_props, ["match_id", "team", "player", "market_type", "line", "safe_probability", "confidence_flag", "calibration_level", "warnings", "explanation"]),
        ("Paper Picks", picks, None),
    ]
    for title, df, cols in sections:
        parts.append(f"<div class='card'><h2>{html.escape(title)}</h2>{_table(df, cols=cols)}</div>")
    parts.append("</body></html>")
    out = _resolve(args.out)
    assert out is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    report = {"status": "DAILY_HTML_REPORT_BUILT", "out": str(out), "sections": {"fixtures": len(fixtures), "match_predictions": len(match_preds), "team_props": len(team_props), "player_props": len(player_props), "picks": len(picks)}, "warnings": warning_bits}
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
