#!/usr/bin/env python3
"""Print a compact console report for Mundialytics matchday outputs.

This reads the CSV files produced by scripts/run_statistical_matchday.py and renders
match predictions, scorelines, dynamic lines, team events, and player-prop candidates
without opening the HTML report.

The console viewer is intentionally pick-oriented: it shows both over and under
sides and ranks rows by statistical signal first, then by price value when odds
are attached. Without odds, these are model signals/fair-odds candidates, not
real betting picks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _fmt_pct(value, decimals: int = 1) -> str:
    if pd.isna(value):
        return "—"
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except Exception:
        return "—"


def _fmt_num(value, decimals: int = 2) -> str:
    if pd.isna(value):
        return "—"
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "—"


def _safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _norm(text: str) -> str:
    return " ".join(str(text).lower().replace("_", " ").replace("-", " ").split())


def _contains_any(row: pd.Series, terms: List[str]) -> bool:
    if not terms:
        return True
    haystack = _norm(" ".join([_safe_str(row.get("home_team", "")), _safe_str(row.get("away_team", "")), _safe_str(row.get("match_id", ""))]))
    return any(_norm(t) in haystack for t in terms)


def _select_matches(matches: pd.DataFrame, match_ids: List[str], teams: List[str]) -> pd.DataFrame:
    if matches.empty:
        return matches
    work = matches.copy()
    if match_ids:
        wanted = {str(m) for m in match_ids}
        work = work[work["match_id"].astype(str).isin(wanted)]
    if teams:
        work = work[work.apply(lambda r: _contains_any(r, teams), axis=1)]
    return work


def _title(text: str) -> None:
    print("\n" + "=" * 92)
    print(text)
    print("=" * 92)


def _section(text: str) -> None:
    print("\n" + text)
    print("-" * len(text))


def _print_table(df: pd.DataFrame, columns: List[str], max_rows: int = 20) -> None:
    if df.empty:
        print("(sin datos)")
        return
    cols = [c for c in columns if c in df.columns]
    if not cols:
        print("(columnas no disponibles)")
        return
    print(df[cols].head(max_rows).to_string(index=False))


def _scoreline_top(scorelines: pd.DataFrame, match_id: str, top_n: int, match: Optional[pd.Series] = None) -> pd.DataFrame:
    work = pd.DataFrame()
    if not scorelines.empty and "match_id" in scorelines.columns:
        work = scorelines[scorelines["match_id"].astype(str) == str(match_id)].copy()
    if work.empty and match is not None and "scoreline_distribution_json" in match.index:
        raw = match.get("scoreline_distribution_json")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else []
            work = pd.DataFrame(parsed)
        except Exception:
            work = pd.DataFrame()
    if work.empty:
        return work
    work = work.sort_values("probability", ascending=False).head(top_n)
    work["probability"] = work["probability"].map(lambda x: _fmt_pct(x, 2))
    return work


def _market_name(market: str) -> str:
    mapping = {
        "goals": "Goles",
        "shots": "Tiros",
        "shots_on_target": "Tiros a puerta",
        "fouls": "Faltas",
        "yellow_cards": "Tarjetas amarillas",
        "corners": "Corners",
        "player_shots": "Jugador tiros",
        "player_shots_on_target": "Jugador tiros a puerta",
        "player_fouls_committed": "Jugador faltas",
        "player_yellow_card": "Jugador amarilla",
    }
    return mapping.get(str(market), str(market))


def _prepare_line_rows(lines: pd.DataFrame, match_id: str, scope: Optional[str] = None) -> pd.DataFrame:
    if lines.empty or "match_id" not in lines.columns:
        return pd.DataFrame()
    work = lines[lines["match_id"].astype(str) == str(match_id)].copy()
    if scope is not None and "scope" in work.columns:
        work = work[work["scope"].astype(str).eq(scope)]
    if work.empty:
        return work
    if "market_label" not in work.columns and "market" in work.columns:
        work["market_label"] = work["market"].map(_market_name)
    for src, dst, func in [
        ("model_probability", "prob", _fmt_pct),
        ("fair_odds", "fair", _fmt_num),
        ("expected_stat", "exp", _fmt_num),
    ]:
        if src in work.columns:
            work[dst] = work[src].map(func)
    return work


def _line_highlights(lines: pd.DataFrame, match_id: str, top_n: int, min_prob: float = 0.52, include_extreme: bool = False) -> pd.DataFrame:
    work = _prepare_line_rows(lines, match_id)
    if work.empty:
        return work
    work = work[work.get("availability", "").astype(str).eq("available")]
    work = work[work.get("scope", "").astype(str).isin(["match", "team"])]
    if work.empty:
        return work

    if "model_probability" in work.columns:
        work["_prob_raw"] = pd.to_numeric(work["model_probability"], errors="coerce").fillna(0.0)
        work = work[work["_prob_raw"] >= float(min_prob)]
        if not include_extreme:
            # Avoid flooding the console with trivial 95-99% unders/overs unless the user asks.
            work = work[work["_prob_raw"] <= 0.92]
    else:
        work["_prob_raw"] = 0.0
    if work.empty:
        return work

    # If odds exist, EV matters. If not, rank by model/evidence signal and fair odds usefulness.
    signal_rank = {
        "high_model_signal": 0,
        "medium_model_signal": 1,
        "fair_or_thin_signal": 2,
        "low_model_signal": 9,
    }
    value_rank = {
        "high_value": 0,
        "medium_value": 1,
        "fair_price": 2,
        "odds_not_available": 5,
        "demo_odds_only": 6,
        "no_value": 9,
        "not_available": 10,
    }
    work["_signal_rank"] = work.get("signal_label", pd.Series(index=work.index, dtype=object)).map(signal_rank).fillna(9)
    work["_value_rank"] = work.get("value_label", pd.Series(index=work.index, dtype=object)).map(value_rank).fillna(5)
    work["_ev_sort"] = pd.to_numeric(work.get("ev", pd.Series(index=work.index, dtype=float)), errors="coerce").fillna(-999)
    work["_edge_sort"] = pd.to_numeric(work.get("edge", pd.Series(index=work.index, dtype=float)), errors="coerce").fillna(-999)
    # Useful betting candidates tend to be probabilities around 55-80%, not only huge favourites.
    work["_useful_band_dist"] = (work["_prob_raw"] - 0.66).abs()
    work = work.sort_values(
        ["_value_rank", "_signal_rank", "_ev_sort", "_edge_sort", "_useful_band_dist", "market", "scope", "team", "line", "over_under"],
        ascending=[True, True, False, False, True, True, True, True, True, True],
    ).head(top_n)
    return work

def _player_highlights(
    lines: pd.DataFrame,
    match_id: str,
    top_n: int,
    include_under: bool = True,
    min_prob: float = 0.45,
    include_extreme: bool = False,
) -> pd.DataFrame:
    work = _prepare_line_rows(lines, match_id, scope="player")
    if work.empty:
        return work
    if "availability" in work.columns:
        work = work[work["availability"].astype(str).eq("available")]
    if work.empty:
        return work
    if not include_under and "over_under" in work.columns:
        over = work[work["over_under"].astype(str).str.lower().eq("over")].copy()
        if not over.empty:
            work = over
    if "data_quality_flag" in work.columns:
        dq = work["data_quality_flag"].astype(str).str.lower()
        filtered = work[~dq.isin(["not_available", "identity_unresolved", "sample_size_zero"])].copy()
        if not filtered.empty:
            work = filtered
    if "model_probability" in work.columns:
        work["_prob_raw"] = pd.to_numeric(work["model_probability"], errors="coerce").fillna(0.0)
        work = work[work["_prob_raw"] >= float(min_prob)]
        if not include_extreme:
            work = work[work["_prob_raw"] <= 0.92]
    else:
        work["_prob_raw"] = 0.0
    if work.empty:
        return work

    # Prefer candidate rows that are not low-confidence-only, but still show both overs and unders.
    policy_rank = {
        "confirmed_lineup_candidate": 0,
        "lineup_candidate": 0,
        "squad_fallback_candidate": 1,
        "squad_low_confidence_basic_only": 2,
    }
    signal_rank = {
        "high_model_signal": 0,
        "medium_model_signal": 1,
        "fair_or_thin_signal": 2,
        "low_model_signal": 9,
    }
    value_rank = {
        "high_value": 0,
        "medium_value": 1,
        "fair_price": 2,
        "odds_not_available": 5,
        "demo_odds_only": 6,
        "no_value": 9,
        "not_available": 10,
    }
    if "candidate_policy" in work.columns:
        work["_policy_rank"] = work["candidate_policy"].map(policy_rank).fillna(5)
    else:
        work["_policy_rank"] = 5
    work["_signal_rank"] = work.get("signal_label", pd.Series(index=work.index, dtype=object)).map(signal_rank).fillna(5)
    work["_value_rank"] = work.get("value_label", pd.Series(index=work.index, dtype=object)).map(value_rank).fillna(5)
    work["_ev_sort"] = pd.to_numeric(work.get("ev", pd.Series(index=work.index, dtype=float)), errors="coerce").fillna(-999)
    # For player props, prefer useful probabilities and good candidate quality.
    work["_useful_band_dist"] = (work["_prob_raw"] - 0.62).abs()
    work = work.sort_values(
        ["_value_rank", "_policy_rank", "_signal_rank", "_ev_sort", "_useful_band_dist", "player", "market", "line", "over_under"],
        ascending=[True, True, True, False, True, True, True, True, True],
    ).head(top_n)
    return work

def _team_stats(team_stats: pd.DataFrame, match_id: str) -> pd.DataFrame:
    if team_stats.empty or "match_id" not in team_stats.columns:
        return pd.DataFrame()
    work = team_stats[team_stats["match_id"].astype(str) == str(match_id)].copy()
    if work.empty:
        return work
    if "expected_count" in work.columns:
        work["expected"] = work["expected_count"].map(_fmt_num)
    if "market" in work.columns:
        work["market_label"] = work["market"].map(_market_name)
    return work


def _print_match(
    match: pd.Series,
    scorelines: pd.DataFrame,
    team_stats: pd.DataFrame,
    lines: pd.DataFrame,
    top_lines: int,
    top_players: int,
    include_under_player_props: bool = True,
    min_line_prob: float = 0.52,
    min_player_prob: float = 0.45,
    include_extreme: bool = False,
) -> None:
    mid = str(match.get("match_id", ""))
    home = _safe_str(match.get("home_team", "home"))
    away = _safe_str(match.get("away_team", "away"))
    kickoff = _safe_str(match.get("kickoff_user", match.get("date", "")))
    event_time = _safe_str(match.get("kickoff_event", ""))
    status = _safe_str(match.get("status_bucket", match.get("status_long", "")))

    _title(f"{home.upper()} vs {away.upper()}  |  {mid}")
    print(f"Estado: {status or '—'}")
    print(f"Hora usuario: {kickoff or '—'}")
    if event_time:
        print(f"Hora evento : {event_time}  ({_safe_str(match.get('event_timezone', ''))})")

    print("\n1X2 / goles")
    print(f"  {home}: {_fmt_pct(match.get('p_home_win'))}   Empate: {_fmt_pct(match.get('p_draw'))}   {away}: {_fmt_pct(match.get('p_away_win'))}")
    print(f"  xG: {home} {_fmt_num(match.get('expected_home_goals'))} - {_fmt_num(match.get('expected_away_goals'))} {away}")
    print(f"  Over 2.5: {_fmt_pct(match.get('p_over_25'))}   BTTS: {_fmt_pct(match.get('p_btts'))}   Marcador modal: {_safe_str(match.get('most_likely_score', '—'))} ({_fmt_pct(match.get('most_likely_score_probability'), 2)})")

    _section("Scorelines más probables")
    st = _scoreline_top(scorelines, mid, 8, match)
    _print_table(st, ["score", "probability"], 8)

    _section("Eventos esperados por equipo")
    ts = _team_stats(team_stats, mid)
    _print_table(ts, ["team", "market_label", "expected", "confidence", "warnings"], 20)

    _section("Líneas partido/equipo con señal estadística (OVER y UNDER)")
    lh = _line_highlights(lines, mid, top_lines, min_prob=min_line_prob, include_extreme=include_extreme)
    _print_table(lh, ["market_label", "scope", "team", "line", "over_under", "prob", "fair", "book_odds", "edge", "ev", "recent_hit_rate", "signal_label", "value_label", "reason_code"], top_lines)

    _section("Player props con señal estadística (OVER y UNDER)")
    ph = _player_highlights(lines, mid, top_players, include_under_player_props, min_prob=min_player_prob, include_extreme=include_extreme)
    _print_table(ph, ["market_label", "team", "player", "line", "over_under", "prob", "fair", "exp", "position_group", "player_selection_confidence", "candidate_policy", "recent_hit_rate", "signal_label", "value_label"], top_players)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Show Mundialytics matchday predictions in the console.")
    parser.add_argument("--out-dir", default="outputs/statistical_matchday_today", help="Directory produced by run_statistical_matchday.py")
    parser.add_argument("--team", action="append", default=[], help="Filter by team text. Can be passed multiple times, e.g. --team ecuador --team japan")
    parser.add_argument("--match-id", action="append", default=[], help="Filter by match_id. Can be passed multiple times")
    parser.add_argument("--top-lines", type=int, default=12, help="Number of match/team dynamic lines to show per match")
    parser.add_argument("--top-players", type=int, default=12, help="Number of player prop rows to show per match")
    parser.add_argument("--overs-only-player-props", action="store_true", help="Only show over rows for player props. Default now shows both over and under.")
    parser.add_argument("--min-line-prob", type=float, default=0.52, help="Minimum probability for match/team statistical signals shown in console")
    parser.add_argument("--min-player-prob", type=float, default=0.45, help="Minimum probability for player-prop statistical signals shown in console")
    parser.add_argument("--include-extreme-lines", action="store_true", help="Also show very high-probability low-price lines, e.g. 95% unders/overs")
    parser.add_argument("--list", action="store_true", help="Only list available matches")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    matches = _read_csv(out_dir / "match_predictions.csv")
    scorelines = _read_csv(out_dir / "scoreline_distribution.csv")
    team_stats = _read_csv(out_dir / "team_stats_predictions.csv")
    lines = _read_csv(out_dir / "dynamic_market_lines.csv")

    if matches.empty:
        print(f"No he encontrado match_predictions.csv en {out_dir}")
        return 1

    selected = _select_matches(matches, args.match_id, args.team)
    if selected.empty:
        print("No hay partidos que coincidan con el filtro.")
        print("Partidos disponibles:")
        _print_table(matches, ["match_id", "home_team", "away_team", "kickoff_user", "kickoff_event", "status_bucket"], 50)
        return 1

    if args.list:
        _print_table(selected, ["match_id", "home_team", "away_team", "kickoff_user", "kickoff_event", "status_bucket", "p_home_win", "p_draw", "p_away_win"], 50)
        return 0

    print("MUNDIALYTICS CONSOLE MATCHDAY")
    print(f"Output dir: {out_dir}")
    print(f"Partidos seleccionados: {len(selected)}")

    for _, match in selected.iterrows():
        _print_match(match, scorelines, team_stats, lines, args.top_lines, args.top_players, include_under_player_props=not args.overs_only_player_props, min_line_prob=args.min_line_prob, min_player_prob=args.min_player_prob, include_extreme=args.include_extreme_lines)

    print("\nNota: sin cuotas, esto NO es pick real: es señal estadística + fair_odds. Pick real = comparar fair_odds/probabilidad con cuota disponible y validar en paper tracking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
