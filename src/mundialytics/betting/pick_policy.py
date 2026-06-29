from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


OUTCOME_MARKET = "1x2"
TOTAL_GOALS_MARKET = "goals"
BTTS_MARKET = "btts"

# Markets that are typically priced as over/under lines by bookmakers.
# These can be evaluated whenever a settled line-signal file exists.
OVER_UNDER_MARKETS = {
    "goals", "team_goals",
    "corners", "team_corners",
    "shots", "team_shots", "player_shots",
    "shots_on_target", "team_shots_on_target", "player_shots_on_target",
    "cards", "yellow_cards", "team_cards", "team_yellow_cards", "player_yellow_card",
    "fouls", "team_fouls", "player_fouls_committed",
    "goalkeeper_saves", "player_saves",
}

MARKET_ALIASES = {
    "total goals": "goals",
    "match goals": "goals",
    "over under goals": "goals",
    "goals total": "goals",
    "team goals": "team_goals",
    "corners total": "corners",
    "total corners": "corners",
    "match corners": "corners",
    "team corners": "team_corners",
    "total shots": "shots",
    "match shots": "shots",
    "team shots": "team_shots",
    "player shots": "player_shots",
    "shots on target": "shots_on_target",
    "shot on target": "shots_on_target",
    "sot": "shots_on_target",
    "total shots on target": "shots_on_target",
    "match shots on target": "shots_on_target",
    "team shots on target": "team_shots_on_target",
    "player shots on target": "player_shots_on_target",
    "cards": "cards",
    "total cards": "cards",
    "yellow cards": "yellow_cards",
    "total yellow cards": "yellow_cards",
    "team cards": "team_cards",
    "team yellow cards": "team_yellow_cards",
    "player yellow card": "player_yellow_card",
    "fouls": "fouls",
    "total fouls": "fouls",
    "team fouls": "team_fouls",
    "player fouls": "player_fouls_committed",
    "goalkeeper saves": "goalkeeper_saves",
    "keeper saves": "goalkeeper_saves",
    "player saves": "player_saves",
    "1 x 2": "1x2",
    "match result": "1x2",
    "full time result": "1x2",
    "both teams to score": "btts",
}



def _norm_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).lower().replace("_", " ").replace("-", " ").split())


def _safe_prob(value: object) -> float:
    try:
        v = float(value)
    except Exception:
        return float("nan")
    if not math.isfinite(v):
        return float("nan")
    return max(1e-6, min(1 - 1e-6, v))


def _fair_odds(prob: float) -> float:
    p = _safe_prob(prob)
    if not math.isfinite(p):
        return float("nan")
    return 1.0 / p


def add_chronological_split(df: pd.DataFrame, train_frac: float = 0.6, validation_frac: float = 0.2) -> pd.DataFrame:
    work = df.copy()
    if "date" in work.columns:
        work["_date_sort"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.sort_values(["_date_sort", "match_id", "market", "selection"], kind="mergesort")
    else:
        work = work.reset_index().rename(columns={"index": "_date_sort"})
    unique_matches = work[["match_id"]].drop_duplicates().reset_index(drop=True)
    n = len(unique_matches)
    if n == 0:
        work["split"] = "test"
        return work.drop(columns=[c for c in ["_date_sort"] if c in work.columns])
    train_cut = max(1, int(round(n * train_frac)))
    validation_cut = max(train_cut + 1, int(round(n * (train_frac + validation_frac))))
    validation_cut = min(validation_cut, n)
    split_by_match = {}
    for i, mid in enumerate(unique_matches["match_id"].astype(str).tolist()):
        if i < train_cut:
            split_by_match[mid] = "train"
        elif i < validation_cut:
            split_by_match[mid] = "validation"
        else:
            split_by_match[mid] = "test"
    work["split"] = work["match_id"].astype(str).map(split_by_match).fillna("test")
    return work.drop(columns=[c for c in ["_date_sort"] if c in work.columns])


def build_match_pick_signals(match_predictions: pd.DataFrame) -> pd.DataFrame:
    """Create settled betting-signal rows from historical match predictions.

    This is market-agnostic and does NOT need bookmaker odds. If odds are added
    later, the same rows can be priced and evaluated for profit.
    """
    rows: list[dict] = []
    if match_predictions.empty:
        return pd.DataFrame()
    for _, r in match_predictions.iterrows():
        mid = r.get("match_id")
        if pd.isna(mid):
            continue
        home = _norm_text(r.get("home_team"))
        away = _norm_text(r.get("away_team"))
        actual_home = r.get("actual_home_goals")
        actual_away = r.get("actual_away_goals")
        try:
            actual_home_i = int(actual_home)
            actual_away_i = int(actual_away)
            total_goals = actual_home_i + actual_away_i
        except Exception:
            continue
        if actual_home_i > actual_away_i:
            actual_outcome = "home"
        elif actual_home_i < actual_away_i:
            actual_outcome = "away"
        else:
            actual_outcome = "draw"
        common = {
            "match_id": str(mid),
            "date": r.get("date"),
            "home_team": home,
            "away_team": away,
            "actual_home_goals": actual_home_i,
            "actual_away_goals": actual_away_i,
            "actual_total_goals": total_goals,
            "competition": r.get("competition"),
            "competition_context": r.get("competition_context"),
            "team_type": r.get("team_type"),
            "gender": r.get("gender"),
        }
        for selection, col in [("home", "p_home_win"), ("draw", "p_draw"), ("away", "p_away_win")]:
            p = _safe_prob(r.get(col))
            if math.isfinite(p):
                rows.append({
                    **common,
                    "market": OUTCOME_MARKET,
                    "scope": "match",
                    "selection": selection,
                    "line": np.nan,
                    "over_under": "",
                    "model_probability": p,
                    "fair_odds": _fair_odds(p),
                    "actual_win": int(actual_outcome == selection),
                    "settled_stat": actual_outcome,
                })
        for line, col in [(0.5, "p_over_05"), (1.5, "p_over_15"), (2.5, "p_over_25"), (3.5, "p_over_35")]:
            p_over = _safe_prob(r.get(col))
            if math.isfinite(p_over):
                over_win = int(total_goals > line)
                for side, p, win in [("over", p_over, over_win), ("under", 1.0 - p_over, 1 - over_win)]:
                    rows.append({
                        **common,
                        "market": TOTAL_GOALS_MARKET,
                        "scope": "match",
                        "selection": side,
                        "line": line,
                        "over_under": side,
                        "model_probability": _safe_prob(p),
                        "fair_odds": _fair_odds(p),
                        "actual_win": int(win),
                        "settled_stat": total_goals,
                    })
        p_btts = _safe_prob(r.get("p_btts"))
        if math.isfinite(p_btts):
            btts_win = int(actual_home_i > 0 and actual_away_i > 0)
            for selection, p, win in [("yes", p_btts, btts_win), ("no", 1.0 - p_btts, 1 - btts_win)]:
                rows.append({
                    **common,
                    "market": BTTS_MARKET,
                    "scope": "match",
                    "selection": selection,
                    "line": np.nan,
                    "over_under": "",
                    "model_probability": _safe_prob(p),
                    "fair_odds": _fair_odds(p),
                    "actual_win": int(win),
                    "settled_stat": bool(btts_win),
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["fair_odds"] = pd.to_numeric(out["fair_odds"], errors="coerce")
    out["model_probability"] = pd.to_numeric(out["model_probability"], errors="coerce")
    out["actual_win"] = pd.to_numeric(out["actual_win"], errors="coerce")
    out = add_signal_group(out)
    return out



def standardize_settled_line_signals(line_signals: pd.DataFrame) -> pd.DataFrame:
    """Standardize generic settled line signals for any bookmaker-style market.

    Expected input can be the output of a future event-line backtest, or a hand-built
    CSV with at least:
      match_id, market, model_probability, line, and either actual_win OR settled_stat.

    Supported sides:
      - over/under markets: selection or over_under = over|under
      - BTTS: selection = yes|no
      - 1X2: selection = home|draw|away

    This is intentionally generic so corners, shots, shots_on_target, cards and
    goalkeeper_saves can be trained/evaluated as soon as real targets are available.
    It does not invent targets; if actual_win/settled_stat is missing, rows are dropped.
    """
    if line_signals is None or line_signals.empty:
        return pd.DataFrame()
    work = line_signals.copy()
    if "market" not in work.columns:
        return pd.DataFrame()
    work["market"] = work["market"].map(lambda x: MARKET_ALIASES.get(_norm_text(x), _norm_text(x)))
    if "selection" not in work.columns:
        if "over_under" in work.columns:
            work["selection"] = work["over_under"]
        elif "side" in work.columns:
            work["selection"] = work["side"]
        else:
            work["selection"] = ""
    work["selection"] = work["selection"].map(_norm_text)
    if "over_under" not in work.columns:
        work["over_under"] = np.where(work["selection"].isin(["over", "under"]), work["selection"], "")
    for col in ["match_id", "home_team", "away_team", "team", "player", "scope", "competition", "competition_context", "team_type", "gender"]:
        if col not in work.columns:
            work[col] = ""
    if "date" not in work.columns:
        work["date"] = ""
    if "line" not in work.columns:
        work["line"] = np.nan
    work["line"] = pd.to_numeric(work["line"], errors="coerce")
    if "model_probability" not in work.columns:
        for c in ["probability", "prob", "p_model"]:
            if c in work.columns:
                work["model_probability"] = work[c]
                break
    if "model_probability" not in work.columns:
        return pd.DataFrame()
    work["model_probability"] = work["model_probability"].map(_safe_prob)
    if "fair_odds" not in work.columns:
        work["fair_odds"] = work["model_probability"].map(_fair_odds)
    else:
        work["fair_odds"] = pd.to_numeric(work["fair_odds"], errors="coerce")
        work["fair_odds"] = work["fair_odds"].fillna(work["model_probability"].map(_fair_odds))

    if "actual_win" not in work.columns:
        if "settled_stat" in work.columns:
            stat = pd.to_numeric(work["settled_stat"], errors="coerce")
            line = pd.to_numeric(work["line"], errors="coerce")
            sel = work["selection"].map(_norm_text)
            work["actual_win"] = np.where(sel.eq("over"), stat > line, np.where(sel.eq("under"), stat < line, np.nan))
        elif "actual_stat" in work.columns:
            stat = pd.to_numeric(work["actual_stat"], errors="coerce")
            line = pd.to_numeric(work["line"], errors="coerce")
            sel = work["selection"].map(_norm_text)
            work["actual_win"] = np.where(sel.eq("over"), stat > line, np.where(sel.eq("under"), stat < line, np.nan))
        else:
            work["actual_win"] = np.nan
    work["actual_win"] = pd.to_numeric(work["actual_win"], errors="coerce")
    work = work[work["model_probability"].notna() & work["actual_win"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    keep = [
        "match_id", "date", "home_team", "away_team", "team", "player", "competition", "competition_context",
        "team_type", "gender", "market", "scope", "selection", "line", "over_under",
        "model_probability", "fair_odds", "actual_win",
        "data_source", "data_quality_flag", "saves_data_quality_flag", "target_quality",
        "expected_stat", "expected_components", "model_family", "goalkeeper",
    ]
    if "settled_stat" in work.columns:
        keep.append("settled_stat")
    elif "actual_stat" in work.columns:
        work["settled_stat"] = work["actual_stat"]
        keep.append("settled_stat")
    out = work[[c for c in keep if c in work.columns]].copy()
    out = add_signal_group(out)
    return out

def _normalise_odds(odds: pd.DataFrame) -> pd.DataFrame:
    if odds is None or odds.empty:
        return pd.DataFrame()
    work = odds.copy()
    rename = {}
    for candidate in ["book_odds", "odds", "decimal_odds", "price"]:
        if candidate in work.columns:
            rename[candidate] = "book_odds"
            break
    if rename:
        work = work.rename(columns=rename)
    if "book_odds" not in work.columns:
        return pd.DataFrame()
    for col in ["match_id", "market", "selection", "over_under", "side", "team", "player"]:
        if col in work.columns:
            work[col] = work[col].map(_norm_text)
    if "selection" not in work.columns:
        if "over_under" in work.columns:
            work["selection"] = work["over_under"]
        elif "side" in work.columns:
            work["selection"] = work["side"]
        else:
            work["selection"] = ""
    if "market" in work.columns:
        work["market"] = work["market"].map(lambda x: MARKET_ALIASES.get(_norm_text(x), _norm_text(x)))
    if "line" not in work.columns:
        work["line"] = np.nan
    work["line"] = pd.to_numeric(work["line"], errors="coerce")
    work["book_odds"] = pd.to_numeric(work["book_odds"], errors="coerce")
    return work


def attach_odds(signals: pd.DataFrame, odds: Optional[pd.DataFrame]) -> pd.DataFrame:
    signals = signals.copy()
    if signals.empty:
        return signals
    signals["book_odds"] = np.nan
    if odds is None or odds.empty:
        signals["implied_probability"] = np.nan
        signals["edge"] = np.nan
        signals["ev"] = np.nan
        signals["profit_1u"] = np.nan
        signals["price_status"] = "no_odds"
        return signals
    odds_n = _normalise_odds(odds)
    if odds_n.empty:
        signals["implied_probability"] = np.nan
        signals["edge"] = np.nan
        signals["ev"] = np.nan
        signals["profit_1u"] = np.nan
        signals["price_status"] = "odds_unusable_schema"
        return signals
    # Key-level merge; use rounded line so 2.5 joins robustly.
    left = signals.copy()
    left["_market_key"] = left["market"].map(_norm_text)
    left["_selection_key"] = left["selection"].map(_norm_text)
    left["_line_key"] = pd.to_numeric(left["line"], errors="coerce").round(3)
    right = odds_n.copy()
    right["_market_key"] = right["market"].map(_norm_text)
    right["_selection_key"] = right["selection"].map(_norm_text)
    right["_line_key"] = pd.to_numeric(right["line"], errors="coerce").round(3)
    # For non-line markets, NaN lines do not merge reliably. Replace with sentinel.
    left["_line_key"] = left["_line_key"].fillna(-9999.0)
    right["_line_key"] = right["_line_key"].fillna(-9999.0)
    keys = ["match_id", "_market_key", "_selection_key", "_line_key"]
    if "match_id" in right.columns:
        right["match_id"] = right["match_id"].astype(str)
    left["match_id"] = left["match_id"].astype(str)
    merged = left.merge(right[keys + ["book_odds"]].dropna(subset=["book_odds"]).drop_duplicates(keys), on=keys, how="left", suffixes=("", "_attached"))
    merged["book_odds"] = merged["book_odds_attached"]
    merged = merged.drop(columns=[c for c in ["book_odds_attached", "_market_key", "_selection_key", "_line_key"] if c in merged.columns])
    merged["implied_probability"] = 1.0 / pd.to_numeric(merged["book_odds"], errors="coerce")
    merged["edge"] = pd.to_numeric(merged["model_probability"], errors="coerce") - merged["implied_probability"]
    merged["ev"] = pd.to_numeric(merged["model_probability"], errors="coerce") * pd.to_numeric(merged["book_odds"], errors="coerce") - 1.0
    merged["profit_1u"] = np.where(merged["actual_win"].eq(1), pd.to_numeric(merged["book_odds"], errors="coerce") - 1.0, -1.0)
    merged.loc[merged["book_odds"].isna(), ["implied_probability", "edge", "ev", "profit_1u"]] = np.nan
    merged["price_status"] = np.where(merged["book_odds"].notna(), "priced", "no_matching_odds")
    return merged


def evaluate_model_performance_by_market(signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate model signal quality by market/split and by market-line bucket.

    This is deliberately not a profitability report. It answers:
    - Which markets are calibrated/useful as predictions?
    - Which markets/lines are unstable or unsupported?
    """
    if signals is None or signals.empty:
        cols = [
            "split", "market", "scope", "selection", "line", "n", "hit_rate", "avg_model_probability",
            "calibration_gap", "brier", "log_loss", "avg_fair_odds",
        ]
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    work = add_signal_group(signals.copy())
    if "split" not in work.columns:
        work = add_chronological_split(work)
    work["_prob"] = pd.to_numeric(work.get("model_probability"), errors="coerce").clip(1e-6, 1 - 1e-6)
    work["_actual"] = pd.to_numeric(work.get("actual_win"), errors="coerce")
    work["_fair"] = pd.to_numeric(work.get("fair_odds"), errors="coerce")
    work = work[work["_prob"].notna() & work["_actual"].notna()].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    def _metrics(g: pd.DataFrame) -> dict:
        p = g["_prob"].astype(float)
        y = g["_actual"].astype(float)
        n = int(len(g))
        hit = float(y.mean()) if n else None
        avgp = float(p.mean()) if n else None
        brier = float(((p - y) ** 2).mean()) if n else None
        logloss = float((-(y * np.log(p) + (1 - y) * np.log(1 - p))).mean()) if n else None
        return {
            "n": n,
            "hit_rate": hit,
            "avg_model_probability": avgp,
            "calibration_gap": (hit - avgp) if hit is not None and avgp is not None else None,
            "brier": brier,
            "log_loss": logloss,
            "avg_fair_odds": float(g["_fair"].mean()) if g["_fair"].notna().any() else None,
        }

    rows = []
    group_cols = ["split", "market"]
    for keys, g in work.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(_metrics(g))
        row["scope"] = "all"
        row["selection"] = "all"
        row["line"] = np.nan
        rows.append(row)
    market_summary = pd.DataFrame(rows)
    if not market_summary.empty:
        market_summary = market_summary[["split", "market", "scope", "selection", "line", "n", "hit_rate", "avg_model_probability", "calibration_gap", "brier", "log_loss", "avg_fair_odds"]]
        market_summary = market_summary.sort_values(["split", "market"], kind="mergesort")

    line_rows = []
    line_group_cols = ["split", "market", "scope", "selection", "line"]
    for keys, g in work.groupby(line_group_cols, dropna=False):
        row = dict(zip(line_group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(_metrics(g))
        line_rows.append(row)
    line_summary = pd.DataFrame(line_rows)
    if not line_summary.empty:
        line_summary = line_summary[["split", "market", "scope", "selection", "line", "n", "hit_rate", "avg_model_probability", "calibration_gap", "brier", "log_loss", "avg_fair_odds"]]
        line_summary = line_summary.sort_values(["split", "market", "line", "selection"], kind="mergesort")
    return market_summary, line_summary


def evaluate_threshold_performance_by_market(signals: pd.DataFrame, thresholds: list[float] | None = None) -> pd.DataFrame:
    """Evaluate high-confidence prediction rows by market and probability threshold.

    This avoids the misleading aggregate issue where paired selections such as
    over/under naturally average around 50%. It answers: when this market gives
    a strong signal, does it settle at roughly that rate?
    """
    if thresholds is None:
        thresholds = [0.52, 0.55, 0.58, 0.60, 0.63, 0.66, 0.70, 0.75, 0.80]
    if signals is None or signals.empty:
        return pd.DataFrame()
    work = add_signal_group(signals.copy())
    if "split" not in work.columns:
        work = add_chronological_split(work)
    work["_prob"] = pd.to_numeric(work.get("model_probability"), errors="coerce")
    work["_actual"] = pd.to_numeric(work.get("actual_win"), errors="coerce")
    work["_fair"] = pd.to_numeric(work.get("fair_odds"), errors="coerce")
    work = work[work["_prob"].notna() & work["_actual"].notna()].copy()
    rows: list[dict] = []
    for split in ["train", "validation", "test", "all"]:
        base = work if split == "all" else work[work["split"].eq(split)]
        if base.empty:
            continue
        base = add_signal_group(base)
        grouping_sets = [
            ("market", ["market"]),
            ("selection", ["market", "selection"]),
            ("signal_group", ["signal_group"]),
            ("line_selection", ["market", "selection", "line"]),
        ]
        for eval_level, group_cols in grouping_sets:
            for keys, g_base in base.groupby(group_cols, dropna=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                key_map = dict(zip(group_cols, keys))
                for thr in thresholds:
                    g = g_base[g_base["_prob"] >= thr]
                    if g.empty:
                        continue
                    n = int(len(g))
                    hit = float(g["_actual"].mean())
                    avgp = float(g["_prob"].mean())
                    rows.append({
                        "split": split,
                        "eval_level": eval_level,
                        "market": key_map.get("market", "all"),
                        "selection": key_map.get("selection", "all"),
                        "signal_group": key_map.get("signal_group", "all"),
                        "line": key_map.get("line", np.nan),
                        "min_model_probability": thr,
                        "n": n,
                        "hit_rate": hit,
                        "avg_model_probability": avgp,
                        "calibration_gap": hit - avgp,
                        "avg_fair_odds": float(g["_fair"].mean()) if g["_fair"].notna().any() else None,
                    })
    out = pd.DataFrame(rows)
    if not out.empty:
        sort_cols = [c for c in ["split", "eval_level", "market", "selection", "line", "signal_group", "min_model_probability"] if c in out.columns]
        out = out.sort_values(sort_cols, kind="mergesort")
    return out


def build_market_takeaways(market_summary: pd.DataFrame, min_test_n: int = 30) -> dict:
    """Create a conservative market status summary from test split metrics."""
    if market_summary is None or market_summary.empty:
        return {"status": "no_market_metrics"}
    test = market_summary[market_summary["split"].eq("test")].copy()
    if test.empty:
        test = market_summary[market_summary["split"].eq("all")].copy()
    rows = []
    for _, r in test.iterrows():
        n = int(r.get("n", 0) or 0)
        gap = r.get("calibration_gap")
        brier = r.get("brier")
        status = "insufficient_test_sample" if n < min_test_n else "monitor"
        if n >= min_test_n:
            if gap is not None and pd.notna(gap) and abs(float(gap)) <= 0.04:
                status = "usable_signal_candidate"
            if gap is not None and pd.notna(gap) and abs(float(gap)) > 0.08:
                status = "calibration_warning"
        rows.append({
            "market": r.get("market"),
            "test_n": n,
            "test_hit_rate": r.get("hit_rate"),
            "test_avg_model_probability": r.get("avg_model_probability"),
            "test_calibration_gap": r.get("calibration_gap"),
            "test_brier": brier,
            "status": status,
        })
    return {
        "status": "completed",
        "warning": "This evaluates predictive signal by market, not betting profitability unless odds are supplied.",
        "markets": rows,
    }


@dataclass(frozen=True)
class PickPolicy:
    policy_id: str
    min_model_probability: float
    min_fair_odds: float
    max_fair_odds: float
    min_edge: float = 0.0
    min_ev: float = 0.0
    allowed_markets: str = "all"
    allowed_signal_group: str = "all"

    def as_dict(self) -> dict:
        return asdict(self)


def _market_allowed(series: pd.Series, allowed: str) -> pd.Series:
    if allowed == "all":
        return pd.Series(True, index=series.index)
    wanted = set(allowed.split("+"))
    return series.astype(str).isin(wanted)


def _signal_group_for_row(row: pd.Series | dict) -> str:
    market = MARKET_ALIASES.get(_norm_text(row.get("market")), _norm_text(row.get("market")))
    selection = _norm_text(row.get("selection")) or _norm_text(row.get("over_under"))
    scope = _norm_text(row.get("scope"))
    if market in OVER_UNDER_MARKETS and selection in {"over", "under"}:
        # Scope prefix prevents mixing match totals vs team/player lines, but avoid
        # double-prefixing markets that are already named like team_shots/player_shots.
        if scope == "team" and not market.startswith("team_"):
            return f"team_{market}_{selection}"
        if scope == "player" and not (market.startswith("player_") or market == "goalkeeper_saves"):
            return f"player_{market}_{selection}"
        return f"{market}_{selection}"
    if market == BTTS_MARKET:
        return f"btts_{selection}" if selection in {"yes", "no"} else "btts_other"
    if market == OUTCOME_MARKET:
        return f"1x2_{selection}" if selection in {"home", "draw", "away"} else "1x2_other"
    return f"{market}_{selection}" if selection else market


def add_signal_group(signals: pd.DataFrame) -> pd.DataFrame:
    work = signals.copy()
    if work.empty:
        work["signal_group"] = []
        work["signal_side"] = []
        return work
    if "signal_group" not in work.columns:
        work["signal_group"] = work.apply(_signal_group_for_row, axis=1)
    if "signal_side" not in work.columns:
        def _side(row: pd.Series) -> str:
            market = _norm_text(row.get("market"))
            selection = _norm_text(row.get("selection"))
            if market == TOTAL_GOALS_MARKET and selection in {"over", "under"}:
                return selection
            if market == BTTS_MARKET and selection in {"yes", "no"}:
                return selection
            if market == OUTCOME_MARKET and selection in {"home", "draw", "away"}:
                return selection
            return selection
        work["signal_side"] = work.apply(_side, axis=1)
    return work


def _signal_group_allowed(df: pd.DataFrame, allowed: str) -> pd.Series:
    if allowed in {"all", ""} or allowed is None:
        return pd.Series(True, index=df.index)
    work = add_signal_group(df)
    group = work["signal_group"].astype(str)
    market = work["market"].astype(str)
    selection = work["selection"].astype(str)
    allowed = str(allowed)
    allowed = MARKET_ALIASES.get(_norm_text(allowed), _norm_text(allowed)).replace(" ", "_")
    if allowed in set(group.astype(str).unique()):
        return group.eq(allowed)
    if allowed == "goals":
        return market.eq(TOTAL_GOALS_MARKET)
    if allowed == "goals_over":
        return group.eq("goals_over")
    if allowed == "goals_under":
        return group.eq("goals_under")
    if allowed == "btts":
        return market.eq(BTTS_MARKET)
    if allowed == "btts_yes":
        return group.eq("btts_yes")
    if allowed == "btts_no":
        return group.eq("btts_no")
    if allowed == "1x2":
        return market.eq(OUTCOME_MARKET)
    if allowed == "1x2_home":
        return group.eq("1x2_home")
    if allowed == "1x2_draw":
        return group.eq("1x2_draw")
    if allowed == "1x2_away":
        return group.eq("1x2_away")
    if allowed == "1x2_home_away":
        return market.eq(OUTCOME_MARKET) & selection.isin(["home", "away"])
    if "+" in allowed:
        pieces = allowed.split("+")
        mask = pd.Series(False, index=df.index)
        for piece in pieces:
            mask |= _signal_group_allowed(work, piece)
        return mask
    wanted = set(allowed.split("|"))
    return group.isin(wanted)


def apply_policy(signals: pd.DataFrame, policy: PickPolicy, require_odds: bool = False) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    work = signals.copy()
    mask = pd.to_numeric(work["model_probability"], errors="coerce") >= policy.min_model_probability
    fair = pd.to_numeric(work["fair_odds"], errors="coerce")
    mask &= fair >= policy.min_fair_odds
    mask &= fair <= policy.max_fair_odds
    mask &= _market_allowed(work["market"], policy.allowed_markets)
    mask &= _signal_group_allowed(work, policy.allowed_signal_group)
    if require_odds:
        mask &= work.get("book_odds", pd.Series(index=work.index, dtype=float)).notna()
        mask &= pd.to_numeric(work.get("edge", pd.Series(index=work.index, dtype=float)), errors="coerce") >= policy.min_edge
        mask &= pd.to_numeric(work.get("ev", pd.Series(index=work.index, dtype=float)), errors="coerce") >= policy.min_ev
    out = work[mask].copy()
    out["policy_id"] = policy.policy_id
    return out


def evaluate_picks(picks: pd.DataFrame, require_odds: bool = False) -> dict:
    if picks.empty:
        return {
            "n_picks": 0,
            "hit_rate": None,
            "avg_model_probability": None,
            "calibration_gap": None,
            "roi": None,
            "profit": None,
            "yield": None,
        }
    actual = pd.to_numeric(picks["actual_win"], errors="coerce")
    prob = pd.to_numeric(picks["model_probability"], errors="coerce")
    n = int(actual.notna().sum())
    hit = float(actual.mean()) if n else None
    avg_prob = float(prob.mean()) if prob.notna().any() else None
    out = {
        "n_picks": n,
        "hit_rate": hit,
        "avg_model_probability": avg_prob,
        "calibration_gap": (hit - avg_prob) if hit is not None and avg_prob is not None else None,
        "avg_fair_odds": float(pd.to_numeric(picks.get("fair_odds"), errors="coerce").mean()) if "fair_odds" in picks.columns else None,
        "markets": picks["market"].value_counts(dropna=False).to_dict() if "market" in picks.columns else {},
        "selections": picks["selection"].value_counts(dropna=False).to_dict() if "selection" in picks.columns else {},
        "signal_groups": add_signal_group(picks)["signal_group"].value_counts(dropna=False).to_dict() if "market" in picks.columns and "selection" in picks.columns else {},
    }
    if require_odds and "profit_1u" in picks.columns:
        priced = picks[picks["profit_1u"].notna()].copy()
        profit = float(pd.to_numeric(priced["profit_1u"], errors="coerce").sum()) if not priced.empty else 0.0
        n_priced = int(len(priced))
        out.update({
            "n_priced_picks": n_priced,
            "profit": profit,
            "roi": profit / n_priced if n_priced else None,
            "yield": profit / n_priced if n_priced else None,
            "avg_edge": float(pd.to_numeric(priced.get("edge"), errors="coerce").mean()) if n_priced else None,
            "avg_ev": float(pd.to_numeric(priced.get("ev"), errors="coerce").mean()) if n_priced else None,
        })
    else:
        out.update({"profit": None, "roi": None, "yield": None})
    return out


def generate_policy_grid(require_odds: bool = False, signal_groups_available: Iterable[str] | None = None) -> list[PickPolicy]:
    policies: list[PickPolicy] = []
    probs = [0.52, 0.55, 0.58, 0.60, 0.63, 0.66, 0.70, 0.75]
    fair_bands = [(1.01, 1.35), (1.10, 1.55), (1.20, 1.80), (1.30, 2.20), (1.40, 3.00), (1.01, 3.50)]
    # allowed_signal_group is the important dimension: it lets the policy learn
    # overs separately from unders, and BTTS Yes separately from BTTS No.
    observed = [str(g) for g in signal_groups_available if str(g)] if signal_groups_available is not None else []
    # Always test broad groups plus the actual sides present in the supplied data.
    # This keeps today's match-only backtests fast, while allowing corners/shots/SOT/cards
    # policies automatically as soon as settled line signals are supplied.
    base_signal_groups = ["all"]
    if any(g.startswith("goals_") for g in observed):
        base_signal_groups += ["goals", "goals_over", "goals_under"]
    if any(g.startswith("btts_") for g in observed):
        base_signal_groups += ["btts", "btts_yes", "btts_no"]
    if any(g.startswith("1x2_") for g in observed):
        base_signal_groups += ["1x2", "1x2_home_away", "1x2_home", "1x2_draw", "1x2_away"]
    if {"goals_under", "btts_no"}.issubset(set(observed)):
        base_signal_groups.append("goals_under+btts_no")
    if {"goals_over", "btts_yes"}.issubset(set(observed)):
        base_signal_groups.append("goals_over+btts_yes")
    signal_groups = list(dict.fromkeys(base_signal_groups + observed))
    if require_odds:
        edges = [0.0, 0.02, 0.04, 0.06, 0.08]
        evs = [0.0, 0.02, 0.04, 0.06]
    else:
        edges = [0.0]
        evs = [0.0]
    idx = 0
    for min_prob, (min_fair, max_fair), signal_group, min_edge, min_ev in product(probs, fair_bands, signal_groups, edges, evs):
        idx += 1
        policies.append(PickPolicy(
            policy_id=f"policy_{idx:04d}",
            min_model_probability=min_prob,
            min_fair_odds=min_fair,
            max_fair_odds=max_fair,
            min_edge=min_edge,
            min_ev=min_ev,
            allowed_markets="all",
            allowed_signal_group=signal_group,
        ))
    return policies


def evaluate_policy_grid(signals: pd.DataFrame, min_picks: int = 30, require_odds: bool = False) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    best_policy: dict = {}
    if signals.empty:
        return pd.DataFrame(), best_policy
    if "split" not in signals.columns:
        signals = add_chronological_split(signals)
    for policy in generate_policy_grid(require_odds=require_odds, signal_groups_available=signals.get("signal_group", pd.Series(dtype=str)).dropna().unique() if "signal_group" in signals.columns else None):
        row = policy.as_dict()
        score_parts = {}
        valid_for_selection = True
        for split in ["train", "validation", "test", "all"]:
            subset = signals if split == "all" else signals[signals["split"].eq(split)]
            picks = apply_policy(subset, policy, require_odds=require_odds)
            metrics = evaluate_picks(picks, require_odds=require_odds)
            for k, v in metrics.items():
                row[f"{split}_{k}"] = v
            if split == "validation":
                if metrics["n_picks"] < min_picks:
                    valid_for_selection = False
                if require_odds:
                    score_parts["primary"] = metrics.get("roi") if metrics.get("roi") is not None else -999.0
                    score_parts["secondary"] = metrics.get("profit") if metrics.get("profit") is not None else -999.0
                else:
                    # This is NOT betting profitability. It selects calibrated/statistically useful signals only.
                    gap = metrics.get("calibration_gap")
                    hit = metrics.get("hit_rate")
                    avg_prob = metrics.get("avg_model_probability")
                    score_parts["primary"] = (gap if gap is not None else -999.0)
                    score_parts["secondary"] = (hit if hit is not None else -999.0) - abs((avg_prob or 0.0) - 0.62) * 0.05
        row["selection_valid"] = bool(valid_for_selection)
        row["selection_score_primary"] = score_parts.get("primary", -999.0) if valid_for_selection else -999.0
        row["selection_score_secondary"] = score_parts.get("secondary", -999.0) if valid_for_selection else -999.0
        rows.append(row)
    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(["selection_score_primary", "selection_score_secondary", "validation_n_picks"], ascending=[False, False, False])
        if leaderboard.iloc[0]["selection_valid"]:
            best_policy = {k: leaderboard.iloc[0][k] for k in PickPolicy.__dataclass_fields__.keys() if k in leaderboard.columns}
    return leaderboard, best_policy


def backtest_best_policy(signals: pd.DataFrame, best_policy: dict, require_odds: bool = False) -> pd.DataFrame:
    if not best_policy:
        return pd.DataFrame()
    policy = PickPolicy(
        policy_id=str(best_policy["policy_id"]),
        min_model_probability=float(best_policy["min_model_probability"]),
        min_fair_odds=float(best_policy["min_fair_odds"]),
        max_fair_odds=float(best_policy["max_fair_odds"]),
        min_edge=float(best_policy.get("min_edge", 0.0)),
        min_ev=float(best_policy.get("min_ev", 0.0)),
        allowed_markets=str(best_policy.get("allowed_markets", "all")),
        allowed_signal_group=str(best_policy.get("allowed_signal_group", "all")),
    )
    return apply_policy(signals, policy, require_odds=require_odds)


def summary_from_outputs(signals: pd.DataFrame, leaderboard: pd.DataFrame, selected_picks: pd.DataFrame, require_odds: bool, best_policy: dict) -> dict:
    status = "priced_pick_policy_backtest" if require_odds else "marketless_signal_policy_backtest"
    return {
        "version": "v0.38.4_large_line_signal_policy_audit",
        "status": status,
        "important_warning": (
            "This is a real betting-value backtest only when historical bookmaker odds are supplied. "
            "Without odds it only tests model signal calibration/hit-rate, not profitability."
        ),
        "signals_rows": int(len(signals)),
        "matches": int(signals["match_id"].nunique()) if not signals.empty and "match_id" in signals.columns else 0,
        "odds_attached": bool(require_odds),
        "best_policy": best_policy,
        "selected_picks_summary": evaluate_picks(selected_picks, require_odds=require_odds),
        "top_policy_ids": leaderboard.head(10)["policy_id"].tolist() if not leaderboard.empty and "policy_id" in leaderboard.columns else [],
    }
