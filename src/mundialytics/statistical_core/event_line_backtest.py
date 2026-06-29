from __future__ import annotations

"""Build settled bookmaker-style over/under line signals for event markets.

v0.38 adds relational baselines: corners can use shot volume, goalkeeper saves can
use opponent shots-on-target pressure, and all markets keep side-level evaluation-ready
metadata. It is still deliberately transparent rather than a black-box model.
"""

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import poisson
except Exception:  # pragma: no cover
    poisson = None


@dataclass(frozen=True)
class LineSpec:
    market: str
    scope: str
    stat_for: str
    stat_against: str | None
    lines: tuple[float, ...]


LINE_SPECS: dict[str, LineSpec] = {
    "team_corners": LineSpec("team_corners", "team", "corners_for", "corners_against", (2.5, 3.5, 4.5, 5.5, 6.5)),
    "corners": LineSpec("corners", "match", "corners_for", "corners_against", (7.5, 8.5, 9.5, 10.5, 11.5)),
    "team_shots": LineSpec("team_shots", "team", "shots_for", "shots_against", (6.5, 8.5, 10.5, 12.5, 14.5)),
    "shots": LineSpec("shots", "match", "shots_for", "shots_against", (18.5, 20.5, 22.5, 24.5, 26.5, 28.5)),
    "team_shots_on_target": LineSpec("team_shots_on_target", "team", "shots_on_target_for", "shots_on_target_against", (2.5, 3.5, 4.5, 5.5, 6.5)),
    "shots_on_target": LineSpec("shots_on_target", "match", "shots_on_target_for", "shots_on_target_against", (5.5, 6.5, 7.5, 8.5, 9.5)),
    "team_yellow_cards": LineSpec("team_yellow_cards", "team", "yellow_cards_for", "yellow_cards_against", (0.5, 1.5, 2.5, 3.5)),
    "yellow_cards": LineSpec("yellow_cards", "match", "yellow_cards_for", "yellow_cards_against", (2.5, 3.5, 4.5, 5.5, 6.5)),
    "team_fouls": LineSpec("team_fouls", "team", "fouls_for", "fouls_against", (8.5, 10.5, 12.5, 14.5, 16.5)),
    "fouls": LineSpec("fouls", "match", "fouls_for", "fouls_against", (18.5, 22.5, 26.5, 30.5)),
    # Team/keeper saves. Football-Data can provide derived team saves. Provider/StatsBomb can provide real player saves.
    "goalkeeper_saves": LineSpec("goalkeeper_saves", "team", "saves_for", "saves_against", (1.5, 2.5, 3.5, 4.5, 5.5)),
}

GK_SAVE_LINES = (1.5, 2.5, 3.5, 4.5, 5.5)


def _safe_num(x: object) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _prob_over_poisson(mu: float, line: float) -> float:
    mu = max(0.05, min(60.0, float(mu)))
    threshold = int(math.floor(float(line)))
    if poisson is not None:
        return float(1.0 - poisson.cdf(threshold, mu))
    cdf = 0.0
    for k in range(0, threshold + 1):
        cdf += math.exp(-mu) * (mu ** k) / math.factorial(k)
    return max(0.0, min(1.0, 1.0 - cdf))


def _clip_expected(value: float, market: str) -> float:
    if not math.isfinite(value):
        return float("nan")
    caps = {
        "corners_for": 12.0,
        "shots_for": 35.0,
        "shots_on_target_for": 14.0,
        "yellow_cards_for": 8.0,
        "fouls_for": 30.0,
        "saves_for": 12.0,
        "saves": 12.0,
    }
    return max(0.02, min(float(value), caps.get(market, 60.0)))


def add_date_and_sort(stats: pd.DataFrame) -> pd.DataFrame:
    work = stats.copy()
    if "date" in work.columns:
        work["_date"] = pd.to_datetime(work["date"], errors="coerce")
    else:
        work["_date"] = pd.NaT
    sort_cols = [c for c in ["_date", "match_id", "is_home", "team"] if c in work.columns]
    return work.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def add_rolling_baselines(stats: pd.DataFrame, stats_cols: Iterable[str], window: int = 10, min_periods: int = 3) -> pd.DataFrame:
    """Add leakage-safe rolling baselines and relational expected values.

    Base expectation for team stat = average(team past production, opponent past allowance).
    v0.38 relational adjustments:
    - expected corners also uses expected shots and expected shots on target.
    - expected goalkeeper saves also uses opponent expected shots on target and goals against pressure.
    """
    work = add_date_and_sort(stats)
    stat_cols = [c for c in stats_cols if c in work.columns]
    for col in stat_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    for col in stat_cols:
        global_mean = float(work[col].dropna().mean()) if work[col].notna().any() else float("nan")
        roll_col = f"rolling_{col}_team"
        work[roll_col] = (
            work.groupby("team", dropna=False)[col]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).mean())
        ).fillna(global_mean)

    for col in [c for c in stat_cols if c.endswith("_for")]:
        against = col.replace("_for", "_against")
        own_for_col = f"rolling_{col}_team"
        opp_allowed_col = f"rolling_{col}_opponent_allowed"
        if against in stat_cols:
            opp_roll_col = f"rolling_{against}_team"
            lookup = work[["match_id", "team", opp_roll_col]].copy()
            lookup = lookup.rename(columns={"team": "opponent", opp_roll_col: opp_allowed_col})
            work = work.merge(lookup, on=["match_id", "opponent"], how="left")
            work[opp_allowed_col] = pd.to_numeric(work.get(opp_allowed_col), errors="coerce").fillna(work[own_for_col])
        else:
            work[opp_allowed_col] = work[own_for_col]
        work[f"rolling_{col}_team_for"] = work[own_for_col]
        work[f"expected_{col}_base"] = ((pd.to_numeric(work[own_for_col], errors="coerce") + pd.to_numeric(work[opp_allowed_col], errors="coerce")) / 2.0)
        work[f"expected_{col}"] = work[f"expected_{col}_base"].fillna(pd.to_numeric(work[own_for_col], errors="coerce"))
        work[f"expected_{col}_components"] = "team_for_rolling+opponent_allowed_rolling"

    # Corners are related to attacking volume, especially shots. Use this as a modest adjustment,
    # not as a replacement for real corner history.
    if "expected_corners_for" in work.columns:
        base = pd.to_numeric(work["expected_corners_for"], errors="coerce")
        comps = "team_corner_history+opponent_corner_allowed"
        if "expected_shots_for" in work.columns:
            shots = pd.to_numeric(work["expected_shots_for"], errors="coerce")
            # Typical corners/shots ratio is noisy; use global ratio from history when available.
            hist_ratio = np.nan
            if "corners_for" in work.columns and "shots_for" in work.columns:
                denom = pd.to_numeric(work["shots_for"], errors="coerce")
                num = pd.to_numeric(work["corners_for"], errors="coerce")
                ratio_series = (num / denom.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
                hist_ratio = float(ratio_series.clip(0.05, 0.75).median()) if not ratio_series.empty else np.nan
            if not math.isfinite(hist_ratio):
                hist_ratio = 0.28
            shot_implied = shots * hist_ratio
            work["expected_corners_for"] = (0.75 * base + 0.25 * shot_implied).fillna(base).map(lambda x: _clip_expected(x, "corners_for"))
            comps += "+shot_volume_adjustment"
        work["expected_corners_for_components"] = comps

    # Team goalkeeper saves are strongly related to opponent SOT pressure. If saves history exists,
    # blend it with opponent expected SOT minus expected goals-against pressure when available.
    if "expected_saves_for" in work.columns:
        base = pd.to_numeric(work["expected_saves_for"], errors="coerce")
        comps = "team_save_history+opponent_save_allowed"
        pressure = pd.Series(np.nan, index=work.index)
        if "expected_shots_on_target_against" in work.columns:
            pressure = pd.to_numeric(work["expected_shots_on_target_against"], errors="coerce")
        elif "expected_shots_on_target_for" in work.columns:
            # Merge opponent expected SOT for from same fixture.
            lookup = work[["match_id", "team", "expected_shots_on_target_for"]].rename(columns={"team": "opponent", "expected_shots_on_target_for": "opponent_expected_sot_for"})
            tmp = work[["match_id", "opponent"]].merge(lookup, on=["match_id", "opponent"], how="left")
            pressure = pd.to_numeric(tmp["opponent_expected_sot_for"], errors="coerce")
        if pressure.notna().any():
            # Not every SOT becomes a save. Goals reduce available saves. Without expected goals against,
            # use a conservative save share of opponent SOT.
            save_pressure = pressure * 0.68
            work["expected_saves_for"] = (0.65 * base + 0.35 * save_pressure).fillna(base).map(lambda x: _clip_expected(x, "saves_for"))
            comps += "+opponent_sot_pressure"
        work["expected_saves_for_components"] = comps
    return work.drop(columns=[c for c in ["_date"] if c in work.columns])


def _match_pair_rows(stats: pd.DataFrame, match_id: object) -> pd.DataFrame:
    g = stats[stats["match_id"].astype(str).eq(str(match_id))]
    if len(g) < 2:
        return pd.DataFrame()
    if "is_home" in g.columns and set(pd.to_numeric(g["is_home"], errors="coerce").dropna().astype(int).unique()) >= {0, 1}:
        home = g[pd.to_numeric(g["is_home"], errors="coerce").eq(1)].iloc[:1]
        away = g[pd.to_numeric(g["is_home"], errors="coerce").eq(0)].iloc[:1]
        if not home.empty and not away.empty:
            return pd.concat([home, away], ignore_index=True)
    return g.iloc[:2].reset_index(drop=True)


def _quality_weight(flag: object) -> str:
    txt = str(flag or "").lower()
    if "provider_player_goalkeeper_saves" in txt or "raw_event_goalkeeper_saves" in txt or "provider_saves_real" in txt:
        return "real_target"
    if "derived" in txt:
        return "derived_target"
    return "unknown_quality"


def _clean_text_flag(value: object) -> str:
    """Return a safe string flag for mixed CSV columns.

    Combined sources can leave NaN/float values in text columns.  Downstream joins must
    never receive floats, and we also do not want literal 'nan' flags polluting audits.
    """
    try:
        if value is None or pd.isna(value):
            return ""
    except Exception:
        pass
    txt = str(value).strip()
    return "" if txt.lower() in {"", "nan", "none", "nat"} else txt


def _join_unique_flags(values: object) -> str:
    if values is None:
        return ""
    if isinstance(values, pd.DataFrame):
        raw = values.to_numpy().ravel().tolist()
    elif isinstance(values, pd.Series):
        raw = values.tolist()
    elif isinstance(values, (list, tuple, set)):
        raw = list(values)
    else:
        raw = [values]
    cleaned = sorted({t for t in (_clean_text_flag(v) for v in raw) if t})
    return ";".join(cleaned)


def build_settled_event_line_signals(team_match_stats: pd.DataFrame, min_history: int = 3) -> pd.DataFrame:
    if team_match_stats is None or team_match_stats.empty:
        return pd.DataFrame()
    stats = team_match_stats.copy()
    stat_cols = sorted({spec.stat_for for spec in LINE_SPECS.values()} | {spec.stat_against for spec in LINE_SPECS.values() if spec.stat_against})
    for c in stat_cols:
        if c in stats.columns:
            stats[c] = pd.to_numeric(stats[c], errors="coerce")
    stats = add_rolling_baselines(stats, [c for c in stat_cols if c.endswith("_for") or c.endswith("_against")], min_periods=min_history)
    rows: list[dict] = []
    for _, r in stats.iterrows():
        common = {
            "match_id": str(r.get("match_id")),
            "date": r.get("date", ""),
            "home_team": r.get("home_team", ""),
            "away_team": r.get("away_team", ""),
            "team": r.get("team", ""),
            "player": "",
            "competition": r.get("competition", ""),
            "competition_context": r.get("competition_context", ""),
            "team_type": r.get("team_type", ""),
            "gender": r.get("gender", ""),
            "data_source": _clean_text_flag(r.get("data_source", "")),
            "data_quality_flag": _clean_text_flag(r.get("data_quality_flag", "")),
            "saves_data_quality_flag": _clean_text_flag(r.get("saves_data_quality_flag", "")),
            "target_quality": _quality_weight(r.get("saves_data_quality_flag") if pd.notna(r.get("saves_data_quality_flag", np.nan)) else r.get("data_quality_flag")),
        }
        for name, spec in LINE_SPECS.items():
            if spec.scope != "team":
                continue
            if spec.stat_for not in stats.columns or pd.isna(r.get(spec.stat_for)):
                continue
            actual = _safe_num(r.get(spec.stat_for))
            expected = _safe_num(r.get(f"expected_{spec.stat_for}"))
            if not math.isfinite(actual) or not math.isfinite(expected):
                continue
            expected = _clip_expected(expected, spec.stat_for)
            comps = r.get(f"expected_{spec.stat_for}_components", "team_for_rolling+opponent_allowed_rolling")
            for line in spec.lines:
                p_over = _prob_over_poisson(expected, line)
                for selection, p in [("over", p_over), ("under", 1.0 - p_over)]:
                    rows.append({
                        **common,
                        "market": name,
                        "scope": spec.scope,
                        "selection": selection,
                        "over_under": selection,
                        "line": float(line),
                        "model_probability": max(1e-6, min(1 - 1e-6, p)),
                        "fair_odds": 1.0 / max(1e-6, min(1 - 1e-6, p)),
                        "settled_stat": actual,
                        "actual_win": int(actual > line) if selection == "over" else int(actual < line),
                        "expected_stat": expected,
                        "expected_components": comps,
                        "model_family": "relational_rolling_market_model_v038",
                    })
    for mid in stats["match_id"].astype(str).dropna().unique():
        pair = _match_pair_rows(stats, mid)
        if pair.empty or len(pair) < 2:
            continue
        base = pair.iloc[0]
        common = {
            "match_id": str(mid),
            "date": base.get("date", ""),
            "home_team": base.get("home_team", ""),
            "away_team": base.get("away_team", ""),
            "team": "match total",
            "player": "",
            "competition": base.get("competition", ""),
            "competition_context": base.get("competition_context", ""),
            "team_type": base.get("team_type", ""),
            "gender": base.get("gender", ""),
            "data_source": _join_unique_flags(pair.get("data_source", pd.Series(dtype=str))),
            "data_quality_flag": _join_unique_flags(pair.get("data_quality_flag", pd.Series(dtype=str))),
            "saves_data_quality_flag": _join_unique_flags(pair.get("saves_data_quality_flag", pd.Series(dtype=str))) if "saves_data_quality_flag" in pair.columns else "",
            "target_quality": "match_total",
        }
        for name, spec in LINE_SPECS.items():
            if spec.scope != "match":
                continue
            if spec.stat_for not in pair.columns:
                continue
            vals = pd.to_numeric(pair[spec.stat_for], errors="coerce")
            if vals.notna().sum() < 2:
                continue
            actual = float(vals.sum())
            exp_col = f"expected_{spec.stat_for}"
            if exp_col not in pair.columns:
                continue
            exp_vals = pd.to_numeric(pair[exp_col], errors="coerce")
            if exp_vals.notna().sum() < 2:
                continue
            expected = _clip_expected(float(exp_vals.sum()), spec.stat_for)
            comp_col = f"expected_{spec.stat_for}_components"
            comps = _join_unique_flags(pair[comp_col]) if comp_col in pair.columns else "team_for_rolling+opponent_allowed_rolling"
            for line in spec.lines:
                p_over = _prob_over_poisson(expected, line)
                for selection, p in [("over", p_over), ("under", 1.0 - p_over)]:
                    rows.append({
                        **common,
                        "market": name,
                        "scope": spec.scope,
                        "selection": selection,
                        "over_under": selection,
                        "line": float(line),
                        "model_probability": max(1e-6, min(1 - 1e-6, p)),
                        "fair_odds": 1.0 / max(1e-6, min(1 - 1e-6, p)),
                        "settled_stat": actual,
                        "actual_win": int(actual > line) if selection == "over" else int(actual < line),
                        "expected_stat": expected,
                        "expected_components": comps,
                        "model_family": "relational_rolling_market_model_v038",
                    })
    return pd.DataFrame(rows)


def build_goalkeeper_save_line_signals(goalkeeper_match_stats: pd.DataFrame, team_match_stats: pd.DataFrame | None = None, min_history: int = 3) -> pd.DataFrame:
    """Build player goalkeeper-saves over/under line signals.

    Uses real player saves where available. Expected saves are leakage-safe blend of:
    player rolling saves, team rolling saves and opponent SOT pressure from team-match stats.

    Important implementation detail: StatsBomb match ids arrive as numeric ids while
    Football-Data/combined sources often use text ids. We normalise all merge keys to
    string before joining. Otherwise pandas may fail with int64-vs-object merge errors.
    """
    if goalkeeper_match_stats is None or goalkeeper_match_stats.empty:
        return pd.DataFrame()
    gk = goalkeeper_match_stats.copy()
    for key in ["match_id", "team", "opponent"]:
        if key in gk.columns:
            gk[key] = gk[key].astype(str)
    if "date" in gk.columns:
        gk["_date"] = pd.to_datetime(gk["date"], errors="coerce")
    else:
        gk["_date"] = pd.NaT
    for c in ["saves", "shots_on_target_against", "goals_against", "team_saves_total"]:
        if c in gk.columns:
            gk[c] = pd.to_numeric(gk[c], errors="coerce")
    gk = gk.sort_values(["_date", "match_id", "team", "goalkeeper"], kind="mergesort").reset_index(drop=True)
    global_save = float(gk["saves"].dropna().mean()) if gk["saves"].notna().any() else 2.5
    gk["rolling_player_saves"] = gk.groupby("goalkeeper", dropna=False)["saves"].transform(lambda s: s.shift(1).rolling(window=10, min_periods=min_history).mean()).fillna(global_save)
    gk["rolling_team_saves"] = gk.groupby("team", dropna=False)["saves"].transform(lambda s: s.shift(1).rolling(window=10, min_periods=min_history).mean()).fillna(gk["rolling_player_saves"])
    gk["opponent_sot_pressure"] = np.nan
    if team_match_stats is not None and not team_match_stats.empty:
        t = team_match_stats.copy()
        for key in ["match_id", "team", "opponent"]:
            if key in t.columns:
                t[key] = t[key].astype(str)
        for c in ["shots_on_target_for", "saves_for"]:
            if c in t.columns:
                t[c] = pd.to_numeric(t[c], errors="coerce")
        if "date" in t.columns:
            t["_date"] = pd.to_datetime(t["date"], errors="coerce")
        else:
            t["_date"] = pd.NaT
        t = t.sort_values(["_date", "match_id", "team"], kind="mergesort")
        if "shots_on_target_for" in t.columns:
            sot_global = float(t["shots_on_target_for"].dropna().mean()) if t["shots_on_target_for"].notna().any() else np.nan
            t["rolling_sot_for"] = t.groupby("team", dropna=False)["shots_on_target_for"].transform(lambda s: s.shift(1).rolling(window=10, min_periods=min_history).mean()).fillna(sot_global)
            lookup = t[["match_id", "team", "rolling_sot_for"]].rename(columns={"team": "opponent", "rolling_sot_for": "opponent_sot_pressure"})
            lookup["match_id"] = lookup["match_id"].astype(str)
            lookup["opponent"] = lookup["opponent"].astype(str)
            lookup = lookup.groupby(["match_id", "opponent"], as_index=False)["opponent_sot_pressure"].mean()
            gk = gk.drop(columns=["opponent_sot_pressure"], errors="ignore").merge(lookup, on=["match_id", "opponent"], how="left")
    pressure = pd.to_numeric(gk.get("opponent_sot_pressure"), errors="coerce")
    player = pd.to_numeric(gk["rolling_player_saves"], errors="coerce")
    team = pd.to_numeric(gk["rolling_team_saves"], errors="coerce")
    pressure_component = pressure * 0.68
    expected = (0.50 * player + 0.30 * team + 0.20 * pressure_component).where(pressure_component.notna(), 0.60 * player + 0.40 * team)
    gk["expected_saves"] = expected.fillna(player).map(lambda x: _clip_expected(x, "saves"))
    rows: list[dict] = []
    for _, r in gk.iterrows():
        actual = _safe_num(r.get("saves"))
        expected_saves = _safe_num(r.get("expected_saves"))
        if not math.isfinite(actual) or not math.isfinite(expected_saves):
            continue
        common = {
            "match_id": str(r.get("match_id")),
            "date": r.get("date", ""),
            "home_team": r.get("home_team", ""),
            "away_team": r.get("away_team", ""),
            "team": r.get("team", ""),
            "player": r.get("goalkeeper") or r.get("player", ""),
            "goalkeeper": r.get("goalkeeper") or r.get("player", ""),
            "competition": r.get("competition", ""),
            "data_source": _clean_text_flag(r.get("data_source", "")),
            "data_quality_flag": _clean_text_flag(r.get("data_quality_flag", "")),
            "saves_data_quality_flag": _clean_text_flag(r.get("saves_data_quality_flag", "")),
            "target_quality": _quality_weight(r.get("saves_data_quality_flag") or r.get("data_quality_flag")),
        }
        for line in GK_SAVE_LINES:
            p_over = _prob_over_poisson(expected_saves, line)
            for selection, p in [("over", p_over), ("under", 1.0 - p_over)]:
                rows.append({
                    **common,
                    "market": "goalkeeper_saves",
                    "scope": "player",
                    "selection": selection,
                    "over_under": selection,
                    "line": float(line),
                    "model_probability": max(1e-6, min(1 - 1e-6, p)),
                    "fair_odds": 1.0 / max(1e-6, min(1 - 1e-6, p)),
                    "settled_stat": actual,
                    "actual_win": int(actual > line) if selection == "over" else int(actual < line),
                    "expected_stat": expected_saves,
                    "expected_components": "player_save_history+team_save_history+opponent_sot_pressure",
                    "model_family": "goalkeeper_relational_saves_model_v038",
                })
    out = pd.DataFrame(rows)
    return out.drop(columns=["_date"], errors="ignore") if "_date" in out.columns else out
