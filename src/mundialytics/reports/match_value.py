from __future__ import annotations

import pandas as pd

from mundialytics.betting.odds import add_implied_probabilities
from mundialytics.betting.value import add_value_columns
from mundialytics.identity.normalization import canonical_team_name


_MATCH_SELECTION_ALIASES = {
    "home": "home",
    "h": "home",
    "1": "home",
    "draw": "draw",
    "d": "draw",
    "x": "draw",
    "away": "away",
    "a": "away",
    "2": "away",
}


def _fixture_key_col(df: pd.DataFrame) -> str:
    if "fixture_id" in df.columns:
        return "fixture_id"
    if "match_id" in df.columns:
        return "match_id"
    raise ValueError("Expected fixture_id or match_id column.")


def _selection_probability(row: pd.Series) -> tuple[str, float]:
    raw = str(row.get("selection", "")).strip()
    normalized = canonical_team_name(raw)
    alias = _MATCH_SELECTION_ALIASES.get(normalized)
    if alias is None:
        if normalized == canonical_team_name(row["home_team"]):
            alias = "home"
        elif normalized == canonical_team_name(row["away_team"]):
            alias = "away"
        else:
            raise ValueError(
                f"Could not map match-winner selection {raw!r}. Use home/draw/away or exact team names."
            )
    if alias == "home":
        return alias, float(row["p_home_win"])
    if alias == "draw":
        return alias, float(row["p_draw"])
    return alias, float(row["p_away_win"])


def build_match_value_picks(
    fixture_predictions: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    min_edge: float = 0.03,
    min_ev: float = 0.03,
    commission: float = 0.0,
) -> pd.DataFrame:
    """Compare 1X2 fixture probabilities against decimal odds.

    ``odds`` should contain fixture_id or match_id, bookmaker, market_type,
    selection and odds. Selections can be home/draw/away, 1/X/2, H/D/A, or
    the canonical home/away team names. The returned dataframe is safe for the
    paper ledger because it includes selection, model probability, implied
    probability, edge and expected return.
    """
    pred = fixture_predictions.copy()
    odd = odds.copy()
    pred_key = _fixture_key_col(pred)
    odd_key = _fixture_key_col(odd)
    if pred_key != odd_key:
        # Rename odds key to prediction key for common fixture-vs-match usage.
        odd = odd.rename(columns={odd_key: pred_key})
    required = {pred_key, "bookmaker", "market_type", "selection", "odds"}
    missing = required - set(odd.columns)
    if missing:
        raise ValueError(f"Odds dataframe missing columns: {sorted(missing)}")
    odd = odd[odd["market_type"].astype(str).str.lower().isin(["match_winner", "1x2", "result"])].copy()
    if odd.empty:
        return pd.DataFrame()

    # Odds extracts can carry descriptive prediction-side columns such as
    # home_team/away_team/date/competition. The prediction dataframe is the
    # canonical source for those fields during value evaluation; keeping odds
    # copies would make pandas suffix them during merge and remove the expected
    # canonical names.
    prediction_side_cols = {
        "date",
        "competition",
        "home_team",
        "away_team",
        "team_scope",
        "p_home_win",
        "p_draw",
        "p_away_win",
        "lambda_home",
        "lambda_away",
        "most_likely_score",
    }
    odd = odd.drop(columns=[c for c in prediction_side_cols if c in odd.columns], errors="ignore")

    for col, default in {
        "date": None,
        "competition": "unknown",
        "team_scope": "unknown",
        "lambda_home": float("nan"),
        "lambda_away": float("nan"),
        "most_likely_score": "unknown",
    }.items():
        if col not in pred.columns:
            pred[col] = default
    required_pred_cols = [
        pred_key, "date", "competition", "home_team", "away_team", "team_scope",
        "p_home_win", "p_draw", "p_away_win", "lambda_home", "lambda_away", "most_likely_score",
    ]
    missing_pred = set(required_pred_cols) - set(pred.columns)
    if missing_pred:
        raise ValueError(f"Prediction dataframe missing columns: {sorted(missing_pred)}")
    # Historical backtests usually predict only the evaluation window after the
    # training period, while odds extracts may contain the full season. Keep only
    # odds rows that have a corresponding prediction snapshot.
    pred_keys = set(pred[pred_key].dropna().astype(str))
    odd = odd[odd[pred_key].astype(str).isin(pred_keys)].copy()
    if odd.empty:
        return pd.DataFrame()

    merged = odd.merge(
        pred[required_pred_cols],
        on=pred_key,
        how="left",
        validate="many_to_one",
    )
    if merged["home_team"].isna().any():
        missing_keys = merged.loc[merged["home_team"].isna(), pred_key].drop_duplicates().head(10).tolist()
        raise ValueError(f"Odds/prediction merge failed after filtering. Examples: {missing_keys}")
    mapped = merged.apply(_selection_probability, axis=1, result_type="expand")
    merged["selection_type"] = mapped[0]
    merged["model_probability"] = mapped[1].astype(float)
    group_cols = [pred_key, "bookmaker", "market_type"]
    merged = add_implied_probabilities(merged, group_cols=group_cols)
    merged["sample_size"] = merged.get("sample_size", pd.NA)
    merged["market_prior"] = merged["selection_type"].map({"home": 0.45, "draw": 0.27, "away": 0.28}).fillna(0.33)
    merged["shrink_strength"] = 0.0  # 1X2 model probabilities already come from match model; do not prop-shrink by default.
    merged = add_value_columns(merged, min_edge=min_edge, min_ev=min_ev, commission=commission)
    merged["reason"] = merged.apply(
        lambda r: (
            f"1X2 model from ELO+Poisson/Skellam: {r['home_team']} λ={r['lambda_home']:.2f}, "
            f"{r['away_team']} λ={r['lambda_away']:.2f}, top score {r['most_likely_score']}"
        ),
        axis=1,
    )
    cols = [
        pred_key, "date", "competition", "team_scope", "bookmaker", "market_type", "selection", "selection_type",
        "home_team", "away_team", "odds", "model_probability", "implied_probability_raw", "book_overround",
        "implied_probability", "edge", "expected_return", "value_flag", "reason",
    ]
    return merged[[c for c in cols if c in merged.columns]].sort_values(
        ["value_flag", "expected_return", "edge"], ascending=False
    ).reset_index(drop=True)
