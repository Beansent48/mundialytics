from __future__ import annotations

import pandas as pd

from mundialytics.betting.market_mapper import is_player_market
from mundialytics.betting.odds import add_implied_probabilities
from mundialytics.betting.value import add_value_columns
from mundialytics.models.minutes_model import MinutesModel
from mundialytics.models.player_event_model import PlayerEventModel
from mundialytics.models.substitute_plus import SubstitutePlusModel


def build_daily_player_props(
    odds: pd.DataFrame,
    player_model: PlayerEventModel,
    minutes_model: MinutesModel,
    lineups: pd.DataFrame | None = None,
    team_context_by_match_team: dict | None = None,
    min_edge: float = 0.03,
    min_ev: float = 0.03,
    commission: float = 0.0,
) -> pd.DataFrame:
    rows = []
    sub_model = SubstitutePlusModel(player_model, lineups=lineups)
    for _, o in odds.iterrows():
        if not is_player_market(o["market_type"]):
            continue
        match_id = int(o["match_id"])
        player = o["player"]
        team = o.get("team")
        ctx = (team_context_by_match_team or {}).get((match_id, team), {})
        minutes = minutes_model.estimate(player, match_id=match_id)
        pred = player_model.predict_market(
            player=player,
            market_type=o["market_type"],
            line=o["line"],
            expected_minutes=minutes["expected_minutes"],
            team_context=ctx,
        )
        p = pred.probability
        sub_note = None
        replacement = None
        if int(o.get("substitute_plus", 0) or 0) == 1:
            adjusted = sub_model.apply(pred, match_id=match_id, team_context=ctx)
            p = adjusted["probability_substitute_plus"]
            sub_note = adjusted["replacement_note"]
            replacement = adjusted["replacement"]
        rows.append({
            "match_id": match_id,
            "bookmaker": o.get("bookmaker", "unknown"),
            "market_type": o["market_type"],
            "team": team,
            "player": player,
            "line": o["line"],
            "odds": float(o["odds"]),
            "substitute_plus": int(o.get("substitute_plus", 0) or 0),
            "model_probability": float(p),
            "expected_count": pred.expected_count,
            "expected_minutes": pred.expected_minutes,
            "sample_size": pred.sample_size,
            "market_prior": 0.50,
            "shrink_strength": 180.0,
            "replacement": replacement,
            "reason": pred.explanation + (f" | {sub_note}" if sub_note else ""),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = add_implied_probabilities(df)
    df = add_value_columns(df, min_edge=min_edge, min_ev=min_ev, commission=commission)
    return df.sort_values(["value_flag", "expected_return", "edge"], ascending=False).reset_index(drop=True)


def format_pick_text(row: pd.Series) -> str:
    return (
        f"{row['player']} {row['market_type']} {row['line']} @ {row['odds']:.2f} | "
        f"model_raw={row['model_probability']:.1%}, model_adj={row.get('model_probability_adjusted', row['model_probability']):.1%}, "
        f"implied={row['implied_probability']:.1%}, edge={row['edge']:.1%}, EV={row['expected_return']:.1%}"
    )
