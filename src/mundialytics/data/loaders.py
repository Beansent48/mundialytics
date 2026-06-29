from __future__ import annotations

import pandas as pd

def to_long_team_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert wide match rows into one row per team/match.

    Keeps all optional match statistics when present. Metrics are represented as
    ``<metric>_for`` and ``<metric>_against`` and later transformed into
    leakage-safe rolling features.
    """
    metric_pairs = [
        ("goals", "home_goals", "away_goals"),
        ("xg", "home_xg", "away_xg"),
        ("npxg", "home_npxg", "away_npxg"),
        ("xa", "home_xa", "away_xa"),
        ("npxg_plus_xa", "home_npxg_plus_xa", "away_npxg_plus_xa"),
        ("shots", "home_shots", "away_shots"),
        ("sot", "home_sot", "away_sot"),
        ("shots_inside_box", "home_shots_inside_box", "away_shots_inside_box"),
        ("shots_outside_box", "home_shots_outside_box", "away_shots_outside_box"),
        ("avg_shot_distance", "home_avg_shot_distance", "away_avg_shot_distance"),
        ("xg_per_shot", "home_xg_per_shot", "away_xg_per_shot"),
        ("header_shots", "home_header_shots", "away_header_shots"),
        ("left_foot_shots", "home_left_foot_shots", "away_left_foot_shots"),
        ("right_foot_shots", "home_right_foot_shots", "away_right_foot_shots"),
        ("header_xg", "home_header_xg", "away_header_xg"),
        ("left_foot_xg", "home_left_foot_xg", "away_left_foot_xg"),
        ("right_foot_xg", "home_right_foot_xg", "away_right_foot_xg"),
        ("penalty_xg", "home_penalty_xg", "away_penalty_xg"),
        ("open_play_xg", "home_open_play_xg", "away_open_play_xg"),
        ("set_piece_xg", "home_set_piece_xg", "away_set_piece_xg"),
        ("corner_xg", "home_corner_xg", "away_corner_xg"),
        ("free_kick_xg", "home_free_kick_xg", "away_free_kick_xg"),
        ("counterattack_xg", "home_counterattack_xg", "away_counterattack_xg"),
        ("big_chances", "home_big_chances", "away_big_chances"),
        ("corners", "home_corners", "away_corners"),
        ("fouls", "home_fouls", "away_fouls"),
        ("yellow_cards", "home_yellow_cards", "away_yellow_cards"),
        ("red_cards", "home_red_cards", "away_red_cards"),
        ("possession", "home_possession", "away_possession"),
        ("field_tilt", "home_field_tilt", "away_field_tilt"),
        ("ppda", "home_ppda", "away_ppda"),
        ("touches_attacking_third", "home_touches_attacking_third", "away_touches_attacking_third"),
        ("touches_box", "home_touches_box", "away_touches_box"),
        ("final_third_entries", "home_final_third_entries", "away_final_third_entries"),
        ("deep_completions", "home_deep_completions", "away_deep_completions"),
        ("progressive_passes", "home_progressive_passes", "away_progressive_passes"),
        ("progressive_carries", "home_progressive_carries", "away_progressive_carries"),
        ("passes_into_final_third", "home_passes_into_final_third", "away_passes_into_final_third"),
        ("passes_into_penalty_area", "home_passes_into_penalty_area", "away_passes_into_penalty_area"),
        ("crosses_into_penalty_area", "home_crosses_into_penalty_area", "away_crosses_into_penalty_area"),
        ("through_balls", "home_through_balls", "away_through_balls"),
        ("shot_creating_actions", "home_shot_creating_actions", "away_shot_creating_actions"),
        ("goal_creating_actions", "home_goal_creating_actions", "away_goal_creating_actions"),
        ("tackles", "home_tackles", "away_tackles"),
        ("interceptions", "home_interceptions", "away_interceptions"),
        ("blocks", "home_blocks", "away_blocks"),
        ("clearances", "home_clearances", "away_clearances"),
        ("pressures", "home_pressures", "away_pressures"),
        ("ball_recoveries", "home_ball_recoveries", "away_ball_recoveries"),
        ("errors", "home_errors", "away_errors"),
        ("keeper_saves", "home_keeper_saves", "away_keeper_saves"),
        ("keeper_psxg", "home_keeper_psxg", "away_keeper_psxg"),
        ("keeper_goals_against", "home_keeper_goals_against", "away_keeper_goals_against"),
        ("keeper_save_pct", "home_keeper_save_pct", "away_keeper_save_pct"),
    ]

    rows = []
    for _, r in matches.iterrows():
        base = {
            "match_id": r.get("match_id"),
            "date": r.get("date"),
            "competition": r.get("competition", "unknown"),
            "season": r.get("season", "unknown"),
            "stage": r.get("stage", "unknown"),
            "team_scope": r.get("team_scope", "unknown"),
            "neutral": r.get("neutral", 0),
        }
        for side, team_col, opp_col, is_home in [
            ("home", "home_team", "away_team", 1),
            ("away", "away_team", "home_team", 0),
        ]:
            d = {**base, "team": r.get(team_col), "opponent": r.get(opp_col), "is_home": is_home}
            for name, hcol, acol in metric_pairs:
                own = hcol if side == "home" else acol
                opp = acol if side == "home" else hcol
                d[f"{name}_for"] = r.get(own)
                d[f"{name}_against"] = r.get(opp)
            rows.append(d)
    out = pd.DataFrame(rows)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def load_matches(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df
