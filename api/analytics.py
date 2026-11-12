"""
Analytics endpoints for the TFG data-science layer.

Exposes the model's internal data trail — lambda decomposition, prediction
errors, Elo trajectory, attack/defense profiles — so the web front end can
render charts that explain *how* the engine works, not just what it predicts.
"""
from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

COMPONENTS = ROOT / "data/processed/enriched/understat_xg/global_components.csv"
WALKFORWARD = ROOT / "data/processed/enriched/understat_xg/walkforward_preds_deployed.csv"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _rps(ph: np.ndarray, pd_: np.ndarray, pa: np.ndarray,
         hg: np.ndarray, ag: np.ndarray) -> np.ndarray:
    """Ranked probability score per match."""
    y_home = (hg > ag).astype(float)
    y_draw = (hg == ag).astype(float)
    cum_p = np.column_stack([ph, ph + pd_])
    cum_y = np.column_stack([y_home, y_home + y_draw])
    return ((cum_p - cum_y) ** 2).sum(axis=1) / 2


def _poisson_pmf(lam: float, k: int) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ── Elo (re-fit, cached) ────────────────────────────────────────────────────


_ELO_CACHE: tuple[float, pd.DataFrame] | None = None


def _elo_history(version: float, df: pd.DataFrame) -> pd.DataFrame:
    global _ELO_CACHE
    if _ELO_CACHE is not None and _ELO_CACHE[0] == version:
        return _ELO_CACHE[1]
    from mundialytics.ratings.elo import EloConfig, EloRater

    elo = EloRater(EloConfig(season_reset_fraction=0.40))
    result = elo.fit(df)
    _ELO_CACHE = (version, result)
    return result


# ── Decomposition ───────────────────────────────────────────────────────────


def decomposition(team: str, df: pd.DataFrame) -> dict:
    """Per-match lambda decomposition for one team."""
    if not COMPONENTS.exists():
        return {"team": team, "matches": []}

    comp = pd.read_csv(COMPONENTS)
    merged = comp.merge(
        df[["match_id", "home_team", "away_team", "date"]].drop_duplicates("match_id"),
        on="match_id", how="inner",
    )

    is_home = merged["home_team"].str.lower() == team.lower()
    is_away = merged["away_team"].str.lower() == team.lower()
    t = merged[is_home | is_away].copy()
    if t.empty:
        return {"team": team, "matches": []}

    rows = []
    for _, r in t.sort_values("date").iterrows():
        home = r["home_team"].lower() == team.lower()
        suffix = "_h" if home else "_a"
        opp_suffix = "_a" if home else "_h"
        rows.append({
            "date": str(r["date"])[:10],
            "opponent": r["away_team"] if home else r["home_team"],
            "venue": "home" if home else "away",
            "goals": int(r["hg"] if home else r["ag"]),
            "conceded": int(r["ag"] if home else r["hg"]),
            "gl": round(float(r[f"gl{suffix}"]), 4),
            "ad": round(float(r[f"ad{suffix}"]), 4),
            "adx": round(float(r[f"adx{suffix}"]), 4),
            "xr": round(float(r[f"xr{suffix}"]), 4),
        })

    return {"team": team, "matches": rows}


# ── Prediction errors ────────────────────────────────────────────────────────


def prediction_errors(df: pd.DataFrame) -> dict:
    """RPS breakdown and Poisson goodness-of-fit from walk-forward predictions."""
    if not WALKFORWARD.exists():
        return {"total_rps": None, "by_season": [], "by_league": [],
                "by_tier": [], "scoreline_gof": None}

    w = pd.read_csv(WALKFORWARD)
    merged = w.merge(
        df[["match_id", "home_team", "away_team", "competition"]].drop_duplicates("match_id"),
        on="match_id", how="inner",
    )

    rps = _rps(
        merged["ph"].to_numpy(), merged["pd"].to_numpy(), merged["pa"].to_numpy(),
        merged["hg"].to_numpy(), merged["ag"].to_numpy(),
    )
    merged["rps"] = rps

    total_rps = round(float(rps.mean()), 6)

    # By season
    by_season = (
        merged.groupby("season")
        .agg(rps=("rps", "mean"), n=("rps", "size"))
        .reset_index()
        .sort_values("season")
    )
    by_season_list = [
        {"season": r.season, "rps": round(float(r.rps), 6), "n": int(r.n)}
        for r in by_season.itertuples()
    ]

    # By league
    by_league = (
        merged.groupby("competition")
        .agg(rps=("rps", "mean"), n=("rps", "size"))
        .reset_index()
        .sort_values("rps")
    )
    by_league_list = [
        {"league": str(r.competition), "rps": round(float(r.rps), 6), "n": int(r.n)}
        for r in by_league.itertuples()
    ]

    # By strength tier (tercile of total lambda)
    merged["total_lambda"] = merged["lh"] + merged["la"]
    merged["tier"] = pd.qcut(merged["total_lambda"], 3, labels=["low", "mid", "high"])
    by_tier = (
        merged.groupby("tier", observed=True)
        .agg(rps=("rps", "mean"), n=("rps", "size"))
        .reset_index()
    )
    by_tier_list = [
        {"tier": str(r.tier), "rps": round(float(r.rps), 6), "n": int(r.n)}
        for r in by_tier.itertuples()
    ]

    # Poisson GOF on scorelines
    max_g = 6
    lh = merged["lh"].to_numpy()
    la = merged["la"].to_numpy()
    hg = merged["hg"].to_numpy().astype(int)
    ag = merged["ag"].to_numpy().astype(int)
    n = len(merged)

    gof_bins = []
    chi2 = 0.0
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            observed = int(((hg == h) & (ag == a)).sum())
            # expected probability from the model's lambdas
            expected_prob = float(np.mean([
                _poisson_pmf(lh_i, h) * _poisson_pmf(la_i, a)
                for lh_i, la_i in zip(lh, la)
            ]))
            expected = expected_prob * n
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected
            gof_bins.append({
                "score": f"{h}-{a}",
                "observed": observed,
                "expected": round(expected, 1),
            })

    df_gof = (max_g + 1) ** 2 - 1
    from scipy.stats import chi2 as chi2_dist
    p_value = float(1 - chi2_dist.cdf(chi2, df_gof))

    # Keep only scorelines with at least 5 expected for the summary
    scoreline_gof = {
        "chi2": round(chi2, 2),
        "p": round(p_value, 6),
        "df": df_gof,
        "bins": sorted(gof_bins, key=lambda x: x["expected"], reverse=True)[:20],
    }

    return {
        "total_rps": total_rps,
        "n": n,
        "by_season": by_season_list,
        "by_league": by_league_list,
        "by_tier": by_tier_list,
        "scoreline_gof": scoreline_gof,
    }


# ── Team profile ─────────────────────────────────────────────────────────────


def team_profile(team: str, engine, df: pd.DataFrame, version: float) -> dict:
    """Elo trajectory + attack/defense radar + head-to-head for one team."""
    from mundialytics.statistical_core.schemas import canonical_name

    # Elo trajectory
    elo_df = _elo_history(version, df)
    cn = canonical_name(team)

    is_home = elo_df["home_team"].apply(canonical_name) == cn
    is_away = elo_df["away_team"].apply(canonical_name) == cn

    trajectory = []
    for _, r in elo_df[is_home | is_away].iterrows():
        home = canonical_name(r["home_team"]) == cn
        trajectory.append({
            "date": str(r["date"])[:10],
            "elo": round(float(r["home_elo_post"] if home else r["away_elo_post"]), 1),
            "opponent": r["away_team"] if home else r["home_team"],
            "venue": "home" if home else "away",
            "goals": int(r["home_goals"] if home else r["away_goals"]),
            "conceded": int(r["away_goals"] if home else r["home_goals"]),
        })

    current_elo = trajectory[-1]["elo"] if trajectory else None
    peak_elo = max((t["elo"] for t in trajectory), default=None)

    # Attack/defense profile from the AD model
    profile = None
    if engine.ad_model_ is not None:
        params = engine.ad_model_.team_params()
        team_row = params[params["team"].apply(canonical_name) == cn]
        if not team_row.empty:
            r = team_row.iloc[0]
            comp_teams = set(
                df.loc[df["competition"] == df.loc[
                    (df["home_team"].apply(canonical_name) == cn) |
                    (df["away_team"].apply(canonical_name) == cn),
                    "competition"
                ].mode().iloc[0] if len(df) else "", "home_team"].unique()
            ) if len(df) else set()
            league_params = params[params["team"].isin(comp_teams)] if comp_teams else params
            profile = {
                "attack": round(float(r["attack"]), 4),
                "defense": round(float(r["defense"]), 4),
                "strength": round(float(r["strength"]), 4),
                "attack_balance": round(float(r["attack_balance"]), 4),
                "league_avg_attack": round(float(league_params["attack"].mean()), 4),
                "league_avg_defense": round(float(league_params["defense"].mean()), 4),
                "league_max_strength": round(float(league_params["strength"].max()), 4),
                "league_min_strength": round(float(league_params["strength"].min()), 4),
            }

    # Head-to-head: top 5 most-played rivals, last 10 each
    is_h = df["home_team"].apply(canonical_name) == cn
    is_a = df["away_team"].apply(canonical_name) == cn
    team_matches = df[is_h | is_a].copy()
    team_matches["opponent"] = np.where(is_h[team_matches.index], df.loc[team_matches.index, "away_team"], df.loc[team_matches.index, "home_team"])
    rival_counts = team_matches["opponent"].value_counts()
    top_rivals = rival_counts.head(5).index.tolist()

    h2h = []
    for rival in top_rivals:
        rival_df = team_matches[team_matches["opponent"] == rival].sort_values("date").tail(10)
        matches = []
        for _, m in rival_df.iterrows():
            home = canonical_name(m["home_team"]) == cn
            matches.append({
                "date": str(m["date"])[:10],
                "venue": "home" if home else "away",
                "goals": int(m["home_goals"] if home else m["away_goals"]),
                "conceded": int(m["away_goals"] if home else m["home_goals"]),
            })
        h2h.append({
            "rival": rival,
            "total": int(rival_counts[rival]),
            "matches": matches,
        })

    return {
        "team": team,
        "elo": {
            "current": current_elo,
            "peak": peak_elo,
            "trajectory": trajectory,
        },
        "profile": profile,
        "h2h": h2h,
    }


# ── Enhanced calibration ─────────────────────────────────────────────────────


def enhanced_calibration() -> dict | None:
    """20-bin calibration with Wilson CIs and Brier decomposition."""
    if not WALKFORWARD.exists():
        return None

    w = pd.read_csv(WALKFORWARD)
    outcome = np.where(w.hg > w.ag, "home", np.where(w.hg < w.ag, "away", "draw"))
    y = np.concatenate([
        (outcome == "home").astype(float),
        (outcome == "draw").astype(float),
        (outcome == "away").astype(float),
    ])
    pr = np.concatenate([
        w["ph"].to_numpy(float), w["pd"].to_numpy(float), w["pa"].to_numpy(float),
    ])

    # 20 bins
    bins = []
    for lo in np.linspace(0, 0.95, 20):
        hi = lo + 0.05
        mask = (pr >= lo) & (pr < hi)
        n = int(mask.sum())
        if n < 50:
            continue
        announced = float(pr[mask].mean())
        observed = float(y[mask].mean())
        ci_lo, ci_hi = _wilson_ci(observed, n)
        bins.append({
            "announced": round(announced, 4),
            "observed": round(observed, 4),
            "n": n,
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
        })

    # Brier score decomposition
    n_total = len(y)
    o_bar = float(y.mean())
    brier = float(np.mean((pr - y) ** 2))

    # Reliability: weighted average (forecast - observed_in_bin)^2
    reliability = 0.0
    resolution = 0.0
    for b in bins:
        w_k = b["n"] / n_total
        reliability += w_k * (b["announced"] - b["observed"]) ** 2
        resolution += w_k * (b["observed"] - o_bar) ** 2
    uncertainty = o_bar * (1 - o_bar)

    return {
        "bins": bins,
        "brier": round(brier, 6),
        "reliability": round(reliability, 6),
        "resolution": round(resolution, 6),
        "uncertainty": round(uncertainty, 6),
        "n": n_total,
    }
