"""
Fitted models for the HTTP API.

The configuration below is a *copy* of the one the Streamlit app serves, down to
every keyword. That is deliberate and it matters: two front ends quoting
different probabilities for the same fixture would make the track record
meaningless, and the benchmark on the landing page would stop describing what
visitors actually see. If the deployed chain ever changes, it changes in both
places in the same commit.

The joblib cache is shared with the Streamlit app on purpose — same directory,
same key — so whichever process starts first pays the ~3 minute fit and the
other one loads it in under a second.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data/processed/cache"


def _code_fingerprint(*rel_paths: str) -> str:
    """8-char hash of the given sources, folded into the cache key so a fitted
    model self-invalidates when the code behind it changes."""
    h = hashlib.sha256()
    for rel in sorted(rel_paths):
        p = ROOT / rel
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"missing")
    return h.hexdigest()[:8]


ENGINE_FP = _code_fingerprint(
    "src/mundialytics/statistical_core/prediction_engine.py",
    "src/mundialytics/statistical_core/attack_defense_model.py",
    "src/mundialytics/statistical_core/distributions.py",
    "src/mundialytics/models/goal_model.py",
    "src/mundialytics/models/xg_rate_model.py",
    "src/mundialytics/ratings/elo.py",
)
PROPS_FP = _code_fingerprint(
    "src/mundialytics/props/team_props.py",
    "src/mundialytics/props/player_props.py",
    # the roster half of the fitted model lives here: change how a squad member
    # is matched to his history and the cached fit is wrong, not merely old
    "src/mundialytics/identity/current_squads.py",
)


def _cached_fit(tag: str, df: pd.DataFrame, build):
    import joblib

    key = f"{tag}_{len(df)}_{str(df['date'].max())[:10]}_{ENGINE_FP}"
    cache_f = CACHE_DIR / f"engine_{key}.joblib"
    if cache_f.exists():
        try:
            return joblib.load(cache_f)
        except Exception:
            pass
    engine = build(df)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(engine, cache_f, compress=3)
        for old in CACHE_DIR.glob(f"engine_{tag}_*.joblib"):
            if old != cache_f:
                old.unlink(missing_ok=True)
    except Exception:
        pass
    return engine


@lru_cache(maxsize=1)
def club_engine():
    from mundialytics.ratings.elo import EloConfig, EloRater
    from mundialytics.statistical_core.engine_utils import load_clubs_data
    from mundialytics.statistical_core.prediction_engine import (
        DEPLOYED_CLUB_ENGINE_KWARGS, PredictionEngine)

    df = load_clubs_data()

    def build(d):
        elo = EloRater(EloConfig(season_reset_fraction=0.40))
        elo.fit(d)
        # Deployed club config is the single source of truth in prediction_engine;
        # never inline the kwargs here (they used to drift between call sites).
        eng = PredictionEngine(**DEPLOYED_CLUB_ENGINE_KWARGS)
        eng.fit(d, elo_history=pd.DataFrame(elo.history))
        return eng

    engine = _cached_fit("clubs", df, build)
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    return engine, teams, df


@lru_cache(maxsize=1)
def props_models():
    """Team-event and player-prop models. Fails soft: (None, None)."""
    import joblib

    from mundialytics.identity.current_squads import load_current_squads, squads_fingerprint

    _, _, df = club_engine()
    squads = load_current_squads()
    key = (f"{len(df)}_{str(df['date'].max())[:10]}_{PROPS_FP}"
           f"_{squads_fingerprint(squads)}")
    cache_f = CACHE_DIR / f"props_models_{key}.joblib"
    if cache_f.exists():
        try:
            return joblib.load(cache_f)
        except Exception:
            pass
    try:
        from mundialytics.props import PlayerPropsModel, TeamPropsModel

        tp = TeamPropsModel().fit(df, root=ROOT)
    except Exception:
        return None, None
    try:
        pmp = ROOT / "data/external/advanced/understat/understat_player_match.csv"
        tmx = (
            pd.read_csv(ROOT / "data/processed/understat_team_match_xg.csv")
            [["provider_match_id", "date"]]
            .rename(columns={"provider_match_id": "game_id"})
            .drop_duplicates("game_id")
        )
        pm = pd.read_csv(pmp).merge(tmx, on="game_id", how="left")
        # `current_squads` decides only WHO is in each roster. Without it the
        # squad is "whoever featured in this club's last ten Understat games",
        # which for a promoted club reaches back to its last top-flight season.
        pp = PlayerPropsModel().fit(
            pm,
            shots_path=ROOT / "data/external/advanced/understat/understat_shots.csv",
            current_squads=squads,
        )
    except Exception:
        pp = None
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump((tp, pp), cache_f, compress=3)
    except Exception:
        pass
    return tp, pp
