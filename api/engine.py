"""
Fitted models for the HTTP API.

The club engine is built from `DEPLOYED_CLUB_ENGINE_KWARGS`, the same dict the
pre-kickoff logger and the walk-forward benchmark use. That matters: a site
quoting different probabilities from the ones logged and benchmarked would make
the track record meaningless, and the benchmark on the landing page would stop
describing what visitors actually see.

Fits are cached with joblib under data/processed/cache, keyed on the data and a
hash of the engine code, so only the first start after a change pays the
~3 minute fit.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CACHE_DIR = ROOT / "data/processed/cache"


from mundialytics.serving.provenance import code_fingerprint as _code_fingerprint  # noqa: E402
from mundialytics.serving.provenance import engine_code_fingerprint  # noqa: E402

# One definition, shared with the pre-kickoff logger (serving/provenance.py).
ENGINE_FP = engine_code_fingerprint()
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


FOUNDATION = ROOT / "data/processed/foundation_big5_multi_season.csv"


def data_version() -> float:
    """The foundation's modification time: changes when a refresh lands.

    The fitted models below used to be cached for the life of the process, so a
    running API kept serving the data it started with, and the daily refresh
    (plus the web's cache revalidation) reached the site only after someone
    restarted uvicorn. Keying the caches on this makes the next request after a
    refresh pick the new data up; the joblib fit cache, warmed by the refresh
    itself, keeps that request from paying for a full refit.
    """
    try:
        return FOUNDATION.stat().st_mtime
    except OSError:
        return 0.0


def club_engine():
    return _club_engine(data_version())


@lru_cache(maxsize=1)
def _club_engine(_version: float):
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


def props_models():
    """Team-event and player-prop models. Fails soft: (None, None)."""
    return _props_models(data_version())


@lru_cache(maxsize=1)
def _props_models(_version: float):
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
