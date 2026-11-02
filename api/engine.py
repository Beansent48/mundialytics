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
LEVEL_FP = _code_fingerprint("src/mundialytics/features/market_level.py")
PROPS_FP = _code_fingerprint(
    "src/mundialytics/props/team_props.py",
    "src/mundialytics/props/player_props.py",
    # the roster half of the fitted model lives here: change how a squad member
    # is matched to his history and the cached fit is wrong, not merely old
    "src/mundialytics/identity/current_squads.py",
)


def _xg_digest(df: pd.DataFrame) -> str:
    """Cheap content fingerprint of the xG columns.

    The key below used to be row count + last date + code + squad values, none of
    which move when xG VALUES change while the row count and the last date stay
    put -- a provider re-score, a backfill, or the 2026-10-17 provider-level
    harmonisation. The cached fit was then reloaded from disk and the new numbers
    never reached a single request. O(n) over 46k rows, microseconds.
    """
    cols = [c for c in ("home_xg", "away_xg") if c in df.columns]
    if not cols:
        return "noxg"
    v = df[cols]
    return f"{int(v.notna().to_numpy().sum())}x{round(float(v.sum(numeric_only=True).sum()), 1)}"


def _cached_fit(tag: str, df: pd.DataFrame, build):
    import joblib

    # the squad values are read at fit time, so a new snapshot must refit
    sv = ROOT / "data/processed/squad_values.csv"
    sv_v = int(sv.stat().st_mtime) if sv.exists() else 0
    key = f"{tag}_{len(df)}_{str(df['date'].max())[:10]}_{ENGINE_FP}_{sv_v}_{_xg_digest(df)}"
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
# everything market_level_detail reads besides the foundation
LEVEL_INPUTS = (ROOT / "data/processed/served_lambdas_current.csv",
                ROOT / "data/processed/logs/predictions_log.csv")
FOOTBALL_DATA = ROOT / "data/raw/football_data"


def _stamp(path: Path) -> str:
    try:
        st = path.stat()
        return f"{int(st.st_mtime)}-{st.st_size}"
    except OSError:
        return "0"


def _dir_stamp(directory: Path, pattern: str) -> str:
    """One short hash of a directory's listing: names, sizes and mtimes.

    Stats rather than contents -- a hundred odds files are tens of megabytes to
    read and a few hundred microseconds to stat, and a changed file changes its
    size or its mtime either way.
    """
    import hashlib

    h = hashlib.sha256()
    for p in sorted(directory.glob(pattern)) if directory.exists() else []:
        h.update(f"{p.name}:{_stamp(p)}|".encode())
    return h.hexdigest()[:8]


def _market_levels(df: pd.DataFrame, k: float) -> dict[str, tuple[int, float]]:
    """`market_level_detail`, memoised on disk. Same numbers, 28 seconds sooner.

    The level is deliberately NOT baked into the engine's joblib (see the note at
    the call site: a level frozen into a fit would be served long after the
    matches it was measured on), so it was recomputed on every process start and
    after every refresh -- and it is not cheap: it re-prices every logged lambda
    of the season against the book's over-2.5 line, fourteen Newton steps each.
    Measured at 28.5 s against 0.8 s for the 29 MB joblib load, which made it
    the single biggest item in the cold start, invisible in the first diagnosis
    because that timed `joblib.load` rather than `club_engine()`.

    The key is everything the function reads: its own source, k, the
    foundation's size and last date, the two lambda logs and every football-data
    file. A new result, a newly logged lambda or an edit to the estimator all
    recompute it; nothing else does. The daily refresh warms it on the way past
    (update_season step 7m calls the endpoints in-process), so the API start
    that follows reads it off disk.
    """
    import json

    from mundialytics.features.market_level import market_level_detail

    key = "_".join([
        LEVEL_FP, f"k{k:g}", str(len(df)), str(df["date"].max())[:10],
        *(_stamp(p) for p in LEVEL_INPUTS),
        _dir_stamp(FOOTBALL_DATA, "*.csv"),
    ])
    cache_f = CACHE_DIR / f"market_level_{key}.json"
    if cache_f.exists():
        try:
            return {c: (int(n), float(m))
                    for c, (n, m) in json.loads(cache_f.read_text(encoding="utf-8")).items()}
        except Exception:
            pass
    detail = market_level_detail(df, None, k=k)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_f.write_text(json.dumps({c: [n, m] for c, (n, m) in detail.items()}),
                           encoding="utf-8")
        for old in CACHE_DIR.glob("market_level_*.json"):
            if old != cache_f:
                old.unlink(missing_ok=True)
    except OSError:
        pass
    return detail


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
    # The market level is set AFTER the fit on purpose: the fit is cached to disk,
    # and a level baked into it would be served long after the matches it was
    # measured on. This runs on every cache miss of _club_engine, which is keyed
    # on the foundation's mtime, so the daily refresh refreshes the level too.
    if DEPLOYED_CLUB_ENGINE_KWARGS.get("market_level_shift"):
        try:
            k = float(DEPLOYED_CLUB_ENGINE_KWARGS["market_level_shift"].get("k", 70.0))
            det = _market_levels(df, k)
            engine.set_market_levels({c: m for c, (_, m) in det.items()})
            print("market level " + (", ".join(f"{c} x{m:.4f} (n={n})"
                                               for c, (n, m) in sorted(det.items()))
                                     or "unavailable, lambdas left alone"))
        except Exception as exc:          # never let it stop the engine loading
            print(f"market level unavailable, lambdas left alone: {exc}")
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
    # the current season's player rows move the player state (see
    # PlayerPropsModel._append_current); ESPN can land before the foundation does
    espn_cur = ROOT / "data/external/advanced/espn/espn_player_match_current.csv"
    comm = sorted((ROOT / "data/external/advanced/espn/commentary").glob("*.jsonl"))
    espn_tag = (f"{espn_cur.stat().st_size}" if espn_cur.exists() else "none") + (
        f"c{comm[-1].stat().st_size}" if comm else "")
    key = (f"{len(df)}_{str(df['date'].max())[:10]}_{PROPS_FP}"
           f"_{squads_fingerprint(squads)}_{espn_tag}")
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
        # this season's penalty attempts, from the commentary text xG downloads
        pens = mins = None
        if espn_cur.exists():
            from mundialytics.enrichment.text_xg import penalties_from_commentary
            seasons = pd.read_csv(espn_cur, usecols=["season"])["season"].dropna()
            if len(seasons):
                comm_f = ROOT / f"data/external/advanced/espn/commentary/{seasons.max()}.jsonl"
                pens = penalties_from_commentary(comm_f)
                # and the minutes each player actually played, from the substitutions
                from mundialytics.enrichment.espn_minutes import season_minutes
                mins = season_minutes(comm_f, pd.read_csv(
                    espn_cur, usecols=["event_id", "player", "starter"]))
        # `current_squads` decides only WHO is in each roster. Without it the
        # squad is "whoever featured in this club's last ten Understat games",
        # which for a promoted club reaches back to its last top-flight season.
        pp = PlayerPropsModel().fit(
            pm,
            shots_path=ROOT / "data/external/advanced/understat/understat_shots.csv",
            current_squads=squads,
            current=espn_cur if espn_cur.exists() else None,
            current_penalties=pens,
            current_minutes=mins,
        )
    except Exception:
        pp = None
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump((tp, pp), cache_f, compress=3)
    except Exception:
        pass
    return tp, pp
