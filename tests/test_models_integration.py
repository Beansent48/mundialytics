"""Integration tests for the deployed model OBJECTS (not just pure math).

Protects the prediction PATH — TeamPropsModel / PlayerPropsModel / the
PredictionEngine and the European simulator — against refactors that would
silently break validated behavior. Uses the disk-cached fitted models when
present (fast, and validates exactly what the app serves); skips if absent.

Run:  .venv/Scripts/python.exe -m pytest tests/test_models_integration.py -q
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CACHE = ROOT / "data/processed/cache"


def _load_cached_props():
    import joblib
    files = sorted(glob.glob(str(CACHE / "props_models_*.joblib")))
    if not files:
        return None, None
    try:
        return joblib.load(files[-1])
    except Exception:
        return None, None


def _load_cached_engine(tag: str):
    import joblib
    files = sorted(glob.glob(str(CACHE / f"engine_{tag}_*.joblib")))
    if not files:
        return None
    try:
        return joblib.load(files[-1])
    except Exception:
        return None


def _prob(x):
    return 0.0 <= float(x) <= 1.0


@pytest.fixture(scope="session")
def team_props_model():
    tp, _ = _load_cached_props()
    if tp is None:
        pytest.skip("no cached props model (run the app once to build it)")
    return tp


@pytest.fixture(scope="session")
def player_props_model():
    _, pp = _load_cached_props()
    if pp is None:
        pytest.skip("no cached player-props model")
    return pp


@pytest.fixture(scope="session")
def clubs_engine():
    eng = _load_cached_engine("clubs")
    if eng is None:
        pytest.skip("no cached clubs engine")
    return eng


# ── team props ──────────────────────────────────────────────────────────────
def test_team_fixture_structure_and_monotone(team_props_model):
    fx = team_props_model.predict_fixture("real madrid", "barcelona", lam_home=1.8, lam_away=1.4)
    assert set(fx) >= {"corners", "yellows", "fouls", "shots", "sot"}
    for mk, d in fx.items():
        if "over" not in d:
            continue
        if "lambda_total" in d:      # booking_pts uses lambda_yellows/reds instead
            assert d["lambda_total"] > 0
        for ln, p in d["over"].items():
            assert _prob(p), f"{mk} O{ln} = {p} out of [0,1]"
        lines = sorted(d["over"])
        ps = [d["over"][ln] for ln in lines]
        assert all(ps[i] >= ps[i + 1] - 1e-9 for i in range(len(ps) - 1)), f"{mk} not monotone"


def test_team_side_lines_present(team_props_model):
    fx = team_props_model.predict_fixture("liverpool", "man city", lam_home=1.7, lam_away=1.5)
    for mk in ["corners", "yellows", "shots"]:
        if mk in fx:
            assert "over_home" in fx[mk] and "over_away" in fx[mk]


def test_team_favourite_more_corners(team_props_model):
    fx = team_props_model.predict_fixture("man city", "bournemouth", lam_home=2.4, lam_away=0.7)
    if "corners" in fx:
        assert fx["corners"]["lambda_home"] > fx["corners"]["lambda_away"]


def test_team_unknown_does_not_crash(team_props_model):
    fx = team_props_model.predict_fixture("atlantis fc", "el dorado utd")
    assert isinstance(fx, dict)   # empty or partial, never an exception


# ── player props ────────────────────────────────────────────────────────────
def test_player_probabilities_valid(player_props_model):
    out = player_props_model.team_players_for_lambda("real madrid", 1.9)
    if out is None or out.empty:
        pytest.skip("real madrid not in the player pool")
    for col in ["p_anytime_scorer", "p_2plus_goals", "p_shots_over_1_5", "p_assist", "p_yellow"]:
        assert out[col].between(0, 1).all(), f"{col} out of [0,1]"
    assert (out["p_2plus_goals"] <= out["p_anytime_scorer"] + 1e-9).all()
    assert out["exp_min"].between(0, 98).all()


# ── prediction engine ───────────────────────────────────────────────────────
def test_engine_1x2_sums_and_ranges(clubs_engine):
    p = clubs_engine.predict_match("real madrid", "barcelona", competition=None, neutral=False)
    assert p.p_home_win + p.p_draw + p.p_away_win == pytest.approx(1.0, abs=1e-6)
    for v in [p.p_home_win, p.p_draw, p.p_away_win, p.p_over_25, p.p_btts]:
        assert _prob(v)
    assert p.lambda_home > 0 and p.lambda_away > 0


def test_engine_home_advantage(clubs_engine):
    a = clubs_engine.predict_match("real madrid", "getafe", competition=None, neutral=False)
    b = clubs_engine.predict_match("getafe", "real madrid", competition=None, neutral=False)
    assert a.p_home_win > a.p_away_win        # RM strong at home
    assert a.p_home_win > b.p_away_win         # RM home > RM away vs same foe


# ── season rollover ─────────────────────────────────────────────────────────
def test_team_props_fits_on_a_just_started_season():
    """A season that has only played its opening matchday must not break the fit.

    Those matches carry no prior-form features, so the Platt walk-forward's test
    fold for that season comes out empty and PoissonRegressor.predict used to
    raise on 0 samples. load_props_models() catches everything and returns
    (None, None), so the whole app silently lost team props every August.
    One league is enough to reproduce it and keeps this test ~12s.
    """
    import pandas as pd
    from mundialytics.props import TeamPropsModel

    found = ROOT / "data/processed/foundation_big5_multi_season.csv"
    if not found.exists():
        pytest.skip("foundation not built")
    df = pd.read_csv(found, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[(df["competition"] == "Premier League") & (df["date"] >= "2016-07-01")]

    seasons = sorted(df["season"].dropna().unique())
    if len(df[df["season"] == seasons[-1]]) > 60:
        pytest.skip("newest season is not a just-started one")

    tp = TeamPropsModel().fit(df, root=ROOT)   # must not raise
    assert getattr(tp, "_platt", None), "Platt calibration came out empty"


# ── european simulator ──────────────────────────────────────────────────────
def test_euro_seeded_reproducible():
    from mundialytics.statistical_core.competition.european import EuropeanTournament
    elos = {f"T{i:02d}": 1900.0 - i * 10 for i in range(36)}
    calib = {"c": 0.2043, "hfa": 0.2806, "b": 0.7394}
    r1 = EuropeanTournament("champions", elos, calib, rng=np.random.default_rng(123)).simulate(300)
    r2 = EuropeanTournament("champions", elos, calib, rng=np.random.default_rng(123)).simulate(300)
    assert np.allclose(r1["p_champion"].to_numpy(), r2["p_champion"].to_numpy())


# ── track-record integrity ──────────────────────────────────────────────────
def test_prediction_log_is_genuinely_pre_match():
    """Every logged prediction must pre-date its own kickoff.

    The log was once filled retroactively: 430 rows written 2026-07-23 for
    matches played 2026-05-23/24, by an engine fitted on those same matches.
    That made the "Track record en vivo" page a fiction. This pins the property
    that makes it real -- a prediction is only evidence if it existed first.
    """
    import pandas as pd

    log_f = ROOT / "data/processed/logs/predictions_log.csv"
    if not log_f.exists():
        pytest.skip("no prediction log")
    log = pd.read_csv(log_f)
    if log.empty:
        pytest.skip("prediction log is empty")

    logged = pd.to_datetime(log["logged_at"], errors="coerce")
    played = pd.to_datetime(log["fecha"], errors="coerce")
    assert logged.notna().all(), "rows without a logged_at timestamp"
    assert played.notna().all(), "rows without a match date"

    # +1 day: `fecha` is date-only, so a prediction made the morning of the
    # match is legitimately later than that date's midnight.
    late = log[logged > played + pd.Timedelta(days=1)]
    assert late.empty, (
        f"{len(late)} predictions logged after their match "
        f"(e.g. {late.iloc[0]['partido']} played {late.iloc[0]['fecha']}, "
        f"logged {late.iloc[0]['logged_at']})")


def test_prediction_log_has_no_duplicate_keys():
    """Re-running the logger must not append duplicates.

    linea="" round-trips through CSV as NaN; a bare astype(str) turned that into
    "nan" for stored rows and "" for fresh ones, so dedupe missed them and every
    re-run added one row per fixture.
    """
    import pandas as pd

    log_f = ROOT / "data/processed/logs/predictions_log.csv"
    if not log_f.exists():
        pytest.skip("no prediction log")
    log = pd.read_csv(log_f)
    if log.empty:
        pytest.skip("prediction log is empty")
    log["linea"] = log["linea"].fillna("").astype(str).replace("nan", "")
    keys = ["season", "jornada", "partido", "mercado", "ambito", "linea", "seleccion"]
    dup = log.duplicated(subset=keys).sum()
    assert dup == 0, f"{dup} duplicated prediction rows"
