"""The prediction that was actually logged before a match, read back.

A played match's page used to re-run the current engine, which by then had been
fitted on that very match, and present the answer as "what we predicted before
kick-off". That grades the model on its own homework. This reads the
pre-kickoff log instead.

Rows logged since 2026-09-23 carry the lambdas, the full 1X2 and the expected
stats, so the whole prediction can be rebuilt exactly ("full"). Older rows hold
only the chosen outcome and its probability ("partial"): enough to grade the
call, not to redraw the distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LOG = ROOT / "data/processed/logs/predictions_log.csv"
STAT_KEYS = ("shots", "sot", "corners", "fouls", "yellows")

_cache: dict[str, object] = {"mtime": None, "df": None}


def _log() -> pd.DataFrame:
    """The log, re-read only when the file changes."""
    try:
        mtime = LOG.stat().st_mtime
    except OSError:
        return pd.DataFrame()
    if _cache["mtime"] != mtime:
        df = pd.read_csv(LOG, low_memory=False)
        df["_fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["_logged"] = pd.to_datetime(df["logged_at"], errors="coerce")
        _cache.update(mtime=mtime, df=df)
    return _cache["df"]  # type: ignore[return-value]


@dataclass
class LoggedPrediction:
    logged_at: str
    full: bool
    pick: str                    # "1" | "X" | "2"
    pick_prob: float
    over25: float | None
    trio: dict | None = None     # {"home", "draw", "away"} when full
    lambdas: tuple | None = None
    model_fp: str | None = None
    train_cutoff: str | None = None
    expected: dict = field(default_factory=dict)   # "shots_home" -> value

    def source(self) -> dict:
        return {
            "kind": "logged" if self.full else "logged-partial",
            "loggedAt": self.logged_at,
            "modelFingerprint": self.model_fp,
            "trainCutoff": self.train_cutoff,
            "pick": self.pick,
            "pickProbability": round(self.pick_prob, 4),
            "over25": None if self.over25 is None else round(self.over25, 4),
        }


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def find_logged(home: str, away: str, match_date, window_days: int = 4) -> LoggedPrediction | None:
    """The earliest pre-kickoff 1X2 row for this fixture, or None.

    A few days of slack on the date absorbs a rescheduled match; a row logged on
    or after the match day is never accepted, since it could not be pre-match.
    """
    df = _log()
    if df.empty:
        return None
    day = pd.Timestamp(match_date).normalize()
    m = df[(df["home"] == str(home).lower()) & (df["away"] == str(away).lower())
           & ((df["_fecha"] - day).abs() <= pd.Timedelta(days=window_days))]
    if m.empty:
        return None
    # pre-kickoff only: the logged timestamp must fall before the match day, or
    # before the recorded kick-off when there is one
    if "kickoff_utc" in m:
        kick = pd.to_datetime(m["kickoff_utc"], errors="coerce", utc=True)
    else:
        kick = pd.Series(pd.NaT, index=m.index, dtype="datetime64[ns, UTC]")
    logged_utc = m["_logged"].dt.tz_localize("Europe/Madrid", ambiguous="NaT",
                                             nonexistent="NaT").dt.tz_convert("UTC")
    before = (logged_utc < kick).where(kick.notna(), m["_logged"] < m["_fecha"])
    m = m[before.fillna(False).astype(bool)].sort_values("_logged")
    x = m[m["mercado"] == "1X2"]
    if x.empty:
        return None
    r = x.iloc[0]
    goals = m[(m["mercado"] == "Goles") & (m["linea"].astype(str).isin(["2.5", "2,5"]))
              & (m["logged_at"] == r["logged_at"])]
    over25 = None
    if len(goals):
        g = goals.iloc[0]
        p = float(g["prob"])
        over25 = p if str(g["seleccion"]).upper() == "OVER" else 1.0 - p

    lh, la = _num(r.get("lambda_home")), _num(r.get("lambda_away"))
    ph, pd_, pa = _num(r.get("p_home")), _num(r.get("p_draw")), _num(r.get("p_away"))
    full = None not in (lh, la, ph, pd_, pa)
    expected = {}
    for k in STAT_KEYS:
        for side in ("home", "away"):
            v = _num(r.get(f"exp_{k}_{side}"))
            if v is not None:
                expected[f"{k}_{side}"] = v
    return LoggedPrediction(
        logged_at=str(r["logged_at"]),
        full=full,
        pick=str(r["seleccion"]),
        pick_prob=float(r["prob"]),
        over25=over25,
        trio={"home": ph, "draw": pd_, "away": pa} if full else None,
        lambdas=(lh, la) if full else None,
        model_fp=None if pd.isna(r.get("model_fp", float("nan"))) else str(r.get("model_fp")),
        train_cutoff=None if pd.isna(r.get("train_cutoff", float("nan"))) else str(r.get("train_cutoff")),
        expected=expected,
    )


def as_prediction(lp: LoggedPrediction, fallback) -> SimpleNamespace:
    """A MatchPrediction-shaped object rebuilt from a full logged row.

    The 1X2 and O/U 2.5 are the logged numbers themselves. The scoreline matrix
    and the other goal lines are rebuilt from the logged lambdas with the
    deployed parameters. Expected stats come from the log when it has them.
    """
    from mundialytics.statistical_core.prediction_engine import deployed_markets_from_lambdas

    assert lp.full and lp.lambdas and lp.trio
    lh, la = lp.lambdas
    probs, dist = deployed_markets_from_lambdas(lh, la)
    over25 = lp.over25 if lp.over25 is not None else probs["p_over_25"]
    ns = SimpleNamespace(
        lambda_home=lh, lambda_away=la,
        p_home_win=lp.trio["home"], p_draw=lp.trio["draw"], p_away_win=lp.trio["away"],
        p_over_15=probs["p_over_15"], p_over_25=over25, p_under_25=1.0 - over25,
        p_over_35=probs["p_over_35"], p_btts=probs["p_btts"],
        top_scorelines=dist.top_scorelines(8), score_matrix=dist.matrix,
    )
    for k in STAT_KEYS:
        for side in ("home", "away"):
            attr = f"expected_{k}_{side}"
            setattr(ns, attr, lp.expected.get(f"{k}_{side}", getattr(fallback, attr, None)))
    return ns
