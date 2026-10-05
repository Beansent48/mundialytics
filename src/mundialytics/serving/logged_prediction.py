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


LINEUP_LOG = ROOT / "data/processed/logs/lineup_pass_log.csv"
MORNING_LOG = ROOT / "data/processed/logs/morning_pass_log.csv"


def find_lineup_pass(home: str, away: str, match_date, window_days: int = 2) -> tuple[LoggedPrediction, dict] | None:
    """(prediction, lineup details) from the pre-kickoff lineup pass, or None.

    scripts/log_lineup_pass.py re-prices a match once both squads are confirmed, about an
    hour before kick-off. The first logged row stays the track record; this is what the
    match page shows in the last hour, when it knows who is playing.
    """
    return _find_pass(LINEUP_LOG, home, away, match_date, window_days)


def find_morning_pass(home: str, away: str, match_date, window_days: int = 2) -> tuple[LoggedPrediction, dict] | None:
    """(prediction, details) from the matchday morning pass, or None.

    scripts/log_morning_pass.py re-prices a team's next league match with the regulars
    who missed its previous squad without a ban (likely still injured). The match page
    shows it until the lineup pass replaces it.
    """
    return _find_pass(MORNING_LOG, home, away, match_date, window_days)


def _find_pass(path: Path, home: str, away: str, match_date, window_days: int) -> tuple[LoggedPrediction, dict] | None:
    try:
        df = pd.read_csv(path, dtype={"event_id": str})
    except (OSError, pd.errors.EmptyDataError):
        return None
    day = pd.Timestamp(match_date).normalize()
    ko = pd.to_datetime(df["kickoff_utc"], errors="coerce").dt.normalize()
    m = df[(df["home"] == str(home).lower()) & (df["away"] == str(away).lower())
           & ((ko - day).abs() <= pd.Timedelta(days=window_days))]
    if m.empty:
        return None
    r = m.sort_values("logged_at_utc").iloc[-1]
    trio = {"home": float(r["p_home"]), "draw": float(r["p_draw"]), "away": float(r["p_away"])}
    pick = max(trio, key=trio.get)
    lp = LoggedPrediction(
        logged_at=str(r["logged_at_utc"]), full=True, pick={"home": "1", "draw": "X", "away": "2"}[pick],
        pick_prob=trio[pick], over25=_num(r.get("p_over_25")), trio=trio,
        lambdas=(float(r["lambda_home"]), float(r["lambda_away"])),
        model_fp=None if pd.isna(r.get("model_fp")) else str(r.get("model_fp")),
        train_cutoff=None if pd.isna(r.get("train_cutoff")) else str(r.get("train_cutoff")),
    )

    def names(v) -> list[str]:
        return [x.strip() for x in str(v).split(";") if x.strip()] if isinstance(v, str) else []

    details = {"loggedAtUtc": str(r["logged_at_utc"]), "kickoffUtc": str(r["kickoff_utc"]),
               "absentHome": _num(r.get("absent_home")), "absentAway": _num(r.get("absent_away")),
               "missingHome": names(r.get("missing_home")), "missingAway": names(r.get("missing_away"))}
    return lp, details


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
    # Which expected counts are the LOGGED pre-kickoff numbers rather than the
    # fallback's. Serving must not quietly replace these with a value recomputed
    # today: for a played match that model has already seen the result, and the
    # whole point of the log is that the record is what we said beforehand.
    ns.expected_from_log = frozenset(lp.expected)
    return ns
