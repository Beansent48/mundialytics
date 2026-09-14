"""
Mundialytics HTTP API.

Serves the same fitted engine the Streamlit app serves, over JSON, so the web
front end can render fixtures and match analysis without importing pandas.

Run it:
    .venv/Scripts/python.exe -m uvicorn api.main:app --port 8000 --reload

The first request pays the model fit (~3 min) unless the joblib cache is warm,
which it is whenever the Streamlit app has been opened since the last data
update — the two share the cache directory and the key.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from api import catalogue as cat  # noqa: E402
from api.engine import club_engine, props_models  # noqa: E402

app = FastAPI(
    title="Mundialytics API",
    version="0.1.0",
    description="Probabilistic football markets for the Big Five leagues.",
)

# The web front end runs on its own origin in development and will run on its
# own domain in production; both need to be allowed explicitly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    # ...and any other local port, because the dev server does not always get
    # 3000: a second session holding it pushes Next.js onto 3001 and every
    # browser call then failed CORS while server-side calls kept working, which
    # looks like a broken app rather than a busy port. The regex echoes the
    # caller's own origin, so allow_credentials still holds (a bare "*" would
    # not). Production is same-origin and unaffected.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    # POST is here for /squadlab/season, which carries the user's chosen eleven
    # in its body. Listing only GET silently blocked it in the browser while
    # every server-side call kept working, because CORS is a browser rule.
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _predict(home: str, away: str, competition: str):
    engine, _, _ = club_engine()
    try:
        return engine.predict_match(home, away, competition=competition, neutral=False)
    except Exception:
        return None


def _fixture_payload(row, comp_slug: str, season: str, with_probs: bool) -> dict:
    """One fixture, in the shape the list screen renders.

    `probabilities` is null for a match already played: quoting a pre-match price
    next to a final score invites reading one as the other, and the settled
    prediction belongs on the match page where it can be compared properly.
    """
    played = pd.notna(row.home_goals) and pd.notna(row.away_goals)
    comp = cat.COMPETITIONS[comp_slug]
    payload = {
        "slug": cat.fixture_slug(row.home_team, row.away_team),
        "competition": comp_slug,
        "competitionName": comp["name"],
        "season": season,
        "matchday": int(row.matchday) if pd.notna(row.matchday) else None,
        "kickoff": row.date.date().isoformat() if pd.notna(row.date) else None,
        "home": str(row.home_team).title(),
        "away": str(row.away_team).title(),
        "homeSlug": cat.slugify(row.home_team),
        "awaySlug": cat.slugify(row.away_team),
        "played": bool(played),
        "score": (
            {"home": int(row.home_goals), "away": int(row.away_goals)} if played else None
        ),
        "probabilities": None,
    }
    if with_probs and not played:
        pred = _predict(row.home_team, row.away_team, comp["id"])
        if pred is not None:
            payload["probabilities"] = {
                "home": round(pred.p_home_win, 4),
                "draw": round(pred.p_draw, 4),
                "away": round(pred.p_away_win, 4),
            }
    return payload


@app.get("/health")
def health() -> dict:
    _, _, df = club_engine()
    return {
        "status": "ok",
        "matches": int(len(df)),
        "dataThrough": str(df["date"].max())[:10],
        "today": date.today().isoformat(),
    }


@app.get("/catalogue")
def catalogue() -> list[dict]:
    """The domestic leagues the matchday browser offers.

    Named `/catalogue` rather than `/competitions`: that path belongs to the
    UEFA competitions, which is what the word means everywhere in the product.
    """
    _, _, df = club_engine()
    out = []
    for slug, meta in cat.COMPETITIONS.items():
        out.append(
            {
                "slug": slug,
                "name": meta["name"],
                "country": meta["country"],
                "type": meta["type"],
                "seasons": cat.seasons_for(df, meta["id"]),
                "currentSeason": cat.current_season(df, meta["id"]),
            }
        )
    return out



# Ninety fixtures priced on every page load is thirteen seconds of work for an
# answer that only changes when a match kicks off or new results land. The key
# carries the dataset's last date, so a refresh invalidates it without a deploy.
_CACHE: dict[tuple, tuple[float, dict]] = {}
_TTL_SECONDS = 600


def _cached(key: tuple, build):
    import time

    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < _TTL_SECONDS:
        return hit[1]
    value = build()
    _CACHE[key] = (now, value)
    return value


@app.get("/fixtures/upcoming")
def upcoming(days: int = Query(8, ge=1, le=30)) -> dict:
    """What is about to be played across every covered league, by day."""
    _, _, df = club_engine()
    key = ("upcoming", days, str(df["date"].max())[:10], date.today().isoformat())
    return _cached(key, lambda: _build_upcoming(df, days))


def _build_upcoming(df, days: int) -> dict:
    window = cat.upcoming_window(df, days=days)
    if window.empty:
        return {"days": [], "count": 0}

    fixtures = [
        _fixture_payload(r, r.competition, r.season, with_probs=True)
        for r in window.itertuples()
    ]
    by_day: dict[str, list[dict]] = {}
    for f in fixtures:
        by_day.setdefault(f["kickoff"] or "", []).append(f)

    return {
        "days": [
            {"date": d, "fixtures": items} for d, items in sorted(by_day.items()) if d
        ],
        "count": len(fixtures),
    }


@app.get("/match/{competition}/{fixture}")
def match(competition: str, fixture: str) -> dict:
    """Full analysis of one fixture, addressed the way its URL spells it."""
    meta = cat.COMPETITIONS.get(competition)
    if meta is None:
        raise HTTPException(404, f"Unknown competition: {competition}")
    _, _, df = club_engine()
    season = cat.current_season(df, meta["id"])
    cal = cat.trusted_calendar(meta["id"], season, df)
    if cal is None or cal.empty:
        raise HTTPException(404, "No calendar available")

    rows = [r for r in cal.itertuples()
            if cat.fixture_slug(r.home_team, r.away_team) == fixture]
    if not rows:
        raise HTTPException(404, f"Unknown fixture: {fixture}")
    row = rows[0]

    key = ("match", competition, fixture, str(df["date"].max())[:10])
    return _cached(key, lambda: _build_match(row, competition, meta, season, df))


def _scorers(pp, home: str, away: str, pred, n: int = 3) -> dict | None:
    """The likeliest scorers per side.

    Ranked, never as a yes/no: over 6,327 settled player-matches the model never
    once put anyone above 50%, so a yes/no read just returns the base rate. The
    ordering is where the skill is — the top pick scores 28.8% of the time,
    3.4x a starter drawn at random from the same eleven.
    """
    if pp is None:
        return None
    try:
        players = pp.predict_fixture(
            home, away, lam_home=pred.lambda_home, lam_away=pred.lambda_away
        )
    except Exception:
        return None
    if players is None or players.empty or "p_anytime_scorer" not in players.columns:
        return None

    out: dict[str, list[dict]] = {}
    for side in ("home", "away"):
        sub = players[players["side"] == side] if "side" in players else players.iloc[0:0]
        if sub.empty:
            continue
        # the likely eleven first: exp_min is minutes-when-featuring, not minutes
        # spread across the squad, so most of a 26-man list clears any threshold
        top = sub.nlargest(11, "exp_min").nlargest(n, "p_anytime_scorer")
        out[side] = [
            {
                "player": str(r.player),
                "p": round(float(r.p_anytime_scorer), 4),
                "minutes": int(r.exp_min),
            }
            for r in top.itertuples()
        ]
    return out or None


# Markets in the order the match page shows them: the ones a reader recognises
# lead, the long tail follows.
_MARKET_ORDER = ["corners", "yellows", "shots", "sot", "fouls", "booking_pts"]


def _team_props(tp, home: str, away: str, pred) -> list[dict] | None:
    if tp is None:
        return None
    try:
        fx = tp.predict_fixture(
            home, away, lam_home=pred.lambda_home, lam_away=pred.lambda_away
        )
    except Exception:
        return None
    if not fx:
        return None

    markets = []
    for key in _MARKET_ORDER:
        d = fx.get(key)
        if not d or "over" not in d:
            continue
        markets.append(
            {
                "key": key,
                "lambdaHome": d.get("lambda_home"),
                "lambdaAway": d.get("lambda_away"),
                "lambdaTotal": d.get("lambda_total"),
                # each line carries its dominant side: a 26% over is a 74% under,
                # and showing only the over hides half the information
                "lines": [
                    {
                        "line": float(ln),
                        "over": round(float(p), 4),
                        "side": "over" if p >= 0.5 else "under",
                        "p": round(float(p if p >= 0.5 else 1 - p), 4),
                    }
                    for ln, p in d["over"].items()
                ],
            }
        )
    return markets or None


def _actual(row, meta: dict, season: str, df) -> dict | None:
    """The real match stats, when the foundation has ingested them.

    Kept separate from the score: the calendar settles a result the night it is
    played, but shots and cards arrive with the next data refresh, and inventing
    zeros in that window would grade the model against numbers nobody measured.
    """
    played = df[
        (df["competition"] == meta["id"])
        & (df["season"] == season)
        & (df["home_team"] == row.home_team)
        & (df["away_team"] == row.away_team)
    ]
    if played.empty:
        return None
    r = played.iloc[0]

    def pair(h, a):
        if pd.isna(r.get(h)) or pd.isna(r.get(a)):
            return None
        return {"home": float(r[h]), "away": float(r[a])}

    stats = {
        "shots": pair("home_shots", "away_shots"),
        "shotsOnTarget": pair("home_sot", "away_sot"),
        "corners": pair("home_corners", "away_corners"),
        "fouls": pair("home_fouls", "away_fouls"),
        "yellows": pair("home_yellow_cards", "away_yellow_cards"),
    }
    if all(v is None for v in stats.values()):
        return None
    return {
        "goals": {"home": int(r["home_goals"]), "away": int(r["away_goals"])},
        "stats": [
            {"key": k, "home": v["home"], "away": v["away"]}
            for k, v in stats.items()
            if v is not None
        ],
    }


# The player-match file is ~600k rows and reads from disk; load it once per data
# version, not once per match page. Keyed on the dataset's last date so a refresh
# rebuilds it without a deploy, exactly like the fixture cache above.
_PLAYER_ACTUALS: dict[str, pd.DataFrame] = {}


def _player_actuals(df) -> pd.DataFrame:
    key = str(df["date"].max())[:10]
    cached = _PLAYER_ACTUALS.get(key)
    if cached is not None:
        return cached
    from mundialytics.serving.track_record import _player_match_actuals

    pa = _player_match_actuals()
    if len(pa):
        pa = pa.copy()
        pa["fecha"] = pd.to_datetime(pa["fecha"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
        pa["eq"] = pa["equipo"].astype(str).str.lower()
    _PLAYER_ACTUALS.clear()  # only the current data version is ever needed
    _PLAYER_ACTUALS[key] = pa
    return pa


def _events(row, meta: dict, season: str, df) -> dict | None:
    """Who scored, assisted and was booked — grouped per side.

    Built from the same settled player-match file the track record scores itself
    on, so it never claims an event the evaluator would not also count. There are
    no minutes in that file (its `minutes` column is minutes played, not the clock
    of each event) and no per-player red cards, so this is a grouped summary, not
    a minute-by-minute timeline. When known scorers do not add up to the final
    score — an own goal, or a name the source never mapped — the gap is surfaced
    honestly as `unattributed` rather than hidden.
    """
    played = df[
        (df["competition"] == meta["id"])
        & (df["season"] == season)
        & (df["home_team"] == row.home_team)
        & (df["away_team"] == row.away_team)
    ]
    if played.empty:
        return None
    r = played.iloc[0]
    match_date = pd.to_datetime(r.get("date"), errors="coerce")
    if pd.isna(match_date):
        return None
    date_str = match_date.strftime("%Y-%m-%d")

    pa = _player_actuals(df)
    if not len(pa):
        return None
    home_l, away_l = str(row.home_team).lower(), str(row.away_team).lower()
    day = pa[(pa["fecha"] == date_str) & (pa["eq"].isin([home_l, away_l]))]
    if day.empty:
        return None

    def side(team_l: str, goals_scored: int) -> dict:
        s = day[day["eq"] == team_l]
        goals = sorted(
            (
                {"player": str(p.player), "count": int(p.goals)}
                for p in s[s["goals"] > 0].itertuples()
            ),
            key=lambda g: -g["count"],
        )
        assists = sorted(
            (
                {"player": str(p.player), "count": int(p.assists)}
                for p in s[s["assists"] > 0].itertuples()
            ),
            key=lambda a: -a["count"],
        )
        yellows = [str(p.player) for p in s[s["yellow_cards"] > 0].itertuples()]
        attributed = sum(g["count"] for g in goals)
        return {
            "goals": goals,
            "assists": assists,
            "yellows": yellows,
            "unattributed": max(0, int(goals_scored) - attributed),
        }

    home = side(home_l, int(r["home_goals"]))
    away = side(away_l, int(r["away_goals"]))
    empty = not any(
        home[k] or away[k] for k in ("goals", "assists", "yellows")
    ) and not (home["unattributed"] or away["unattributed"])
    if empty:
        return None
    return {"home": home, "away": away}


def _build_match(row, competition: str, meta: dict, season: str, df) -> dict:
    base = _fixture_payload(row, competition, season, with_probs=False)
    pred = _predict(row.home_team, row.away_team, meta["id"])
    if pred is None:
        raise HTTPException(503, "The model could not price this fixture")

    # The score matrix is the object every other market is read from; sending it
    # whole lets the client show the correct-score grid without a second call.
    matrix = pred.score_matrix.iloc[:7, :7]

    # Headline that does NOT collapse to "1-1" for every fixture. The global modal
    # exact score is a true-but-useless number (product-Poisson always peaks on a
    # low draw); lead instead with the modal 1X2 outcome, the expected score, and
    # the most likely score *conditional on each outcome*. See
    # ScoreDistribution.most_likely_by_outcome.
    from mundialytics.statistical_core.distributions import ScoreDistribution
    _dist = ScoreDistribution(lambda_home=pred.lambda_home,
                              lambda_away=pred.lambda_away,
                              matrix=pred.score_matrix)
    _by_outcome = _dist.most_likely_by_outcome()
    _probs = {"home": pred.p_home_win, "draw": pred.p_draw, "away": pred.p_away_win}
    _modal = max(_probs, key=_probs.get)
    headline = {
        "modalOutcome": _modal,
        "modalOutcomeProbability": round(_probs[_modal], 4),
        "expectedScore": f"{round(pred.lambda_home)}-{round(pred.lambda_away)}",
        "likelyScore": _by_outcome[_modal]["score"],
        "byOutcome": {
            k: {"score": v["score"],
                "outcomeProbability": round(float(v["p_outcome"]), 4),
                "conditionalProbability": round(float(v["p_conditional"]), 4)}
            for k, v in _by_outcome.items()
        },
    }

    base["prediction"] = {
        "headline": headline,
        "probabilities": {
            "home": round(pred.p_home_win, 4),
            "draw": round(pred.p_draw, 4),
            "away": round(pred.p_away_win, 4),
        },
        "expectedGoals": {
            "home": round(pred.lambda_home, 2),
            "away": round(pred.lambda_away, 2),
        },
        "goals": {
            "over15": round(pred.p_over_15, 4),
            "over25": round(pred.p_over_25, 4),
            "over35": round(pred.p_over_35, 4),
            "under25": round(pred.p_under_25, 4),
            "btts": round(pred.p_btts, 4),
        },
        "scorelines": [
            {"score": i["score"], "p": round(i["probability"], 4)}
            for i in pred.top_scorelines[:8]
        ],
        "scoreMatrix": [[round(float(v), 5) for v in r] for r in matrix.values],
        "expectedStats": [
            {"key": "shots", "home": round(pred.expected_shots_home, 1),
             "away": round(pred.expected_shots_away, 1)},
            {"key": "shotsOnTarget", "home": round(pred.expected_sot_home, 1),
             "away": round(pred.expected_sot_away, 1)},
            {"key": "corners", "home": round(pred.expected_corners_home, 1),
             "away": round(pred.expected_corners_away, 1)},
            {"key": "fouls", "home": round(pred.expected_fouls_home, 1),
             "away": round(pred.expected_fouls_away, 1)},
            {"key": "yellows", "home": round(pred.expected_yellows_home, 1),
             "away": round(pred.expected_yellows_away, 1)},
        ],
    }

    tp, pp = props_models()
    base["scorers"] = _scorers(pp, row.home_team, row.away_team, pred)
    base["teamProps"] = _team_props(tp, row.home_team, row.away_team, pred)
    base["actual"] = _actual(row, meta, season, df) if base["played"] else None
    base["events"] = _events(row, meta, season, df) if base["played"] else None
    return base


@app.get("/fixtures")
def fixtures(
    competition: str,
    season: str | None = None,
    matchday: int | None = None,
) -> dict:
    """One matchday of one competition."""
    meta = cat.COMPETITIONS.get(competition)
    if meta is None:
        raise HTTPException(404, f"Unknown competition: {competition}")
    _, _, df = club_engine()
    season = season or cat.current_season(df, meta["id"])

    cal = cat.trusted_calendar(meta["id"], season, df)
    if cal is None or cal.empty:
        raise HTTPException(404, f"No calendar for {meta['name']} {season}")

    rounds = sorted(int(m) for m in cal["matchday"].dropna().unique())
    if matchday is None:
        today = pd.Timestamp(date.today())
        pending = cal.loc[cal["date"] >= today, "matchday"]
        matchday = int(pending.min()) if len(pending) else rounds[-1]

    rows = cal[cal["matchday"] == matchday].sort_values("date")
    return {
        "competition": competition,
        "competitionName": meta["name"],
        "season": season,
        "matchday": matchday,
        "rounds": rounds,
        "fixtures": [
            _fixture_payload(r, competition, season, with_probs=True)
            for r in rows.itertuples()
        ],
    }

# ── Track record ───────────────────────────────────────────────────────────────
# The RPS ladder is a fixed measurement, reproduced by scripts/benchmark_vs_bet365.py
# over 10,080 Big Five matches (2020/21–2025/26). It changes when the model does,
# not when someone reloads the page.
BENCHMARK = {
    "sample": 10_080,
    "window": "2020/21–2025/26",
    "rows": [
        {"key": "market", "rps": 0.1946},
        {"key": "engine", "rps": 0.2025},
        {"key": "baseRates", "rps": 0.2308},
        {"key": "uniform", "rps": 0.2356},
    ],
    "byLeague": [
        {"league": "Bundesliga", "gap": 0.0060},
        {"league": "LaLiga", "gap": 0.0064},
        {"league": "Ligue 1", "gap": 0.0075},
        {"league": "Serie A", "gap": 0.0092},
        {"league": "Premier League", "gap": 0.0099},
    ],
}


def _calibration() -> list[dict] | None:
    """Announced probability against observed frequency, from the walk-forward cache.

    Read from the deployed chain's own predictions rather than recomputed here:
    a calibration curve produced by different code than the one being graded is
    not a check, it is a second opinion.
    """
    p = ROOT / "data/processed/enriched/understat_xg/walkforward_preds_deployed.csv"
    if not p.exists():
        return None
    import numpy as np

    w = pd.read_csv(p)
    outcome = np.where(w.hg > w.ag, "home", np.where(w.hg < w.ag, "away", "draw"))
    y = np.concatenate([
        (outcome == "home").astype(float),
        (outcome == "draw").astype(float),
        (outcome == "away").astype(float),
    ])
    pr = np.concatenate([
        w["ph"].to_numpy(float), w["pd"].to_numpy(float), w["pa"].to_numpy(float)
    ])
    bins = []
    for lo in np.linspace(0, 0.9, 10):
        mask = (pr >= lo) & (pr < lo + 0.1)
        if mask.sum() > 200:
            bins.append({
                "announced": round(float(pr[mask].mean()), 4),
                "observed": round(float(y[mask].mean()), 4),
                "n": int(mask.sum()),
            })
    return bins or None


@app.get("/track-record")
def track_record() -> dict:
    """Benchmark, calibration and the settled live log."""
    from mundialytics.serving.track_record import evaluate_log

    _, _, df = club_engine()
    key = ("track", str(df["date"].max())[:10])

    def build() -> dict:
        ev = evaluate_log(df)
        # Player markets are left out of this page on purpose: their accuracy is
        # dominated by the base rate (only 8.6% of starters score), so pooling
        # them would lift the headline without the model being any better.
        team = ev[~ev["es_jugador"]] if len(ev) and "es_jugador" in ev else ev
        live = None
        if len(team):
            by_market = (
                team.groupby("mercado_id")
                .agg(n=("acierto", "size"), hit=("acierto", "mean"),
                     announced=("confianza", "mean"))
                .reset_index()
                .sort_values("n", ascending=False)
            )
            live = {
                "count": int(len(team)),
                "matches": int(team["partido"].nunique()),
                "hitRate": round(float(team["acierto"].mean()), 4),
                "announced": round(float(team["confianza"].mean()), 4),
                "byMarket": [
                    {
                        "key": str(r.mercado_id),
                        "n": int(r.n),
                        "hitRate": round(float(r.hit), 4),
                        "announced": round(float(r.announced), 4),
                    }
                    for r in by_market.itertuples()
                ],
            }
        return {
            "benchmark": BENCHMARK,
            "calibration": _calibration(),
            "live": live,
            "dataThrough": str(df["date"].max())[:10],
        }

    return _cached(key, build)

# ── League forecast ────────────────────────────────────────────────────────────
def _league_cache_path(comp_id: str, season: str) -> Path:
    slug = f"{comp_id}_{season}".lower().replace(" ", "-").replace("/", "-")
    return ROOT / "data/processed/competition_cache" / f"live_{slug}.json"


@app.get("/league/{competition}")
def league(competition: str) -> dict:
    """Where the season stands and how it is likely to finish.

    The simulation costs ~20 seconds, so it is written to disk and keyed on how
    many matches the season has played: a page that only recomputes when a
    matchday lands opens instantly for everyone who is not the first visitor
    after a round, and is never stale for anyone.
    """
    import json

    meta = cat.COMPETITIONS.get(competition)
    if meta is None:
        raise HTTPException(404, f"Unknown competition: {competition}")

    _, _, df = club_engine()
    season = cat.current_season(df, meta["id"])
    played = int(((df["competition"] == meta["id"]) & (df["season"] == season)).sum())
    path = _league_cache_path(meta["id"], season)

    snap = None
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("_n_played") == played:
                snap = cached
        except Exception:
            snap = None

    if snap is None:
        from mundialytics.statistical_core.competition import forecast_cache as fc

        try:
            snap = fc.build_live_snapshot(meta["id"], season, df, n_sims=10_000, root=ROOT)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, f"Could not forecast: {str(exc)[:140]}") from exc
        snap["_n_played"] = played
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(snap, default=str), encoding="utf-8")
        except OSError:
            pass

    return _shape_league(snap, competition, meta, season)


def _shape_league(snap: dict, competition: str, meta: dict, season: str) -> dict:
    """Turn the engine's snapshot into the shape the page renders.

    Team names are title-cased here rather than in the client: the foundation
    stores them lower-case, and leaving that to the front end means every screen
    that forgets is the one that ships "real madrid".
    """
    probs = {p["team"]: p for p in snap["forecast"]["team_probs"]}
    standings = []
    for row in snap["standings"]:
        team = row["team"]
        p = probs.get(team, {})
        standings.append({
            "rank": int(row["rank"]),
            "team": str(team).title(),
            "played": int(row["played"]),
            "won": int(row["won"]),
            "drawn": int(row["drawn"]),
            "lost": int(row["lost"]),
            "goalsFor": int(row["goals_for"]),
            "goalsAgainst": int(row["goals_against"]),
            "points": int(row["points"]),
            "pTitle": p.get("p_champion"),
            "pTop4": p.get("p_top4"),
            "pRelegation": p.get("p_relegation"),
            "expectedPoints": p.get("exp_points"),
        })

    pm = snap["forecast"]["position_matrix"]
    remaining = snap.get("fixtures", {}).get("remaining", []) or []

    return {
        "competition": competition,
        "competitionName": meta["name"],
        "season": season,
        "matchday": snap.get("matchday"),
        "remaining": snap.get("n_remaining", 0),
        "standings": standings,
        "positionMatrix": {
            "teams": [str(t).title() for t in pm["teams"]],
            "positions": pm["positions"],
            "values": pm["values"],
        },
        "upcoming": [
            {
                "home": str(f["home_team"]).title(),
                "away": str(f["away_team"]).title(),
                "date": str(f.get("date", ""))[:10],
                "pHome": f.get("p_home"),
                "pDraw": f.get("p_draw"),
                "pAway": f.get("p_away"),
                "slug": cat.fixture_slug(f["home_team"], f["away_team"]),
            }
            for f in remaining[:10]
        ],
    }

@app.get("/leagues")
def leagues() -> list[dict]:
    """One card per competition for the index page.

    Standings come straight from the foundation, which is instant. The title
    odds are read from the forecast cache only — never computed here. Five
    leagues at twenty seconds each would make an index page cost a minute and a
    half, so a league nobody has opened yet shows its table and fills in its
    probabilities the first time someone visits it.
    """
    import json

    _, _, df = club_engine()
    out = []
    for slug, meta in cat.COMPETITIONS.items():
        season = cat.current_season(df, meta["id"])
        rows = df[(df["competition"] == meta["id"]) & (df["season"] == season)]
        leader, favourite, p_title = None, None, None

        if len(rows):
            # points, then goal difference, then goals for — the same order the
            # forecast's standings use. Ranking on points alone broke ties
            # arbitrarily, and the index named a different leader than the
            # league page it linked to.
            table: dict[str, list[int]] = {}
            for r in rows.itertuples():
                if pd.isna(r.home_goals) or pd.isna(r.away_goals):
                    continue
                h, a = int(r.home_goals), int(r.away_goals)
                for team, gf, ga in ((r.home_team, h, a), (r.away_team, a, h)):
                    e = table.setdefault(team, [0, 0, 0])
                    e[0] += 3 if gf > ga else 1 if gf == ga else 0
                    e[1] += gf - ga
                    e[2] += gf
            if table:
                best = max(table.items(), key=lambda kv: tuple(kv[1]))
                leader = {"team": str(best[0]).title(), "points": best[1][0]}

        cached = _league_cache_path(meta["id"], season)
        if cached.exists():
            try:
                snap = json.loads(cached.read_text(encoding="utf-8"))
                probs = snap.get("forecast", {}).get("team_probs", [])
                if probs:
                    top = max(probs, key=lambda p: p.get("p_champion", 0))
                    favourite = str(top["team"]).title()
                    p_title = top.get("p_champion")
            except Exception:
                pass

        out.append({
            "slug": slug,
            "name": meta["name"],
            "country": meta["country"],
            "season": season,
            "played": int(len(rows)),
            "leader": leader,
            "favourite": favourite,
            "pTitle": p_title,
        })
    return out

# ── UEFA club competitions ─────────────────────────────────────────────────────
@app.get("/competitions")
def competitions_index() -> list[dict]:
    """The three UEFA competitions, with the favourite when one is on disk."""
    from api import european as eu

    out = []
    for slug, meta in eu.COMPETITIONS.items():
        year = eu.current_year()
        cached = eu._cache_path(meta["id"], year)
        favourite, p_champion, teams, played = None, None, None, None
        if cached.exists():
            try:
                import json

                snap = json.loads(cached.read_text(encoding="utf-8"))
                rows = [r for r in snap.get("standings", []) if r.get("champion")]
                if rows:
                    top = max(rows, key=lambda r: r["champion"])
                    favourite, p_champion = top["team"], top["champion"]
                teams, played = snap.get("teams"), snap.get("played")
            except Exception:
                pass
        out.append({
            "slug": slug,
            "name": meta["name"],
            "season": f"{year}/{str(year + 1)[2:]}",
            "favourite": favourite,
            "pChampion": p_champion,
            "teams": teams,
            "played": played,
        })
    return out


@app.get("/competition/{slug}")
def competition(slug: str, matchday: int | None = None) -> dict:
    """Round-by-round probabilities plus one matchday of the league phase."""
    from api import european as eu

    meta = eu.COMPETITIONS.get(slug)
    if meta is None:
        raise HTTPException(404, f"Unknown competition: {slug}")
    year = eu.current_year()

    try:
        fc = eu.forecast(meta["id"], year)
        phase = eu.league_phase_round(meta["id"], year, matchday)
    except Exception as exc:
        raise HTTPException(503, f"European layer unavailable: {str(exc)[:140]}") from exc

    return {
        "slug": slug,
        "name": meta["name"],
        "season": f"{year}/{str(year + 1)[2:]}",
        "teams": fc["teams"],
        "played": fc["played"],
        "leaguePhaseTotal": fc["leaguePhaseTotal"],
        "preDraw": fc["preDraw"],
        "simulations": fc["simulations"],
        "standings": fc["standings"],
        "phase": phase,
    }

# ── Awards ─────────────────────────────────────────────────────────────────────
def _award_cache_path(comp_id: str, season: str) -> Path:
    slug = f"{comp_id}_{season}".lower().replace(" ", "-").replace("/", "-")
    return ROOT / "data/processed/competition_cache" / f"scorers_{slug}.json"


@app.get("/awards/{competition}")
def awards(competition: str) -> dict:
    """The top-scorer race: who finishes as the league's leading scorer.

    Written to disk and keyed on how many matches the season has played, because
    walking ~350 remaining fixtures through the player model costs about half a
    minute. It recomputes when a matchday lands, and only then.
    """
    import json

    meta = cat.COMPETITIONS.get(competition)
    if meta is None:
        raise HTTPException(404, f"Unknown competition: {competition}")

    _, _, df = club_engine()
    season = cat.current_season(df, meta["id"])
    rows = df[(df["competition"] == meta["id"]) & (df["season"] == season)]
    n_played = int(len(rows))

    path = _award_cache_path(meta["id"], season)
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("_nPlayed") == n_played:
                return cached
        except Exception:
            pass

    from mundialytics.serving import scorer_race as sr
    from mundialytics.serving.track_record import _player_match_actuals

    cal = cat.season_calendar(meta["id"], season)
    if cal.empty or rows.empty:
        raise HTTPException(404, "No calendar or results for this season")
    remaining = cal[~cal["completed"].astype(bool)]

    engine, _, _ = club_engine()
    _, pp = props_models()
    if pp is None:
        raise HTTPException(503, "The player model is not available")

    def predict(home: str, away: str):
        try:
            return engine.predict_match(home, away, competition=meta["id"], neutral=False)
        except Exception:
            return None

    teams = set(rows["home_team"]) | set(rows["away_team"])
    goals = sr.goals_so_far(
        _player_match_actuals(), teams,
        since=str(rows["date"].min())[:10],
        until=str(pd.Timestamp.today())[:10],
    )
    rates = sr.remaining_rates(remaining, predict, pp.team_players_for_lambda)
    race = sr.simulate(goals, rates)

    leader = None
    if not race.empty:
        top_now = race.sort_values("goals", ascending=False).iloc[0]
        leader = {"player": str(top_now["player"]), "goals": int(top_now["goals"])}

    out = {
        "_nPlayed": n_played,
        "competition": competition,
        "competitionName": meta["name"],
        "season": season,
        "leader": leader,
        "inTheRace": int((race["pTopScorer"] > 0.005).sum()) if not race.empty else 0,
        "players": [
            {
                "player": str(r.player),
                "team": str(r.team).title(),
                "goals": int(r.goals),
                "expected": float(r.expected),
                "p": float(r.pTopScorer),
            }
            for r in race.head(25).itertuples()
        ],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, default=str), encoding="utf-8")
    except OSError:
        pass
    return out

# ── SquadLab ───────────────────────────────────────────────────────────────────
from pydantic import BaseModel, Field  # noqa: E402


class SeasonRequest(BaseModel):
    squad: list[str] = Field(min_length=11, max_length=11)
    # Draft vs Sandbox is only a build method now — both play the Champions.
    mode: str = "draft"
    # Omit for a fresh random draw; pass one to replay the same path.
    seed: int | None = None


@app.get("/squadlab/pool")
def squadlab_pool() -> dict:
    """Every player available to draft, from any league, by position, best first.

    The squad always plays the Champions, whose field is pan-European, so the
    pool is no longer scoped to a single competition.
    """
    from api import squadlab as sl

    data = sl.champions_pool()
    if not any(data["players"].values()):
        raise HTTPException(503, "No player profiles available")
    return {"competition": "champions", "competitionName": "Champions League", **data}


@app.post("/squadlab/season")
def squadlab_season(req: SeasonRequest) -> dict:
    """Play the whole Champions League once with the chosen eleven.

    Not cached: the squad is the user's and each playthrough takes a random slot
    in the draw, so there is no key that would ever be hit twice.
    """
    from api import squadlab as sl

    try:
        return sl.play_champions(req.squad, seed=req.seed)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, f"Could not play the Champions: {str(exc)[:180]}") from exc

@app.get("/fixtures/day")
def fixtures_day(day: str | None = None) -> dict:
    """One calendar day, grouped by competition.

    Grouping happens here rather than in the client so every screen that shows a
    day shows it in the same order — the catalogue's order, which is the one the
    rest of the product uses.
    """
    from datetime import date as _date

    _, _, df = club_engine()
    try:
        target = _date.fromisoformat(day) if day else _date.today()
    except ValueError as exc:
        raise HTTPException(400, f"Bad date: {day}") from exc

    key = ("day", target.isoformat(), str(df["date"].max())[:10])
    return _cached(key, lambda: _build_day(df, target))


def _build_day(df, target) -> dict:
    rows = cat.window(df, target, target)
    groups: dict[str, list[dict]] = {}
    for r in rows.itertuples():
        payload = _fixture_payload(r, r.competition, r.season, with_probs=True)
        groups.setdefault(r.competition, []).append(payload)

    return {
        "date": target.isoformat(),
        "count": sum(len(v) for v in groups.values()),
        # catalogue order, not dictionary order: the leagues always line up the
        # same way whichever day you land on
        "competitions": [
            {
                "slug": slug,
                "name": cat.COMPETITIONS[slug]["name"],
                "country": cat.COMPETITIONS[slug]["country"],
                "fixtures": groups[slug],
            }
            for slug in cat.COMPETITIONS
            if slug in groups
        ],
    }


@app.get("/fixtures/calendar")
def fixtures_calendar(days_back: int = Query(7, ge=0, le=30),
                      days_forward: int = Query(14, ge=1, le=60)) -> dict:
    """How many fixtures each nearby day holds, for the date strip."""
    from datetime import date as _date, timedelta as _td

    _, _, df = club_engine()
    today = _date.today()
    start, end = today - _td(days=days_back), today + _td(days=days_forward)
    key = ("calendar", start.isoformat(), end.isoformat(), str(df["date"].max())[:10])
    return _cached(key, lambda: {
        "today": today.isoformat(),
        "from": start.isoformat(),
        "to": end.isoformat(),
        "counts": cat.day_counts(df, start, end),
    })
