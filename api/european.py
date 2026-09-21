"""
UEFA club competitions for the HTTP API.

Champions, Europa and Conference are priced on ClubElo rather than on the
domestic engine: their fixtures cross leagues, and a model fitted per league has
no way to say whether a good Dutch side is better than a mid-table Italian one.
The Elo→goals mapping is calibrated on 20,000 of this project's own matches.

The tournament simulation costs real time, so results are written to disk and
keyed on how much of the competition has been played — the same contract the
league forecast uses.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data/processed/competition_cache"
UEFA_DIR = ROOT / "data/external/uefa"

COMPETITIONS: dict[str, dict] = {
    "champions-league": {"id": "champions", "name": "Champions League"},
    "europa-league": {"id": "europa", "name": "Europa League"},
    "conference-league": {"id": "conference", "name": "Conference League"},
}

# Where each team sits if the draw has not happened yet: the Elo ranking cut into
# the three tiers, so a pre-draw page can still say something rather than nothing.
PRE_DRAW_TIERS = {"champions": (0, 36), "europa": (36, 72), "conference": (72, 108)}

ROUND_KEYS = [
    ("p_top24", "leaguePhase"),
    ("p_top8", "topEight"),
    ("p_playoff", "playoff"),
    ("p_r16", "roundOf16"),
    ("p_qf", "quarterFinal"),
    ("p_sf", "semiFinal"),
    ("p_final", "final"),
    ("p_champion", "champion"),
]


def current_year() -> int:
    today = date.today()
    return today.year if today.month >= 7 else today.year - 1


def available_years(comp_id: str) -> list[int]:
    """Seasons on disk, plus the one starting now."""
    from mundialytics.statistical_core.competition.european import FD_SLUG

    cached = {
        int(m.group(1))
        for p in (ROOT / "data/external/uefa").glob(f"raw_{FD_SLUG[comp_id]}_*.csv")
        if (m := re.search(r"_(\d{4})\.csv$", p.name))
    }
    return sorted(cached | {current_year()}, reverse=True)


def load_state(comp_id: str, year: int) -> dict:
    """Calibration, Elo ratings and the season's fixtures, parsed."""
    from mundialytics.statistical_core.competition.european import (
        fetch_current_elo,
        fetch_season_fixtures,
        load_calibration,
        make_resolver,
        normalize_club,
        parse_fixturedownload,
    )

    calib = load_calibration(ROOT)
    elo_all = fetch_current_elo(ROOT)
    resolver = make_resolver(list(elo_all))
    elo_by_norm = {normalize_club(k): v for k, v in elo_all.items()}

    raw = fetch_season_fixtures(ROOT, comp_id, year)
    league, ko, teams = None, None, {}
    if raw is not None:
        league, ko = parse_fixturedownload(raw, resolver)
        real = sorted(set(league.home) | set(league.away))
        teams = {t: elo_by_norm.get(normalize_club(t)) for t in real}
        teams = {t: e for t, e in teams.items() if e is not None}

    return {
        "calib": calib,
        "elo_all": elo_all,
        "elo_by_norm": elo_by_norm,
        "resolver": resolver,
        "raw": raw,
        "league": league,
        "ko": ko,
        "teams": teams,
    }


def probabilities(elo_home: float, elo_away: float, calib: dict) -> dict:
    """1X2 and goal markets from the Elo gap, on the calibrated European scale."""
    from mundialytics.statistical_core.distributions import outcome_probabilities

    d400 = (elo_home - elo_away) / 400.0
    lh = float(np.exp(calib["c"] + calib["hfa"] + calib["b"] * d400))
    la = float(np.exp(calib["c"] - calib["b"] * d400))
    p = outcome_probabilities(lh, la, dixon_coles_rho=-0.07)
    return {**p, "lambda_home": lh, "lambda_away": la}


def _cache_path(comp_id: str, year: int) -> Path:
    return CACHE_DIR / f"uefa_{comp_id}_{year}.json"


def forecast(comp_id: str, year: int, n_sims: int = 2000) -> dict:
    """Round-by-round probabilities for every team, from the current state."""
    from mundialytics.statistical_core.competition.european import EuropeanTournament

    state = load_state(comp_id, year)
    league, ko, teams = state["league"], state["ko"], state["teams"]

    played = 0
    if league is not None and len(league):
        played = int(league["home_goals"].notna().sum())
        if ko is not None and len(ko):
            played += int(ko["hg"].notna().sum())

    pre_draw = state["raw"] is None
    if pre_draw:
        # No calendar yet: seed the field by Elo rank and let each simulation
        # draw its own valid league phase. Stated as an estimate on the page,
        # never presented as the real participant list.
        lo, hi = PRE_DRAW_TIERS[comp_id]
        teams = dict(sorted(state["elo_all"].items(), key=lambda kv: -kv[1])[lo:hi])

    path = _cache_path(comp_id, year)
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            # The field size belongs in the key, not just the match count. A club
            # that starts resolving to an Elo (Slavia Praha did not, which is why
            # the Champions cache said 35 teams) changes the tournament without
            # changing `played`, and the stale answer would never expire.
            if (cached.get("_played") == played
                    and cached.get("_preDraw") == pre_draw
                    and cached.get("_teams") == len(teams)):
                return cached
        except Exception:
            pass

    tour = EuropeanTournament(comp_id, teams, state["calib"], league, ko)
    res = tour.simulate(n_sims)

    rows = []
    for r in res.itertuples():
        row = {"team": str(r.team), "elo": round(float(r.elo), 1)}
        for src, dst in ROUND_KEYS:
            v = getattr(r, src, None)
            row[dst] = None if v is None or pd.isna(v) else round(float(v), 4)
        rows.append(row)

    out = {
        "_played": played,
        "_preDraw": pre_draw,
        "_teams": len(teams),
        "competition": comp_id,
        "year": year,
        "teams": len(teams),
        "played": played,
        "leaguePhaseTotal": int(len(league)) if league is not None else 0,
        "preDraw": pre_draw,
        "simulations": n_sims,
        "standings": rows,
    }
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, default=str), encoding="utf-8")
    except OSError:
        pass
    return out



# ── one calendar day, for the matchday screen ─────────────────────────────────
# league_phase_round indexes by ROUND, which is the right handle for a
# competition page and the wrong one for "what is on today". These read the same
# parsed state and price with the same model, so the two screens can never drift
# apart -- and unlike the round view they keep the knockout rows, which are real
# matchdays from February on.

def _season_year(day: date) -> int:
    """European seasons run July-June, so January belongs to the year before."""
    return day.year if day.month >= 7 else day.year - 1


def _raw_with_dates(comp_id: str, year: int):
    state = load_state(comp_id, year)
    raw = state["raw"]
    if raw is None or not len(raw):
        return state, None
    rr = raw.copy()
    # dayfirst: these files write 09/12/2026 for 9 December, and a month-first
    # parser silently reads it as 12 September. See european.py's own note.
    rr["fecha"] = pd.to_datetime(rr["Date"], dayfirst=True, errors="coerce",
                                 format="mixed")
    rr["played"] = rr["Result"].astype(str).str.contains(r"\d+\s*-\s*\d+")
    return state, rr


def _price(state, home: str, away: str) -> dict | None:
    from mundialytics.statistical_core.competition.european import normalize_club

    h_ce, a_ce = state["resolver"](home), state["resolver"](away)
    eh = state["elo_by_norm"].get(normalize_club(h_ce)) if h_ce else None
    ea = state["elo_by_norm"].get(normalize_club(a_ce)) if a_ce else None
    if not eh or not ea:
        return None
    p = probabilities(eh, ea, state["calib"])
    return {"home": round(p["p_home_win"], 4), "draw": round(p["p_draw"], 4),
            "away": round(p["p_away_win"], 4)}


def fixtures_fingerprint() -> str:
    """How fresh the European fixture files are, for a cache key.

    Their results arrive on a different clock from the domestic foundation, so
    a day cached off the league feed alone would never notice a European score
    landing.
    """
    from mundialytics.statistical_core.competition.european import FD_SLUG

    parts = []
    for slug in sorted(FD_SLUG.values()):
        for path in sorted(UEFA_DIR.glob(f"raw_{slug}_*.csv")):
            parts.append(f"{path.name}:{int(path.stat().st_mtime)}")
    return "|".join(parts)


def day_fixtures(day: date) -> list[dict]:
    """Every European match on one day, in the shape the day screen renders.

    Same keys as the domestic payload so the client needs no second code path,
    plus two that tell the truth about what it is looking at: `analysis` is
    false because there is no match page for these, and `model` says the price
    came from the European Elo scale rather than the big-five engine. Pricing a
    Madrid-Bayern with team strengths fitted PER LEAGUE is exactly the mistake
    the Elo layer exists to avoid.
    """
    year = _season_year(day)
    out: list[dict] = []
    for slug, meta in COMPETITIONS.items():
        try:
            state, rr = _raw_with_dates(meta["id"], year)
        except Exception:
            continue
        if rr is None:
            continue
        sel = rr[rr["fecha"].dt.date == day].sort_values("fecha")
        for _, r in sel.iterrows():
            home, away = str(r["Home Team"]), str(r["Away Team"])
            played = bool(r["played"])
            score = None
            if played:
                got = re.search(r"(\d+)\s*-\s*(\d+)", str(r["Result"]))
                if got:
                    score = {"home": int(got.group(1)), "away": int(got.group(2))}
            rnum = pd.to_numeric(r.get("Round Number"), errors="coerce")
            out.append({
                "slug": f"{slugify(home)}-vs-{slugify(away)}",
                "competition": slug,
                "competitionName": meta["name"],
                "season": f"{year}-{year + 1}",
                "matchday": None if pd.isna(rnum) else int(rnum),
                "kickoff": None if pd.isna(r["fecha"]) else r["fecha"].date().isoformat(),
                "home": home,
                "away": away,
                "homeSlug": slugify(home),
                "awaySlug": slugify(away),
                "played": played,
                "score": score,
                "probabilities": None if played else _price(state, home, away),
                # no match page exists for these; the row must not link into a 404
                "analysis": False,
                "model": "elo",
            })
    return out


def day_counts(start: date, end: date) -> dict[str, int]:
    """How many European matches each day in the window holds."""
    counts: dict[str, int] = {}
    for year in {_season_year(start), _season_year(end)}:
        for meta in COMPETITIONS.values():
            try:
                _, rr = _raw_with_dates(meta["id"], year)
            except Exception:
                continue
            if rr is None:
                continue
            days = rr["fecha"].dropna().dt.date
            for d in days[(days >= start) & (days <= end)]:
                key = d.isoformat()
                counts[key] = counts.get(key, 0) + 1
    return counts


def slugify(value: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def league_phase_round(comp_id: str, year: int, round_number: int | None) -> dict:
    """One matchday of the league phase, with a price on every pending fixture."""
    state = load_state(comp_id, year)
    raw = state["raw"]
    if raw is None or state["league"] is None or not len(state["league"]):
        return {"rounds": [], "round": None, "fixtures": []}

    rr = raw.copy()
    rr["rnum"] = pd.to_numeric(rr["Round Number"], errors="coerce")
    lg = rr[rr["rnum"].notna()].copy()
    lg["played"] = lg["Result"].astype(str).str.contains(r"\d+\s*-\s*\d+")
    rounds = sorted(int(x) for x in lg["rnum"].unique())
    if not rounds:
        return {"rounds": [], "round": None, "fixtures": []}

    if round_number is None:
        pending = [r for r in rounds if not lg[lg.rnum == r]["played"].all()]
        round_number = pending[0] if pending else rounds[-1]

    sel = lg[lg.rnum == round_number].copy()
    sel["fecha"] = pd.to_datetime(sel["Date"], dayfirst=True, errors="coerce",
                                  format="mixed")
    sel = sel.sort_values("fecha")

    from mundialytics.statistical_core.competition.european import normalize_club

    fixtures = []
    for _, r in sel.iterrows():
        home, away = str(r["Home Team"]), str(r["Away Team"])
        item = {
            "home": home,
            "away": away,
            "slug": f"{slugify(home)}-vs-{slugify(away)}",
            "date": None if pd.isna(r["fecha"]) else r["fecha"].date().isoformat(),
            "played": bool(r["played"]),
            "result": str(r["Result"]) if r["played"] else None,
            "probabilities": None,
            "over25": None,
        }
        if not r["played"]:
            h_ce, a_ce = state["resolver"](home), state["resolver"](away)
            eh = state["elo_by_norm"].get(normalize_club(h_ce)) if h_ce else None
            ea = state["elo_by_norm"].get(normalize_club(a_ce)) if a_ce else None
            if eh and ea:
                p = probabilities(eh, ea, state["calib"])
                item["probabilities"] = {
                    "home": round(p["p_home_win"], 4),
                    "draw": round(p["p_draw"], 4),
                    "away": round(p["p_away_win"], 4),
                }
                item["over25"] = round(p["p_over_25"], 4)
        fixtures.append(item)

    return {"rounds": rounds, "round": round_number, "fixtures": fixtures}
