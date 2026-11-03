"""
Settling the prediction log against real results.

This lives in one place on purpose. The log is the product's whole claim to
honesty, and two copies of "how often were we right" would eventually disagree
— at which point neither number means anything. The HTTP API imports from here.

Nothing in this module knows about HTTP or about any language: it takes
dataframes and gives back one row per settled prediction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mundialytics.identity.normalization import canonical_team_name

ROOT = Path(__file__).resolve().parents[3]
PRED_LOG = ROOT / "data/processed/logs/predictions_log.csv"


def _elo_resolver():
    """Map a team name in any of the project's spellings to its ClubElo name.

    Returns None when the Elo table cannot be read. European rows then cannot be
    settled at all, which is a reporting hole and not something to raise inside
    an HTTP request: `scripts/audit_settlement.py` is what notices it.
    """
    try:
        from mundialytics.statistical_core.competition.european import (
            fetch_current_elo, make_resolver)
        return make_resolver(list(fetch_current_elo(ROOT)))
    except Exception:
        return None


def _european_results() -> pd.DataFrame:
    """Real European results from the cached season CSVs, ClubElo names."""
    resolver = _elo_resolver()
    if resolver is None:
        return pd.DataFrame()
    rows = []
    for p in (ROOT / "data/external/uefa").glob("raw_*.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        res = df["Result"].astype(str).str.extract(r"(\d+)\s*-\s*(\d+)")
        df["hg"] = pd.to_numeric(res[0], errors="coerce")
        df["ag"] = pd.to_numeric(res[1], errors="coerce")
        df["fecha"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce",
                                     format="mixed").dt.strftime("%Y-%m-%d")
        df["home"] = df["Home Team"].astype(str).map(resolver)
        df["away"] = df["Away Team"].astype(str).map(resolver)
        rows.append(df.dropna(subset=["home", "away", "hg", "fecha"])[
            ["fecha", "home", "away", "hg", "ag"]])
    if not rows:
        return pd.DataFrame()
    return (pd.concat(rows, ignore_index=True)
            .drop_duplicates(subset=["fecha", "home", "away"])
            .rename(columns={"hg": "home_goals", "ag": "away_goals"}))


CALENDAR_DIR = ROOT / "data/processed/fixture_calendars"


def _calendar_results() -> pd.DataFrame:
    """Completed results from the season fixture calendars, foundation names.

    The foundation trails the calendars by a round or two, so the freshest
    results sit on disk before they reach it — lorient vs brest was played 0-2
    on 2026-11-02 and settled nowhere, because settlement only ever read the
    foundation. That made the live record systematically exclude the newest
    round. Used strictly to fill what the foundation does not have: the
    foundation stays authoritative, and only goal-outcome markets can settle
    from here, since the calendars carry no shots or cards.
    """
    if not CALENDAR_DIR.exists():
        return pd.DataFrame()
    rows = []
    need = {"date", "home_team", "away_team", "completed", "home_goals", "away_goals"}
    for p in CALENDAR_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if not need <= set(df.columns):
            continue
        d = df[df["completed"].astype(str).str.lower() == "true"].copy()
        d["fecha"] = d["date"].astype(str).str[:10]
        d = d.rename(columns={"home_team": "home", "away_team": "away"})
        rows.append(d.dropna(subset=["home_goals", "away_goals"])[
            ["fecha", "home", "away", "home_goals", "away_goals"]])
    if not rows:
        return pd.DataFrame()
    return (pd.concat(rows, ignore_index=True)
            .drop_duplicates(subset=["fecha", "home", "away"]))


EVENT_COLS = {"corners": ("home_corners", "away_corners"),
              "yellows": ("home_yellow_cards", "away_yellow_cards"),
              "fouls": ("home_fouls", "away_fouls"),
              "shots": ("home_shots", "away_shots"),
              "sot": ("home_sot", "away_sot")}


PLAYER_MARKETS_ES = {"jug_goleador": "Goleador", "jug_2goles": "2+ goles",
                     "jug_tiros": "Tiros jugador", "jug_asistencia": "Asistencia",
                     "jug_amarilla": "Amarilla jugador"}


def _player_match_actuals() -> pd.DataFrame:
    """(fecha, equipo, jugador) -> what the player actually did.

    Built from the Understat player-match file, keyed onto football-data team
    names via understat_team_match_xg. Without this the jug_* markets would be
    logged and never scored -- exactly the booking-points failure, where a market
    was priced and served for months while the evaluator silently skipped it.
    """
    pm = ROOT / "data/external/advanced/understat/understat_player_match.csv"
    tm = ROOT / "data/processed/understat_team_match_xg.csv"
    if not pm.exists() or not tm.exists():
        return pd.DataFrame()
    try:
        p = pd.read_csv(pm, usecols=["game_id", "team", "player", "minutes",
                                     "goals", "shots", "assists", "yellow_cards"],
                        low_memory=False)
        g = pd.read_csv(tm, usecols=["provider_match_id", "date", "home_team",
                                     "away_team", "home_team_fd", "away_team_fd"])
    except Exception:
        return pd.DataFrame()
    g = g.rename(columns={"provider_match_id": "game_id"}).drop_duplicates("game_id")
    g["fecha"] = pd.to_datetime(g["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    m = p.merge(g, on="game_id", how="inner")
    # the player file names teams the Understat way; the log uses football-data's
    m["equipo"] = np.where(m["team"] == m["home_team"], m["home_team_fd"],
                           np.where(m["team"] == m["away_team"], m["away_team_fd"], None))
    m = m.dropna(subset=["equipo", "fecha"])
    out = m[["fecha", "equipo", "player", "minutes", "goals", "shots",
             "assists", "yellow_cards"]]
    extra = [x for x in (_espn_player_actuals(), _fbref_player_actuals()) if len(x)]
    if extra:
        # Understat kept first: it is the source the model was fitted on, so its
        # naming matches best where both have a match. ESPN comes before FBref
        # for the current season because it covers all five leagues and says who
        # actually played, which the FBref file does not.
        out = pd.concat([out, *extra], ignore_index=True)
        out = out.drop_duplicates(subset=["fecha", "equipo", "player"], keep="first")
    return out


def _espn_player_actuals() -> pd.DataFrame:
    """Current-season player results from ESPN's public JSON, all five leagues.

    FBref only ever yielded the Premier League before stalling, which left 79% of
    the logged player predictions with nothing able to score them -- the
    booking-points failure once more. ESPN is a JSON API asked by date, so there
    is no season label to get wrong, and it carries the full roster.

    That roster is the point. ESPN's event feed names only scorers and booked
    players; settling from it alone would score a striker's market only in the
    matches where he scored, and the hit rate would approach 100% by
    construction.
    """
    f = ROOT / "data/external/advanced/espn/espn_player_match_current.csv"
    if not f.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(f, low_memory=False)
    except Exception:
        return pd.DataFrame()
    if d.empty or "date" not in d.columns:
        return pd.DataFrame()
    d["fecha"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    # the connector already writes football-data names, via the same alias file
    d["equipo"] = d["team"].astype(str)
    for c in ("appearances", "goals", "shots", "assists", "yellow_cards"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    # only players who actually took the field. An unused substitute never had
    # the chance, so settling him as "did not score" would hand the track record
    # a free hit; dropped here, he falls through the evaluator's `act is None`
    # branch and is skipped, the way a book voids the bet.
    d = d[d["appearances"].fillna(0) > 0].copy()
    d["minutes"] = pd.NA
    keep = ["fecha", "equipo", "player", "minutes", "goals", "shots",
            "assists", "yellow_cards"]
    return d.dropna(subset=["fecha", "equipo", "player"])[keep]


def _fbref_player_actuals() -> pd.DataFrame:
    """Current-season player results from FBref, for seasons Understat lacks.

    Understat had not published 2026/27 while the player markets were already
    being logged, which would have left them unsettleable -- the booking-points
    failure again. FBref does have it (scripts/fetch_fbref_player_stats.py), so
    settlement no longer depends on a single provider publishing on time.
    """
    f = ROOT / "data/external/advanced/fbref/fbref_player_match_current.csv"
    if not f.exists():
        return pd.DataFrame()
    try:
        d = pd.read_csv(f, low_memory=False)
    except Exception:
        return pd.DataFrame()
    if d.empty or "date" not in d.columns:
        return pd.DataFrame()
    d["fecha"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    # FBref writes full club names ("Manchester United") where the foundation
    # carries football-data's short forms ("man united"), so the normaliser alone
    # resolved only 10 of 20 Premier League sides. The curated alias file closes
    # the rest; without it half the settlements would silently find no player.
    alias_f = ROOT / "data/curated/fixture_team_aliases.csv"
    alias = {}
    if alias_f.exists():
        try:
            alias = dict(pd.read_csv(alias_f).itertuples(index=False, name=None))
        except Exception:
            alias = {}
    d["equipo"] = [alias.get(t, canonical_team_name(t)) for t in d["team"].astype(str)]
    keep = ["fecha", "equipo", "player", "minutes", "goals", "shots",
            "assists", "yellow_cards"]
    for c in keep:
        if c not in d.columns:
            d[c] = pd.NA
    return d.dropna(subset=["fecha", "equipo", "player"])[keep]


def _player_finder(plk: dict):
    """(fecha, team, name) -> that player's row: the exact name first, then a
    tolerant match among the players of THAT team in THAT match only.

    The log names players the way the model knows them (Understat), while the
    current season settles from ESPN: "Kylian Mbappe-Lottin" is ESPN's "Kylian
    Mbappé", "Pape Alassane Gueye" is "Pape Gueye". Exact keys alone left those
    unsettled -- 112 player-matches of 2026/27 by 2026-09-29. Confining the
    search to one team-match's players is what keeps a short-key match safe."""
    from mundialytics.identity.current_squads import NameIndex

    by_match: dict = {}
    for (fecha, team, _), row in plk.items():
        by_match.setdefault((fecha, team), []).append(row)
    indexes: dict = {}

    def find(fecha, team, name):
        row = plk.get((fecha, team, name))
        if row is not None:
            return row
        rows = by_match.get((fecha, team))
        if not rows:
            return None
        if (fecha, team) not in indexes:
            idx = NameIndex()
            for x in rows:
                idx.add(x.player, x)
            indexes[(fecha, team)] = idx
        return indexes[(fecha, team)].lookup(name)

    return find


def _fill_near_date(m: pd.DataFrame, res: pd.DataFrame,
                    home_col: str = "home", away_col: str = "away",
                    tol_days: int = 2) -> pd.DataFrame:
    """Settle rows whose fixture moved after the prediction was logged.

    Kick-offs get rescheduled for television, so the date stored with the
    prediction is the date the calendar promised, not always the date the match
    was played: four of 2026/27's logged rows were off by exactly one day
    (toulouse-troyes, paris sg-lyon, nice-rennes, lyon-angers). An exact-date
    join drops those, which reads as "never predicted" instead of "moved".

    Only an unambiguous match counts: one result for that pair inside the
    window, or the row is left unsettled. A tolerance of a couple of days cannot
    reach the reverse fixture or another season's meeting.
    """
    if not len(res) or not m["home_goals"].isna().any():
        return m
    # Carry every column the source has, not just the goals. Filling goals
    # alone settled the 1X2 of a moved fixture and left its half-time and event
    # markets unscoreable even though the foundation held them one day over —
    # 14 of 2026/27's fixtures, which read as missing coverage rather than as a
    # join that stopped early. `date` stays out: the row keeps the date it was
    # predicted for.
    cols = [c for c in res.columns
            if c not in ("home", "away", "fecha", "date") and c in m.columns]
    idx: dict[tuple, list] = {}
    for r in res.itertuples(index=False):
        idx.setdefault((r.home, r.away), []).append(r)
    tol = pd.Timedelta(days=tol_days)
    for i in m.index[m["home_goals"].isna()]:
        cands = idx.get((m.at[i, home_col], m.at[i, away_col]))
        if not cands:
            continue
        want = pd.Timestamp(m.at[i, "fecha"])
        near = [c for c in cands if abs(pd.Timestamp(c.fecha) - want) <= tol]
        if len(near) == 1:
            for c in cols:
                m.at[i, c] = getattr(near[0], c)
    return m


def attach_results(log: pd.DataFrame, df_clubs: pd.DataFrame) -> pd.DataFrame:
    """The prediction log with real goals attached, from every source we hold.

    One place decides what counts as "there is a result for this fixture": the
    foundation first, then the UEFA results for European ties, then the fixture
    calendars for rounds the foundation has not ingested, and last a two-day
    window for fixtures that moved. `evaluate_log` grades from this and
    `scripts/audit_settlement.py` audits it, so the audit cannot disagree with
    the settlement it audits — which it did while each kept its own copy of the
    join, reporting as holes four fixtures that settle perfectly well.

    Rows keep their NaN goals when nothing has a result: the caller decides
    whether that is "not played yet" or a hole.
    """
    extra = [c for c in ("home_red_cards", "away_red_cards",
                         "home_goals_ht", "away_goals_ht") if c in df_clubs.columns]
    res = df_clubs[["home_team", "away_team", "date", "home_goals", "away_goals"]
                   + [c for cc in EVENT_COLS.values() for c in cc] + extra].copy()
    res["fecha"] = res["date"].astype(str).str[:10]
    res_log = res.rename(columns={"home_team": "home", "away_team": "away"})
    m = log.merge(res_log, on=["home", "away", "fecha"], how="left")
    # European rows won't match the domestic foundation — fill from the UEFA
    # results. Those carry ClubElo names ("Bayern", "Man United") while the log
    # carries foundation names ("bayern munich", "manchester united"), so the log
    # side has to be resolved into Elo space first. Merging the two spellings
    # directly matched zero rows of 162, which left every European prediction
    # silently unsettled; resolving first settles all 108 that have a result.
    eu_res = _european_results()
    resolve = _elo_resolver()
    if len(eu_res) and resolve is not None:
        miss = m["home_goals"].isna()
        side = m.loc[miss, ["home", "away", "fecha"]].copy()
        side["home"] = side["home"].astype(str).map(resolve)
        side["away"] = side["away"].astype(str).map(resolve)
        fill = side.merge(eu_res, on=["home", "away", "fecha"], how="left")
        m.loc[miss, "home_goals"] = fill["home_goals"].to_numpy()
        m.loc[miss, "away_goals"] = fill["away_goals"].to_numpy()
        # keep the Elo spelling so the moved-fixture pass can use it too
        m["_elo_home"] = m["home"].astype(str).map(resolve)
        m["_elo_away"] = m["away"].astype(str).map(resolve)

    # The foundation trails the calendars, so fill from them what it has not
    # ingested yet. Foundation first, always: this only sees untouched rows.
    cal = _calendar_results()
    if len(cal):
        miss = m["home_goals"].isna()
        fill = (m.loc[miss, ["home", "away", "fecha"]]
                .merge(cal, on=["home", "away", "fecha"], how="left"))
        m.loc[miss, "home_goals"] = fill["home_goals"].to_numpy()
        m.loc[miss, "away_goals"] = fill["away_goals"].to_numpy()

    # Anything still open may simply have been played on another day.
    m = _fill_near_date(m, res_log)
    m = _fill_near_date(m, cal)
    if "_elo_home" in m.columns and len(eu_res):
        m = _fill_near_date(m, eu_res, "_elo_home", "_elo_away")
        m = m.drop(columns=["_elo_home", "_elo_away"])
    return m


def evaluate_log(df_clubs: pd.DataFrame,
                 labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Join the served-predictions log with real results -> hit per prediction.

    `labels` renames a market for display. Left out, the raw market id comes
    back and the caller translates — which is what the web front end does, and
    why this returns ids rather than one language's wording."""
    labels = labels or {}
    if not PRED_LOG.exists():
        return pd.DataFrame()
    log = pd.read_csv(PRED_LOG)
    m = attach_results(log, df_clubs)
    # player markets settle from a different source (Understat player-match),
    # so they are joined separately rather than through the team-level frame
    pl = _player_match_actuals()
    plk = {}
    if len(pl):
        for r in pl.itertuples(index=False):
            plk[(r.fecha, str(r.equipo), str(r.player))] = r
    find_player = _player_finder(plk)

    m = m.dropna(subset=["home_goals"])
    out = []
    for r in m.itertuples(index=False):
        mk = r.mercado
        if mk == "1X2":
            real = "1" if r.home_goals > r.away_goals else ("X" if r.home_goals == r.away_goals else "2")
            hit = float(r.seleccion == real)
        elif mk == "Goles":
            over = (r.home_goals + r.away_goals) > float(r.linea)
            hit = float(over == (r.seleccion == "OVER"))
        elif mk in EVENT_COLS:
            hc, ac = EVENT_COLS[mk]
            hv, av = getattr(r, hc), getattr(r, ac)
            if pd.isna(hv):
                continue
            actual = {"Total": hv + av, "Local": hv, "Visitante": av}[r.ambito]
            over = actual > float(r.linea)
            hit = float(over == (r.seleccion == "OVER"))
        elif mk.startswith("jug_"):
            # `ambito` carries the player for these markets
            act = (find_player(r.fecha, str(r.home), str(r.ambito))
                   or find_player(r.fecha, str(r.away), str(r.ambito)))
            if act is None:
                continue          # player data for this match not published yet
            if mk == "jug_goleador":
                happened = act.goals >= 1
            elif mk == "jug_2goles":
                happened = act.goals >= 2
            elif mk == "jug_tiros":
                happened = act.shots > float(r.linea)
            elif mk == "jug_asistencia":
                happened = act.assists >= 1
            else:
                happened = act.yellow_cards >= 1
            hit = float(bool(happened) == (r.seleccion == "SI"))
        elif mk in ("ht_1x2", "ht_goles", "ht_ft"):
            # Half-time markets, settled from home/away_goals_ht (in the
            # foundation since 2026-09-03). See props/half_time.py.
            hh = getattr(r, "home_goals_ht", float("nan"))
            ah = getattr(r, "away_goals_ht", float("nan"))
            if pd.isna(hh) or pd.isna(ah):
                continue
            ht_res = "1" if hh > ah else ("X" if hh == ah else "2")
            if mk == "ht_1x2":
                hit = float(r.seleccion == ht_res)
            elif mk == "ht_goles":
                over = (hh + ah) > float(r.linea)
                hit = float(over == (r.seleccion == "OVER"))
            else:
                ft_res = ("1" if r.home_goals > r.away_goals
                          else ("X" if r.home_goals == r.away_goals else "2"))
                hit = float(r.seleccion == f"{ht_res}/{ft_res}")
        elif mk == "booking_pts":
            # 10 per yellow + 25 per red, the standard scale the model prices on.
            # This branch used to be a bare `continue` because reds were not in
            # the foundation, so 7% of every logged round was silently
            # unscoreable. home/away_red_cards are now carried through.
            hy, ay = r.home_yellow_cards, r.away_yellow_cards
            hr = getattr(r, "home_red_cards", float("nan"))
            ar = getattr(r, "away_red_cards", float("nan"))
            if pd.isna(hy) or pd.isna(hr):
                continue
            actual = 10.0 * (hy + ay) + 25.0 * (hr + ar)
            over = actual > float(r.linea)
            hit = float(over == (r.seleccion == "OVER"))
        else:
            continue
        # For pick-one markets the stored prob IS the selection's probability;
        # for over/under it is the OVER probability, so the confidence in an
        # UNDER pick is its complement.
        pick_one = mk in ("1X2", "ht_1x2", "ht_ft")
        conf = r.prob if pick_one else max(r.prob, 1 - r.prob)
        label = labels.get(mk, mk)
        linea = str(getattr(r, "linea", "")).strip()
        has_line = bool(linea) and linea.lower() != "nan"
        # Player shot markets are logged at two lines (1.5 and 2.5), and the
        # results table groups by this label: pooled, they became one bucket
        # whose numbers belonged to neither. Team markets stay pooled on purpose
        # -- spelling out every corner and shot line would turn that page into a
        # published price ladder.
        if has_line and mk.startswith("jug_"):
            try:
                label = f"{label} >{float(linea):g}"
            except ValueError:
                pass
        out.append({"mercado": label, "ambito": r.ambito,
                    "lado": r.seleccion, "confianza": conf, "acierto": hit,
                    "jornada": r.jornada, "season": r.season,
                    "linea": linea if has_line else "",
                    # kept for the player panel: `confianza` is max(p, 1-p), which
                    # loses the raw probability, and ranking needs both it and the
                    # fixture to sort players within a match
                    "es_jugador": mk.startswith("jug_"), "mercado_id": mk,
                    "prob": float(r.prob), "partido": r.partido})
    return pd.DataFrame(out)
