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


def _european_results() -> pd.DataFrame:
    """Real European results from the cached season CSVs, ClubElo names."""
    try:
        from mundialytics.statistical_core.competition.european import (
            fetch_current_elo, make_resolver)
        resolver = make_resolver(list(fetch_current_elo(ROOT)))
    except Exception:
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
    extra = [c for c in ("home_red_cards", "away_red_cards",
                         "home_goals_ht", "away_goals_ht") if c in df_clubs.columns]
    res = df_clubs[["home_team", "away_team", "date", "home_goals", "away_goals"]
                   + [c for cc in EVENT_COLS.values() for c in cc] + extra].copy()
    res["fecha"] = res["date"].astype(str).str[:10]
    m = log.merge(res.rename(columns={"home_team": "home", "away_team": "away"}),
                  on=["home", "away", "fecha"], how="left")
    # European rows won't match the domestic foundation — fill from the UEFA results
    eu_res = _european_results()
    if len(eu_res):
        miss = m["home_goals"].isna()
        fill = (m.loc[miss, ["home", "away", "fecha"]]
                .merge(eu_res, on=["home", "away", "fecha"], how="left"))
        m.loc[miss, "home_goals"] = fill["home_goals"].to_numpy()
        m.loc[miss, "away_goals"] = fill["away_goals"].to_numpy()
    # player markets settle from a different source (Understat player-match),
    # so they are joined separately rather than through the team-level frame
    pl = _player_match_actuals()
    plk = {}
    if len(pl):
        for r in pl.itertuples(index=False):
            plk[(r.fecha, str(r.equipo), str(r.player))] = r

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
            act = (plk.get((r.fecha, str(r.home), str(r.ambito)))
                   or plk.get((r.fecha, str(r.away), str(r.ambito))))
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
