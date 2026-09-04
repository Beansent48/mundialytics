from __future__ import annotations

"""Log predictions for matches that have NOT been played yet — a real track record.

WHY THIS EXISTS. The app's "Track record en vivo" was fed by a button that had to
be clicked every week, and it could only ever see matches ALREADY in the
foundation (matchdays are derived from played results). The consequence, found on
2026-08-26: all 430 logged rows were written on 2026-07-23 for matches played
2026-05-23/24 — two months late — by an engine fitted on the full foundation,
i.e. trained on the very matches it was "predicting". Retroactive AND in-sample:
not a track record at all, on the one page whose purpose is to show reliability.

This script fixes the root cause. It pulls the FORWARD fixture list from
fixturedownload (the same source already used for the European competitions),
maps team names through data/curated/fixture_team_aliases.csv, and logs
predictions only for fixtures whose kickoff is still in the future. The engine is
fitted on the foundation, which by construction holds only played matches — so a
prediction written here is genuinely pre-match and genuinely out-of-sample.

Run weekly, before the round kicks off:
    .venv/Scripts/python.exe scripts/log_upcoming_round.py
    .venv/Scripts/python.exe scripts/log_upcoming_round.py --days 10 --dry-run
"""

import argparse
import io
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402
from mundialytics.props.half_time import HalfTimeModel  # noqa: E402
from mundialytics.ratings.elo import EloConfig, EloRater  # noqa: E402

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
ALIASES = ROOT / "data/curated/fixture_team_aliases.csv"
PRED_LOG = ROOT / "data/processed/logs/predictions_log.csv"
LOG_KEYS = ["season", "jornada", "partido", "mercado", "ambito", "linea"]
SLUGS = {"epl": "Premier League", "la-liga": "LaLiga", "serie-a": "Serie A",
         "bundesliga": "Bundesliga", "ligue-1": "Ligue 1"}


def fetch_fixtures(year: int) -> pd.DataFrame:
    """Forward fixture list for the five domestic leagues."""
    alias = dict(pd.read_csv(ALIASES).itertuples(index=False, name=None))
    rows = []
    for slug, comp in SLUGS.items():
        url = f"https://fixturedownload.com/download/{slug}-{year}-UTC.csv"
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200 or len(r.text) < 500:
                print(f"  {slug}: no publicado todavia (HTTP {r.status_code})")
                continue
            r.encoding = "utf-8"
            f = pd.read_csv(io.StringIO(r.text))
        except Exception as exc:
            print(f"  {slug}: FALLA {str(exc)[:70]}")
            continue
        f["date"] = pd.to_datetime(f["Date"], dayfirst=True, errors="coerce")
        f["competition"] = comp
        f["jornada"] = pd.to_numeric(f.get("Round Number"), errors="coerce")
        f["home"] = f["Home Team"].map(alias)
        f["away"] = f["Away Team"].map(alias)
        f["played"] = f["Result"].astype(str).str.contains(r"\d+\s*-\s*\d+", regex=True)
        miss = sorted(set(f.loc[f.home.isna(), "Home Team"])
                      | set(f.loc[f.away.isna(), "Away Team"]))
        if miss:
            print(f"  {slug}: sin alias -> {miss} (esos partidos se omiten)")
        rows.append(f.dropna(subset=["home", "away", "date"]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# Player markets: the model's column -> (log market name, line). `ambito` carries
# the player, since the log's schema has no column of its own for one.
# Deliberately NOT every market the model exposes. Settled over 120 real
# matches, the average confidence per market was: 2+ goals 99.3%, shots 2.5
# 91.0%, assist 93.9%, scorer 92.4%, yellow 86.0%, shots 1.5 82.2%. The high
# ones are almost all "NO" — a given player usually does not score — so being
# right 99% of the time on "he will not score twice" is the base rate, not skill.
# Logging those would lift the track record's headline accuracy from ~65% to
# ~80% without the model being any better, which is padding. Kept: the four
# where the call carries information; dropped: 2+ goals and shots over 2.5.
PLAYER_MARKETS = {
    "p_anytime_scorer":  ("jug_goleador", ""),
    "p_shots_over_1_5":  ("jug_tiros", 1.5),
    "p_assist":          ("jug_asistencia", ""),
    "p_yellow":          ("jug_amarilla", ""),
}
# Only the likely XI. The model returns 26-29 players a side, and `exp_min` is
# minutes-when-featuring rather than minutes-per-fixture spread over the squad:
# 22 Barcelona players clear 45, which cannot be a starting eleven. Ranking by
# exp_min and taking the top 11 is the honest read of "who probably starts".
PLAYERS_PER_TEAM = 11


def _load_player_props():
    """The cached PlayerPropsModel — fitted on Understat history already on disk.

    Worth being precise about the dependency: PREDICTING needs no new data (the
    model is already fitted on 620k player-match rows). Only SETTLEMENT needs the
    current season's player stats, which Understat has not published yet. That is
    a temporary gap in a source that has published every season since 2014 — not
    the booking-points situation, where the data sat on disk for years and the
    evaluator simply never read it.
    """
    import glob

    import joblib
    files = sorted(glob.glob(str(ROOT / "data/processed/cache/props_models_*.joblib")))
    if not files:
        print("  player props: sin cache (arranca la app una vez para generarla)")
        return None
    try:
        _, pp = joblib.load(files[-1])
    except Exception as exc:
        print(f"  player props: cache ilegible ({str(exc)[:50]})")
        return None
    if pp is None:
        print("  player props: el cache no trae modelo de jugador")
    else:
        print("  player props: OK")
    return pp


def player_rows(pp, base: dict, team: str, lam: float) -> list[dict]:
    """Per-player markets for one side of a fixture."""
    try:
        out = pp.team_players_for_lambda(team, float(lam))
    except Exception:
        return []
    if out is None or out.empty:
        return []
    out = out.nlargest(PLAYERS_PER_TEAM, "exp_min")
    rows = []
    for r in out.itertuples(index=False):
        for col, (mk, line) in PLAYER_MARKETS.items():
            p = getattr(r, col, None)
            if p is None or pd.isna(p):
                continue
            p = float(p)
            rows.append({**base, "mercado": mk, "ambito": str(r.player),
                         "linea": line, "prob": round(p, 4),
                         "seleccion": "SI" if p >= 0.5 else "NO"})
    return rows


def european_rows(year: int, horizon: pd.Timestamp, now: pd.Timestamp,
                  stamp: str, season: str) -> list[dict]:
    """Pre-kickoff predictions for upcoming UCL/UEL/UECL ties.

    The European layer prices off ClubElo rather than the domestic engine — it
    is the only cross-league scale, and its Elo->goals constants were calibrated
    on 1,000 real European matches. Ratings come from the locally-advanced table
    (ratings/clubelo_local.py), so this keeps working while the ClubElo API is
    down.

    Clubs the rating table cannot cover are skipped, never guessed.
    """
    from mundialytics.statistical_core.competition.european import (
        FD_SLUG, fetch_current_elo, fetch_season_fixtures, load_calibration,
        load_event_calibration, make_resolver, predict_euro_events)
    from mundialytics.statistical_core.distributions import scoreline_distribution

    try:
        elo = fetch_current_elo(ROOT)
        calib = load_calibration(ROOT)
    except Exception as exc:
        print(f"  europa no disponible: {str(exc)[:70]}")
        return []
    ev_calib = load_event_calibration(ROOT)
    resolver = make_resolver(elo.keys())
    htm = HalfTimeModel()
    c, hfa, b = calib["c"], calib["hfa"], calib["b"]

    rows: list[dict] = []
    for comp, slug in FD_SLUG.items():
        fx = fetch_season_fixtures(ROOT, comp, year)
        if fx is None or fx.empty:
            continue
        fx = fx.copy()
        fx["date"] = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce")
        fx["played"] = fx["Result"].astype(str).str.contains(r"\d+\s*-\s*\d+", regex=True)
        up = fx[(~fx.played) & (fx.date > now) & (fx.date <= horizon)]
        if up.empty:
            continue
        skipped = []
        n_before = len(rows)
        for _, row in up.iterrows():
            # explicit column names: fixturedownload's headers contain spaces,
            # so itertuples renames them positionally (_1, _2, ...) and the
            # indices shift whenever the file's layout changes
            h_raw, a_raw = row["Home Team"], row["Away Team"]
            rnd = row.get("Round Number")
            r_date = row["date"]
            h, a = resolver(str(h_raw)), resolver(str(a_raw))
            if not h or not a or h not in elo or a not in elo:
                skipped.append(f"{h_raw} vs {a_raw}")
                continue
            d400 = (elo[h] - elo[a]) / 400.0
            lam_h = float(np.exp(c + hfa + b * d400))
            lam_a = float(np.exp(c - b * d400))
            dist = scoreline_distribution(lam_h, lam_a, max_goals=10, normalize=True)
            label = f"{str(h_raw).lower()} vs {str(a_raw).lower()}"
            base = dict(logged_at=stamp, season=season,
                        jornada=int(rnd) if pd.notna(rnd) else 0,
                        partido=label, fecha=f"{r_date:%Y-%m-%d}",
                        home=str(h_raw).lower(), away=str(a_raw).lower())
            trio = {"1": dist.p_home_win, "X": dist.p_draw, "2": dist.p_away_win}
            pick = max(trio, key=trio.get)
            rows.append({**base, "mercado": "1X2", "ambito": "Total", "linea": "",
                         "prob": round(trio[pick], 4), "seleccion": pick})
            # scoreline_distribution exposes no p_over_25 attribute; ask its own
            # total-goals helper rather than inventing a number
            p_o25 = float(dist.total_goals_probability(2.5, "over"))
            rows.append({**base, "mercado": "Goles", "ambito": "Total", "linea": 2.5,
                         "prob": round(p_o25, 4),
                         "seleccion": "OVER" if p_o25 >= 0.5 else "UNDER"})
            paths = htm.predict_ht_ft(lam_h, lam_a)
            bestp = max(paths, key=paths.get)
            rows.append({**base, "mercado": "ht_ft", "ambito": "Total", "linea": "",
                         "prob": round(paths[bestp], 4), "seleccion": bestp})
            ht = htm.predict_half_time(lam_h, lam_a)
            ht_trio = {"1": ht["p_home"], "X": ht["p_draw"], "2": ht["p_away"]}
            hp = max(ht_trio, key=ht_trio.get)
            rows.append({**base, "mercado": "ht_1x2", "ambito": "Total", "linea": "",
                         "prob": round(ht_trio[hp], 4), "seleccion": hp})
            for ln, pr in ht["over"].items():
                rows.append({**base, "mercado": "ht_goles", "ambito": "Total", "linea": ln,
                             "prob": round(float(pr), 4),
                             "seleccion": "OVER" if pr >= 0.5 else "UNDER"})
            if ev_calib:
                for mk, dd in predict_euro_events(elo[h], elo[a], ev_calib).items():
                    for ln, pr in dd.get("over", {}).items():
                        rows.append({**base, "mercado": mk, "ambito": "Total", "linea": ln,
                                     "prob": round(float(pr), 4),
                                     "seleccion": "OVER" if pr >= 0.5 else "UNDER"})
        print(f"  {comp:12s} {len(up):3d} partidos -> {len(rows)-n_before} predicciones"
              + (f"  (omitidos {len(skipped)}: {skipped[:3]})" if skipped else ""))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=8, help="ventana hacia delante")
    ap.add_argument("--dry-run", action="store_true", help="no escribe el log")
    ap.add_argument("--year", type=int, default=None, help="temporada fixturedownload")
    ap.add_argument("--skip-europe", action="store_true", help="solo ligas domesticas")
    ap.add_argument("--skip-players", action="store_true", help="omitir props de jugador")
    args = ap.parse_args()

    now = pd.Timestamp.now()
    year = args.year or (now.year if now.month >= 7 else now.year - 1)
    print(f"fixtures {year}/{year+1}, ventana {args.days} dias desde {now:%Y-%m-%d %H:%M}")

    fx = fetch_fixtures(year)
    if fx.empty:
        print("Sin fixtures. Nada que registrar.")
        return

    horizon = now + timedelta(days=args.days)
    up = fx[(~fx.played) & (fx.date > now) & (fx.date <= horizon)].sort_values("date")
    print(f"\npartidos por jugar en la ventana: {len(up)}")
    if up.empty:
        print("Nada dentro de la ventana. Prueba --days mayor.")
        return
    for c, g in up.groupby("competition"):
        js = sorted(set(g.jornada.dropna().astype(int)))
        print(f"  {c:16s} {len(g):3d} partidos  jornada(s) {js}")

    # engine fitted on PLAYED matches only -> predictions are genuinely pre-match
    df = pd.read_csv(FOUND, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print(f"\nmotor: ajustando sobre {len(df):,} partidos jugados "
          f"(hasta {df.date.max():%Y-%m-%d})...", flush=True)
    elo = EloRater(EloConfig(season_reset_fraction=0.40))
    elo.fit(df)
    eng = PredictionEngine(blend_weight_gl=0.30, ad_rho=-0.07, sharpen_gamma_1x2=1.3,
                           rescale_lambda_to_goals=True, outcome_rho=-0.17,
                           xg_rate_kwargs={"use_ewma": True})
    eng.fit(df, elo_history=pd.DataFrame(elo.history))
    known = set(df.home_team) | set(df.away_team)

    htm = HalfTimeModel()   # stateless: a scaling of the lambdas above

    tp = None
    try:
        from mundialytics.props import TeamPropsModel
        tp = TeamPropsModel().fit(df, root=ROOT)
        print("  team props: OK")
    except Exception as exc:
        print(f"  team props no disponibles ({str(exc)[:60]}) — solo 1X2/Goles")

    pp = None
    if not args.skip_players:
        pp = _load_player_props()

    season = f"{year}-{year+1}"
    stamp = now.isoformat(timespec="seconds")
    rows, skipped = [], []
    for r in up.itertuples(index=False):
        if r.home not in known or r.away not in known:
            skipped.append(f"{r.home} vs {r.away}")
            continue
        base = dict(logged_at=stamp, season=season,
                    jornada=int(r.jornada) if pd.notna(r.jornada) else 0,
                    partido=f"{r.home} vs {r.away}", fecha=f"{r.date:%Y-%m-%d}",
                    home=r.home, away=r.away)
        try:
            p = eng.predict_match(r.home, r.away, competition=r.competition, neutral=False)
        except Exception:
            skipped.append(f"{r.home} vs {r.away}")
            continue
        trio = {"1": p.p_home_win, "X": p.p_draw, "2": p.p_away_win}
        pick = max(trio, key=trio.get)
        rows.append({**base, "mercado": "1X2", "ambito": "Total", "linea": "",
                     "prob": round(trio[pick], 4), "seleccion": pick})
        rows.append({**base, "mercado": "Goles", "ambito": "Total", "linea": 2.5,
                     "prob": round(p.p_over_25, 4),
                     "seleccion": "OVER" if p.p_over_25 >= 0.5 else "UNDER"})

        # Half-time markets, derived from the same lambdas (see props/half_time.py).
        ht = htm.predict_half_time(p.lambda_home, p.lambda_away)
        ht_trio = {"1": ht["p_home"], "X": ht["p_draw"], "2": ht["p_away"]}
        ht_pick = max(ht_trio, key=ht_trio.get)
        rows.append({**base, "mercado": "ht_1x2", "ambito": "Total", "linea": "",
                     "prob": round(ht_trio[ht_pick], 4), "seleccion": ht_pick})
        for ln, pr in ht["over"].items():
            rows.append({**base, "mercado": "ht_goles", "ambito": "Total", "linea": ln,
                         "prob": round(float(pr), 4),
                         "seleccion": "OVER" if pr >= 0.5 else "UNDER"})
        paths = htm.predict_ht_ft(p.lambda_home, p.lambda_away)
        best = max(paths, key=paths.get)
        rows.append({**base, "mercado": "ht_ft", "ambito": "Total", "linea": "",
                     "prob": round(paths[best], 4), "seleccion": best})

        if tp is None:
            continue
        try:
            fxp = tp.predict_fixture(r.home, r.away,
                                     lam_home=p.lambda_home, lam_away=p.lambda_away)
        except Exception:
            continue
        if pp is not None:
            rows += player_rows(pp, base, r.home, p.lambda_home)
            rows += player_rows(pp, base, r.away, p.lambda_away)
        for mk, dd in fxp.items():
            if not isinstance(dd, dict):
                continue
            for ln, pr in dd.get("over", {}).items():
                rows.append({**base, "mercado": mk, "ambito": "Total", "linea": ln,
                             "prob": round(float(pr), 4),
                             "seleccion": "OVER" if pr >= 0.5 else "UNDER"})
            for skey, amb in [("over_home", "Local"), ("over_away", "Visitante")]:
                for ln, pr in dd.get(skey, {}).items():
                    rows.append({**base, "mercado": mk, "ambito": amb, "linea": ln,
                                 "prob": round(float(pr), 4),
                                 "seleccion": "OVER" if pr >= 0.5 else "UNDER"})

    if not args.skip_europe:
        print("\ncompeticiones europeas:")
        rows += european_rows(year, horizon, now, stamp, season)

    if skipped:
        print(f"\nomitidos (equipo sin historial): {len(skipped)} -> {skipped[:6]}")
    new = pd.DataFrame(rows)
    n_matches = new.partido.nunique() if len(new) else 0
    print(f"predicciones generadas: {len(new):,} filas sobre {n_matches} partidos")
    if new.empty:
        return

    if args.dry_run:
        print("\n--dry-run: no se escribe nada. Muestra:")
        print(new.head(8).to_string(index=False))
        return

    PRED_LOG.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(PRED_LOG) if PRED_LOG.exists() else pd.DataFrame()
    comb = pd.concat([old, new], ignore_index=True) if len(old) else new
    # 1X2 rows carry linea="", which round-trips through CSV as NaN. Casting
    # straight to str would turn those into "nan" for the stored rows and "" for
    # the fresh ones, so they never matched and every re-run appended one
    # duplicate per fixture. Normalise the missing value BEFORE stringifying.
    comb["linea"] = comb["linea"].fillna("").astype(str).replace("nan", "")
    comb = comb.drop_duplicates(subset=LOG_KEYS, keep="first")  # NOT + seleccion:
    # including the pick in the key let the SAME line survive twice once its
    # probability crossed 0.5 between runs, logging both OVER and UNDER for it.
    comb.to_csv(PRED_LOG, index=False)
    print(f"\nREGISTRADAS {len(comb) - len(old):,} filas nuevas -> {PRED_LOG} "
          f"(total {len(comb):,})")
    print("Todas con fecha de partido en el futuro: track record honesto.")


if __name__ == "__main__":
    main()
