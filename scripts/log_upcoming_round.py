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

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=8, help="ventana hacia delante")
    ap.add_argument("--dry-run", action="store_true", help="no escribe el log")
    ap.add_argument("--year", type=int, default=None, help="temporada fixturedownload")
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

    tp = None
    try:
        from mundialytics.props import TeamPropsModel
        tp = TeamPropsModel().fit(df, root=ROOT)
        print("  team props: OK")
    except Exception as exc:
        print(f"  team props no disponibles ({str(exc)[:60]}) — solo 1X2/Goles")

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
        if tp is None:
            continue
        try:
            fxp = tp.predict_fixture(r.home, r.away,
                                     lam_home=p.lambda_home, lam_away=p.lambda_away)
        except Exception:
            continue
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
    comb = comb.drop_duplicates(subset=LOG_KEYS + ["seleccion"], keep="first")
    comb.to_csv(PRED_LOG, index=False)
    print(f"\nREGISTRADAS {len(comb) - len(old):,} filas nuevas -> {PRED_LOG} "
          f"(total {len(comb):,})")
    print("Todas con fecha de partido en el futuro: track record honesto.")


if __name__ == "__main__":
    main()
