from __future__ import annotations

"""Dress rehearsal for the live track record, before real results land.

In a few days the Resultados page will show, publicly, "acierto real vs acierto
anunciado" for the 4,321 predictions now logged. Two things are worth knowing
BEFORE that moment rather than after:

  1. Does the settlement path actually work end to end on real matches? It has
     never run on genuine results. Its bugs so far were found by inspection, not
     by use: booking points silently unsettleable for want of red cards, and a
     confidence formula that inflated pick-one markets whenever p < 0.5.
  2. Is each market calibrated NOW? The props were validated in July on
     different folds. A market that claims 70% and hits 55% would be embarrassing
     in public and is worth catching here.

Method: fit on everything before a cutoff, predict the whole following season
with the SAME calls the logger makes, settle with the SAME rules the app uses,
and report calibration per market. Honest out-of-sample — the models never see
the season they are scored on.

Run: .venv/Scripts/python.exe scripts/rehearse_track_record.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.props.half_time import HalfTimeModel  # noqa: E402
from mundialytics.ratings.elo import EloConfig, EloRater  # noqa: E402
from mundialytics.statistical_core.prediction_engine import PredictionEngine  # noqa: E402

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
CUTOFF = "2025-08-01"
EVENT_COLS = {"corners": ("home_corners", "away_corners"),
              "yellows": ("home_yellow_cards", "away_yellow_cards"),
              "fouls": ("home_fouls", "away_fouls"),
              "shots": ("home_shots", "away_shots"),
              "sot": ("home_sot", "away_sot")}


def settle(mk: str, sel: str, line, row, ambito: str = "Total") -> float | None:
    """Mirror of app.evaluate_prediction_log's rules.

    Must cover EVERYTHING the logger writes, including booking points and the
    per-side lines — a rehearsal that skips a published market proves nothing
    about it. The first version of this mirror omitted both, which is how 4,926
    of 53,314 predictions came back unsettled.
    """
    hg, ag = row["home_goals"], row["away_goals"]
    if mk == "1X2":
        real = "1" if hg > ag else ("X" if hg == ag else "2")
        return float(sel == real)
    if mk == "Goles":
        return float(((hg + ag) > float(line)) == (sel == "OVER"))
    if mk == "booking_pts":
        hy, ay = row.get("home_yellow_cards"), row.get("away_yellow_cards")
        hr, ar = row.get("home_red_cards"), row.get("away_red_cards")
        if pd.isna(hy) or pd.isna(hr):
            return None
        actual = 10.0 * (hy + ay) + 25.0 * (hr + ar)
        return float((actual > float(line)) == (sel == "OVER"))
    if mk in EVENT_COLS:
        hc, ac = EVENT_COLS[mk]
        if pd.isna(row.get(hc)):
            return None
        actual = {"Total": row[hc] + row[ac], "Local": row[hc],
                  "Visitante": row[ac]}[ambito]
        return float((actual > float(line)) == (sel == "OVER"))
    if mk in ("ht_1x2", "ht_goles", "ht_ft"):
        hh, ah = row.get("home_goals_ht"), row.get("away_goals_ht")
        if pd.isna(hh):
            return None
        ht = "1" if hh > ah else ("X" if hh == ah else "2")
        if mk == "ht_1x2":
            return float(sel == ht)
        if mk == "ht_goles":
            return float(((hh + ah) > float(line)) == (sel == "OVER"))
        ft = "1" if hg > ag else ("X" if hg == ag else "2")
        return float(sel == f"{ht}/{ft}")
    return None


def main() -> None:
    df = pd.read_csv(FOUND, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_goals", "away_goals"]).sort_values("date")
    train = df[df.date < CUTOFF]
    test = df[(df.date >= CUTOFF) & (df.date < "2026-07-01")]
    print(f"entreno: {len(train):,} partidos (hasta {CUTOFF})")
    print(f"evaluo : {len(test):,} partidos ({test.date.min():%Y-%m-%d} -> {test.date.max():%Y-%m-%d})")

    elo = EloRater(EloConfig(season_reset_fraction=0.40))
    elo.fit(train)
    eng = PredictionEngine(blend_weight_gl=0.30, ad_rho=-0.07, sharpen_gamma_1x2=1.3,
                           rescale_lambda_to_goals=True, outcome_rho=-0.17,
                           xg_rate_kwargs={"use_ewma": True})
    eng.fit(train, elo_history=pd.DataFrame(elo.history))
    htm = HalfTimeModel()
    try:
        from mundialytics.props import TeamPropsModel
        tp = TeamPropsModel().fit(train, root=ROOT)
    except Exception as exc:
        print(f"  team props no disponibles: {str(exc)[:60]}")
        tp = None
    print("modelos ajustados; prediciendo...", flush=True)

    rows = []
    for r in test.itertuples(index=False):
        try:
            p = eng.predict_match(r.home_team, r.away_team,
                                  competition=str(r.competition), neutral=False)
        except Exception:
            continue
        row = r._asdict()
        trio = {"1": p.p_home_win, "X": p.p_draw, "2": p.p_away_win}
        pick = max(trio, key=trio.get)
        rows.append(("1X2", "", pick, trio[pick], row, "Total"))
        rows.append(("Goles", 2.5, "OVER" if p.p_over_25 >= .5 else "UNDER",
                     max(p.p_over_25, 1 - p.p_over_25), row, "Total"))
        ht = htm.predict_half_time(p.lambda_home, p.lambda_away)
        ht_trio = {"1": ht["p_home"], "X": ht["p_draw"], "2": ht["p_away"]}
        hp = max(ht_trio, key=ht_trio.get)
        rows.append(("ht_1x2", "", hp, ht_trio[hp], row, "Total"))
        for ln, pr in ht["over"].items():
            rows.append(("ht_goles", ln, "OVER" if pr >= .5 else "UNDER",
                         max(pr, 1 - pr), row, "Total"))
        paths = htm.predict_ht_ft(p.lambda_home, p.lambda_away)
        bp = max(paths, key=paths.get)
        rows.append(("ht_ft", "", bp, paths[bp], row, "Total"))
        if tp is not None:
            try:
                fx = tp.predict_fixture(r.home_team, r.away_team,
                                        lam_home=p.lambda_home, lam_away=p.lambda_away)
            except Exception:
                fx = {}
            for mk, d in fx.items():
                if not isinstance(d, dict):
                    continue
                for key, amb in (("over", "Total"), ("over_home", "Local"),
                                 ("over_away", "Visitante")):
                    for ln, pr in d.get(key, {}).items():
                        rows.append((mk, ln, "OVER" if pr >= .5 else "UNDER",
                                     max(pr, 1 - pr), row, amb))

    print(f"predicciones generadas: {len(rows):,}")

    out = []
    for mk, ln, sel, conf, row, amb in rows:
        hit = settle(mk, sel, ln, row, amb)
        if hit is not None:
            out.append({"mercado": mk, "ambito": amb,
                        "confianza": float(conf), "acierto": hit})
    ev = pd.DataFrame(out)
    print(f"liquidadas: {len(ev):,}  ({len(rows)-len(ev):,} sin datos para liquidar)")

    print("\n=== LO QUE MOSTRARA LA PAGINA (por mercado) ===")
    print(f"  {'mercado':12s} {'n':>6s} {'anunciado':>10s} {'real':>8s} {'brecha':>8s}")
    for mk, g in ev.groupby("mercado"):
        exp, act = g.confianza.mean(), g.acierto.mean()
        flag = "  <-- REVISAR" if abs(exp - act) > 0.05 else ""
        print(f"  {mk:12s} {len(g):6d} {exp:10.1%} {act:8.1%} {act-exp:+8.1%}{flag}")
    exp, act = ev.confianza.mean(), ev.acierto.mean()
    print(f"  {'GLOBAL':12s} {len(ev):6d} {exp:10.1%} {act:8.1%} {act-exp:+8.1%}")

    print("\n=== POR BANDA DE CONFIANZA (calibracion) ===")
    ev["banda"] = pd.cut(ev.confianza, [0.5, 0.55, 0.6, 0.7, 0.8, 1.0],
                         labels=["50-55", "55-60", "60-70", "70-80", "80+"])
    print(f"  {'banda':8s} {'n':>6s} {'anunciado':>10s} {'real':>8s} {'brecha':>8s}")
    ece = 0.0
    for b, g in ev.groupby("banda", observed=True):
        exp, act = g.confianza.mean(), g.acierto.mean()
        ece += len(g) / len(ev) * abs(exp - act)
        print(f"  {b:8s} {len(g):6d} {exp:10.1%} {act:8.1%} {act-exp:+8.1%}")
    print(f"\n  ECE (error de calibracion esperado) = {ece:.4f}")
    print("  -> " + ("bien calibrado (<0.02)" if ece < 0.02 else
                     "DESCALIBRADO: revisar antes de publicar"))


if __name__ == "__main__":
    main()
