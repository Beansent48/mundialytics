"""
Fail if a logged prediction that could have been checked was not.

`audit_prediction_coverage.py` asks the opposite question — did every played
match get a prediction. This one asks whether every prediction got checked,
which is how three holes went unnoticed for weeks in 2026/27:

  * every European row stayed unsettled, because the UEFA results carry ClubElo
    names and the log carries foundation names (162 rows, 0 matched);
  * four fixtures moved by a day after being logged, and an exact-date join
    dropped them;
  * the foundation trails the fixture calendars, so the newest round could not
    settle at all.

A hole here is worse than a crash: the track record keeps publishing a number,
just over fewer matches than it claims.

Two levels, because the first one alone let the European hole through for weeks:

  1. **Fixture level.** Is there a result we failed to use? The question is
     answered by `attach_results`, the same function settlement uses, so this
     cannot disagree with what it audits.
  2. **Row level.** Every row that has a result but was not scored must name the
     column it is missing — half-time goals, an event count, player rows. Rows
     that cannot name one are a hole, even while the market's overall rate looks
     plausible. "Explainable" is exactly what the European rows looked like.

Exit codes: 0 clean · 1 an unexplained hole · 2 could not run.
Accepted holes live in data/curated/settlement_known_gaps.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.serving.track_record import (  # noqa: E402
    EVENT_COLS,
    PRED_LOG,
    _player_finder,
    _player_match_actuals,
    attach_results,
    evaluate_log,
)

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
CAL_DIR = ROOT / "data/processed/fixture_calendars"
KNOWN = ROOT / "data/curated/settlement_known_gaps.csv"

HT_MARKETS = ("ht_1x2", "ht_goles", "ht_ft")
GOAL_MARKETS = ("1X2", "Goles")


def _known_gaps() -> set[tuple[str, str, str]]:
    if not KNOWN.exists():
        return set()
    k = pd.read_csv(KNOWN)
    return {(str(r.date), str(r.home), str(r.away))
            for r in k.itertuples(index=False)}


def _not_played_pairings() -> set[tuple[str, str]]:
    """Pairings the calendars still list as unplayed, in any round.

    A fixture postponed past its logged date is not a hole, it has no result
    yet: vallecano vs ath bilbao was logged for 2026-10-10 and the calendar
    moved it to 2026-12-09.
    """
    out: set[tuple[str, str]] = set()
    if not CAL_DIR.exists():
        return out
    for p in CAL_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if not {"home_team", "away_team", "completed"} <= set(df.columns):
            continue
        open_rows = df[df["completed"].astype(str).str.lower() != "true"]
        for r in open_rows.itertuples(index=False):
            out.add((str(r.home_team), str(r.away_team)))
    return out


def _missing_column(row, find_player) -> str | None:
    """Why this row cannot be scored although its fixture has a result.

    None means it should have been scored. The branches mirror `evaluate_log`:
    whatever makes it `continue` there has to name itself here, or the shortfall
    is a hole rather than a limitation.
    """
    mk = str(row.mercado)
    if mk in GOAL_MARKETS:
        return None                       # goals are present by construction
    if mk in EVENT_COLS:
        hc, _ = EVENT_COLS[mk]
        return None if not pd.isna(getattr(row, hc, float("nan"))) else f"falta {hc}"
    if mk in HT_MARKETS:
        hh = getattr(row, "home_goals_ht", float("nan"))
        ah = getattr(row, "away_goals_ht", float("nan"))
        return None if not (pd.isna(hh) or pd.isna(ah)) else "falta marcador al descanso"
    if mk == "booking_pts":
        hy = getattr(row, "home_yellow_cards", float("nan"))
        hr = getattr(row, "home_red_cards", float("nan"))
        return None if not (pd.isna(hy) or pd.isna(hr)) else "faltan tarjetas"
    if mk.startswith("jug_"):
        act = (find_player(row.fecha, str(row.home), str(row.ambito))
               or find_player(row.fecha, str(row.away), str(row.ambito)))
        return None if act is not None else "faltan datos de jugador"
    return f"mercado desconocido: {mk}"


def main() -> int:
    if not PRED_LOG.exists():
        print("No hay log de predicciones.")
        return 2

    log = pd.read_csv(PRED_LOG, low_memory=False)
    season = str(log["season"].dropna().max())
    print(f"=== AUDITORIA DE LIQUIDACION · temporada {season} ===\n")

    found = pd.read_csv(FOUND, low_memory=False)
    found["date"] = pd.to_datetime(found["date"], errors="coerce")

    m = attach_results(log, found)
    m = m[m["season"] == season].copy()
    with_res = m[m["home_goals"].notna()]
    print(f"Predicciones registradas  : {len(m)}")
    print(f"  con resultado disponible: {len(with_res)}")
    print(f"  sin resultado           : {int(m['home_goals'].isna().sum())}\n")

    ev = evaluate_log(found)
    ev = ev[ev["season"] == season] if len(ev) else ev
    print(f"Predicciones puntuadas    : {len(ev)}\n")

    # ---------- level 1: is there a result we failed to use? ----------
    later = _not_played_pairings()
    known = _known_gaps()
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    holes, pending = [], 0
    for r in (m[m["home_goals"].isna()][["fecha", "home", "away"]]
              .drop_duplicates().itertuples(index=False)):
        if str(r.fecha) >= today or (str(r.home), str(r.away)) in later:
            pending += 1
        elif (str(r.fecha), str(r.home), str(r.away)) not in known:
            holes.append({"fecha": r.fecha, "home": r.home, "away": r.away})

    print(f"Partidos sin jugar o aplazados: {pending}")
    print("\n--- partidos jugados sin comprobar ---")
    print(pd.DataFrame(holes).to_string(index=False) if holes else "ninguno")

    # ---------- level 2: does every unscored row name its missing column? ----------
    pl = _player_match_actuals()
    plk = {}
    for r in pl.itertuples(index=False):
        plk[(r.fecha, str(r.equipo), str(r.player))] = r
    find_player = _player_finder(plk)

    reasons: dict[str, dict[str, int]] = {}
    expected: dict[str, int] = {}
    for row in with_res.itertuples(index=False):
        mk = str(row.mercado)
        why = _missing_column(row, find_player)
        if why is None:
            expected[mk] = expected.get(mk, 0) + 1
        else:
            reasons.setdefault(mk, {})
            reasons[mk][why] = reasons[mk].get(why, 0) + 1

    scored = ev.groupby("mercado_id").size() if len(ev) else pd.Series(dtype=int)
    rows, shortfalls = [], []
    for mk in sorted(set(m["mercado"].astype(str))):
        logged = int((m["mercado"].astype(str) == mk).sum())
        exp = expected.get(mk, 0)
        act = int(scored.get(mk, 0))
        why = reasons.get(mk, {})
        rows.append({
            "mercado": mk, "registradas": logged, "puntuables": exp,
            "puntuadas": act, "falta": exp - act,
            "no puntuables": "; ".join(f"{k} ({v})" for k, v in sorted(why.items())) or "-",
        })
        if act < exp:
            shortfalls.append((mk, exp - act))

    print("\n--- por mercado ---")
    print(pd.DataFrame(rows).to_string(index=False))

    bad = False
    if holes:
        print(f"\nFALLO: {len(holes)} partidos jugados cuya prediccion no se "
              f"comprueba. Arregla la liquidacion, o anotalos en "
              f"{KNOWN.relative_to(ROOT).as_posix()} con su causa.")
        bad = True
    if shortfalls:
        print("\nFALLO: estos mercados puntuan menos filas de las que pueden, "
              "y ninguna razon lo explica:")
        for mk, n in shortfalls:
            print(f"  {mk}: faltan {n}")
        bad = True
    if not bad:
        print("\nOK: toda prediccion comprobable esta comprobada, y cada fila "
              "no puntuada nombra la columna que le falta.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
