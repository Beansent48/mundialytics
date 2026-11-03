"""
Fail if a logged prediction whose result exists was never settled.

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

The "is there a result" question is answered by `attach_results`, the same
function settlement uses, so this cannot disagree with what it audits.

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
    PRED_LOG,
    attach_results,
    evaluate_log,
)

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
CAL_DIR = ROOT / "data/processed/fixture_calendars"
KNOWN = ROOT / "data/curated/settlement_known_gaps.csv"


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


def main() -> int:
    if not PRED_LOG.exists():
        print("No hay log de predicciones.")
        return 2

    log = pd.read_csv(PRED_LOG, low_memory=False)
    season = str(log["season"].dropna().max())
    print(f"=== AUDITORIA DE LIQUIDACION · temporada {season} ===\n")

    found = pd.read_csv(FOUND, low_memory=False)
    found["date"] = pd.to_datetime(found["date"], errors="coerce")

    # Same join settlement uses, so "unsettled" means the same thing here.
    m = attach_results(log, found)
    m = m[m["season"] == season]
    print(f"Predicciones registradas  : {len(m)}")
    print(f"  con resultado disponible: {int(m['home_goals'].notna().sum())}")
    print(f"  sin resultado           : {int(m['home_goals'].isna().sum())}\n")

    ev = evaluate_log(found)
    ev = ev[ev["season"] == season] if len(ev) else ev
    print(f"Predicciones puntuadas    : {len(ev)}\n")

    # A market that scores nothing is the loudest possible symptom.
    logged = m.groupby("mercado").size()
    scored = ev.groupby("mercado_id").size() if len(ev) else pd.Series(dtype=int)
    rows, dead = [], []
    for mk, n in logged.sort_values(ascending=False).items():
        s = int(scored.get(mk, 0))
        rows.append({"mercado": mk, "registradas": int(n), "puntuadas": s,
                     "pct": round(100.0 * s / n, 1) if n else 0.0})
        if s == 0:
            dead.append(str(mk))
    print("--- por mercado ---")
    print(pd.DataFrame(rows).to_string(index=False))

    # Fixture level: a result exists and we did not use it.
    later = _not_played_pairings()
    known = _known_gaps()
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    holes, pending = [], 0
    for r in (m[m["home_goals"].isna()][["fecha", "home", "away"]]
              .drop_duplicates().itertuples(index=False)):
        if str(r.fecha) >= today:
            pending += 1
            continue
        if (str(r.home), str(r.away)) in later:
            pending += 1
            continue
        if (str(r.fecha), str(r.home), str(r.away)) in known:
            continue
        holes.append({"fecha": r.fecha, "home": r.home, "away": r.away})

    print(f"\nPartidos sin jugar o aplazados: {pending}")
    print("\n--- huecos sin explicar ---")
    print(pd.DataFrame(holes).to_string(index=False) if holes else "ninguno")

    bad = False
    if dead:
        print(f"\nFALLO: estos mercados no puntuan NADA: {', '.join(dead)}")
        bad = True
    if holes:
        print(f"\nFALLO: {len(holes)} partidos jugados cuya prediccion no se "
              f"comprueba. Arregla la liquidacion, o anotalos en "
              f"{KNOWN.relative_to(ROOT).as_posix()} con su causa.")
        bad = True
    if not bad:
        print("\nOK: toda prediccion con resultado disponible esta comprobada.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
