from __future__ import annotations

"""Market drift report: is any market drifting away from what actually happens?

The 2026/27 shots drift was in the log from round 2: the model said 63% of
matches would go over, 77% did, in all five leagues. Nobody saw it until a
one-off scorecard on 2026-09-29, because the track record reports hit rates
and hit rates hide a bias that pushes every line the same way. This report
looks for exactly that, every day, over the pre-kickoff log:

  counts       every count the log carries an expectation for (goals, shots,
               shots on target, corners, fouls, yellows): mean actual minus
               mean expected per match, per league, as a z-score
  calibration  every over/under line and player market: mean predicted
               probability against the share that happened, as a z-score
               (sum of y - p over sqrt(sum p(1-p)))

Windows: the current season so far and the last 28 days. A row is an ALERT
when |z| >= 3 on at least MIN_N observations. Informational only: it never
changes a prediction. Written to data/processed/logs/market_drift.json.

    python scripts/report_market_drift.py
    python scripts/report_market_drift.py --as-of 2026-08-31   # what it said then
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.serving.track_record import PRED_LOG, evaluate_log  # noqa: E402

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
OUT = ROOT / "data/processed/logs/market_drift.json"
Z_ALERT = 3.0
MIN_N = 60
# expectation column pairs in the log -> actual column pairs in the foundation.
# For the five event counts the log carries two expectations: tp_* from
# TeamPropsModel (the model the board is actually priced with) and exp_* from the
# engine's own EventLambdaModel. tp_* wins where present -- a drift report should
# measure what we serve -- and exp_* covers the rows logged before 2026-10-06,
# when only it existed. Mixing the two in one series is the point: the question
# the report answers is "is the board drifting", not "which model was it".
COUNTS = {
    "goals": ((("lambda_home",), ("lambda_away",)), ("home_goals", "away_goals")),
    "shots": ((("tp_shots_home", "exp_shots_home"), ("tp_shots_away", "exp_shots_away")),
              ("home_shots", "away_shots")),
    "sot": ((("tp_sot_home", "exp_sot_home"), ("tp_sot_away", "exp_sot_away")),
            ("home_sot", "away_sot")),
    "corners": ((("tp_corners_home", "exp_corners_home"), ("tp_corners_away", "exp_corners_away")),
                ("home_corners", "away_corners")),
    "fouls": ((("tp_fouls_home", "exp_fouls_home"), ("tp_fouls_away", "exp_fouls_away")),
              ("home_fouls", "away_fouls")),
    "yellows": ((("tp_yellows_home", "exp_yellows_home"), ("tp_yellows_away", "exp_yellows_away")),
                ("home_yellow_cards", "away_yellow_cards")),
}


def _coalesce(m: pd.DataFrame, names: tuple[str, ...]) -> pd.Series | None:
    """First column of `names` present in `m`, filled from the ones after it."""
    have = [n for n in names if n in m.columns]
    if not have:
        return None
    out = m[have[0]].astype(float)
    for n in have[1:]:
        out = out.combine_first(m[n].astype(float))
    return out
PICK_ONE = {"1X2", "ht_1x2", "ht_ft"}


def count_drift(log: pd.DataFrame, found: pd.DataFrame) -> pd.DataFrame:
    """Per count and league: expected vs actual per match, one row per match."""
    one = (log[log["mercado"] == "1X2"].sort_values("logged_at")
           .drop_duplicates(["home", "away", "fecha"], keep="last"))
    f = found.assign(fecha=found["date"].astype(str).str[:10])
    m = one.merge(f.rename(columns={"home_team": "home", "away_team": "away"}),
                  on=["home", "away", "fecha"], how="inner")
    rows = []
    for name, ((hnames, anames), (ah, aa)) in COUNTS.items():
        eh, ea = _coalesce(m, hnames), _coalesce(m, anames)
        if eh is None or ea is None or not {ah, aa} <= set(m.columns):
            continue
        d = m.assign(_eh=eh, _ea=ea).dropna(subset=["_eh", "_ea", ah, aa])
        d = d.assign(exp=d["_eh"] + d["_ea"], act=d[ah] + d[aa])
        for league, g in [("all", d)] + list(d.groupby("competition")):
            if len(g) < 2:
                continue
            diff = g["act"] - g["exp"]
            se = diff.std(ddof=1) / np.sqrt(len(g))
            rows.append({"kind": "count", "market": name, "scope": league, "n": len(g),
                         "expected": round(float(g["exp"].mean()), 3),
                         "actual": round(float(g["act"].mean()), 3),
                         "z": round(float(diff.mean() / se), 2) if se > 0 else 0.0})
    return pd.DataFrame(rows)


def calibration_drift(ev: pd.DataFrame) -> pd.DataFrame:
    """Per over/under line and player market: predicted vs happened."""
    ev = ev[~ev["mercado_id"].isin(PICK_ONE)].copy()
    if ev.empty:
        return pd.DataFrame()
    yes = ev["lado"].isin(["OVER", "SI"])
    ev["y"] = np.where(yes, ev["acierto"], 1 - ev["acierto"])
    ev["p"] = ev["prob"].clip(1e-4, 1 - 1e-4)
    ev["scope"] = np.where(ev["es_jugador"], "players", ev["ambito"].astype(str))
    rows = []
    for (mk, scope, line), g in ev.groupby(["mercado_id", "scope", "linea"], dropna=False):
        var = float((g["p"] * (1 - g["p"])).sum())
        rows.append({"kind": "calibration", "market": f"{mk} {line}".strip(), "scope": scope,
                     "n": len(g), "expected": round(float(g["p"].mean()), 3),
                     "actual": round(float(g["y"].mean()), 3),
                     "z": round(float((g["y"] - g["p"]).sum() / np.sqrt(var)), 2) if var > 0 else 0.0})
    return pd.DataFrame(rows)


def report(as_of: pd.Timestamp | None = None) -> dict:
    found = pd.read_csv(FOUND, low_memory=False)
    found["date"] = pd.to_datetime(found["date"], errors="coerce")
    if as_of is not None:
        found = found[found["date"] <= as_of]
    log = pd.read_csv(PRED_LOG, low_memory=False)
    season = str(log["season"].dropna().max())
    log = log[log["season"] == season]
    ev = evaluate_log(found)
    ev = ev[ev["season"] == season] if len(ev) else ev
    out = {"season": season, "as_of": str((as_of or found["date"].max()).date()), "windows": {}}
    for window, days in (("season", None), ("last_28_days", 28)):
        lg = log
        if days is not None:
            since = (as_of or found["date"].max()) - pd.Timedelta(days=days)
            lg = log[pd.to_datetime(log["fecha"], errors="coerce") > since]
        c = count_drift(lg, found)
        keep = set(zip(lg["partido"], lg["jornada"]))
        e = ev[[k in keep for k in zip(ev["partido"], ev["jornada"])]] if len(ev) else ev
        k = calibration_drift(e) if len(e) else pd.DataFrame()
        t = pd.concat([c, k], ignore_index=True)
        if t.empty:
            out["windows"][window] = {"rows": [], "alerts": []}
            continue
        t["alert"] = (t["z"].abs() >= Z_ALERT) & (t["n"] >= MIN_N)
        out["windows"][window] = {"rows": t.to_dict("records"),
                                  "alerts": t[t["alert"]].sort_values("z", key=abs, ascending=False)
                                  .to_dict("records")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Market drift report over the pre-kickoff log.")
    ap.add_argument("--as-of", default=None, help="only results up to this date (YYYY-MM-DD)")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    as_of = pd.Timestamp(args.as_of) if args.as_of else None
    r = report(as_of)
    for window, w in r["windows"].items():
        rows = pd.DataFrame(w["rows"])
        print(f"\n== {window} ({r['season']}, results to {r['as_of']}) ==")
        if rows.empty:
            print("  nothing settled")
            continue
        counts = rows[(rows["kind"] == "count") & (rows["scope"] == "all")] if "kind" in rows else rows
        if len(counts):
            print(counts[["market", "n", "expected", "actual", "z"]].to_string(index=False))
        else:
            # the log carries exp_*/lambda only for rows logged since 2026-09-14
            print("  medias por partido: aun sin partidos jugados con esperados en el log")
        if w["alerts"]:
            print(f"  ALERTA: {len(w['alerts'])} deriva(s) con |z| >= {Z_ALERT:g}")
            for a in w["alerts"][:12]:
                print(f"    {a['kind']:11s} {a['market']:22s} {a['scope']:16s} n={a['n']:<5} "
                      f"dice {a['expected']:.3f} / pasa {a['actual']:.3f}  z={a['z']:+.1f}")
        else:
            print("  sin derivas")
    if not args.no_write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
        print(f"\n-> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
