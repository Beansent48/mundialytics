from __future__ import annotations

"""Produce a locally-advanced ClubElo table, so the European layer survives the API.

Seeds from the newest cached ClubElo snapshot and applies every result we hold
since that date — Big5 domestic from the foundation, plus UEFA competition
results — using the update rule fitted by scripts/fit_local_elo.py (K=16,
home advantage=65, chosen on held-out matches; rolling forward beat freezing in
5/5 cutoffs).

Clubs with no match since the seed keep their seed value and are flagged stale.
That is honest and unavoidable: our results cover Big5 and European ties, not
the Portuguese, Dutch, Czech or Scandinavian leagues, so those clubs cannot move
until they play in Europe.

Run: .venv/Scripts/python.exe scripts/build_local_clubelo.py
"""

import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.ratings.clubelo_local import (  # noqa: E402
    EloParams, load_seed, roll_forward, to_frame)
from mundialytics.statistical_core.competition.european import make_resolver  # noqa: E402

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
PARAMS = ROOT / "data/processed/local_elo_params.json"
OUT = ROOT / "data/processed/clubelo_local.csv"


def european_results() -> pd.DataFrame:
    """Played UEFA ties from the cached fixturedownload files."""
    rows = []
    for f in glob.glob(str(ROOT / "data/external/uefa/raw_*.csv")):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if not {"Date", "Home Team", "Away Team", "Result"} <= set(d.columns):
            continue
        d = d[d["Result"].astype(str).str.contains(r"\d+\s*-\s*\d+", regex=True)]
        if d.empty:
            continue
        sc = d["Result"].astype(str).str.extract(r"(\d+)\s*-\s*(\d+)")
        out = pd.DataFrame({
            "date": pd.to_datetime(d["Date"], dayfirst=True, errors="coerce"),
            "home_team": d["Home Team"].astype(str),
            "away_team": d["Away Team"].astype(str),
            "home_goals": pd.to_numeric(sc[0], errors="coerce"),
            "away_goals": pd.to_numeric(sc[1], errors="coerce"),
            "neutral": 0,
        })
        rows.append(out.dropna(subset=["date", "home_goals", "away_goals"]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    seed, seed_date = load_seed(ROOT)
    print(f"semilla ClubElo: {seed_date}  ({len(seed)} clubes)")

    p = EloParams()
    if PARAMS.exists():
        cfg = json.loads(PARAMS.read_text(encoding="utf-8"))
        p = EloParams(k=float(cfg["k"]), hfa=float(cfg["hfa"]))
        print(f"parametros ajustados: K={p.k:.0f}  factor campo={p.hfa:.0f}  "
              f"(avanzar>congelar {cfg.get('rolling_beats_freezing', '?')})")

    dom = pd.read_csv(FOUND, low_memory=False)
    dom["date"] = pd.to_datetime(dom["date"], errors="coerce")
    euro = european_results()
    cutoff = pd.Timestamp(seed_date)
    dom = dom[dom.date > cutoff]
    euro = euro[euro.date > cutoff] if len(euro) else euro
    matches = pd.concat([dom[["date", "home_team", "away_team", "home_goals",
                              "away_goals", "neutral"]], euro], ignore_index=True)
    print(f"partidos posteriores a la semilla: {len(matches)} "
          f"({len(dom)} domesticos + {len(euro)} europeos)")

    resolver = make_resolver(seed.keys())
    elo, last = roll_forward(seed, matches, p, resolver=resolver)

    df = to_frame(elo, last, seed_date)
    df.to_csv(OUT, index=False)
    moved = int((~df["stale"]).sum())
    print(f"\nESCRITO {OUT}")
    print(f"  {len(df)} clubes, {moved} actualizados, {len(df)-moved} sin partidos (stale)")

    # sanity: the scale must not drift, or elo_lambda_calibration_euro.json
    # (validated against ClubElo's scale) stops applying
    before = pd.Series(list(seed.values()))
    after = df["elo"]
    print(f"  escala  media {before.mean():.1f} -> {after.mean():.1f}   "
          f"desv {before.std():.1f} -> {after.std():.1f}")

    ch = df[~df["stale"]].copy()
    ch["delta"] = [elo[c] - seed[c] for c in ch["club"]]
    ch = ch.reindex(ch["delta"].abs().sort_values(ascending=False).index)
    print("\n  mayores movimientos:")
    for r in ch.head(8).itertuples(index=False):
        print(f"    {r.club:22s} {seed[r.club]:7.1f} -> {r.elo:7.1f}  ({r.delta:+6.1f})")


if __name__ == "__main__":
    main()
