from __future__ import annotations

"""Are we benchmarking against the wrong price?

Every comparison so far has been against Bet365 CLOSING odds — the market's
final, most-informed price, and the hardest benchmark in the sport. But nobody
bets into the closing line; you bet at a price available EARLIER, before the
market has absorbed team news and sharp money.

So the product question is not "can we beat the close?" (almost nobody can, and
we showed the residual gap is irreducible from public data) but "can we beat the
price actually on offer?". football-data.co.uk carries both:
    B365H/D/A     pre-match (earlier, softer)
    B365CH/CD/CA  closing (final, sharpest)

This measures our model against BOTH, on the same matches. If the opening line
is materially worse than the close AND our model sits between them, there is a
real edge at a price you can actually take.

EVALUATION ONLY — odds are never model inputs.
"""

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_market_gap import rps3  # noqa: E402

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
PREDS = ROOT / "data/processed/enriched/understat_xg/walkforward_preds_deployed.csv"
WANT = {"Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A",
        "B365CH", "B365CD", "B365CA"}


def load_both_prices() -> pd.DataFrame:
    rows = []
    for p in glob.glob(str(ROOT / "data/raw/football_data/**/*.csv"), recursive=True):
        m = re.search(r"(\d{2})(\d{2})_(E0|SP1|D1|I1|F1)\.csv$", Path(p).name)
        if not m or int(m.group(1)) < 19:
            continue
        try:
            df = pd.read_csv(p, encoding="latin-1", on_bad_lines="skip",
                             usecols=lambda c: c in WANT)
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed")
        rows.append(df)
    r = pd.concat(rows, ignore_index=True).dropna(subset=["date"])
    r = r.drop_duplicates(subset=["date", "HomeTeam", "AwayTeam"])
    r["home_team"] = r["HomeTeam"].astype(str).str.lower().str.strip()
    r["away_team"] = r["AwayTeam"].astype(str).str.lower().str.strip()
    for c in ["B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA"]:
        r[c] = pd.to_numeric(r.get(c), errors="coerce")
    keep = ["date", "home_team", "away_team", "B365H", "B365D", "B365A",
            "B365CH", "B365CD", "B365CA"]
    r = r[keep].dropna()
    # decimal odds are > 1; football-data encodes some missing values as 0
    cols = ["B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA"]
    return r[(r[cols] > 1.0).all(axis=1)]


def devig(a, b, c):
    inv = np.c_[1 / a, 1 / b, 1 / c]
    return inv / inv.sum(axis=1, keepdims=True)


def main() -> None:
    preds = pd.read_csv(PREDS)
    found = pd.read_csv(FOUND, low_memory=False)
    found["date"] = pd.to_datetime(found["date"], errors="coerce")
    m = preds.merge(found[["match_id", "date", "competition", "home_team", "away_team"]],
                    on="match_id", how="left").dropna(subset=["date"])
    m["date"] = m["date"].dt.normalize()
    od = load_both_prices()
    od["date"] = od["date"].dt.normalize()
    m = m.merge(od, on=["date", "home_team", "away_team"], how="inner")
    print(f"matches with BOTH pre-match and closing odds: {len(m):,}"
          f"  ({m.date.min():%Y-%m-%d} -> {m.date.max():%Y-%m-%d})")

    y = np.where(m.hg > m.ag, 0, np.where(m.hg == m.ag, 1, 2))
    P = m[["ph", "pd", "pa"]].to_numpy()
    P = P / P.sum(axis=1, keepdims=True)
    OPEN = devig(m.B365H, m.B365D, m.B365A)
    CLOSE = devig(m.B365CH, m.B365CD, m.B365CA)

    r_us, r_open, r_close = rps3(y, P), rps3(y, OPEN), rps3(y, CLOSE)
    print("\n=== RPS SOBRE LOS MISMOS PARTIDOS ===")
    print(f"  nuestro modelo        {r_us:.4f}")
    print(f"  Bet365 PRE-PARTIDO    {r_open:.4f}   (el precio que puedes tomar)")
    print(f"  Bet365 CIERRE         {r_close:.4f}   (el benchmark imposible)")
    print(f"\n  cuanto se afila el mercado de apertura a cierre: {r_open - r_close:+.4f}")
    print(f"  nosotros vs PRE-PARTIDO: {r_us - r_open:+.4f}"
          f"  ({'GANAMOS' if r_us < r_open else 'perdemos'})")
    print(f"  nosotros vs CIERRE     : {r_us - r_close:+.4f}")

    # where does our model sit between the two prices?
    span = r_open - r_close
    if span > 0:
        pos = (r_open - r_us) / span
        print(f"\n  posicion del modelo en el recorrido apertura->cierre: {pos:.0%}")
        print("  (100% = tan bueno como el cierre; 0% = como la apertura; "
              "<0% = peor que la apertura)")

    print("\n=== POR TEMPORADA (nosotros | apertura | cierre) ===")
    for s, g in m.groupby("season"):
        i = g.index.to_numpy()
        gi = np.isin(m.index.to_numpy(), i)
        print(f"  {s:12s} n={gi.sum():5d}  {rps3(y[gi], P[gi]):.4f} | "
              f"{rps3(y[gi], OPEN[gi]):.4f} | {rps3(y[gi], CLOSE[gi]):.4f}")

    print("\n=== POR LIGA (nosotros | apertura | cierre) ===")
    for c, g in m.groupby("competition"):
        i = g.index.to_numpy()
        gi = np.isin(m.index.to_numpy(), i)
        if gi.sum() < 200:
            continue
        a, o, cl = rps3(y[gi], P[gi]), rps3(y[gi], OPEN[gi]), rps3(y[gi], CLOSE[gi])
        flag = "  <-- ganamos a la apertura" if a < o else ""
        print(f"  {c:16s} n={gi.sum():5d}  {a:.4f} | {o:.4f} | {cl:.4f}{flag}")


if __name__ == "__main__":
    main()
