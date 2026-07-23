from __future__ import annotations

"""Validate the Elo->goals mapping on REAL European matches (UCL/UEL 2021-2025
from fixturedownload.com), and refit the EUROPEAN home advantage.

The mapping constants were calibrated on domestic big-5 games; literature says
European-night home advantage is smaller and two-legged dynamics differ. This
measures exactly that, walk-forward:
  ARM A: domestic constants as-is (c, hfa, b from big-5)
  ARM B: same b (Elo slope, 17k-match estimate) but c+hfa REFIT on past
         European seasons only.
Metrics: RPS / LL / home-bias on ~1,500 European matches with pre-match Elo.
"""

import json
import re
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
UEFA_DIR = ROOT / "data/external/uefa"
TEAMS_DIR = ROOT / "data/external/clubelo/teams"
CALIB = json.loads((ROOT / "data/processed/elo_lambda_calibration.json").read_text())

COMPS = {"champions-league": ["2021", "2022", "2023", "2024"],
         "europa-league": ["2021", "2022", "2023", "2024"],
         "conference-league": ["2024"]}

ALIASES = {
    "bayern munich": "Bayern", "atletico madrid": "Atletico", "atletico de madrid": "Atletico",
    "man city": "ManCity", "manchester city": "ManCity", "man utd": "ManUnited",
    "manchester united": "ManUnited", "paris saint-germain": "ParisSG", "psg": "ParisSG",
    "inter milan": "Inter", "ac milan": "Milan", "bayer leverkusen": "Leverkusen",
    "borussia dortmund": "Dortmund", "rb leipzig": "RBLeipzig", "sporting cp": "Sporting",
    "sporting lisbon": "Sporting", "fc porto": "Porto", "sl benfica": "Benfica",
    "psv eindhoven": "PSV", "az alkmaar": "AZAlkmaar", "crvena zvezda": "CrvenaZvezda",
    "red star belgrade": "CrvenaZvezda", "fc copenhagen": "FCKobenhavn",
    "copenhagen": "FCKobenhavn", "club brugge": "Brugge", "union berlin": "UnionBerlin",
    "eintracht frankfurt": "Frankfurt", "real sociedad": "Sociedad", "celtic fc": "Celtic",
    "rangers fc": "Rangers", "young boys": "YoungBoys", "red bull salzburg": "Salzburg",
    "rb salzburg": "Salzburg", "shakhtar donetsk": "Shakhtar", "dinamo zagreb": "DinamoZagreb",
    "slavia prague": "SlaviaPraha", "sparta prague": "SpartaPraha", "viktoria plzen": "Plzen",
    "bodo/glimt": "Bodoe/Glimt", "bodo / glimt": "Bodoe/Glimt", "malmo ff": "Malmoe",
    "besiktas": "Besiktas", "fenerbahce": "Fenerbahce", "galatasaray": "Galatasaray",
    "olympiacos": "Olympiakos", "olympiacos piraeus": "Olympiakos", "paok": "PAOK",
    "aek athens": "AEK", "sturm graz": "SturmGraz", "slovan bratislava": "SlovanBratislava",
    "st gilloise": "UnionStGilloise", "union saint-gilloise": "UnionStGilloise",
    "royale union sg": "UnionStGilloise", "girona fc": "Girona", "stade brestois": "Brest",
    "vfb stuttgart": "Stuttgart", "sc freiburg": "Freiburg", "olympique marseille": "Marseille",
    "olympique lyonnais": "Lyon", "lyon": "Lyon", "nice": "Nice", "lens": "Lens", "lille": "Lille",
    "spurs": "Tottenham", "tottenham hotspur": "Tottenham", "west ham united": "WestHam",
    "wolverhampton": "Wolves", "leicester city": "Leicester", "sevilla fc": "Sevilla",
    "villarreal cf": "Villarreal", "real betis": "Betis", "athletic club": "Bilbao",
    "athletic bilbao": "Bilbao", "feyenoord rotterdam": "Feyenoord", "ajax amsterdam": "Ajax",
    "twente": "Twente", "sc braga": "Braga", "vitoria de guimaraes": "Guimaraes",
    "ludogorets razgrad": "Ludogorets", "qarabag fk": "Qarabag", "ferencvaros": "Ferencvaros",
    "maccabi haifa": "MaccabiHaifa", "maccabi tel aviv": "MaccabiTelAviv",
    "molde fk": "Molde", "rosenborg": "Rosenborg", "legia warsaw": "LegiaWarszawa",
    "lech poznan": "LechPoznan", "rakow": "Rakow", "servette fc": "Servette",
    "fc basel": "Basel", "fc zurich": "Zuerich", "fc lugano": "Lugano",
    "atleti": "Atletico", "atlético": "Atletico", "atlético de madrid": "Atletico",
    "bayern": "Bayern", "bayern münchen": "Bayern", "fenerbahçe": "Fenerbahce",
    "ferencváros": "Ferencvaros", "djurgården": "Djurgarden", "häcken": "Haecken",
    "bröndby": "Broendby", "brøndby": "Broendby", "malmö": "Malmoe",
    "köln": "Koeln", "gnk dinamo": "DinamoZagreb", "fcsb": "FCSB",
    "basaksehir": "Basaksehir", "başakşehir": "Basaksehir",
    "paris": "ParisSG", "s. bratislava": "SlovanBratislava", "shakhtar": "ShakhtarDonetsk",
    "stuttgart": "Stuttgart", "losc": "Lille", "leicester": "Leicester",
    "m. haifa": "MaccabiHaifa", "raków": "Rakow", "sk rapid": "RapidWien",
    "omonoia": "Omonia", "qarabag": "Qarabag", "ludogorets": "Ludogorets",
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def download_results() -> pd.DataFrame:
    rows = []
    for comp, seasons in COMPS.items():
        for yr in seasons:
            cache = UEFA_DIR / f"raw_{comp}_{yr}.csv"
            if cache.exists():
                txt = cache.read_text(encoding="utf-8")
            else:
                r = requests.get(f"https://fixturedownload.com/download/{comp}-{yr}-UTC.csv",
                                 timeout=30, headers={"User-Agent": "Mozilla/5.0"})
                r.encoding = "utf-8"   # requests guesses latin-1 -> mojibake in club names
                if r.status_code != 200 or len(r.text) < 500:
                    print(f"  no data: {comp} {yr}", flush=True)
                    continue
                txt = r.text
                UEFA_DIR.mkdir(parents=True, exist_ok=True)
                cache.write_text(txt, encoding="utf-8")
                time.sleep(0.4)
            df = pd.read_csv(StringIO(txt))
            df["comp"] = comp
            df["edition"] = f"{yr}-{int(yr) + 1}"
            rows.append(df)
            print(f"  OK {comp} {yr}: {len(df)} matches", flush=True)
    m = pd.concat(rows, ignore_index=True)
    m["date"] = pd.to_datetime(m["Date"], dayfirst=True, errors="coerce", format="mixed").dt.normalize()
    res = m["Result"].astype(str).str.extract(r"(\d+)\s*-\s*(\d+)")
    m["hg"] = pd.to_numeric(res[0], errors="coerce")
    m["ag"] = pd.to_numeric(res[1], errors="coerce")
    return m.dropna(subset=["date", "hg", "ag"])


def resolve_names(m: pd.DataFrame, elo_names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    by_norm = {norm(n): n for n in elo_names}
    unmatched: set[str] = set()

    def to_clubelo(name: str) -> str | None:
        low = name.lower().strip()
        if low in ALIASES:
            return ALIASES[low]
        n = norm(name)
        if n in by_norm:
            return by_norm[n]
        cands = [v for k, v in by_norm.items() if n in k or k in n]
        if len(cands) == 1:
            return cands[0]
        unmatched.add(name)
        return None

    m = m.copy()
    m["home"] = m["Home Team"].astype(str).map(to_clubelo)
    m["away"] = m["Away Team"].astype(str).map(to_clubelo)
    return m.dropna(subset=["home", "away"]), sorted(unmatched)


API_NAME = {"M Tel Aviv": "MaccabiTelAviv", "M Haifa": "MaccabiHaifa"}


def fetch_history(club: str) -> pd.DataFrame | None:
    api = API_NAME.get(club, club).replace(" ", "")   # daily names have spaces, the API doesn't
    safe = api.replace("/", "_")
    cache = TEAMS_DIR / f"{safe}.csv"
    if cache.exists() and cache.stat().st_size > 200:
        h = pd.read_csv(cache)
    else:
        try:
            r = requests.get(f"http://api.clubelo.com/{api}", timeout=30)
            if r.status_code != 200 or len(r.text) < 100:
                return None
            h = pd.read_csv(StringIO(r.text))
            if h.empty or "Elo" not in h.columns:
                return None
            TEAMS_DIR.mkdir(parents=True, exist_ok=True)
            h.to_csv(cache, index=False)
            time.sleep(0.4)
        except Exception:
            return None
    h = h.copy()
    h["From"] = pd.to_datetime(h["From"], errors="coerce")
    h["To"] = pd.to_datetime(h["To"], errors="coerce")
    return h[["From", "To", "Elo"]].dropna()


def rps3(y_idx, P):
    Y = np.zeros_like(P)
    Y[np.arange(len(y_idx)), y_idx] = 1.0
    cp, cy = np.cumsum(P, axis=1), np.cumsum(Y, axis=1)
    return float(((cp - cy) ** 2)[:, :2].sum(axis=1).mean() / 2)


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from mundialytics.statistical_core.distributions import outcome_probabilities

    from mundialytics.statistical_core.competition.european import fetch_current_elo

    elo_now = fetch_current_elo(ROOT)
    print("descargando resultados europeos...", flush=True)
    m = download_results()
    m, unmatched = resolve_names(m, list(elo_now))
    print(f"partidos: {len(m)} | equipos sin resolver: {len(unmatched)} -> {unmatched[:15]}", flush=True)

    clubs = sorted(set(m.home) | set(m.away))
    print(f"fetching {len(clubs)} club histories (cached where possible)...", flush=True)
    hist = {c: fetch_history(c) for c in clubs}
    missing_h = [c for c, h in hist.items() if h is None]
    print(f"histories ok: {len(clubs) - len(missing_h)} | missing: {missing_h[:10]}", flush=True)

    def elo_at(team, when):
        h = hist.get(team)
        if h is None:
            return None
        r = h[(h.From <= when) & (h.To >= when)]
        return float(r.Elo.iloc[0]) if len(r) else None

    rows = []
    for r in m.itertuples(index=False):
        eh, ea = elo_at(r.home, r.date), elo_at(r.away, r.date)
        if eh is None or ea is None:
            continue
        rows.append((r.comp, r.edition, r.date, int(r.hg), int(r.ag), eh, ea))
    d = pd.DataFrame(rows, columns=["comp", "edition", "date", "hg", "ag", "eh", "ea"])
    d["d400"] = (d.eh - d.ea) / 400.0
    print(f"partidos europeos con Elo pre-partido: {len(d)}", flush=True)

    c0, hfa0, b0 = CALIB["c"], CALIB["hfa"], CALIB["b"]

    def eval_arm(df, c, hfa, b, tag):
        P, y = [], []
        for r in df.itertuples(index=False):
            lh = float(np.exp(c + hfa + b * r.d400))
            la = float(np.exp(c - b * r.d400))
            p = outcome_probabilities(lh, la, dixon_coles_rho=-0.07)
            P.append([p["p_home_win"], p["p_draw"], p["p_away_win"]])
            y.append(0 if r.hg > r.ag else (1 if r.hg == r.ag else 2))
        P, y = np.array(P), np.array(y)
        ll = float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).mean())
        bias_h = float(P[:, 0].mean() - (y == 0).mean())
        return rps3(y, P), ll, bias_h

    editions = sorted(d.edition.unique())
    print("\nARM A (constantes domésticas) vs ARM B (hfa+c refit europeo walk-forward):")
    resA, resB = [], []
    from sklearn.linear_model import PoissonRegressor
    for ed in editions[1:]:
        te = d[d.edition == ed]
        tr = d[d.edition < ed]
        if len(tr) < 300 or len(te) == 0:
            continue
        X = np.concatenate([np.column_stack([np.ones(len(tr)), tr.d400]),
                            np.column_stack([np.zeros(len(tr)), -tr.d400])])
        yg = np.concatenate([tr.hg, tr.ag])
        reg = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(X, yg)
        cE, hfaE = float(reg.intercept_), float(reg.coef_[0])
        a = eval_arm(te, c0, hfa0, b0, "A")
        bm = eval_arm(te, cE, hfaE, b0, "B")
        resA.append((*a, len(te), ed))
        resB.append((*bm, len(te), ed))
        print(f"  {ed}: A rps {a[0]:.4f} ll {a[1]:.4f} biasH {a[2]:+.3f} | "
              f"B rps {bm[0]:.4f} ll {bm[1]:.4f} biasH {bm[2]:+.3f} (hfaE={hfaE:.3f}, n={len(te)})", flush=True)
    pool = lambda rs, i: sum(r[i] * r[3] for r in rs) / sum(r[3] for r in rs)
    print(f"\nPOOLED: A rps {pool(resA,0):.4f} ll {pool(resA,1):.4f} biasH {pool(resA,2):+.3f}")
    print(f"        B rps {pool(resB,0):.4f} ll {pool(resB,1):.4f} biasH {pool(resB,2):+.3f}")

    # final european constants on ALL euro data (for the simulator)
    X = np.concatenate([np.column_stack([np.ones(len(d)), d.d400]),
                        np.column_stack([np.zeros(len(d)), -d.d400])])
    yg = np.concatenate([d.hg, d.ag])
    reg = PoissonRegressor(alpha=1e-4, max_iter=1000).fit(X, yg)
    out = {"c": float(reg.intercept_), "hfa": float(reg.coef_[0]), "b": b0,
           "n_matches": len(d), "note": "c+hfa refit on European matches; b from big-5 (17k)",
           "domestic": {"c": c0, "hfa": hfa0, "b": b0}}
    (ROOT / "data/processed/elo_lambda_calibration_euro.json").write_text(json.dumps(out, indent=2))
    print(f"\nWROTE euro constants: c={out['c']:.4f} hfa={out['hfa']:.4f} "
          f"(domestico: c={c0:.4f} hfa={hfa0:.4f})")


if __name__ == "__main__":
    main()
