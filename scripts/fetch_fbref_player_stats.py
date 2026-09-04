from __future__ import annotations

"""Per-player match stats for the CURRENT season, from FBref.

WHY THIS EXISTS. Player markets (scorer, shots, assists, cards) were being logged
pre-kickoff with nothing able to settle them: Understat has not published
2026/27, and its site blocks plain HTTP (soccerdata gets in with a TLS-spoofing
client, which is why the historical file still works). Logging a market that
cannot be settled is the booking-points failure, so this closes the loop from a
source that DOES have the season.

WHY THE SEASON IS VERIFIED BY DATE, NOT BY ITS CODE. soccerdata's season labels
cannot be trusted here. Asking FBref for "2627" returns a complete, entirely
plausible 462-match Premier League season -- from **1926-27**. And "2526", which
by the pattern of "2425" -> 2024/25 ought to be last season, actually returns
2026/27: verified by cross-checking its results against football-data.co.uk,
which has the same three fixtures with the same scores on the same dates
(Man Utd 5-2 Ipswich and Aston Villa 0-1 Arsenal, 30-31 Aug 2026).

Relying on a label that is wrong-but-working is how a silent regression starts,
so this tries candidate codes and accepts one only if the dates it returns land
in the season actually asked for. A source that renames its seasons then becomes
a loud failure instead of quietly correct-looking data from the wrong year.

Run: .venv/Scripts/python.exe scripts/fetch_fbref_player_stats.py --season 2026-2027
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

OUT = ROOT / "data/external/advanced/fbref/fbref_player_match_current.csv"
LEAGUES = ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga",
           "ITA-Serie A", "FRA-Ligue 1"]
# codes worth trying, in order; which one is "current" has moved before
CANDIDATES = ["2526", "2627", "2425"]


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    df.columns = ["_".join(str(x) for x in c if x and str(x) != "nan").strip("_")
                  if isinstance(c, tuple) else str(c) for c in df.columns]
    return df


def _pick(cols: list[str], *suffixes: str) -> str | None:
    for s in suffixes:
        for c in cols:
            if c == s or c.endswith("_" + s):
                return c
    return None


def season_window(season: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """A season labelled 2026-2027 must fall between July 2026 and July 2027."""
    y1 = int(str(season)[:4])
    return pd.Timestamp(f"{y1}-07-01"), pd.Timestamp(f"{y1 + 1}-07-31")


def resolve_code(league: str, season: str) -> str | None:
    """Find the soccerdata code whose DATES match the season we want."""
    from soccerdata import FBref

    lo, hi = season_window(season)
    for code in CANDIDATES:
        try:
            sch = _flatten(FBref(leagues=league, seasons=code).read_schedule())
        except Exception as exc:
            print(f"    code {code}: {type(exc).__name__} {str(exc)[:50]}")
            continue
        d = pd.to_datetime(sch.get("date"), errors="coerce").dropna()
        if d.empty:
            print(f"    code {code}: sin fechas")
            continue
        if lo <= d.min() <= hi and lo <= d.max() <= hi:
            print(f"    code {code}: OK -> {d.min():%Y-%m} a {d.max():%Y-%m}")
            return code
        print(f"    code {code}: RECHAZADO -> {d.min():%Y-%m} a {d.max():%Y-%m} "
              f"(fuera de {lo:%Y-%m}..{hi:%Y-%m})")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=None, help="p.ej. 2026-2027 (por defecto, la actual)")
    ap.add_argument("--leagues", nargs="*", default=LEAGUES)
    args = ap.parse_args()

    now = pd.Timestamp.now()
    y = now.year if now.month >= 7 else now.year - 1
    season = args.season or f"{y}-{y + 1}"
    print(f"temporada objetivo: {season}")

    frames = []
    for league in args.leagues:
        print(f"\n{league}")
        code = resolve_code(league, season)
        if code is None:
            print("    sin codigo valido — se omite esta liga")
            continue
        try:
            from soccerdata import FBref
            ps = _flatten(FBref(leagues=league, seasons=code)
                          .read_player_match_stats(stat_type="summary"))
        except Exception as exc:
            print(f"    fallo al leer jugadores: {str(exc)[:70]}")
            continue

        cols = list(ps.columns)
        gl, ast = _pick(cols, "Gls"), _pick(cols, "Ast")
        sh, sot = _pick(cols, "Sh"), _pick(cols, "SoT")
        crd = _pick(cols, "CrdY")
        mins = _pick(cols, "Min")
        player, team, game = _pick(cols, "player"), _pick(cols, "team"), _pick(cols, "game")
        if not (gl and player and team and game):
            print(f"    columnas inesperadas: {cols[:12]}")
            continue
        out = pd.DataFrame({
            "league": league, "season": season,
            "game": ps[game], "team": ps[team], "player": ps[player],
            "minutes": pd.to_numeric(ps[mins], errors="coerce") if mins else pd.NA,
            "goals": pd.to_numeric(ps[gl], errors="coerce"),
            "assists": pd.to_numeric(ps[ast], errors="coerce") if ast else pd.NA,
            "shots": pd.to_numeric(ps[sh], errors="coerce") if sh else pd.NA,
            "sot": pd.to_numeric(ps[sot], errors="coerce") if sot else pd.NA,
            "yellow_cards": pd.to_numeric(ps[crd], errors="coerce") if crd else pd.NA,
        })
        # FBref's `game` reads "2026-08-31 Aston Villa-Arsenal"
        out["date"] = pd.to_datetime(out["game"].astype(str).str.slice(0, 10),
                                     errors="coerce")
        out = out.dropna(subset=["date", "player"])
        print(f"    {len(out):,} filas jugador-partido, "
              f"{out['date'].min():%Y-%m-%d} a {out['date'].max():%Y-%m-%d}, "
              f"{int(out['goals'].fillna(0).gt(0).sum())} con gol")
        frames.append(out)

    if not frames:
        print("\nNada descargado.")
        raise SystemExit(1)
    allp = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    allp.to_csv(OUT, index=False)
    print(f"\nESCRITO {OUT}")
    print(f"  {len(allp):,} filas · {allp['player'].nunique():,} jugadores · "
          f"{allp['date'].nunique()} fechas · "
          f"{int(allp['goals'].fillna(0).sum())} goles")


if __name__ == "__main__":
    main()
