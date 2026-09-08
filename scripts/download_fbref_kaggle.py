#!/usr/bin/env python3
"""Pull the published FBref season dumps from Kaggle.

WHY THIS EXISTS. The defensive axis of every player card came from StatsBomb's
free release, whose coverage is a publishing decision rather than a sample: the
ten best-measured defenders in the pool are ten Barcelona defenders, and
`defense_creation_matches` is zero for 62% of them. FBref carries the numbers
that would fix it — challenge success rate, aerial win rate, times dribbled
past, clearances, blocks, interceptions — but answers a plain HTTP client with
403 and stalls `soccerdata` indefinitely.

These Kaggle datasets are FBref's season tables already scraped and republished,
and Kaggle serves them over an open endpoint with no credentials. That is the
route: take the published dataset, not the site that does not want to be read.

WHAT ARRIVES.
  2024/25  267 columns, the big five. Includes Tkl%, Tkld%, aerial Won%, Clr,
           Blocks, Int, Recov, and the full modern keeper set (PSxG, PSxG+/-,
           Save%, Stp%, #OPA/90) at 100% coverage of keepers.
  2023/24  the same tables across SEVENTEEN leagues — Championship, Segunda,
           Serie B, 2. Bundesliga, Ligue 2, Eredivisie, Primeira, MLS, Liga MX,
           Brazil. That breadth is why it matters: it is where the players of
           the promoted clubs actually come from.

  2025/26  hubertsidorowicz's own dump is a thin 102-column version (basics
           only). The advanced tables come instead from chuongtrinh's scouting
           merge (FBref + Understat + SofaScore), 171 columns at 96% coverage of
           creation, defence and duel rates plus keeper goals-prevented — saved
           as scouting_2526.csv and wired in as the PRIMARY season.

WHAT DOES NOT. 2026/27 does not exist yet, so the current season still only has
ESPN counting stats for form. See [[project-current-squads]].

Run: .venv/Scripts/python.exe scripts/download_fbref_kaggle.py
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/external/advanced/fbref_kaggle"
BASE = "https://www.kaggle.com/api/v1/datasets/download/{slug}"

# (kaggle slug, season tag, which files to keep)
SOURCES = [
    ("hubertsidorowicz/football-players-stats-2024-2025", "2425", "wide"),
    ("hubertsidorowicz/football-players-stats-2025-2026", "2526", "wide"),
    ("anisguechtouli/football-leagues-data-2023-2024", "2324", "tables"),
    # the one 25/26 dump that carries advanced tables — a FBref+Understat+
    # SofaScore merge. Kept as scouting_2526.csv; it is what makes 25/26 usable
    # for creation and defence, not just goals and assists.
    ("chuongtrinh/top-5-league-individual-stats-season-20252026", "2526adv", "scouting"),
]


def fetch(slug: str, timeout: int = 300) -> zipfile.ZipFile:
    req = urllib.request.Request(BASE.format(slug=slug),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return zipfile.ZipFile(io.BytesIO(fh.read()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for slug, season, kind in SOURCES:
        try:
            z = fetch(slug)
        except Exception as exc:
            print(f"  {slug}: FALLO {type(exc).__name__} {str(exc)[:70]}", flush=True)
            continue
        if kind == "scouting":
            # a single scouting CSV, saved under a stable name
            name = next((n for n in z.namelist() if n.endswith(".csv")), None)
            if not name:
                print(f"  {slug}: sin CSV")
                continue
            path = out / "scouting_2526.csv"
            path.write_bytes(z.read(name))
            written.append(path.name)
        elif kind == "wide":
            # one row per player, every stat table joined side by side
            name = next((n for n in z.namelist()
                         if n.endswith(".csv") and "light" not in n), None)
            if not name:
                print(f"  {slug}: sin CSV completo")
                continue
            path = out / f"fbref_players_{season}.csv"
            path.write_bytes(z.read(name))
            written.append(path.name)
        else:
            # one CSV per stat table
            for n in z.namelist():
                if n.startswith("denormalized/") and n.endswith(".csv"):
                    path = out / f"{season}_{Path(n).name}"
                    path.write_bytes(z.read(n))
                    written.append(path.name)
        print(f"  {slug}: OK", flush=True)

    print(f"\nESCRITO en {out}")
    for name in sorted(written):
        p = out / name
        try:
            n = len(pd.read_csv(p, nrows=None, usecols=[0]))
        except Exception:
            n = -1
        print(f"  {name:34} {p.stat().st_size / 1024:8.0f} KB  {n:6d} filas")


if __name__ == "__main__":
    main()
