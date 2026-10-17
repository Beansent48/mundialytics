"""Put the whole xG series on ONE provider's level, and say what it costs.

Finding 1 of docs/FOOTBALL_SEMANTICS_AUDIT_2026_10_17.md: history is Understat
(goals/xG 0.893 in 2025/26) and 2026/27 is Sofascore/Opta level (0.979), so the
xG-rate model is trained at one level and scored at another, and `lambda_scale_`
-- fitted over a 730-day window that is 90.7% Understat -- corrects an inflation
the current season's data no longer carries. The inflation is corrected twice.

This writes a foundation copy whose current-season xG is lifted onto the
UNDERSTAT level: the level the model was trained at and that `lambda_scale_`
measures. Historical rows are untouched, so every historical fold is identical
by construction and only 2026/27 can move.

The bridge is the text xG. `build_espn_text_xg.py` step 1 maps it onto
Understat's level per league; `--align history` stops step 3 pulling it back
down to Sofascore. Understat froze 2026-05-24 and Sofascore starts 2026-08-15,
so they never share a match and this is the only way to measure the offset.

    python scripts/build_espn_text_xg.py --offline --align history \
        --out outputs/xg_level/text_xg_understat_level.csv
    python scripts/xg_level/harmonise_providers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
SOFA = ROOT / "data/processed/sofascore_team_match_xg.csv"
TEXT_UNDERSTAT = ROOT / "outputs/xg_level/text_xg_understat_level.csv"
OUT = ROOT / "outputs/xg_level/foundation_xg_understat_level.csv"
XG = [f"{s}_{v}" for s in ("home", "away") for v in ("xg", "npxg", "xg_op", "xg_sp")]
KEY = ["date", "home_team_fd", "away_team_fd"]


def main() -> int:
    if not TEXT_UNDERSTAT.exists():
        print(f"missing {TEXT_UNDERSTAT.relative_to(ROOT)} -- run build_espn_text_xg.py "
              f"--offline --align history --out that path first")
        return 2
    txt = pd.read_csv(TEXT_UNDERSTAT)
    sofa = pd.read_csv(SOFA)
    j = txt.merge(sofa, on=KEY, suffixes=("_u", "_s"))
    if j.empty:
        print("no shared matches between the two providers")
        return 2
    # Understat level / Sofascore level, per league, on the matches both cover
    lift = (j.groupby("competition_u")[["home_xg_u", "away_xg_u"]].sum().sum(axis=1)
            / j.groupby("competition_u")[["home_xg_s", "away_xg_s"]].sum().sum(axis=1))
    overall = (j[["home_xg_u", "away_xg_u"]].values.sum()
               / j[["home_xg_s", "away_xg_s"]].values.sum())
    print(f"shared matches: {len(j)}")
    print(f"Understat/Sofascore level per league: {lift.round(3).to_dict()}")
    print(f"  overall {overall:.4f}  (so the deployed series runs {1 - 1 / overall:+.1%} "
          f"cold against the level the models were trained at)")

    sofa_up = sofa.copy()
    f = sofa_up["competition"].map(lift).fillna(overall)
    for c in XG:
        if c in sofa_up.columns:
            sofa_up[c] = sofa_up[c] * f

    found = pd.read_csv(FOUND, low_memory=False)
    # Sofascore first so it wins any overlap, exactly as augment_foundation_with_xg does
    new = (pd.concat([sofa_up[KEY + XG], txt[KEY + XG]], ignore_index=True)
           .rename(columns={"home_team_fd": "home_team", "away_team_fd": "away_team"}))
    new["date"] = pd.to_datetime(new["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    new = new.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
    found["date"] = pd.to_datetime(found["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    idx = ["date", "home_team", "away_team"]
    before = found.set_index(idx)
    patch = new.set_index(idx)
    touched = before.index.isin(patch.index)
    out = found.copy()
    for c in XG:
        if c in out.columns and c in patch.columns:
            mapped = out.set_index(idx).index.map(patch[c])
            out[c] = pd.Series(mapped, index=out.index).combine_first(out[c])
    out.to_csv(OUT, index=False)

    cur = out[out["season"].astype(str).str.startswith("2026")]
    old = found[found["season"].astype(str).str.startswith("2026")]
    g = (cur["home_goals"] + cur["away_goals"]).mean()
    print(f"\n2026/27 rows patched: {touched.sum()} of {len(old)}")
    print(f"  xG total/match  {(old.home_xg + old.away_xg).mean():.4f} -> "
          f"{(cur.home_xg + cur.away_xg).mean():.4f}   (goals {g:.4f})")
    print(f"  goals/xG        {g / (old.home_xg + old.away_xg).mean():.4f} -> "
          f"{g / (cur.home_xg + cur.away_xg).mean():.4f}")
    hist_old = found[~found["season"].astype(str).str.startswith("2026")]["home_xg"]
    hist_new = out[~out["season"].astype(str).str.startswith("2026")]["home_xg"]
    same = hist_old.fillna(-1).equals(hist_new.fillna(-1))
    print(f"  historical rows unchanged: {same}")
    print(f"\n-> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
