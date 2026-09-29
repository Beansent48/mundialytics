from __future__ import annotations

"""Current-season team xG from ESPN match commentary (text xG).

Why: every live xG source is gone -- Understat froze in 05/2026, FBref serves pages without
xG, and Sofascore answers 403 since 2026-09-26. Without fresh xG the engine's walk-forward
form freezes, which cost +0.0026 RPS over 2025/26 (scripts/espn_xg/score_xg_source.py).
ESPN's commentary describes every attempt in the Opta standard (zone, body part, situation,
assist); a shot model on that text (src/mundialytics/enrichment/text_xg.py), trained on
2024/25 only and applied to 2025/26, correlates 0.83 with Understat per team-match and
keeps all but +0.0003 of Understat's RPS (not significant) -- ~89% of what was at stake.

Steps:
  1. fetch the commentary of the current season's new completed matches (one ESPN summary
     each, resumable; data/external/advanced/espn/commentary/<season>.jsonl)
  2. train TextXG on every earlier season on disk, and read Understat's level per league
     off the foundation (cross-fitted text xG vs Understat, same matches) -- the engine's
     history is Understat, which runs hotter than goals
  3. score the current season and write data/processed/espn_text_team_match_xg.csv in the
     Sofascore/Understat schema; update_season.augment_foundation_with_xg appends it after
     the real-xG sources, so real xG wins wherever a match has both

    python scripts/build_espn_text_xg.py [--offline]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/espn_xg")]

from mundialytics.enrichment import text_xg as tx  # noqa: E402
from mundialytics.identity.normalization import canonical_team_name  # noqa: E402

COMM = ROOT / "data/external/advanced/espn/commentary"
FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
OUT = ROOT / "data/processed/espn_text_team_match_xg.csv"
ALIASES = ROOT / "data/curated/fixture_team_aliases.csv"
SOFA = ROOT / "data/processed/sofascore_team_match_xg.csv"
MIN_HISTORY_SHOTS = 20_000
MIN_OVERLAP = 100      # matches this season with both real and text xG to align the level


def _canon():
    alias = {}
    if ALIASES.exists():
        a = pd.read_csv(ALIASES)
        alias = dict(zip(a.iloc[:, 0].astype(str), a.iloc[:, 1].astype(str)))
    return lambda name: alias.get(name, canonical_team_name(name))


def align_to_real(out: pd.DataFrame) -> pd.DataFrame:
    """Put this season's text xG on the level of the real xG it continues.

    The foundation keeps real xG where a match has it (Sofascore gave the first 250 of
    2026/27) and text xG after that. Text is calibrated to Understat, which runs ~7% hotter
    than Sofascore's Opta numbers, and a level jump mid-season would push the xG form up by
    itself. With enough matches covered by both this season, rescale per league to match.
    """
    if not SOFA.exists():
        return out
    k = ["date", "home_team_fd", "away_team_fd"]
    j = out.merge(pd.read_csv(SOFA), on=k, suffixes=("", "_real"))
    if len(j) < MIN_OVERLAP:
        return out
    ratio = (j.groupby("competition")[["home_xg_real", "away_xg_real"]].sum().sum(1)
             / j.groupby("competition")[["home_xg", "away_xg"]].sum().sum(1))
    overall = j[["home_xg_real", "away_xg_real"]].values.sum() / j[["home_xg", "away_xg"]].values.sum()
    f = out["competition"].map(ratio).fillna(overall)
    cols = [f"{side}_{v}" for side in ("home", "away") for v in ("xg", "npxg", "xg_op", "xg_sp")]
    out[cols] = out[cols].mul(f, axis=0).round(4)
    print(f"  aligned to the real xG on {len(j)} shared matches: {ratio.round(3).to_dict()}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use the commentary already on disk")
    args = ap.parse_args()
    now = pd.Timestamp.now()
    year = now.year if now.month >= 7 else now.year - 1
    cur = f"{year}-{year + 1}"

    if not args.offline:
        from fetch_commentary import fetch_season
        fetch_season(year, log=lambda m: print(f"  {m}", flush=True))

    # 1. history: the model, and Understat's level per league
    hist = [p for p in sorted(COMM.glob("*.jsonl")) if p.stem != cur]
    parts = [tx.read_commentary(p, p.stem) for p in hist]
    shots = pd.concat([s for s, _ in parts], ignore_index=True)
    games = pd.concat([g for _, g in parts], ignore_index=True)
    if len(shots) < MIN_HISTORY_SHOTS:
        print(f"only {len(shots)} historical attempts in {COMM}; run scripts/espn_xg/fetch_commentary.py "
              f"for past seasons first", flush=True)
        return 2
    model = tx.TextXG().fit(shots)
    oof = tx.per_match(shots.assign(xg=tx.cross_fit(shots)))
    oof = oof.join(games.set_index("event_id")["competition"])
    found = pd.read_csv(FOUND, low_memory=False,
                        usecols=["date", "home_team", "away_team", "home_goals", "away_goals", "home_xg", "away_xg"])
    found = found[pd.to_datetime(found["date"], errors="coerce") < pd.Timestamp(f"{year}-07-01")]
    ref = tx.join_to_matches(games, found).set_index("event_id")
    ref = ref.rename(columns={"home_xg_m": "home_xg", "away_xg_m": "away_xg"})
    scale = tx.level(oof, ref)
    print(f"text xG trained on {len(shots):,} attempts ({', '.join(p.stem for p in hist)}); "
          f"Understat level {scale.round(3).to_dict()}", flush=True)

    # 2. the current season
    path = COMM / f"{cur}.jsonl"
    if not path.exists():
        print(f"no commentary for {cur} yet", flush=True)
        return 0
    cs, cg = tx.read_commentary(path, cur)
    if cs.empty:
        print(f"no attempts parsed for {cur}", flush=True)
        return 0
    pm = tx.per_match(cs.assign(xg=model.predict(cs))).join(cg.set_index("event_id"))
    k = pm["competition"].map(scale).fillna(float(scale.mean()))
    canon = _canon()
    rows = []
    for eid, r in pm.iterrows():
        f = k[eid]
        rows.append({
            "provider": "espn_text", "provider_match_id": eid, "date": r["date"],
            "competition": r["competition"], "season": f"{year % 100:02d}{(year + 1) % 100:02d}",
            "home_team": r["home_raw"], "away_team": r["away_raw"],
            "home_team_fd": canon(r["home_raw"]), "away_team_fd": canon(r["away_raw"]),
            **{f"{side}_{v}": round(float(r[f"{side}_{v}"]) * f, 4)
               for side in ("home", "away") for v in ("xg", "npxg", "xg_op", "xg_sp")},
            "home_shots": int(r["home_n"]), "away_shots": int(r["away_n"]),
            "home_goals": int(r["home_goals"]), "away_goals": int(r["away_goals"]),
            "xg_match_confidence": "espn_commentary_text_model",
        })
    out = pd.DataFrame(rows).sort_values("date")
    out = align_to_real(out)
    out.to_csv(OUT, index=False)
    print(f"{cur}: {len(out)} matches, mean xG {out[['home_xg', 'away_xg']].values.mean():.2f} "
          f"(up to {out['date'].max()}) -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
