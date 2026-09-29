"""Weekly-refit walk-forward of one season with the season's xG from a chosen source.

Same harness as scripts/squad_value/wf_refit.py (refit every week, xG form updated after
each match, debutants priced by the serving stand-in), but the TEST season's xG comes from:
    understat  what we had while a live xG source worked (the reference)
    text       text xG from ESPN commentary (build_text_xg_matches.py); unmatched -> none
    none       no xG for the season at all: what happens if nothing replaces Sofascore
History before the season keeps Understat's xG in every mode.

    python scripts/espn_xg/wf_xg_source.py 2025-2026 text OUT.csv [TEXT_XG_FILE]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mundialytics.serving.debutants import relegated_last_season  # noqa: E402
from mundialytics.statistical_core.prediction_engine import DEPLOYED_CLUB_ENGINE_KWARGS, PredictionEngine  # noqa: E402
from mundialytics.statistical_core.schemas import canonical_name  # noqa: E402

season, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
# the squad-value and team-news tilts are applied when scoring, on top of these components
KW = {k: v for k, v in DEPLOYED_CLUB_ENGINE_KWARGS.items()
      if k not in ("squad_value_shift", "lineup_shift", "morning_shift")}
df = pd.read_csv(ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv", low_memory=False)
df = df[df["xg_available"] == True].copy()  # noqa: E712
for c in ["home_goals", "away_goals", "home_xg", "away_xg"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["home_goals", "away_goals", "home_xg", "away_xg", "date"]).sort_values("date")
df["home_team"] = df["home_team"].map(canonical_name)
df["away_team"] = df["away_team"].map(canonical_name)

cur = df["season"] == season
if mode == "none":
    df.loc[cur, ["home_xg", "away_xg"]] = np.nan
elif mode == "text":
    src = sys.argv[4] if len(sys.argv) > 4 else "text_xg_matches.csv"   # build_text_xg_matches.py --out
    t = pd.read_csv(ROOT / "data/external/transfermarkt/work" / src).set_index("match_id")
    df.loc[cur, "home_xg"] = df.loc[cur, "match_id"].map(t.home_xg_text).values
    df.loc[cur, "away_xg"] = df.loc[cur, "match_id"].map(t.away_xg_text).values
    print(f"text xG on {df.loc[cur, 'home_xg'].notna().mean():.1%} of the season", flush=True)
elif mode != "understat":
    raise SystemExit(f"unknown mode {mode}")

test = df[cur].sort_values("date")
weeks = test["date"].dt.to_period("W-SUN")
rows = []
t0 = time.time()
for wk, blk in test.groupby(weeks, sort=True):
    cutoff = blk["date"].min()
    train = df[df["date"] < cutoff]
    eng = PredictionEngine(**KW).fit(train)
    known = set(train["home_team"]) | set(train["away_team"])
    for _, r in blk.iterrows():
        h, a, comp = r.home_team, r.away_team, r.competition
        cur_teams = set(test.loc[test.competition == comp, "home_team"])
        hs = [h] if h in known else [x for x in relegated_last_season(train, comp, season, cur_teams) if x in known]
        as_ = [a] if a in known else [x for x in relegated_last_season(train, comp, season, cur_teams) if x in known]
        comps = []
        for hh in hs:
            for aa in as_:
                if hh != aa:
                    lh_ad, la_ad = eng._lambdas_ad(hh, aa, comp, False)
                    xr_h, xr_a = eng.xg_rate_model_.predict_lambda(hh, aa, neutral=False)
                    comps.append((lh_ad, la_ad, xr_h, xr_a))
        c = np.mean(np.array(comps), axis=0)
        rows.append({"season": season, "match_id": r.match_id, "date": r.date, "comp": comp,
                     "home": h, "away": a, "hg": int(r.home_goals), "ag": int(r.away_goals),
                     "lh_ad": c[0], "la_ad": c[1], "xr_h": c[2], "xr_a": c[3],
                     "w": eng.xg_rate_weight, "ls": eng.lambda_scale_})
        if eng.xg_rate_model_ is not None and pd.notna(r.home_xg) and pd.notna(r.away_xg):
            eng.xg_rate_model_.update_form(h, a, r.home_xg, r.away_xg)
    print(f"{season} {mode} {wk} n={len(rows)} {time.time() - t0:.0f}s", flush=True)
pd.DataFrame(rows).to_csv(out, index=False)
print("DONE", season, mode, len(rows), flush=True)
