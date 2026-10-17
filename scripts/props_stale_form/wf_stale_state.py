"""Does re-basing a long-absent club's EVENT state beat its ancient form?

Audit finding 2 reaches the props layer too: `TeamPropsModel._latest_team_state`
takes `g["ev_for"].tail(w).mean()`, the last w rows whatever their date, past a
`len(g) < 3` guard that a club with 38 ancient rows sails through. That is the
model the board's event markets are priced with, and `_side_lambda` reads the
team's `for_*` AND the opponent's `against_*`, so one stale club spoils its own
λ and its opponents'.

On the goal arm the obvious fixes failed, because 2001 form is no worse as a
*predictor* than the league mean. The screen says the event markets split, and
split along the league-drift axis -- per-match, per-team error at predicting the
club's own rate this season, 58 club-seasons with a gap over two years:

  market    ancient form   league level   league drift 2014->2026
  fouls            2.433          0.887   13.90 -> 11.69  (-16%)
  yellows          0.440          0.294    2.08 ->  1.84
  corners          0.905          0.891    5.05 ->  4.88
  shots            1.586          1.797   12.46 -> 14.13 (non-monotonic)
  sot              0.655          0.873

Where the LEAGUE level drifts, a club's ancient rate measures a refereeing era
that no longer exists; where it does not, the rate is a real team trait and
worth keeping. So the arm tested here replaces only fouls and yellows, and only
while the club lacks `MIN_RECENT` matches inside `STALE_DAYS`, which makes it
self-limiting: once the club has played enough this season the windows are
current anyway.

One fit per block serves both arms -- the defect is in the served state, not in
the coefficients -- so this costs one TeamPropsModel fit per block (~44 s).

    python scripts/props_stale_form/wf_stale_state.py SEASON OUT.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mundialytics.props.team_props import MARKETS, TeamPropsModel  # noqa: E402
from mundialytics.statistical_core.engine_utils import load_clubs_data  # noqa: E402

STALE_DAYS = 730
MIN_RECENT = 19          # = max(WINDOWS): below this the state is not yet current
PATCH = ("fouls", "yellows")
BLOCK_DAYS = 28
EV_COLS = {"corners": ("home_corners", "away_corners"),
           "yellows": ("home_yellow_cards", "away_yellow_cards"),
           "fouls": ("home_fouls", "away_fouls"),
           "shots": ("home_shots", "away_shots"),
           "sot": ("home_sot", "away_sot")}
LINES = {"corners": [8.5, 9.5, 10.5], "yellows": [3.5, 4.5, 5.5],
         "fouls": [21.5, 23.5], "shots": [22.5, 24.5], "sot": [7.5, 8.5]}


def league_levels(train: pd.DataFrame) -> dict[tuple[str, str], float]:
    """{(competition, market): per-side level over the most recent season on hand}."""
    if train.empty:
        return {}
    last = str(train["season"].astype(str).max())
    d = train[train["season"].astype(str) == last]
    out = {}
    for comp, g in d.groupby("competition"):
        for mk, (ch, ca) in EV_COLS.items():
            v = pd.concat([pd.to_numeric(g[ch], errors="coerce"),
                           pd.to_numeric(g[ca], errors="coerce")])
            if v.notna().any():
                out[(str(comp), mk)] = float(v.mean())
    return out


def stale_teams(train: pd.DataFrame, cutoff: pd.Timestamp,
                teams: set[str]) -> set[str]:
    """Clubs without MIN_RECENT matches inside STALE_DAYS of the cutoff."""
    recent = train[train["date"] >= cutoff - pd.Timedelta(days=STALE_DAYS)]
    n = pd.concat([recent["home_team"], recent["away_team"]]).value_counts()
    seen = pd.concat([train["home_team"], train["away_team"]]).value_counts()
    return {t for t in teams
            if int(n.get(t, 0)) < MIN_RECENT and int(seen.get(t, 0)) >= 3}


def patched_feats(model: TeamPropsModel, stale: set[str],
                  levels: dict, comp_of: dict) -> dict:
    """A copy of _team_feats with the stale clubs' PATCH markets re-based."""
    out = {}
    for mk, per_team in model._team_feats.items():
        out[mk] = per_team if mk not in PATCH else {
            t: ({**st, **{k: levels[(comp_of.get(t), mk)]
                          for k in st if k.startswith(("for_r", "against_r"))
                          or k in ("for_ewm", "against_ewm")}}
                if t in stale and (comp_of.get(t), mk) in levels else st)
            for t, st in per_team.items()}
    return out


def main() -> None:
    season, out_path = sys.argv[1], sys.argv[2]
    df = load_clubs_data()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_goals", "away_goals"]).sort_values("date")
    test = df[df["season"].astype(str) == season]
    if test.empty:
        raise SystemExit(f"no rows for {season}")
    comp_of = {**dict(zip(test.home_team, test.competition)),
               **dict(zip(test.away_team, test.competition))}

    rows = []
    blocks = ((test["date"] - test["date"].min()).dt.days // BLOCK_DAYS)
    for blk, part in test.groupby(blocks, sort=True):
        cutoff = part["date"].min()
        train = df[df["date"] < cutoff]
        if len(train) < 5000:
            continue
        model = TeamPropsModel().fit(train)
        lv = league_levels(train)
        stale = stale_teams(train, cutoff, set(comp_of))
        base = model._team_feats
        patched = patched_feats(model, stale, lv, comp_of)
        print(f"{season} block {blk} n={len(part)} stale={len(stale)} "
              f"{sorted(stale)[:4]}", flush=True)
        for r in part.itertuples(index=False):
            touched = bool({r.home_team, r.away_team} & stale)
            preds = {}
            for arm, feats in (("base", base), ("arm", patched)):
                model._team_feats = feats
                preds[arm] = model.predict_fixture(r.home_team, r.away_team)
            model._team_feats = base
            for mk, (ch, ca) in EV_COLS.items():
                actual = pd.to_numeric(pd.Series([getattr(r, ch), getattr(r, ca)]),
                                       errors="coerce").sum()
                if pd.isna(actual) or mk not in preds["base"]:
                    continue
                for line in LINES[mk]:
                    pb = preds["base"][mk]["over"].get(line)
                    pa = preds["arm"].get(mk, {}).get("over", {}).get(line)
                    if pb is None or pa is None:
                        continue
                    rows.append({"season": season, "date": r.date, "mk": mk,
                                 "line": line, "touched": touched,
                                 "over": float(actual > line),
                                 "p_base": float(pb), "p_arm": float(pa)})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"DONE {season} {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
