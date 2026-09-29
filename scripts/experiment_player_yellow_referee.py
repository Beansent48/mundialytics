from __future__ import annotations

"""Player yellow card: does the referee move it, as it moves the team's cards?

v0.56.0 gave the team card/foul models the referee (ESPN, LaLiga/Bundesliga/
Serie A/Ligue 1): the referee's shrunk mean deviation of the match's yellows
from his league's level, ref_dev (team_props._referee_deviation). The player
yellow card never sees it. Here each player's deployed yellow mu is scaled by
exp(b x ref_dev), ref_dev in cards per match, for a few b, on the matches of
the four leagues whose referee ESPN names.

Harness = scripts/backtest_player_props.py (identical panel and recipe);
the referee joins through the canonical match id. Folds 2022/23-2025/26
(ESPN referees start in 2021/22, so the first season only builds history).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import backtest_player_props as bp  # noqa: E402
from mundialytics.props import team_props as tp  # noqa: E402

TEST = ["2022-2023", "2023-2024", "2024-2025", "2025-2026"]
BS = [0.03, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.4]
PICK, HOLD = TEST[:2], TEST[2:]   # b is chosen on the first two folds, judged on the last two


def bll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main() -> None:
    t0 = time.time()
    pm = bp.add_context(bp.build_panel())
    found = pd.read_csv(bp.FOUND, low_memory=False, usecols=["provider_match_id", "match_id"])
    found["game_id"] = pd.to_numeric(found["provider_match_id"], errors="coerce")
    f = pd.read_csv(ROOT / "data/processed/foundation_big5_multi_season.csv", low_memory=False)
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    f = f[f["season"] >= "2014-2015"].dropna(subset=["date", "home_yellow_cards", "away_yellow_cards"])
    refs = tp.load_espn_referees(ROOT if (ROOT / "data/processed/espn_referees.csv").exists()
                                 else ROOT.parents[2])
    f["referee"] = f["match_id"].map(tp.attach_espn_referees(f, refs))
    rd, _ = tp._referee_deviation(f, "home_yellow_cards", "away_yellow_cards")
    f["ref_dev"] = f["match_id"].map(rd).fillna(0.0)
    link = (found.dropna(subset=["game_id"]).drop_duplicates("game_id")
            .merge(f[["match_id", "ref_dev", "referee", "competition"]], on="match_id", how="inner"))
    pm = pm.merge(link[["game_id", "ref_dev", "referee", "competition"]], on="game_id", how="left")
    print(f"panel ready ({time.time() - t0:.0f}s)", flush=True)

    played = pm["minutes"] > 0
    train = pm[(~pm["season"].isin(bp.TEST_SEASONS)) & played]
    pri = train.groupby("pgroup").apply(
        lambda gr: gr["yellow_cards"].sum() / max(gr["minutes"].sum(), 1) * 90.0, include_groups=False)
    glob = train["yellow_cards"].sum() / max(train["minutes"].sum(), 1) * 90.0
    prior = pm["pgroup"].map(pri).fillna(glob)
    r = bp.shrunk_rate(pm["c_yellow_cards"], pm["cmin"], prior)
    pos_min = train.groupby("pgroup")["minutes"].mean()
    prior_min = pm["pgroup"].map(pos_min).fillna(float(train["minutes"].mean()))
    cred = pm["nplayed10"].fillna(0) / (pm["nplayed10"].fillna(0) + 3.0)
    em = (cred * pm["avg_minp10"].fillna(prior_min) + (1 - cred) * prior_min).clip(20, 95) / 90.0
    pm["mu_yc"] = r * em ** 0.7

    test = (pm["season"].isin(TEST) & played & pm["referee"].notna()
            & ~pm["competition"].isin(tp.REF_DEV_SKIP)).to_numpy()
    t = pm[test]
    y = (t["yellow_cards"] >= 1).astype(float).to_numpy()
    seas = t["season"].to_numpy()
    p0 = 1 - poisson.cdf(0, t["mu_yc"].to_numpy())
    print(f"\n{len(t)} appearances with an ESPN referee (4 leagues); ref_dev sd {t['ref_dev'].std():.2f}")
    print(f"  base yellow LL {bll(y, p0):.5f}")
    pick, hold = np.isin(seas, PICK), np.isin(seas, HOLD)
    scores = {}
    for b in BS:
        p1 = 1 - poisson.cdf(0, t["mu_yc"].to_numpy() * np.exp(b * t["ref_dev"].to_numpy()))
        wins = sum(bll(y[seas == s], p1[seas == s]) < bll(y[seas == s], p0[seas == s]) for s in TEST)
        scores[b] = bll(y[pick], p1[pick]) - bll(y[pick], p0[pick])
        print(f"  b={b:<5g} all {bll(y, p1) - bll(y, p0):+.5f} ({wins}/{len(TEST)}) | pick-folds {scores[b]:+.5f}"
              f" | hold-out {bll(y[hold], p1[hold]) - bll(y[hold], p0[hold]):+.5f}")
    print(f"  -> b picked on {PICK}: {min(scores, key=scores.get)}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
