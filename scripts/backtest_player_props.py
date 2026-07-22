from __future__ import annotations

"""Player-props backtest harness (validation-FIRST, before any engine wiring).

Data: understat_player_match.csv (620k player-match rows: minutes, position with
'Sub' marking substitutes, goals, shots, xG, xA, assists, yellow_cards) + the
deployed engine's cached walk-forward team lambdas (match context).

Per player-match, ALL walk-forward (shifted, leakage-safe):
  rates      career-to-date per-90 (xG, goals, shots, xA, assists, yellows),
             credibility-shrunk toward the position-group mean (K=900 minutes).
  minutes    appearance rate over the team's last 10 games x avg minutes over the
             player's last 5 appearances.
  context    team match lambda / team baseline lambda (attacking props scale).

Props scored on appearances (played >= 1'), bookmaker-style:
  anytime scorer, 2+ goals, shots >=2 / >=3 (lines 1.5/2.5), assist anytime,
  yellow card.
Baselines: global rate, position-group rate, and the NAIVE player model (career
rate x avg minutes, no shrinkage, no context) — our machinery must beat naive.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PM = ROOT / "data/external/advanced/understat/understat_player_match.csv"
TM = ROOT / "data/processed/understat_team_match_xg.csv"
PREDS = ROOT / "data/processed/enriched/understat_xg/walkforward_preds.csv"
FOUND = ROOT / "data/processed/enriched/understat_xg/canonical_matches_with_xg.csv"
TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
K_MIN = 900.0
POS_GROUP = {"GK": "GK", "D": "DEF", "M": "MID", "A": "ATT", "F": "FW", "S": "SUB"}


def pos_group(p: str) -> str:
    p = str(p)
    if p == "GK":
        return "GK"
    if p.startswith("D") and not p.startswith("DM"):
        return "DEF"
    if p.startswith("DM") or p.startswith("M"):
        return "MID"
    if p.startswith("AM"):
        return "ATT"
    if p.startswith("F"):
        return "FW"
    return "SUB"


def build_panel() -> pd.DataFrame:
    pm = pd.read_csv(PM)
    tm = (pd.read_csv(TM)[["provider_match_id", "date"]]
          .rename(columns={"provider_match_id": "game_id"}).drop_duplicates("game_id"))
    pm = pm.merge(tm, on="game_id", how="left")
    pm["date"] = pd.to_datetime(pm["date"], errors="coerce")
    pm = pm.dropna(subset=["date", "minutes"]).copy()
    pm = pm.drop(columns=["season"], errors="ignore")  # understat int season clashes with foundation season joined later
    for c in ["minutes", "goals", "shots", "xg", "xa", "assists", "yellow_cards"]:
        pm[c] = pd.to_numeric(pm[c], errors="coerce").fillna(0.0)
    pm["started"] = (pm["position"].astype(str) != "Sub").astype(int)

    # position group: career MODE of non-Sub positions (approx-static; Sub rows inherit it)
    mode_pos = (pm[pm.position != "Sub"].groupby("player_id")["position"]
                .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "MC"))
    pm["pgroup"] = pm["player_id"].map(mode_pos).map(pos_group).fillna("MID")

    pm = pm.sort_values(["player_id", "date", "game_id"])
    g = pm.groupby("player_id", group_keys=False)
    pm["cmin"] = g["minutes"].apply(lambda s: s.shift(1).cumsum()).fillna(0.0)
    pm["rmin15"] = g["minutes"].apply(lambda s: s.shift(1).rolling(15, min_periods=1).sum()).fillna(0.0)
    for c in ["xg", "goals", "shots", "xa", "assists", "yellow_cards"]:
        pm[f"c_{c}"] = g[c].apply(lambda s: s.shift(1).cumsum()).fillna(0.0)
        pm[f"rr_{c}"] = g[c].apply(lambda s: s.shift(1).rolling(15, min_periods=1).sum()).fillna(0.0)

    # appearance rate: this player's appearances among the team's last 10 games
    tg = pm[["team", "game_id", "date"]].drop_duplicates().sort_values(["team", "date"])
    tg["tgn"] = tg.groupby("team").cumcount()
    pm = pm.merge(tg[["team", "game_id", "tgn"]], on=["team", "game_id"], how="left")
    pm = pm.sort_values(["player_id", "team", "tgn"])
    g2 = pm.groupby(["player_id", "team"], group_keys=False)
    pm["app_n"] = g2.cumcount()
    pm["prev_tgn"] = g2["tgn"].shift(1)
    # appearances in the team's last 10 games = count of my rows with tgn in [tgn-10, tgn-1]
    def apps_last10(s):
        tgns = s.to_numpy()
        return pd.Series([np.sum((tgns[:i] >= t - 10) & (tgns[:i] < t)) for i, t in enumerate(tgns)], index=s.index)
    pm["apps10"] = g2["tgn"].apply(apps_last10)
    # E[minutes | plays]: rolling over last 10 PLAYED games only (unused-bench rows
    # have minutes=0 and must not drag the average down — props condition on playing)
    pm["min_played"] = pm["minutes"].where(pm["minutes"] > 0)
    pm["avg_minp10"] = g2["min_played"].apply(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    pm["nplayed10"] = g2["min_played"].apply(lambda s: s.shift(1).rolling(10, min_periods=1).count())
    pm["start_rate10"] = g2["started"].apply(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    return pm


def add_context(pm: pd.DataFrame) -> pd.DataFrame:
    found = pd.read_csv(FOUND, low_memory=False)[["match_id", "provider_match_id", "home_team", "away_team", "season"]]
    preds = pd.read_csv(PREDS).merge(found, on="match_id", how="left", suffixes=("", "_f"))
    preds["game_id"] = pd.to_numeric(preds["provider_match_id"], errors="coerce")
    sys.path.insert(0, str(ROOT / "src"))
    from mundialytics.enrichment.understat_team_aliases import to_foundation_name
    pm["team_fd"] = pm["team"].map(to_foundation_name)
    h = preds[["game_id", "home_team", "lh", "season"]].rename(columns={"home_team": "team_fd", "lh": "team_lam"})
    a = preds[["game_id", "away_team", "la", "season"]].rename(columns={"away_team": "team_fd", "la": "team_lam"})
    lam = pd.concat([h, a], ignore_index=True).dropna(subset=["game_id"])
    pm = pm.merge(lam, on=["game_id", "team_fd"], how="left")
    # team baseline lambda: team's rolling mean of match lambdas (shifted)
    pm = pm.sort_values(["team_fd", "date"])
    base = (pm[["team_fd", "game_id", "date", "team_lam"]].drop_duplicates(["team_fd", "game_id"])
            .sort_values(["team_fd", "date"]))
    base["team_lam_base"] = (base.groupby("team_fd", group_keys=False)["team_lam"]
                             .apply(lambda s: s.shift(1).rolling(19, min_periods=5).mean()))
    pm = pm.merge(base[["team_fd", "game_id", "team_lam_base"]], on=["team_fd", "game_id"], how="left")
    pm["atk_factor"] = (pm["team_lam"] / pm["team_lam_base"]).clip(0.4, 2.5)
    return pm


def shrunk_rate(cnum, cmin, prior_per90, k=K_MIN):
    raw = np.where(cmin > 0, cnum / np.clip(cmin, 1e-9, None) * 90.0, prior_per90)
    cred = cmin / (cmin + k)
    return cred * raw + (1 - cred) * prior_per90


def main() -> None:
    t0 = time.time()
    pm = build_panel()
    print(f"panel: {len(pm)} rows ({time.time()-t0:.0f}s)", flush=True)
    pm = add_context(pm)
    print(f"context joined: {pm['team_lam'].notna().sum()} rows with match lambda", flush=True)

    played = pm["minutes"] > 0
    train_mask = (~pm["season"].isin(TEST_SEASONS)) & played
    test_mask = pm["season"].isin(TEST_SEASONS) & pm["team_lam"].notna() & played
    train = pm[train_mask]
    print(f"test appearances: {int(test_mask.sum())}  train: {len(train)}", flush=True)

    # position-group priors from TRAIN only
    pri = train.groupby("pgroup").apply(
        lambda gr: pd.Series({c: gr[c].sum() / max(gr["minutes"].sum(), 1) * 90.0
                              for c in ["xg", "goals", "shots", "xa", "assists", "yellow_cards"]}),
        include_groups=False)
    glob = {c: train[c].sum() / max(train["minutes"].sum(), 1) * 90.0
            for c in ["xg", "goals", "shots", "xa", "assists", "yellow_cards"]}

    # per-stat recency blend (v4 A/B): recent-15-app form helps shots/goals, HURTS
    # cards (noisy) and is neutral for assists -> career-only there
    RECENT_W = {"xg": 0.5, "goals": 0.5, "shots": 0.5, "xa": 0.0, "assists": 0.0, "yellow_cards": 0.0}
    for c in ["xg", "goals", "shots", "xa", "assists", "yellow_cards"]:
        prior = pm["pgroup"].map(pri[c]).fillna(glob[c])
        r_car = shrunk_rate(pm[f"c_{c}"], pm["cmin"], prior)
        w = RECENT_W[c]
        if w > 0:
            # recent form rate, shrunk toward the CAREER rate (not the position prior)
            r_rec = shrunk_rate(pm[f"rr_{c}"], pm["rmin15"], r_car, k=450.0)
            pm[f"r_{c}"] = w * r_rec + (1 - w) * r_car
        else:
            pm[f"r_{c}"] = r_car

    # expected minutes GIVEN the player plays: credibility blend of the player's
    # last-10-played average toward the position-group mean minutes (train)
    pos_min = train.groupby("pgroup")["minutes"].mean()
    prior_min = pm["pgroup"].map(pos_min).fillna(float(train["minutes"].mean()))
    cred_m = pm["nplayed10"].fillna(0) / (pm["nplayed10"].fillna(0) + 3.0)
    pm["exp_min"] = cred_m * pm["avg_minp10"].fillna(prior_min) + (1 - cred_m) * prior_min
    emins_all = pm["exp_min"].clip(20, 95) / 90.0         # model's pre-match expectation
    af_all = pm["atk_factor"].fillna(1.0) ** 0.7          # team uplift doesn't transfer 1:1 to each player

    # OUR model: xG-based scoring rate x expected minutes x match context
    pm["mu_goal"] = (0.7 * pm["r_xg"] + 0.3 * pm["r_goals"]) * emins_all * af_all
    pm["mu_shots"] = pm["r_shots"] * emins_all * af_all
    pm["mu_ass"] = (0.7 * pm["r_xa"] + 0.3 * pm["r_assists"]) * emins_all * af_all
    pm["mu_yc"] = pm["r_yellow_cards"] * emins_all ** 0.7  # cards sub-linear in minutes (late-game refs, subs get carded)

    test = pm[test_mask].copy()
    emins = test["exp_min"].clip(20, 95) / 90.0
    # NAIVE player model: raw career rate x expected minutes, no shrink/context
    raw_g = np.where(test["cmin"] > 0, test["c_goals"] / test["cmin"].clip(lower=1) * 90, glob["goals"])
    test["mu_goal_naive"] = raw_g * emins
    raw_s = np.where(test["cmin"] > 0, test["c_shots"] / test["cmin"].clip(lower=1) * 90, glob["shots"])
    test["mu_shots_naive"] = raw_s * emins

    from scipy.stats import nbinom, poisson

    def p_ge(mu, k, disp=1.0):
        mu = np.clip(np.asarray(mu, dtype=float), 1e-6, 10)
        if disp > 1.05:
            r = mu / (disp - 1.0)
            return 1 - nbinom.cdf(k - 1, r, 1.0 / disp)
        return 1 - poisson.cdf(k - 1, mu)

    # shots are over-dispersed per appearance: pick dispersion on TRAIN (grid),
    # judged by binary log-loss of the 2+ line under NB with train mus
    tr_mu = pm[train_mask & pm["mu_shots"].notna()]
    y_tr2 = (tr_mu["shots"] >= 2).astype(float).to_numpy()
    best = (1.0, 1e9)
    for d in (1.0, 1.15, 1.3, 1.5, 1.7):
        p = np.clip(p_ge(tr_mu["mu_shots"], 2, d), 1e-6, 1 - 1e-6)
        ll = float(-(y_tr2 * np.log(p) + (1 - y_tr2) * np.log(1 - p)).mean())
        if ll < best[1]:
            best = (d, ll)
    disp_shots = best[0]
    print(f"shots dispersion (train-picked): {disp_shots}", flush=True)

    def bll(y, p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    def report(name, y, p_model, p_naive, base_by_pos):
        base_p = test["pgroup"].map(base_by_pos).fillna(y.mean()).to_numpy()
        print(f"\n===== {name} (n={len(y)}, rate {y.mean():.3f}) =====")
        print(f"  log-loss  OURS {bll(y, p_model):.4f} | naive {bll(y, p_naive):.4f} | pos-base {bll(y, base_p):.4f} | global {bll(y, np.full(len(y), y.mean())):.4f}")
        # per-fold consistency: OURS vs pos-base per test season
        seas = test["season"].to_numpy()
        marks = []
        for s in TEST_SEASONS:
            msk = seas == s
            if msk.sum() < 100:
                continue
            marks.append(f"{s[-2:]}{'+' if bll(y[msk], np.asarray(p_model)[msk]) < bll(y[msk], base_p[msk]) else '-'}")
        print(f"  folds vs pos-base: [{' '.join(marks)}]")
        # calibration deciles for ours
        q = pd.qcut(p_model, 8, duplicates="drop")
        cal = pd.DataFrame({"p": p_model, "y": y}).groupby(q, observed=True).agg(n=("y", "size"), pred=("p", "mean"), emp=("y", "mean"))
        ece = float((cal.n / cal.n.sum() * (cal.pred - cal.emp).abs()).sum())
        print(f"  ECE {ece:.4f}")
        print(cal.round(3).to_string())

    y_goal = (test["goals"] >= 1).astype(float).to_numpy()
    base_goal = train[train.minutes > 0].groupby("pgroup").apply(lambda g: (g.goals >= 1).mean(), include_groups=False)
    report("ANYTIME SCORER", y_goal, p_ge(test["mu_goal"], 1), p_ge(test["mu_goal_naive"], 1), base_goal)

    y_2g = (test["goals"] >= 2).astype(float).to_numpy()
    report("2+ GOALS", y_2g, p_ge(test["mu_goal"], 2), p_ge(test["mu_goal_naive"], 2),
           train[train.minutes > 0].groupby("pgroup").apply(lambda g: (g.goals >= 2).mean(), include_groups=False))

    y_s2 = (test["shots"] >= 2).astype(float).to_numpy()
    report("SHOTS 2+ (line 1.5)", y_s2, p_ge(test["mu_shots"], 2, disp_shots), p_ge(test["mu_shots_naive"], 2, disp_shots),
           train[train.minutes > 0].groupby("pgroup").apply(lambda g: (g.shots >= 2).mean(), include_groups=False))

    y_s3 = (test["shots"] >= 3).astype(float).to_numpy()
    report("SHOTS 3+ (line 2.5)", y_s3, p_ge(test["mu_shots"], 3, disp_shots), p_ge(test["mu_shots_naive"], 3, disp_shots),
           train[train.minutes > 0].groupby("pgroup").apply(lambda g: (g.shots >= 3).mean(), include_groups=False))

    y_a = (test["assists"] >= 1).astype(float).to_numpy()
    report("ASSIST ANYTIME", y_a, p_ge(test["mu_ass"], 1), p_ge(test["mu_ass"] * 0 + glob["assists"] * emins, 1),
           train[train.minutes > 0].groupby("pgroup").apply(lambda g: (g.assists >= 1).mean(), include_groups=False))

    y_y = (test["yellow_cards"] >= 1).astype(float).to_numpy()
    report("YELLOW CARD", y_y, p_ge(test["mu_yc"], 1), p_ge(test["mu_yc"] * 0 + glob["yellow_cards"] * emins, 1),
           train[train.minutes > 0].groupby("pgroup").apply(lambda g: (g.yellow_cards >= 1).mean(), include_groups=False))

    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
