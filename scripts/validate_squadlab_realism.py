from __future__ import annotations

"""END-TO-END realism validation for SquadLab (never done before).

For each big-5 (club, season) with enough data:
  1. reconstruct the club's real best-XI for THAT season,
  2. bridge it to (attack_param, defense_param) via the DEPLOYED calibration,
  3. compute the club's EXPECTED SEASON POINTS from those params vs the rest of
     the league at their real fitted params (analytic Poisson round-robin),
  4. compare to the club's ACTUAL final points that season.

Two arms isolate SquadLab's OWN error from the match engine's:
  - REAL params -> expected pts  (ceiling: how well the validated AD model
    itself explains the real table),
  - BRIDGED params -> expected pts  (what SquadLab actually produces).
The gap between the two = the realism cost of the player->strength bridge,
reported separately for the attack and defense axes (defense R^2=0.25 is the
suspected weak link).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mundialytics.statistical_core.attack_defense_model import AttackDefenseModel  # noqa: E402
from mundialytics.statistical_core.schemas import canonical_name  # noqa: E402
from mundialytics.statistical_core.squadlab import calibration_constants as CC  # noqa: E402

# reuse the exact reconstruction the calibration uses (same best-XI, same idx)
from fit_squad_lambda_calibration_season_scoped import (  # noqa: E402
    BIG5_COMP_IDS, MIN_TEAM_MATCHES, POSITION_SLOTS, _discover_season_matches, _team_strength)


def _expected_points(a, d, opp_params, mu, ha, max_goals=10) -> float:
    """Expected league points for a team (attack a, defense d) over a full
    home+away round-robin vs opp_params [(a_o, d_o), ...]. Axis 0 of the grid
    is ALWAYS this team's goals; home advantage goes to whoever is home."""
    k = np.arange(max_goals + 1)
    pts = 0.0
    for a_o, d_o in opp_params:
        # fixture A: this team HOME  (this gets ha)
        lam_us_h = np.exp(mu + ha + a - d_o)
        lam_op_h = np.exp(mu + a_o - d)
        # fixture B: this team AWAY  (opponent gets ha)
        lam_us_a = np.exp(mu + a - d_o)
        lam_op_a = np.exp(mu + ha + a_o - d)
        for lam_us, lam_op in ((lam_us_h, lam_op_h), (lam_us_a, lam_op_a)):
            M = np.outer(poisson.pmf(k, lam_us), poisson.pmf(k, lam_op))
            p_win = np.tril(M, -1).sum()    # our goals (rows) > opp goals (cols)
            p_draw = np.trace(M)
            pts += 3 * p_win + p_draw
    return float(pts)


def main() -> None:
    prof = pd.read_csv(ROOT / "data/processed/player_profiles_by_season.csv")
    prof["team_c"] = prof["team"].map(canonical_name)
    prof["pass_completion"] = (prof["complete_passes_per_match"]
                               / prof["passes_per_match"].clip(lower=0.01)).fillna(0.75).clip(0, 1)
    prof["finishing_per_shot"] = (prof["goals_per_match"]
                                  / prof["shots_per_match"].clip(lower=0.01)).fillna(0.0).clip(0, 1)
    season_matches = _discover_season_matches()

    rows = []
    for (comp, season), gm in season_matches.groupby(["competition", "season"]):
        if len(gm) < 20:
            continue
        ad = AttackDefenseModel()
        ad.fit(gm)
        params = ad.team_params().set_index("team")
        li = ad.league_index_.get(comp, 0)
        mu = float(ad.league_effect_[li]) if li < len(ad.league_effect_) else ad.mu_
        ha = float(ad.league_home_adv_[li]) if li < len(ad.league_home_adv_) else ad.home_adv_
        all_teams = list(params.index)
        real_all = {t: (float(params.loc[t, "attack"]), float(params.loc[t, "defense"])) for t in all_teams}

        # actual final points this season from the match results
        pts_actual = {t: 0.0 for t in all_teams}
        for r in gm.itertuples(index=False):
            h, a = canonical_name(r.home_team), canonical_name(r.away_team)
            if h not in pts_actual or a not in pts_actual:
                continue
            if r.home_goals > r.away_goals:
                pts_actual[h] += 3
            elif r.home_goals < r.away_goals:
                pts_actual[a] += 3
            else:
                pts_actual[h] += 1
                pts_actual[a] += 1

        sp = prof[(prof["competition"] == comp) & (prof["season"] == season)]
        for team_c, tg in sp.groupby("team_c"):
            if team_c not in params.index or ad.match_counts_.get(team_c, 0) < MIN_TEAM_MATCHES:
                continue
            squad = pd.concat([tg[tg["position"] == pos].sort_values("matches", ascending=False).head(n)
                               for pos, n in POSITION_SLOTS.items()])
            if len(squad) < 8:
                continue
            atk_idx, def_idx = _team_strength(squad)
            a_real, d_real = real_all[team_c]
            a_brdg = float(np.clip(CC.GOAL_ATTACK_SLOPE * atk_idx + CC.GOAL_ATTACK_INTERCEPT, *CC.ATTACK_PARAM_CLIP))
            d_brdg = float(np.clip(CC.GOAL_DEFENSE_SLOPE * def_idx + CC.GOAL_DEFENSE_INTERCEPT, *CC.DEFENSE_PARAM_CLIP))
            opps = [real_all[o] for o in all_teams if o != team_c]
            rows.append({
                "team": team_c, "season": season, "comp": comp,
                "atk_idx": atk_idx, "def_idx": def_idx, "a_real": a_real, "d_real": d_real,
                "_opps": opps, "_mu": mu, "_ha": ha,
                "pts_actual": pts_actual[team_c],
                "pts_real": _expected_points(a_real, d_real, opps, mu, ha),
                "pts_bridge": _expected_points(a_brdg, d_brdg, opps, mu, ha),
                "pts_bridge_atkonly": _expected_points(a_brdg, d_real, opps, mu, ha),
                "pts_bridge_defonly": _expected_points(a_real, d_brdg, opps, mu, ha),
            })

    df = pd.DataFrame(rows)
    print(f"\n{len(df)} (club, season) reconstructions across {df.season.nunique()} seasons\n")

    # ── VARIANCE-MATCHED bridge (de-attenuation): map the reconstructed index
    # onto the REAL param distribution's spread (same mean+std), preserving
    # ordering. Fixes the least-squares compression that squashes every squad
    # to mid-table. Constants computed on THIS population (leave-one-out would
    # be cleaner but the moments are population-stable).
    ai_mean, ai_std = df["atk_idx"].mean(), df["atk_idx"].std()
    di_mean, di_std = df["def_idx"].mean(), df["def_idx"].std()
    ap_mean, ap_std = df["a_real"].mean(), df["a_real"].std()
    dp_mean, dp_std = df["d_real"].mean(), df["d_real"].std()
    a_vm = ap_mean + (df["atk_idx"] - ai_mean) / ai_std * ap_std
    d_vm = dp_mean + (df["def_idx"] - di_mean) / di_std * dp_std
    a_vm = a_vm.clip(*CC.ATTACK_PARAM_CLIP)
    d_vm = d_vm.clip(*CC.DEFENSE_PARAM_CLIP)
    opps_by_row = df["_opps"]
    df["pts_vm"] = [
        _expected_points(av, dv, op, mu_, ha_)
        for av, dv, op, mu_, ha_ in zip(a_vm, d_vm, opps_by_row, df["_mu"], df["_ha"])
    ]
    print("VARIANCE-MATCHED bridge constants (attack, defense):")
    print(f"  attack:  param = {ap_mean:.4f} + (idx - {ai_mean:.4f})/{ai_std:.4f} * {ap_std:.4f}")
    print(f"  defense: param = {dp_mean:.4f} + (idx - {di_mean:.4f})/{di_std:.4f} * {dp_std:.4f}\n")

    def mae(col):
        return (df[col] - df["pts_actual"]).abs().mean()

    def corr(col):
        return df[col].corr(df["pts_actual"])

    print("Mean abs error vs REAL final points (over ~38-game seasons):")
    print(f"  REAL params (engine ceiling)     : {mae('pts_real'):5.1f} pts   corr {corr('pts_real'):.3f}")
    print(f"  BRIDGED params (deployed)         : {mae('pts_bridge'):5.1f} pts   corr {corr('pts_bridge'):.3f}")
    print(f"  VARIANCE-MATCHED (candidate)      : {mae('pts_vm'):5.1f} pts   corr {corr('pts_vm'):.3f}")
    print(f"    -> deployed added error         : {mae('pts_bridge') - mae('pts_real'):+5.1f} pts")
    print(f"    -> var-matched added error      : {mae('pts_vm') - mae('pts_real'):+5.1f} pts")
    print(f"  bridge attack only (real defense) : {mae('pts_bridge_atkonly'):5.1f} pts")
    print(f"  bridge defense only (real attack) : {mae('pts_bridge_defonly'):5.1f} pts")
    print(f"\n  points spread (std) — real {df['pts_real'].std():.1f} | deployed "
          f"{df['pts_bridge'].std():.1f} | var-matched {df['pts_vm'].std():.1f} | actual {df['pts_actual'].std():.1f}")

    # where does the bridge error come from? correlation of param error with pts error
    df["atk_err"] = df["pts_bridge_atkonly"] - df["pts_real"]
    df["def_err"] = df["pts_bridge_defonly"] - df["pts_real"]
    print(f"\n  attack-axis pts error (mean abs)  : {df['atk_err'].abs().mean():.1f}")
    print(f"  defense-axis pts error (mean abs) : {df['def_err'].abs().mean():.1f}   <- the weak link if larger")

    # worst misses (SquadLab vs reality)
    df["miss"] = (df["pts_bridge"] - df["pts_actual"]).abs()
    print("\nWorst 8 SquadLab misses (bridged expected vs actual):")
    for r in df.nlargest(8, "miss").itertuples(index=False):
        print(f"  {r.team:16s} {r.season}  actual {r.pts_actual:5.1f} | "
              f"real-model {r.pts_real:5.1f} | squadlab {r.pts_bridge:5.1f}")


if __name__ == "__main__":
    main()
