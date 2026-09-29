from __future__ import annotations

"""Grade the matchday re-prices against the first logged prediction, same matches.

For every played match with a pass row and a full first-logged row (predictions_log.csv,
1X2 with p_home/p_draw/p_away), print both 1X2 RPS and the difference. Two passes:
  lineup  (data/processed/logs/lineup_pass_log.csv, ~1h before kick-off): the backtest
          expects about -0.0012 per match on average;
  morning (data/processed/logs/morning_pass_log.csv, matchday morning): about -0.0003.
Both are small against match-to-match noise: it takes a few hundred matches to see them.

    python scripts/evaluate_lineup_pass.py [--morning]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.serving.logged_prediction import find_logged  # noqa: E402

LINEUP = ROOT / "data/processed/logs/lineup_pass_log.csv"
MORNING = ROOT / "data/processed/logs/morning_pass_log.csv"
RESULTS = ROOT / "data/external/advanced/espn/espn_matches_current.csv"
PLAYER_LOG = ROOT / "data/processed/logs/lineup_pass_players_log.csv"
TEAM_LOG = ROOT / "data/processed/logs/lineup_pass_team_log.csv"
FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
PRED_LOG = ROOT / "data/processed/logs/predictions_log.csv"


def rps(p: np.ndarray, outcome: int) -> float:
    y = np.zeros(3)
    y[outcome] = 1
    return float(((np.cumsum(p) - np.cumsum(y))[:2] ** 2).sum() / 2)


def main() -> int:
    path, what = (MORNING, "morning") if "--morning" in sys.argv else (LINEUP, "lineup")
    if not path.exists():
        print(f"no {what} passes logged yet")
        return 0
    lp = pd.read_csv(path, dtype={"event_id": str})
    res = pd.read_csv(RESULTS, dtype={"event_id": str})
    x = lp.merge(res[["event_id", "home_goals", "away_goals"]], on="event_id", how="inner")
    rows = []
    for r in x.itertuples():
        morning = find_logged(r.home, r.away, pd.Timestamp(r.kickoff_utc))
        if morning is None or not morning.full:
            continue
        o = 0 if r.home_goals > r.away_goals else 1 if r.home_goals == r.away_goals else 2
        m = np.array([morning.trio["home"], morning.trio["draw"], morning.trio["away"]])
        l_ = np.array([r.p_home, r.p_draw, r.p_away])
        rows.append({"match": f"{r.home} {r.home_goals}-{r.away_goals} {r.away}",
                     "absent": f"{r.absent_home:.2f}/{r.absent_away:.2f}" if pd.notna(r.absent_home) else "n/a",
                     "rps_morning": rps(m, o), "rps_lineup": rps(l_, o)})
    if not rows:
        print(f"{len(lp)} {what} passes logged, none played yet with a full first-logged row")
        return 0
    d = pd.DataFrame(rows)
    d["delta"] = d.rps_lineup - d.rps_morning
    print(d.round(4).to_string(index=False))
    se = d.delta.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    print(f"\n{len(d)} matches: RPS first logged {d.rps_morning.mean():.4f} -> {what} pass {d.rps_lineup.mean():.4f} "
          f"(delta {d.delta.mean():+.4f} ± {se:.4f}); {what} pass better in {(d.delta < 0).mean():.0%}")
    if what == "lineup":     # the player markets are only re-priced on the confirmed squads
        grade_players()
        grade_team()
    return 0


def grade_team() -> None:
    """Team event markets: lineup-pass price (with the referee) vs the morning price,
    same (match, market, side, line), settled from the foundation."""
    if not TEAM_LOG.exists() or not FOUND.exists():
        return
    from mundialytics.serving.track_record import EVENT_COLS

    lp = pd.read_csv(TEAM_LOG, dtype={"event_id": str})
    f = pd.read_csv(FOUND, low_memory=False)
    f["fecha"] = pd.to_datetime(f["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    res = {(r.home_team, r.away_team, r.fecha): r for r in f.itertuples(index=False)}
    log = pd.read_csv(PRED_LOG, low_memory=False)
    mo = log[log["mercado"].isin(list(EVENT_COLS) + ["booking_pts"])].copy()
    mo["lk"] = mo["linea"].map(_line_key)
    mk_ = {(r.home, r.away, r.mercado, r.ambito, r.lk, str(r.fecha)): float(r.prob)
           for r in mo.itertuples(index=False)}
    rows = []
    for r in lp.itertuples(index=False):
        ko = pd.Timestamp(r.kickoff_utc)
        days = [ko.strftime("%Y-%m-%d"), ko.tz_localize("UTC").tz_convert("Europe/Madrid").strftime("%Y-%m-%d")]
        act = next((res[(r.home, r.away, d)] for d in days if (r.home, r.away, d) in res), None)
        morning = next((mk_[k] for d in days
                        if (k := (r.home, r.away, r.mercado, r.ambito, _line_key(r.linea), d)) in mk_), None)
        if act is None or morning is None:
            continue
        if r.mercado == "booking_pts":
            hv = 10 * act.home_yellow_cards + 25 * getattr(act, "home_red_cards", 0)
            av = 10 * act.away_yellow_cards + 25 * getattr(act, "away_red_cards", 0)
        else:
            hc, ac = EVENT_COLS[r.mercado]
            hv, av = getattr(act, hc), getattr(act, ac)
        if pd.isna(hv) or pd.isna(av):
            continue
        val = {"Total": hv + av, "Local": hv, "Visitante": av}[r.ambito]
        rows.append({"mercado": r.mercado, "referee": bool(str(r.referee or "").strip() and r.referee == r.referee),
                     "y": float(val > float(r.linea)), "p_lineup": float(r.prob), "p_morning": morning})
    if not rows:
        print(f"\n{lp.event_id.nunique()} matches with lineup team prices, none settled yet")
        return
    d = pd.DataFrame(rows)
    for col in ("lineup", "morning"):
        p = d[f"p_{col}"].clip(1e-4, 1 - 1e-4)
        d[f"ll_{col}"] = -(d.y * np.log(p) + (1 - d.y) * np.log(1 - p))
    g = d.groupby(["mercado", "referee"]).agg(n=("y", "size"), ll_morning=("ll_morning", "mean"),
                                              ll_lineup=("ll_lineup", "mean"))
    g["delta"] = g.ll_lineup - g.ll_morning
    print(f"\nteam markets, {len(d)} settled lines (referee = named by ESPN at lineup time):")
    print(g.round(4).to_string())


def _happened(mk: str, line, act) -> bool:
    if mk == "jug_goleador":
        return act.goals >= 1
    if mk == "jug_2goles":
        return act.goals >= 2
    if mk == "jug_tiros":
        return act.shots > float(line)
    if mk == "jug_asistencia":
        return act.assists >= 1
    return act.yellow_cards >= 1


def grade_players() -> None:
    """Player markets: lineup-pass price vs the morning price, same (match, player,
    market), settled only for players who took the field (as the track record does)."""
    if not PLAYER_LOG.exists():
        return
    from mundialytics.serving.track_record import _player_match_actuals

    lp = pd.read_csv(PLAYER_LOG, dtype={"event_id": str})
    acts = _player_match_actuals()
    if lp.empty or acts.empty:
        return
    ak = {(r.fecha, str(r.equipo), str(r.player)): r for r in acts.itertuples(index=False)}
    log = pd.read_csv(PRED_LOG, low_memory=False)
    mo = log[log["mercado"].astype(str).str.startswith("jug_")].copy()
    mo["lk"] = mo["linea"].map(_line_key)
    mk_ = {(str(r.ambito), r.mercado, r.lk, str(r.fecha)): float(r.prob) for r in mo.itertuples(index=False)}
    rows = []
    for r in lp.itertuples(index=False):
        ko = pd.Timestamp(r.kickoff_utc)
        days = {ko.strftime("%Y-%m-%d"), ko.tz_localize("UTC").tz_convert("Europe/Madrid").strftime("%Y-%m-%d")}
        act = next((ak[k] for d in days if (k := (d, str(r.team), str(r.player))) in ak), None)
        if act is None:
            continue
        morning = next((mk_[k] for d in days
                        if (k := (str(r.player), r.mercado, _line_key(r.linea), d)) in mk_), None)
        y = float(_happened(r.mercado, r.linea, act))
        rows.append({"mercado": f"{r.mercado}{'>' + str(r.linea) if _line_key(r.linea) else ''}",
                     "started": r.started, "y": y, "p_lineup": r.prob, "p_morning": morning})
    if not rows:
        print(f"\n{lp.event_id.nunique()} matches with lineup player prices, none settled yet")
        return
    d = pd.DataFrame(rows)

    def ll(p, y):
        p = np.clip(p.astype(float), 1e-4, 1 - 1e-4)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    d["ll_lineup"] = ll(d.p_lineup, d.y)
    both = d.dropna(subset=["p_morning"]).copy()
    both["ll_morning"] = ll(both.p_morning, both.y)
    print(f"\nplayer markets, {len(d)} settled rows ({len(both)} also priced in the morning):")
    g = both.groupby("mercado").agg(n=("y", "size"), real=("y", "mean"), morning=("p_morning", "mean"),
                                    lineup=("p_lineup", "mean"), ll_morning=("ll_morning", "mean"),
                                    ll_lineup=("ll_lineup", "mean"))
    g["delta"] = g.ll_lineup - g.ll_morning
    print(g.round(4).to_string())


def _line_key(x) -> str:
    try:
        return f"{float(x):g}" if str(x).strip() not in ("", "nan") else ""
    except ValueError:
        return ""


if __name__ == "__main__":
    sys.exit(main())
