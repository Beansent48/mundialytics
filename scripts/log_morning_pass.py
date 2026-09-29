from __future__ import annotations

"""Matchday morning pass: re-price a team's next league match with who is likely still out.

The first logged prediction of a fixture (log_upcoming_round.py, up to 8 days ahead) is
the track record, and at that point the team usually has another match to play first,
so it cannot know who will be missing. On matchday morning it can: a regular who was
not even in the squad for the previous league match, and was not serving a ban there,
is most likely still injured (features/lineup_absence.py: still_out). This pass logs the
deployed engine's prediction with that tilt to data/processed/logs/morning_pass_log.csv,
once per match, for Big Five matches kicking off within LOOKAHEAD_H. The match page
shows it until the lineup pass (log_lineup_pass.py) replaces it; both are graded next to
the track record (scripts/evaluate_lineup_pass.py).

Backtest: 1X2 RPS -0.00026, better in 6 of 6 seasons on top of the squad-value shift,
O/U not worse (scripts/lineup_pass/eval_signals.py inj_cnt). Runs in the daily refresh
(update_season.py step 7b3) after the ESPN history is refreshed.

    python scripts/log_morning_pass.py [--dry-run]
"""

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

from log_lineup_pass import BASE, HISTORY, PRED_LOG, _canon, revalidate_web, upcoming  # noqa: E402
from mundialytics.features.lineup_absence import still_out  # noqa: E402
from mundialytics.providers.espn_fixtures import espn_json  # noqa: E402

LOG = ROOT / "data/processed/logs/morning_pass_log.csv"
LOOKAHEAD_H = 30      # the 06:30 run covers today and the early hours of tomorrow


@lru_cache(maxsize=1)
def _logged_fixtures() -> pd.DataFrame:
    if not PRED_LOG.exists():
        return pd.DataFrame(columns=["home", "away", "fecha"])
    p = pd.read_csv(PRED_LOG, usecols=["home", "away", "fecha"], low_memory=False).drop_duplicates()
    return p.assign(fecha=pd.to_datetime(p["fecha"], errors="coerce"))


def fixtures_before(team: str, last_played: pd.Timestamp, kickoff: pd.Timestamp) -> bool:
    """Does `team` have a logged league fixture after its last played match and before
    this kick-off? Then this is not its next match and last week's squad says nothing."""
    p = _logged_fixtures()
    f = p["fecha"]
    mine = (p["home"] == team) | (p["away"] == team)
    return bool((mine & (f > last_played.normalize()) & (f < kickoff.tz_localize(None).normalize())).any())


def run(dry_run: bool = False) -> int:
    now = pd.Timestamp.now(tz="UTC")
    canon = _canon()
    done = set(pd.read_csv(LOG, dtype={"event_id": str})["event_id"]) if LOG.exists() else set()
    todo = [e for e in upcoming(now, LOOKAHEAD_H * 60) if e["event_id"] not in done]
    if not todo:
        print(f"{now:%Y-%m-%d %H:%M} UTC: no Big Five match in the next {LOOKAHEAD_H} h without a morning pass")
        return 0
    hist = pd.read_csv(HISTORY)
    hist["team"] = hist["team"].map(canon)
    hist["date"] = pd.to_datetime(hist["date"])
    engine = None
    rows = []
    for e in todo:
        s = espn_json(f"{BASE.format(code=e['code'])}/summary?event={e['event_id']}", timeout=30)
        teams = {c.get("homeAway"): canon(c.get("team", {}).get("displayName", ""))
                 for c in (s.get("header", {}).get("competitions", [{}])[0].get("competitors", []) or [])}
        if not {"home", "away"} <= set(teams):
            print(f"  {e['event_id']}: teams not readable; skipped", flush=True)
            continue
        out, names = {}, {}
        for side, team in teams.items():
            th = hist[(hist["competition"] == e["competition"]) & (hist["team"] == team)]
            if th.empty or fixtures_before(team, th["date"].max(), e["kickoff"]):
                out[side], names[side] = None, []
                continue
            r = still_out(th, e["kickoff"].tz_localize(None).normalize(), e["competition"])
            out[side], names[side] = (r if r is not None else (None, []))
        if out["home"] is None and out["away"] is None:
            print(f"  {teams['home']} vs {teams['away']}: no usable history; skipped", flush=True)
            continue
        if engine is None:
            from api.engine import club_engine
            from mundialytics.serving.provenance import model_fingerprint, train_cutoff
            from mundialytics.statistical_core.prediction_engine import DEPLOYED_CLUB_ENGINE_KWARGS
            engine, known, df = club_engine()
            known = set(known)
            prov = dict(train_cutoff=train_cutoff(df), model_fp=model_fingerprint(df, DEPLOYED_CLUB_ENGINE_KWARGS))
        h, a = teams["home"], teams["away"]
        if h not in known or a not in known:
            print(f"  {h} vs {a}: a side unknown to the engine (debutant); skipped", flush=True)
            continue
        # a side without history counts as full strength, as in the backtest
        p = engine.predict_match(h, a, competition=e["competition"],
                                 morning_absent=(out["home"] or 0.0, out["away"] or 0.0))
        rows.append({
            "logged_at_utc": now.strftime("%Y-%m-%dT%H:%M:%S"), "event_id": e["event_id"],
            "kickoff_utc": e["kickoff"].strftime("%Y-%m-%dT%H:%M:%S"), "competition": e["competition"],
            "home": h, "away": a, "absent_home": out["home"], "absent_away": out["away"],
            "missing_home": "; ".join(names["home"]), "missing_away": "; ".join(names["away"]),
            "lambda_home": round(p.lambda_home, 4), "lambda_away": round(p.lambda_away, 4),
            "p_home": round(p.p_home_win, 4), "p_draw": round(p.p_draw, 4), "p_away": round(p.p_away_win, 4),
            "p_over_25": round(p.p_over_25, 4), "model_source": p.model_source, **prov,
        })
        print(f"  {h} vs {a}: likely still out {len(names['home'])}/{len(names['away'])} -> "
              f"1X2 {p.p_home_win:.2f}/{p.p_draw:.2f}/{p.p_away_win:.2f} ({p.model_source})", flush=True)

    if rows and not dry_run:
        new = pd.DataFrame(rows)
        if LOG.exists():
            new = pd.concat([pd.read_csv(LOG, dtype={"event_id": str}), new], ignore_index=True)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(LOG, index=False)
        print(f"logged {len(rows)} match(es) to {LOG}", flush=True)
        revalidate_web()
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print, do not write the log")
    run(ap.parse_args().dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
