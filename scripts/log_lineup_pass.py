from __future__ import annotations

"""Pre-kickoff lineup pass: re-price a match once both starting XIs are out.

The morning logger (log_upcoming_round.py) prices the round at 06:30, before anyone knows
who plays. About an hour before kick-off ESPN publishes the lineups. This pass finds Big
Five matches kicking off within LOOKAHEAD_MIN, waits until both XIs are confirmed, measures
how many of each side's usual starters are missing (features/lineup_absence.py), and logs
the deployed engine's prediction with that tilt to data/processed/logs/lineup_pass_log.csv,
once per match. The morning prediction stays the track record of record; this log is
graded next to it (scripts/evaluate_lineup_pass.py).

Backtest of the signal: 1X2 RPS better in 6 of 6 seasons on top of the squad-value
shift, O/U unchanged (scripts/lineup_pass/backtest_absence.py).

Run every ~10 minutes on match days (run_lineup_pass.ps1 / the scheduled task):
    python scripts/log_lineup_pass.py [--lookahead 75] [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from mundialytics.features.lineup_absence import absent_share, regular_xi  # noqa: E402
from mundialytics.identity.normalization import canonical_team_name  # noqa: E402
from mundialytics.providers.espn_fixtures import espn_json  # noqa: E402

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}"
LEAGUES = {"eng.1": "Premier League", "esp.1": "LaLiga", "ger.1": "Bundesliga",
           "ita.1": "Serie A", "fra.1": "Ligue 1"}
LOG = ROOT / "data/processed/logs/lineup_pass_log.csv"
HISTORY = ROOT / "data/external/advanced/espn/espn_player_match_current.csv"
ALIASES = ROOT / "data/curated/fixture_team_aliases.csv"
LOOKAHEAD_MIN = 75


def _canon():
    alias = {}
    if ALIASES.exists():
        a = pd.read_csv(ALIASES)
        alias = dict(zip(a.iloc[:, 0].astype(str), a.iloc[:, 1].astype(str)))
    return lambda name: alias.get(name, canonical_team_name(name))


def upcoming(now: pd.Timestamp, lookahead_min: int) -> list[dict]:
    """Big Five events kicking off in [now, now + lookahead] that have not started."""
    out = []
    days = sorted({now.strftime("%Y%m%d"), (now + pd.Timedelta(minutes=lookahead_min)).strftime("%Y%m%d")})
    for code, comp in LEAGUES.items():
        for day in days:
            for ev in espn_json(f"{BASE.format(code=code)}/scoreboard?dates={day}", timeout=30).get("events", []) or []:
                ko = pd.Timestamp(ev["date"]).tz_convert("UTC")
                state = ev.get("status", {}).get("type", {}).get("state")
                if state == "pre" and now <= ko <= now + pd.Timedelta(minutes=lookahead_min):
                    out.append({"event_id": str(ev["id"]), "code": code, "competition": comp, "kickoff": ko})
    return out


def confirmed_xis(summary: dict, canon) -> dict[str, dict] | None:
    """{home|away: {team, starters}} when both sides list exactly eleven starters."""
    sides = {}
    for side in summary.get("rosters", []) or []:
        starters = [(p.get("athlete") or {}).get("displayName") for p in side.get("roster", []) or []
                    if p.get("starter")]
        starters = [s for s in starters if s]
        if len(starters) != 11:
            return None
        sides[side.get("homeAway")] = {"team": canon(side.get("team", {}).get("displayName", "")),
                                       "starters": starters}
    return sides if {"home", "away"} <= set(sides) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookahead", type=int, default=LOOKAHEAD_MIN)
    ap.add_argument("--dry-run", action="store_true", help="print, do not write the log")
    args = ap.parse_args()
    now = pd.Timestamp.now(tz="UTC")
    canon = _canon()

    done = set(pd.read_csv(LOG, dtype={"event_id": str})["event_id"]) if LOG.exists() else set()
    todo = [e for e in upcoming(now, args.lookahead) if e["event_id"] not in done]
    if not todo:
        print(f"{now:%Y-%m-%d %H:%M} UTC: nothing kicking off in the next {args.lookahead} min "
              f"without a lineup pass", flush=True)
        return 0

    hist = pd.read_csv(HISTORY)
    hist["team"] = hist["team"].map(canon)
    engine = None
    rows = []
    for e in todo:
        s = espn_json(f"{BASE.format(code=e['code'])}/summary?event={e['event_id']}", timeout=30)
        xis = confirmed_xis(s, canon)
        if xis is None:
            print(f"  {e['event_id']} {e['competition']} {e['kickoff']:%H:%M} UTC: lineups not out yet", flush=True)
            continue
        season_hist = hist[hist["competition"] == e["competition"]]
        absent, missing = {}, {}
        for side in ("home", "away"):
            team = xis[side]["team"]
            regs = regular_xi(season_hist[season_hist["team"] == team],
                              before=e["kickoff"].tz_localize(None).normalize())
            absent[side] = absent_share(regs, xis[side]["starters"]) if regs else None
            missing[side] = [p for p in (regs or []) if p not in set(xis[side]["starters"])]
        if engine is None:
            from api.engine import club_engine
            from mundialytics.serving.provenance import model_fingerprint, train_cutoff
            from mundialytics.statistical_core.prediction_engine import DEPLOYED_CLUB_ENGINE_KWARGS
            engine, known, df = club_engine()
            known = set(known)
            prov = dict(train_cutoff=train_cutoff(df), model_fp=model_fingerprint(df, DEPLOYED_CLUB_ENGINE_KWARGS))
        h, a = xis["home"]["team"], xis["away"]["team"]
        if h not in known or a not in known:
            print(f"  {h} vs {a}: a side unknown to the engine (debutant); the morning stand-in stays", flush=True)
            continue
        use = (absent["home"], absent["away"]) if None not in absent.values() else None
        p = engine.predict_match(h, a, competition=e["competition"], lineup_absent=use)
        rows.append({
            "logged_at_utc": now.strftime("%Y-%m-%dT%H:%M:%S"), "event_id": e["event_id"],
            "kickoff_utc": e["kickoff"].strftime("%Y-%m-%dT%H:%M:%S"), "competition": e["competition"],
            "home": h, "away": a,
            "absent_home": absent["home"], "absent_away": absent["away"],
            "missing_home": "; ".join(missing["home"]), "missing_away": "; ".join(missing["away"]),
            "lambda_home": round(p.lambda_home, 4), "lambda_away": round(p.lambda_away, 4),
            "p_home": round(p.p_home_win, 4), "p_draw": round(p.p_draw, 4), "p_away": round(p.p_away_win, 4),
            "p_over_25": round(p.p_over_25, 4), "model_source": p.model_source, **prov,
        })
        print(f"  {h} vs {a}: missing regulars {len(missing['home'])}/{len(missing['away'])} -> "
              f"1X2 {p.p_home_win:.2f}/{p.p_draw:.2f}/{p.p_away_win:.2f} ({p.model_source})", flush=True)

    if rows and not args.dry_run:
        out = pd.DataFrame(rows)
        if LOG.exists():
            out = pd.concat([pd.read_csv(LOG, dtype={"event_id": str}), out], ignore_index=True)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(LOG, index=False)
        print(f"logged {len(rows)} match(es) to {LOG}", flush=True)
        revalidate_web()
    return 0


def revalidate_web() -> None:
    """Tell the site to drop its cached pages, as run_update.ps1 does after a refresh.

    The API keys the match page on this log's mtime, but the web layer caches API
    responses under a shared tag. Best-effort: if the site is not running, its own
    revalidate window picks the lineup price up a few minutes later.
    """
    import os
    import urllib.request

    url = os.environ.get("MUNDIALYTICS_WEB_URL", "http://localhost:3000") + "/api/revalidate"
    if os.environ.get("REVALIDATE_SECRET"):
        url += f"?secret={os.environ['REVALIDATE_SECRET']}"
    try:
        urllib.request.urlopen(url, timeout=15).read()
        print("revalidated the web cache", flush=True)
    except Exception as exc:
        print(f"web cache revalidation skipped ({type(exc).__name__})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
