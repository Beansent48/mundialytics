from __future__ import annotations

"""Pre-kickoff lineup pass: re-price a match once both starting XIs are out.

The morning logger (log_upcoming_round.py) prices the round at 06:30, before anyone knows
who plays. About an hour before kick-off ESPN publishes the lineups. This pass finds Big
Five matches kicking off within LOOKAHEAD_MIN, waits until both XIs are confirmed, measures
the market-value share of each side's usual starters who are not even in the matchday
squad (features/lineup_absence.py), and logs the deployed engine's prediction with that
tilt to data/processed/logs/lineup_pass_log.csv, once per match. The morning prediction
stays the track record of record; this log is graded next to it
(scripts/evaluate_lineup_pass.py).

Backtest of the signal: 1X2 RPS -0.00118, better in 6 of 6 seasons on top of the
squad-value shift, O/U not worse (scripts/lineup_pass/eval_signals.py). A regular on the
bench is rotation and carries nothing; the first version counted him and was worth half.

How it runs: the daily refresh (run_update.ps1) starts it once with --watch. The watcher
reads the day's Big Five kick-offs, sleeps until 70 minutes before the first, polls every
5 minutes until both XIs are out, sleeps again until the next match, and exits after the
last kick-off (or at once on a day without matches). One watcher at a time (a lock file).
    python scripts/log_lineup_pass.py --watch          # the daily watcher
    python scripts/log_lineup_pass.py [--dry-run]      # one pass now (manual / debugging)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from mundialytics.features.lineup_absence import match_values, regular_xi, unavailable_share  # noqa: E402
from mundialytics.identity.normalization import canonical_team_name  # noqa: E402
from mundialytics.providers.espn_fixtures import espn_json  # noqa: E402

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}"
LEAGUES = {"eng.1": "Premier League", "esp.1": "LaLiga", "ger.1": "Bundesliga",
           "ita.1": "Serie A", "fra.1": "Ligue 1"}
LOG = ROOT / "data/processed/logs/lineup_pass_log.csv"
HISTORY = ROOT / "data/external/advanced/espn/espn_player_match_current.csv"
ALIASES = ROOT / "data/curated/fixture_team_aliases.csv"
LOOKAHEAD_MIN = 75
WAKE_BEFORE_MIN = 70     # the watcher starts polling this long before a kick-off
POLL_MIN = 5
LOCK = ROOT / "data/processed/logs/lineup_watch.lock"
PRED_LOG = ROOT / "data/processed/logs/predictions_log.csv"
PLAYER_LOG = ROOT / "data/processed/logs/lineup_pass_players_log.csv"
# Team markets re-priced with the referee (and the XI's lambdas): ESPN names the
# referee in the summary it serves the lineups from, so this is where the
# referee feature (props/team_props.py REF_DEV_*) is sure to have him.
TEAM_LOG = ROOT / "data/processed/logs/lineup_pass_team_log.csv"
# The morning player markets, re-priced on the confirmed matchday squad: each
# player on the minutes of his role tonight (PlayerPropsModel.predict_lineup).
# Backtest: every prop 5/5 folds better than the morning price.
PLAYER_COLS = {"p_anytime_scorer": ("jug_goleador", ""), "p_2plus_goals": ("jug_2goles", ""),
               "p_shots_over_1_5": ("jug_tiros", 1.5), "p_shots_over_2_5": ("jug_tiros", 2.5),
               "p_assist": ("jug_asistencia", ""), "p_yellow": ("jug_amarilla", "")}
ROSTERS = ROOT / "data/external/advanced/espn/espn_team_rosters_current.csv"
PLAYER_VALUES = ROOT / "data/processed/squad_player_values.csv"   # scripts/build_squad_values.py


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
    """{home|away: {team, starters, bench, squad}} when both sides list exactly eleven
    starters. `squad` is everyone named for the match, starters and bench."""
    sides = {}
    for side in summary.get("rosters", []) or []:
        roster = side.get("roster", []) or []
        starters = [(p.get("athlete") or {}).get("displayName") for p in roster if p.get("starter")]
        starters = [s for s in starters if s]
        if len(starters) != 11:
            return None
        squad = [n for n in ((p.get("athlete") or {}).get("displayName") for p in roster) if n]
        bench = [(p.get("athlete") or {}).get("displayName") for p in roster if not p.get("starter")]
        sides[side.get("homeAway")] = {"team": canon(side.get("team", {}).get("displayName", "")),
                                       "starters": starters, "bench": [b for b in bench if b],
                                       "squad": squad}
    return sides if {"home", "away"} <= set(sides) else None


def team_values(team: str, names, table: pd.DataFrame | None) -> dict:
    """{ESPN name: EUR} for `names` from the team's latest live Transfermarkt squad."""
    if table is None:
        return {}
    t = table[table["team"] == team]
    if t.empty:
        return {}
    return match_values(names, t[t["snap"] == t["snap"].max()])


def referee_of(summary: dict) -> str | None:
    """The referee ESPN names in a match summary, or None."""
    for o in (summary.get("gameInfo", {}) or {}).get("officials", []) or []:
        if str((o.get("position") or {}).get("name", "")).lower() == "referee" or o.get("order") == 1:
            return str(o.get("fullName") or o.get("displayName") or "").strip() or None
    return None


def lineup_team_rows(tp, e: dict, h: str, a: str, lams: dict, referee: str | None,
                     now: pd.Timestamp) -> list[dict]:
    """Team event markets (totals, sides, booking points) with the referee, one row per line."""
    try:
        fx = tp.predict_fixture(h, a, referee=referee, lam_home=lams["home"], lam_away=lams["away"])
    except Exception as exc:
        print(f"    team props {h} vs {a}: failed ({str(exc)[:60]})", flush=True)
        return []
    base = {"logged_at_utc": now.strftime("%Y-%m-%dT%H:%M:%S"), "event_id": e["event_id"],
            "kickoff_utc": e["kickoff"].strftime("%Y-%m-%dT%H:%M:%S"), "competition": e["competition"],
            "home": h, "away": a, "referee": referee or ""}
    out = []
    for mk, d in fx.items():
        for key, amb in (("over", "Total"), ("over_home", "Local"), ("over_away", "Visitante")):
            for ln, p in (d.get(key) or {}).items():
                out.append({**base, "mercado": mk, "ambito": amb, "linea": ln, "prob": round(float(p), 4)})
    return out


def lineup_player_rows(pp, e: dict, xis: dict, lams: dict, now: pd.Timestamp,
                       ref_dev: float | None = None) -> list[dict]:
    """Player markets for both confirmed squads, one row per (player, market)."""
    h, a = xis["home"]["team"], xis["away"]["team"]
    out = []
    for side in ("home", "away"):
        team = xis[side]["team"]
        try:
            props, missing = pp.predict_lineup(team, xis[side]["starters"], xis[side]["bench"],
                                               atk_factor=pp._atk_factor(team, float(lams[side]), None),
                                               ref_dev=ref_dev)
        except Exception as exc:
            print(f"    player props {team}: failed ({str(exc)[:60]})", flush=True)
            continue
        for r in props.itertuples(index=False):
            for col, (mk, line) in PLAYER_COLS.items():
                prob = getattr(r, col, None)
                if prob is None or pd.isna(prob):
                    continue
                out.append({"logged_at_utc": now.strftime("%Y-%m-%dT%H:%M:%S"), "event_id": e["event_id"],
                            "kickoff_utc": e["kickoff"].strftime("%Y-%m-%dT%H:%M:%S"),
                            "competition": e["competition"], "partido": f"{h} vs {a}",
                            "team": team, "side": side, "player": r.player, "started": bool(r.started),
                            "exp_min": int(r.exp_min), "mercado": mk, "linea": line,
                            "prob": round(float(prob), 4)})
        print(f"    {team}: {len(props)} players priced on their role"
              + (f", {len(missing)} without big-five history" if missing else ""), flush=True)
    return out


def logged_ids() -> set[str]:
    return set(pd.read_csv(LOG, dtype={"event_id": str})["event_id"]) if LOG.exists() else set()


def run_pass(lookahead: int = LOOKAHEAD_MIN, dry_run: bool = False) -> int:
    """One pass: log every match in the window whose two XIs are out. Returns rows logged."""
    now = pd.Timestamp.now(tz="UTC")
    canon = _canon()

    done = logged_ids()
    todo = [e for e in upcoming(now, lookahead) if e["event_id"] not in done]
    if not todo:
        print(f"{now:%Y-%m-%d %H:%M} UTC: nothing kicking off in the next {lookahead} min "
              f"without a lineup pass", flush=True)
        return 0

    hist = pd.read_csv(HISTORY)
    hist["team"] = hist["team"].map(canon)
    values = pd.read_csv(PLAYER_VALUES) if PLAYER_VALUES.exists() else None
    engine = None
    pp = None
    tp = None
    rows, player_rows, team_rows = [], [], []
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
            squad = set(xis[side]["squad"])
            absent[side] = (unavailable_share(regs, squad, team_values(team, regs, values))
                            if regs else None)
            missing[side] = [p for p in (regs or []) if p not in squad]
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
        if pp is None:
            from api.engine import props_models
            tp, pp = props_models()
            pp = pp if pp is not None else False
            tp = tp if tp is not None else False
        lams = {"home": p.lambda_home, "away": p.lambda_away}
        referee = referee_of(s)
        rd = tp.referee_deviation("yellows", referee, h, a) if (tp and referee) else None
        if tp:
            team_rows.extend(lineup_team_rows(tp, e, h, a, lams, referee, now))
        if pp:
            prow = lineup_player_rows(pp, e, xis, lams, now, ref_dev=rd)
            player_rows.extend(prow)
        print(f"    referee: {referee or 'not named yet'}"
              + (f" (card deviation {rd:+.2f}/match)" if rd is not None else ""), flush=True)
        print(f"  {h} vs {a}: regulars out of the squad {len(missing['home'])}/{len(missing['away'])} -> "
              f"1X2 {p.p_home_win:.2f}/{p.p_draw:.2f}/{p.p_away_win:.2f} ({p.model_source})", flush=True)

    if rows and not dry_run:
        out = pd.DataFrame(rows)
        if LOG.exists():
            out = pd.concat([pd.read_csv(LOG, dtype={"event_id": str}), out], ignore_index=True)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(LOG, index=False)
        print(f"logged {len(rows)} match(es) to {LOG}", flush=True)
        if team_rows:
            tm = pd.DataFrame(team_rows)
            if TEAM_LOG.exists():
                tm = pd.concat([pd.read_csv(TEAM_LOG, dtype={"event_id": str}), tm], ignore_index=True)
            tm.to_csv(TEAM_LOG, index=False)
            print(f"logged {len(team_rows)} team-market rows to {TEAM_LOG}", flush=True)
        if player_rows:
            pl = pd.DataFrame(player_rows)
            if PLAYER_LOG.exists():
                pl = pd.concat([pd.read_csv(PLAYER_LOG, dtype={"event_id": str}), pl], ignore_index=True)
            pl.to_csv(PLAYER_LOG, index=False)
            print(f"logged {len(player_rows)} player-market rows to {PLAYER_LOG}", flush=True)
        revalidate_web()
    return len(rows)


def _pid_alive(pid: int) -> bool:
    """Whether a process is still running, without touching it.

    Not os.kill(pid, 0): on Windows that TERMINATES the process.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259     # STILL_ACTIVE
    try:
        os.kill(pid, 0)                           # POSIX: signal 0 only checks
    except OSError:
        return False
    return True


def fixtures_logged_ahead(now: pd.Timestamp, horizon_h: int) -> int | None:
    """Big Five fixtures the morning logger wrote for the next `horizon_h` hours, from the
    local log -- no network. None when the local files cannot answer (then ask ESPN).

    The daily refresh logs every fixture of the coming days before this runs (and fails
    loudly when it misses one), so an empty answer means no match today.
    """
    try:
        log = pd.read_csv(PRED_LOG, usecols=["fecha", "kickoff_utc", "home", "mercado"], low_memory=False)
        big5 = set(pd.read_csv(ROSTERS, usecols=["team"])["team"].str.lower())
    except Exception:
        return None
    x = log[(log["mercado"] == "1X2") & log["home"].str.lower().isin(big5)]
    kick = pd.to_datetime(x["kickoff_utc"], errors="coerce", utc=True)
    end = now + pd.Timedelta(hours=horizon_h)
    by_kick = (kick >= now) & (kick <= end)
    days = {now.tz_convert("Europe/Madrid").strftime("%Y-%m-%d"), end.tz_convert("Europe/Madrid").strftime("%Y-%m-%d")}
    by_day = kick.isna() & x["fecha"].astype(str).isin(days)
    return int((by_kick | by_day).sum())


def watch(horizon_h: int = 20) -> int:
    """Sleep until each kick-off's lineup window, poll through it, exit after the last."""
    ahead = fixtures_logged_ahead(pd.Timestamp.now(tz="UTC"), horizon_h)
    if ahead == 0:
        print(f"watch: no Big Five fixture logged for the next {horizon_h} h; exiting without "
              f"contacting ESPN", flush=True)
        return 0
    if LOCK.exists():
        try:
            other = int(LOCK.read_text().strip())
        except ValueError:
            other = -1
        if other != os.getpid() and _pid_alive(other):
            print(f"another watcher is running (pid {other}); exiting", flush=True)
            return 0
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(os.getpid()))
    try:
        schedule = upcoming(pd.Timestamp.now(tz="UTC"), horizon_h * 60)
        print(f"watch: {len(schedule)} Big Five kick-off(s) in the next {horizon_h} h", flush=True)
        while True:
            now = pd.Timestamp.now(tz="UTC")
            done = logged_ids()
            left = [e for e in schedule if e["kickoff"] > now and e["event_id"] not in done]
            if not left:
                print(f"{now:%H:%M} UTC: no kick-off left to watch; done", flush=True)
                return 0
            wake = min(e["kickoff"] for e in left) - pd.Timedelta(minutes=WAKE_BEFORE_MIN)
            if now < wake:
                # sleep in slices so the log shows it is alive and a clock change is absorbed
                time.sleep(min((wake - now).total_seconds(), 1800))
                continue
            try:
                run_pass(LOOKAHEAD_MIN)
            except Exception as exc:  # ESPN hiccup: keep watching, the next poll retries
                print(f"{now:%H:%M} UTC: pass failed ({type(exc).__name__}: {exc}); retrying", flush=True)
            time.sleep(POLL_MIN * 60)
    finally:
        if LOCK.exists() and LOCK.read_text().strip() == str(os.getpid()):
            LOCK.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="the daily watcher (see the module docstring)")
    ap.add_argument("--lookahead", type=int, default=LOOKAHEAD_MIN)
    ap.add_argument("--dry-run", action="store_true", help="print, do not write the log")
    args = ap.parse_args()
    if args.watch:
        return watch()
    run_pass(args.lookahead, args.dry_run)
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
