from __future__ import annotations

"""ONE-COMMAND season update — keeps the whole system fresh for 2026/27+.

    .venv/Scripts/python.exe scripts/update_season.py            # weekly in-season
    .venv/Scripts/python.exe scripts/update_season.py --full     # + regenerate the
                                                                 # Resultados walk-forward cache

Steps (each tolerant — a failed download never corrupts existing data; all
Understat writes are append + dedupe):
  1. football-data.co.uk CSVs for the current (and previous) season
  2. Understat shots + player-match stats for the current season
  3. rebuild the Understat xG match aggregation (canonical + team-match)
  4. rebuild the modeling foundation from the raw CSVs
  5. re-attach the xG columns to the foundation (the builder predates xG)
  6. refresh canonical_matches_with_xg (walk-forward context joins)
  7. prune the fitted-props cache (the app refits + recaches on next load)
  7m. pre-compute the league forecasts and Golden Boot races the web serves
  8. [--full] regenerate the deployed-chain walk-forward cache (Resultados page)

The engines and props models need no manual retrain: they fit from the
foundation at app load, and the props joblib cache key includes the data's
max date, so it self-invalidates.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mundialytics.utils import atomic_to_csv  # noqa: E402

PY = sys.executable
STEPS_JSON = ROOT / "data/processed/logs/last_steps.json"
# Exit codes: 0 all good, 1 aborted (foundation rolled back), 2 finished but a
# required step failed. Before this, a failed step only printed "FAILED" and the
# run still exited 0 -- ESPN answered 403 three days running (2026-09-16..18),
# the player markets went unsettled, and last_run.json said ok every time.
EXIT_STEP_FAILED = 2
STEPS: list[dict] = []


def record(name: str, ok: bool, optional: bool = False, seconds: float = 0.0,
           detail: str = "") -> bool:
    STEPS.append({"step": name, "ok": bool(ok), "optional": bool(optional),
                  "seconds": round(seconds, 1), "detail": detail[:200]})
    return ok


def failed_steps() -> list[str]:
    return [s["step"] for s in STEPS if not s["ok"] and not s["optional"]]


def write_steps(exit_code: int) -> None:
    import json

    STEPS_JSON.parent.mkdir(parents=True, exist_ok=True)
    STEPS_JSON.write_text(json.dumps({
        "finished_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "exit_code": exit_code,
        "failed": failed_steps(),
        "failed_optional": [s["step"] for s in STEPS if not s["ok"] and s["optional"]],
        "steps": STEPS,
    }, indent=2), encoding="utf-8")


UNDERSTAT_LEAGUES = ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga",
                     "ITA-Serie A", "FRA-Ligue 1"]
SHOTS_CSV = ROOT / "data/external/advanced/understat/understat_shots.csv"
PLAYER_CSV = ROOT / "data/external/advanced/understat/understat_player_match.csv"
FOUND = ROOT / "data/processed/foundation_big5_multi_season.csv"
TEAM_MATCH = ROOT / "data/processed/understat_team_match_xg.csv"
# Live current-season xG from Sofascore (Understat froze 2026-05-24). Same schema
# as TEAM_MATCH; augment_foundation_with_xg concats both before the join.
SOFA_TEAM_MATCH = ROOT / "data/processed/sofascore_team_match_xg.csv"


def season_codes(today: date) -> tuple[str, str]:
    """('2627', '2526') style current + previous season codes (July = new season)."""
    y = today.year % 100
    if today.month >= 7:
        return f"{y:02d}{(y + 1) % 100:02d}", f"{(y - 1) % 100:02d}{y:02d}"
    return f"{(y - 1) % 100:02d}{y:02d}", f"{(y - 2) % 100:02d}{(y - 1) % 100:02d}"


class IntegrityError(RuntimeError):
    pass


def check_foundation_integrity(prev_rows: int | None) -> None:
    """Abort before caches/models are touched if the rebuilt foundation looks
    corrupt — a bad download must never silently poison the deployed models."""
    df = pd.read_csv(FOUND, low_memory=False)
    problems = []
    n = len(df)
    if n < 40000:                                   # we carry 26 seasons (~45k)
        problems.append(f"only {n} rows (expected 40k+)")
    if prev_rows is not None and n < prev_rows - 5:  # rebuild should never LOSE matches
        problems.append(f"row count dropped {prev_rows} -> {n}")
    for col in ["home_goals", "away_goals", "home_team", "away_team", "date", "competition"]:
        if col not in df.columns:
            problems.append(f"missing column {col}")
        elif df[col].isna().mean() > 0.02:
            problems.append(f"{col} has {df[col].isna().mean():.0%} NaN")
    d = pd.to_datetime(df.get("date"), errors="coerce")
    if d.notna().mean() < 0.98:
        problems.append("dates unparseable")
    elif d.max() < pd.Timestamp("2024-01-01"):
        problems.append(f"newest match {d.max():%Y-%m-%d} — data looks stale")
    for gc in ["home_goals", "away_goals"]:
        if gc in df.columns:
            g = pd.to_numeric(df[gc], errors="coerce")
            if g.max() > 20 or g.min() < 0:
                problems.append(f"{gc} out of range [{g.min()}, {g.max()}]")
    if df.duplicated(subset=["match_id"]).any():
        problems.append("duplicate match_id")
    if problems:
        raise IntegrityError("; ".join(problems))
    print(f"    integrity OK: {n} rows, dates to {d.max():%Y-%m-%d}, "
          f"xG cov {df.get('home_xg', pd.Series(dtype=float)).notna().mean():.0%}", flush=True)


def _kill_process_tree(pid: int) -> None:
    """Terminate a process and all its descendants (only this pid's subtree).

    A timed-out step's Chrome/chromedriver children survive a plain kill and pile
    up over daily runs. On Windows `taskkill /T` walks the descendant tree by
    parent-pid, so it reaches those grandchildren and nothing else -- an
    unrelated browser is not a descendant of this step and is never touched.
    """
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def run_step(name: str, cmd: list[str], optional: bool = False,
             timeout: int | None = None) -> bool:
    """Run one refresh step.

    `optional` marks a step whose failure must not stop the refresh -- a second
    data source, say. `timeout` caps a step that can hang rather than fail: the
    FBref fetch once ran for minutes without writing a byte, which is exactly
    the kind of stall that must not hold up the weekly run.

    On timeout the WHOLE process tree is killed, not just the direct child. The
    FBref fetch drives Chrome through soccerdata/Selenium; killing only the Python
    child (subprocess.run's default) orphans the chromedriver + Chrome
    grandchildren, which then accumulate across daily runs. `taskkill /T` targets
    only descendants of this step's own process, so a user's own browser -- never
    a descendant -- is untouched.
    """
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT)
    try:
        proc.wait(timeout=timeout)
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc.pid)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(f"    TIMEOUT tras {timeout}s (árbol de proceso terminado)", flush=True)
        ok = False
    tag = "OK" if ok else ("FAILED (opcional, se continua)" if optional else "FAILED")
    print(f"    {tag} ({time.time()-t0:.0f}s)", flush=True)
    return record(name, ok, optional, time.time() - t0)


def update_understat(season: str) -> None:
    """Append + dedupe shots and player-match stats for one season code."""
    from soccerdata import Understat
    for out, reader, dedupe in [
        (SHOTS_CSV, "read_shot_events", ["shot_id"]),
        (PLAYER_CSV, "read_player_match_stats", ["game_id", "player_id"]),
    ]:
        existing = pd.read_csv(out) if out.exists() else pd.DataFrame()
        frames = []
        for lg in UNDERSTAT_LEAGUES:
            try:
                u = Understat(leagues=lg, seasons=season)
                df = getattr(u, reader)().reset_index()
                frames.append(df)
                print(f"    OK   {out.name} {lg} {season}: {len(df)} rows", flush=True)
            except Exception as exc:
                print(f"    FAIL {out.name} {lg} {season}: {str(exc)[:120]}", flush=True)
        if not frames:
            continue
        combined = pd.concat([existing, *frames], ignore_index=True)
        keys = [k for k in dedupe if k in combined.columns]
        if keys:
            combined = combined.drop_duplicates(subset=keys, keep="first")
        atomic_to_csv(combined, out)
        print(f"    {out.name}: {len(existing)} -> {len(combined)} rows", flush=True)


def augment_foundation_with_xg() -> None:
    """Re-attach the 8 xG columns (the foundation builder predates xG)."""
    found = pd.read_csv(FOUND, low_memory=False)
    xg_cols = ["home_xg", "away_xg", "home_npxg", "away_npxg",
               "home_xg_op", "away_xg_op", "home_xg_sp", "away_xg_sp"]
    found = found.drop(columns=[c for c in xg_cols if c in found.columns])
    cols = ["date", "home_team_fd", "away_team_fd"] + xg_cols
    # Understat carries the history; Sofascore carries the current season (Understat
    # froze 2026-05-24). Concat both, Understat first so it wins any overlap.
    sources = [pd.read_csv(TEAM_MATCH)[cols]]
    if SOFA_TEAM_MATCH.exists():
        sources.append(pd.read_csv(SOFA_TEAM_MATCH)[cols])
    tm = pd.concat(sources, ignore_index=True)
    tm = tm.rename(columns={"home_team_fd": "home_team", "away_team_fd": "away_team"})
    tm["date"] = pd.to_datetime(tm["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    tm = tm.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
    merged = found.merge(tm, on=["date", "home_team", "away_team"], how="left")
    atomic_to_csv(merged, FOUND)
    cov = merged["home_xg"].notna().mean()
    cur_mask = merged["date"] >= "2026-07-01"
    cur_cov = merged.loc[cur_mask, "home_xg"].notna().mean() if cur_mask.any() else float("nan")
    print(f"    foundation: {len(merged)} matches, xG coverage {cov:.1%} "
          f"(current season {cur_cov:.1%})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=None, help="Season code like 2627 (auto from date)")
    ap.add_argument("--skip-understat", action="store_true", help="Skip the network-heavy Understat pulls")
    ap.add_argument("--skip-players", action="store_true",
                    help="Skip the slow FBref player-stats download")
    ap.add_argument("--skip-logging", action="store_true",
                    help="Skip logging predictions for the upcoming round")
    ap.add_argument("--full", action="store_true",
                    help="Also regenerate the deployed walk-forward cache (Resultados page, ~15 min)")
    args = ap.parse_args()

    cur, prev = season_codes(date.today())
    cur = args.season or cur
    print(f"Season update: current={cur} (also refreshing {prev})", flush=True)

    run_step("1/8 football-data CSVs",
             [PY, "scripts/download_football_data_stats.py", "--seasons", cur, prev,
              "--mode", "csv", "--leagues", "E0", "SP1", "I1", "D1", "F1"])

    if not args.skip_understat:
        # Optional: Understat has served a JS stub since 2026-05 and fails on
        # every league; the history it holds is already on disk.
        print("\n=== 2/8 Understat (shots + player-match) ===", flush=True)
        t0 = time.time()
        try:
            update_understat(cur)
            record("2/8 Understat", True, optional=True, seconds=time.time() - t0)
        except Exception as exc:
            print(f"    FAILED (opcional): {str(exc)[:120]}", flush=True)
            record("2/8 Understat", False, optional=True, seconds=time.time() - t0,
                   detail=str(exc))
    else:
        print("\n=== 2/8 Understat SKIPPED ===", flush=True)

    run_step("3/8 Understat xG aggregation", [PY, "scripts/build_understat_xg_matches.py"])

    # Live current-season xG from Sofascore (Understat froze 2026-05-24). Optional:
    # a Sofascore outage must not break the daily refresh -- the augment step falls
    # back to whatever sofascore_team_match_xg.csv already holds. Timeout caps a
    # hung API pull the way the FBref step is capped.
    run_step("3b/8 Sofascore xG (current season)",
             [PY, "scripts/build_sofascore_xg_matches.py"],
             optional=True, timeout=600)

    prev_rows = len(pd.read_csv(FOUND, low_memory=False)) if FOUND.exists() else None
    backup_found = FOUND.with_suffix(".csv.prev")
    if FOUND.exists():
        backup_found.write_bytes(FOUND.read_bytes())   # rollback safety net
    # Steps 4-5b land together or not at all. Any failure in between, not only a
    # failed integrity check, restores the backup and stops the run: the xG
    # augment raising used to leave a half-built foundation in place.
    try:
        if not run_step("4/8 foundation rebuild",
                        [PY, "scripts/build_foundation_big5_historical.py"]):
            raise IntegrityError("the foundation builder failed")

        print("\n=== 5/8 foundation xG augment ===", flush=True)
        augment_foundation_with_xg()
        record("5/8 foundation xG augment", True)

        print("\n=== 5b/8 foundation integrity check ===", flush=True)
        check_foundation_integrity(prev_rows)
        record("5b/8 foundation integrity", True)
    except Exception as exc:
        print(f"    ABORT: foundation step failed -> {str(exc)[:200]}", flush=True)
        record("4-5b/8 foundation", False, detail=str(exc))
        if backup_found.exists():
            os.replace(backup_found, FOUND)
            print("    rolled back to the previous foundation; models untouched.", flush=True)
        write_steps(1)
        raise SystemExit(1) from exc
    finally:
        backup_found.unlink(missing_ok=True)

    run_step("6/8 canonical_matches_with_xg",
             [PY, "scripts/enrich_matches_with_xg.py",
              "--matches", "data/processed/foundation_big5_multi_season.csv",
              "--xg", "data/external/xg/understat/understat_xg_matches.csv",
              "--provider", "understat",
              "--out-dir", "data/processed/enriched/understat_xg",
              "--allow-missing-xg"])

    print("\n=== 6b/8 European fixtures refresh (UCL/UEL/UECL) ===", flush=True)
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from mundialytics.statistical_core.competition.european import (
            FD_SLUG, fetch_season_fixtures)
        yr = date.today().year if date.today().month >= 7 else date.today().year - 1
        for comp, slug in FD_SLUG.items():
            # Fetch BEFORE dropping the cache. Deleting first made the cache
            # useless as a fallback exactly when it was needed: with both hosts
            # down the file was gone and the fetch had nothing to fall back on.
            df = fetch_season_fixtures(ROOT, comp, yr)
            if df is not None:
                res_n = df["Result"].astype(str).str.contains(r"\d+\s*-\s*\d+").sum()
                print(f"    OK   {slug} {yr}: {len(df)} partidos, {res_n} con resultado", flush=True)
            else:
                print(f"    ---  {slug} {yr}: aún no publicado", flush=True)
        record("6b/8 European fixtures", True, optional=True)
    except Exception as exc:
        print(f"    FAIL europeo: {str(exc)[:120]}", flush=True)
        record("6b/8 European fixtures", False, optional=True, detail=str(exc))

    print("\n=== 7/8 prune fitted-model caches + backup prediction log ===", flush=True)
    n = 0
    for pat in ["props_models_*.joblib", "engine_*.joblib"]:
        for f in (ROOT / "data/processed/cache").glob(pat):
            f.unlink(missing_ok=True)
            n += 1
    print(f"    removed {n} cache file(s); the app refits + recaches on next load", flush=True)
    log_f = ROOT / "data/processed/logs/predictions_log.csv"
    if log_f.exists():
        bdir = ROOT / "data/processed/logs/backup"
        bdir.mkdir(parents=True, exist_ok=True)
        dest = bdir / f"predictions_log_{date.today().isoformat()}.csv"
        dest.write_bytes(log_f.read_bytes())
        backups = sorted(bdir.glob("predictions_log_*.csv"))
        for old in backups[:-10]:
            old.unlink(missing_ok=True)
        print(f"    prediction log backed up -> {dest.name} ({len(backups)} kept, max 10)", flush=True)

    # Per-player results for the current season. Understat had not published
    # 2026/27 while the player markets were already being logged, which would
    # have left them unsettleable -- the booking-points failure again. FBref has
    # it, so settlement no longer waits on one provider. Slow (it drives Chrome),
    # which is why it lives in the weekly refresh and not in the logger.
    # ESPN runs first and carries the load: FBref only ever yielded the Premier
    # League before stalling, leaving 79% of the logged player predictions
    # unsettleable. ESPN is a JSON API asked by date, covers all five leagues,
    # and reports the full roster, so "played and did not score" settles as the
    # miss it is. FBref stays as a second opinion and is allowed to fail.
    if not args.skip_players:
        run_step("7d/8 ESPN player stats (settles the jug_* markets)",
                 [PY, "scripts/fetch_espn_match_events.py"])
        run_step("7e/8 FBref player stats (segunda fuente, puede fallar)",
                 [PY, "scripts/fetch_fbref_player_stats.py"],
                 optional=True, timeout=600)
        # Europe writes to its own files, so this cannot corrupt the league
        # ones, and it is optional: a bad day at UEFA must not stop the
        # domestic markets settling. It feeds the scorers of the other 35
        # clubs in a SquadLab Champions run, and nothing that settles a bet.
        run_step("7d2/8 ESPN player stats, UEFA (CL/EL/UECL)",
                 [PY, "scripts/fetch_espn_match_events.py",
                  "--competitions", "uefa"],
                 optional=True, timeout=900)
        # Squads last: it reads both halves of the player-match data above.
        run_step("7f/8 current squads (big-5 + los 108 de Europa)",
                 [PY, "scripts/build_current_squads.py"],
                 optional=True, timeout=1200)
    else:
        print("\n=== 7d/8 player stats SKIPPED ===", flush=True)

    # Rebuild the locally-advanced ClubElo before logging, so European
    # predictions use ratings carried forward to today's results rather than a
    # snapshot frozen whenever the API last worked (it was 502 for three days
    # running with the Champions League five days out). See
    # src/mundialytics/ratings/clubelo_local.py.
    run_step("7c/8 local ClubElo roll-forward",
             [PY, "scripts/build_local_clubelo.py"])

    if not args.skip_logging:
        # Log the upcoming round BEFORE it is played. This is what makes the
        # track record real: predictions must be written pre-kickoff, by an
        # engine fitted only on played matches. Doing it here means it happens
        # every weekly refresh instead of depending on someone remembering to
        # click a button -- which is exactly why the log held a single
        # retroactive matchday for a whole season (see the quarantine README).
        # Fails (exit 3/4) when any fixture in the window is left without a
        # prediction or a league's calendar is missing -- a lost matchday is a
        # test of the model that can never be run again.
        run_step("7b/8 log upcoming round (pre-match track record)",
                 [PY, "scripts/log_upcoming_round.py"])
        # And looking back: anything played this week with no prediction logged
        # before kick-off (catches days the logger never ran at all).
        run_step("7b2/8 prediction coverage audit (last 7 days)",
                 [PY, "scripts/audit_prediction_coverage.py", "--days", "7"])
    else:
        print("\n=== 7b/8 upcoming-round logging SKIPPED ===", flush=True)

    # League forecasts and Golden Boot races cost ~30 s each and are keyed on the
    # matches played, so this refresh has just invalidated them. Computing them
    # here means no visitor waits past the web client's 20 s timeout — which
    # rendered the league page as a 404 for the first one after every refresh.
    run_step("7m/8 warm league + awards caches (web)",
             [PY, "scripts/warm_api_caches.py"], optional=True, timeout=900)

    if args.full:
        run_step("8/8 deployed walk-forward cache (Resultados)",
                 [PY, "scripts/generate_deployed_walkforward.py"])
    else:
        print("\n=== 8/8 walk-forward cache SKIPPED (use --full) ===", flush=True)

    # ── summary ────────────────────────────────────────────────────────────────
    print("\n===== SUMMARY =====", flush=True)
    for label, p, datecol in [
        ("foundation", FOUND, "date"),
        ("team xG", TEAM_MATCH, "date"),
        ("player-match", PLAYER_CSV, None),
        ("shots", SHOTS_CSV, None),
    ]:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            last = f" | last: {pd.to_datetime(df[datecol], errors='coerce').max():%Y-%m-%d}" if datecol else ""
            print(f"  {label:14s} {len(df):>8,} rows{last}", flush=True)

    failed = failed_steps()
    code = EXIT_STEP_FAILED if failed else 0
    write_steps(code)
    if failed:
        print(f"\n  PASOS FALLIDOS: {', '.join(failed)}", flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
