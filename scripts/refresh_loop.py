"""The daily refresh as a long-lived process, for the container that runs it.

`scripts/run_update.ps1` is the Windows Task Scheduler version of this: it runs
`update_season.py`, writes `last_run.json`, invalidates the web's cache and
starts the lineup watcher. This is the same contract expressed as a process that
sleeps until the next run, because a container has no Task Scheduler and option
(b) of docs/DEPLOYMENT_PLAN.md -- the user's choice on 2026-11-02 -- puts the
refresh inside the stack.

What it keeps from the PowerShell wrapper, deliberately:

- **the light/heavy split** -- `--full` only on REFRESH_FULL_DAY, because
  regenerating the walk-forward cache behind the Resultados page is ~15 minutes
  and does not need to happen daily;
- **stdout teed, stderr filed.** `--full` emits hundreds of thousands of
  sklearn/soccerdata warning lines; sending those to the container log would
  bury the summary. stdout (the summary) goes to the container log and to the
  run log, stderr goes to the run log only and its tail is printed if the run
  fails;
- **`last_run.json`**, same shape, so anything that reads it on the PC reads it
  here too;
- **the cache invalidation** after a successful run, so the new matchday appears
  on the site at once instead of waiting out each page's revalidate window;
- **the lineup watcher**, which re-prices each Big Five match once both XIs are
  out and exits after the last one.

What it does NOT do, and what that costs: `scripts/commit_track_record.py`.
The prediction log and the drift report are the two refreshed files that live in
git, and the container has no checkout and no credentials -- by design, since
nothing in this project pushes unattended. In the stack they accumulate on the
data volume; getting them into git stays a job for the machine that holds the
repository. docs/DEPLOYMENT_PLAN.md records this as the one piece of the daily
pass that option (b) does not move.

Configured entirely by environment, because that is what a compose file can set:

    REFRESH_AT=10:00              local time of the daily run (TZ picks the zone)
    REFRESH_FULL_DAY=Monday       weekday that also passes --full ("never" to disable)
    REFRESH_ON_START=0            1 runs once at startup instead of waiting
    REFRESH_ARGS=                 extra arguments for update_season.py
    REFRESH_KEEP_LOGS=30          how many run logs to keep
    REFRESH_LINEUP_WATCHER=1      start the lineup watcher after each run
    REVALIDATE_URL=http://web:3000/api/revalidate
    REVALIDATE_SECRET=            if the web app requires one

    python scripts/refresh_loop.py
    REFRESH_ON_START=1 REFRESH_ARGS=--skip-understat python scripts/refresh_loop.py
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

LOG_DIR = ROOT / "data/processed/logs/runs"
LAST_RUN = ROOT / "data/processed/logs/last_run.json"
LAST_STEPS = ROOT / "data/processed/logs/last_steps.json"

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Set by SIGTERM/SIGINT so a `docker compose down` does not wait out a sleep.
_STOP = threading.Event()


def say(msg: str) -> None:
    """One line on the container log, timestamped, flushed."""
    print(f"[refresh {datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def next_run(now: datetime, at: str) -> datetime:
    """The next occurrence of HH:MM, today if it is still ahead, else tomorrow."""
    try:
        hour, minute = (int(p) for p in at.split(":", 1))
    except ValueError:
        say(f"REFRESH_AT={at!r} is not HH:MM; falling back to 10:00")
        hour, minute = 10, 0
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return target if target > now else target + timedelta(days=1)


def is_full_day(when: datetime, full_day: str) -> bool:
    if full_day.lower() in {"", "never", "none"}:
        return False
    if full_day.lower() == "daily":
        return True
    if full_day.capitalize() not in WEEKDAYS:
        say(f"REFRESH_FULL_DAY={full_day!r} is not a weekday; no --full run scheduled")
        return False
    return WEEKDAYS[when.weekday()] == full_day.capitalize()


def _pump(stream, log_handle, echo: bool) -> None:
    """Forward a child's output to the run log, and to our stdout if echo."""
    for raw in stream:
        line = raw.rstrip("\n")
        log_handle.write(line + "\n")
        if echo:
            print(line, flush=True)
    log_handle.flush()


def run_update(full: bool) -> tuple[int, Path, str]:
    """Run update_season.py. Returns (exit code, log path, stdout tail)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    log = LOG_DIR / f"update_{stamp}.log"
    errlog = LOG_DIR / f"update_{stamp}.log.stderr"

    argv = [sys.executable, str(ROOT / "scripts/update_season.py")]
    if full:
        argv.append("--full")
    argv += [a for a in env("REFRESH_ARGS").split() if a]

    say(f"starting {'--full ' if full else ''}refresh -> {log.name}")
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as out, errlog.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(
            argv, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        threads = [
            threading.Thread(target=_pump, args=(proc.stdout, out, True), daemon=True),
            threading.Thread(target=_pump, args=(proc.stderr, err, False), daemon=True),
        ]
        for t in threads:
            t.start()
        code = proc.wait()
        for t in threads:
            t.join(timeout=30)

    mins = (time.perf_counter() - started) / 60
    say(f"refresh exited {code} after {mins:.1f} min")
    tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
    if code != 0:
        err_tail = errlog.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        if err_tail:
            say("last stderr lines:")
            for line in err_tail:
                print(f"    {line}", flush=True)
    return code, log, tail


def read_failed_steps(started_iso: str) -> tuple[list[str], list[str]]:
    """What update_season.py recorded, if it recorded it for THIS run."""
    try:
        steps = json.loads(LAST_STEPS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], []
    if str(steps.get("finished_at", "")) < started_iso[:19]:
        return [], []                      # a previous run's file
    return list(steps.get("failed") or []), list(steps.get("failed_optional") or [])


def write_last_run(started: datetime, full: bool, code: int, log: Path, tail: str) -> None:
    """The heartbeat `run_update.ps1` writes, in the same shape and place."""
    failed, failed_optional = read_failed_steps(started.isoformat())
    state = {
        "finished_at": datetime.now().isoformat(),
        "started_at": started.isoformat(),
        "full": bool(full),
        "exit_code": code,
        "ok": code == 0,
        "failed_steps": failed,
        "failed_optional": failed_optional,
        "log": str(log),
        "tail": tail,
    }
    LAST_RUN.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN.write_text(json.dumps(state, indent=2), encoding="utf-8")
    if code != 0:
        # The Windows wrapper raises a toast here. A container has nobody to
        # toast, so this is the loud line: it is what a log alert would match on.
        what = f" failed steps: {', '.join(failed)}" if failed else ""
        say(f"REFRESH FAILED (exit {code}).{what} Log: {log}")


def revalidate() -> None:
    """Invalidate the web app's shared `data` cache tag, best effort."""
    url = env("REVALIDATE_URL")
    if not url:
        return
    secret = env("REVALIDATE_SECRET")
    if secret:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode({"secret": secret})
    try:
        with urllib.request.urlopen(url, timeout=20) as res:  # noqa: S310 (our own URL)
            say(f"revalidated the web cache ({res.status})")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # Never fails the run: without it each page still refreshes on its own
        # revalidate window.
        say(f"web cache revalidation skipped ({type(exc).__name__}: {str(exc)[:120]})")


def start_lineup_watcher() -> None:
    """Sleep-until-kickoff watcher that re-prices each match once the XIs are out.

    Detached and fire-and-forget, exactly as on Windows: it exits by itself after
    the day's last kick-off (immediately on a day with no matches) and a second
    start is a no-op because it holds a lock.
    """
    if env("REFRESH_LINEUP_WATCHER", "1") != "1":
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    log = LOG_DIR / f"lineup_watch_{stamp}.log"
    try:
        with log.open("w", encoding="utf-8") as out:
            subprocess.Popen(
                [sys.executable, str(ROOT / "scripts/log_lineup_pass.py"), "--watch"],
                cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        say(f"lineup watcher started ({log.name})")
    except OSError as exc:
        say(f"lineup watcher not started ({exc})")


def prune_logs(keep: int) -> None:
    """Keep the newest `keep` runs; each is a .log plus its sibling .stderr."""
    for pattern, limit in (("update_*.log*", keep * 2), ("lineup_watch_*.log*", keep)):
        files = sorted(LOG_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[limit:]:
            try:
                old.unlink()
            except OSError:
                pass


def one_pass(full: bool) -> int:
    started = datetime.now()
    code, log, tail = run_update(full)
    write_last_run(started, full, code, log, tail)
    # Exit 1 rolled the foundation back, so there is nothing new to show; exit 2
    # finished with a non-foundation step failed and the data did move.
    if code in (0, 2):
        revalidate()
    start_lineup_watcher()
    try:
        prune_logs(int(env("REFRESH_KEEP_LOGS", "30")))
    except ValueError:
        prune_logs(30)
    return code


def ensure_writable_home() -> None:
    """Create HOME and the scrape cache if the image pointed them at the volume.

    The container may run as a uid with no home directory (compose sets `user:`
    to the owner of the data mount), and a scraper that cannot write its cache
    fails in a way that reads like a network error.
    """
    for var in ("HOME", "SOCCERDATA_DIR"):
        path = env(var)
        if not path:
            continue
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            say(f"could not create {var}={path} ({exc})")


def main() -> int:
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: _STOP.set())
    ensure_writable_home()

    at = env("REFRESH_AT", "10:00")
    full_day = env("REFRESH_FULL_DAY", "Monday")
    say(f"daily refresh at {at} ({time.tzname[0]}), --full on {full_day or 'never'}")

    # Said once, at startup, rather than discovered as a 401 after the first
    # refresh: without a secret the web app only accepts the call from a
    # localhost Host header, and from here the Host is `web`. The refresh still
    # works -- each page refreshes on its own revalidate window -- but the new
    # matchday stops appearing immediately, which is the whole point of the call.
    if env("REVALIDATE_URL") and not env("REVALIDATE_SECRET"):
        say("REVALIDATE_SECRET is not set: /api/revalidate will answer 401 from "
            "this container (it only trusts a localhost Host otherwise). Set the "
            "same value here and on the web service.")

    if env("REFRESH_ON_START") == "1":
        one_pass(is_full_day(datetime.now(), full_day))

    while not _STOP.is_set():
        now = datetime.now()
        when = next_run(now, at)
        wait = (when - now).total_seconds()
        say(f"next run {when:%Y-%m-%d %H:%M} (in {wait / 3600:.1f} h)")
        # Wake every minute rather than sleeping the whole gap: a suspended host
        # or a clock change would otherwise leave this process asleep past its
        # slot, and the Event makes a stop signal land immediately.
        while not _STOP.is_set() and datetime.now() < when:
            _STOP.wait(min(60.0, max(1.0, (when - datetime.now()).total_seconds())))
        if _STOP.is_set():
            break
        one_pass(is_full_day(datetime.now(), full_day))

    say("stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
