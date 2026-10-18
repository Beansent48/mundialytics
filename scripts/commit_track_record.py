"""Commit the track record, so it outlives this machine.

Everything the daily refresh touches is gitignored except two files: the
prediction log and the morning drift report. Nothing committed them on its own
-- the four "automated daily settlement" commits in the history were typed by
hand, and by 2026-10-18 the committed log was 19 days behind the one on disk,
so anything reading the track record from GitHub read a stale file.

Two files, two cadences, for one reason. market_drift.json is a few KB and its
verdict moves every morning, so it goes in on every run. predictions_log.csv is
14 MB and grows; a daily commit would pay that delta for a handful of appended
rows, so it goes in once a round.

"Once a round" is a cadence in days, not a round read out of the file, and that
is a measured decision rather than a lazy one. The log pre-registers whole
rounds in bulk -- 33,984 rows on 2026-09-25 covering jornadas 1 to 10 and dates
out to 2026-10-31 -- so every (season, jornada) pair is in git already. What
actually arrives in between is the lineup pass re-pricing matches that were
logged weeks ago: of the 4,416 rows added on 2026-10-18, not one brought a new
round, match or date. A round key would therefore have reported "nothing new"
forever while the file drifted 19 days out of date, which is the exact failure
this script exists to end.

Commits, never pushes: nothing in this project talks to a remote unattended.
Stages by name, because this working directory is shared with other sessions
and `git add .` would sweep in their work.

Run:  .venv/Scripts/python.exe scripts/commit_track_record.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = "data/processed/logs/predictions_log.csv"
DRIFT = "data/processed/logs/market_drift.json"

# A Big Five round lands over a weekend and the next one is a week away, so
# five days is "a round has happened" while still collapsing Saturday and
# Sunday into a single commit.
LOG_EVERY_DAYS = 5.0


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=check)


def log_is_due(days_since_commit: float | None, every_days: float = LOG_EVERY_DAYS) -> bool:
    """True when the log's turn has come round again.

    None means git has never been given the file, which is always due: the
    point of the script is that the track record reaches the remote at all.
    """
    if days_since_commit is None:
        return True
    return days_since_commit >= every_days


def days_since_last_commit(path: str, now: float | None = None) -> float | None:
    """Age in days of the newest commit touching a path, or None if there is none."""
    done = git("log", "-1", "--format=%ct", "--", path, check=False)
    stamp = done.stdout.decode("utf-8", "replace").strip()
    if done.returncode != 0 or not stamp:
        return None
    return ((time.time() if now is None else now) - int(stamp)) / 86400.0


def changed(paths: list[str]) -> list[str]:
    """Which of these paths differ from HEAD -- modified, or not yet tracked."""
    done = git("status", "--porcelain", "--", *paths, check=False)
    seen = []
    for line in done.stdout.decode("utf-8", "replace").splitlines():
        name = line[3:].strip().strip('"')
        if name in paths:
            seen.append(name)
    return seen


def mid_operation() -> str | None:
    """The git operation in progress, if any -- never commit into one."""
    gitdir = Path(git("rev-parse", "--git-dir").stdout.decode("utf-8").strip())
    if not gitdir.is_absolute():
        gitdir = ROOT / gitdir
    for marker, what in (("MERGE_HEAD", "merge"), ("rebase-merge", "rebase"),
                         ("rebase-apply", "rebase"), ("CHERRY_PICK_HEAD", "cherry-pick")):
        if (gitdir / marker).exists():
            return what
    return None


def drift_as_of() -> str:
    try:
        return json.loads((ROOT / DRIFT).read_text(encoding="utf-8")).get("as_of", "")
    except (OSError, ValueError):
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Commit the prediction log and the drift report.")
    ap.add_argument("--dry-run", action="store_true",
                    help="say what it would commit, and commit nothing")
    ap.add_argument("--log-every-days", type=float, default=LOG_EVERY_DAYS,
                    help=f"days between prediction-log commits (default {LOG_EVERY_DAYS:g})")
    args = ap.parse_args()

    busy = mid_operation()
    if busy:
        print(f"track record: skipped, a {busy} is in progress")
        return 0

    pending = changed([DRIFT, LOG])
    staging: list[str] = []
    parts: list[str] = []

    if DRIFT in pending:
        staging.append(DRIFT)
        parts.append(f"drift report {drift_as_of()}".strip())

    if LOG in pending:
        age = days_since_last_commit(LOG)
        if log_is_due(age, args.log_every_days):
            staging.append(LOG)
            parts.append("prediction log")
        else:
            print(f"track record: log committed {age:.1f}d ago, "
                  f"next one at {args.log_every_days:g}d")

    if not staging:
        print("track record: nothing to commit")
        return 0

    message = "Track record: " + ", ".join(p for p in parts if p)
    if args.dry_run:
        print(f"track record: would commit {' '.join(staging)}\n  {message}")
        return 0

    git("add", "--", *staging)
    # Pathspec on the commit as well as the add: the index may already hold a
    # parallel session's staged work, and this must never carry it along.
    done = git("commit", "-m", message, "--", *staging, check=False)
    if done.returncode != 0:
        print("track record: commit failed -- "
              + done.stderr.decode("utf-8", "replace").strip())
        return 1
    print(f"track record: {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
