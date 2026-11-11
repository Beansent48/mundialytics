"""What the serving path actually opens, measured instead of guessed.

`data/` is 2.7 GB and `outputs/` is 7.6 GB. The deployment plan says the image
carries code and the data is a volume, which only answers half the question: a
volume still has to be filled, and filling it with 2.7 GB to serve a site that
reads a fraction of it is the same waste one layer down.

So this script records every file the API opens while every endpoint is called,
by installing an audit hook on `open` **before** `api.main` is imported and then
driving a real uvicorn in a thread. The hook sees what the code does, not what
the code says it does: pandas, joblib and numpy all open through CPython's
`open`, and `sys.addaudithook` is fired by that one place.

Two outputs, both of which A2 needs:

- the **read set** -- the files that must exist for the site to serve, with
  sizes, which is the serving bundle;
- the **write set** -- because `data/` cannot be mounted read-only if the
  serving path writes to it, and the deployment plan assumed it could.

    python scripts/measure_serving_bundle.py
    python scripts/measure_serving_bundle.py --out outputs/serving_bundle

Measured 2026-11-02 on this machine; the numbers are in
docs/DEPLOYMENT_PLAN.md section 1. Re-run it after anything that changes what
the serving path reads -- a new market, a new page, a new data source.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Run from anywhere: `api` and `mundialytics` have to come from THIS checkout,
# not from wherever an editable install happens to point (in a worktree those
# are two different trees, with two different `data/`).
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

# ── The hook ──────────────────────────────────────────────────────────────────
# Installed at import time, before anything of the app is imported, because the
# module-level reads (config/default.yaml, the catalogue, the fixture snapshot)
# are part of the answer and happen on import.

READS: dict[str, int] = {}      # repo-relative path -> times opened for reading
WRITES: dict[str, int] = {}     # repo-relative path -> times opened for writing
WALKED: dict[str, int] = {}     # repo-relative directory -> times listed
_RECORDING = threading.Event()

# Why directory listings are recorded and not just opens, which cost an
# afternoon on 2026-11-03: the market level is memoised on disk under
# `data/processed/cache/`, and its key is built by *stat-ing* every file in
# `data/raw/football_data/` through a glob. A glob is `os.scandir`, not `open`.
# So a measurement taken with the memo warm reads the memo, never opens a single
# football-data CSV, and reports a bundle that cannot recompute the level -- in
# which case the engine fails soft and serves lambdas with no market shift at
# all. Silently different probabilities, from a bundle that looked complete.
_DIR_EVENTS = {"os.listdir", "os.scandir"}


_ROOT_PREFIX = str(ROOT).lower().replace("\\", "/")


def _rel(path) -> str | None:
    """A repo-relative posix path, or None for anything outside the repo."""
    try:
        raw = os.fsdecode(path).replace("\\", "/")
    except (TypeError, ValueError):
        return None
    if raw.lower().startswith(_ROOT_PREFIX):
        return raw[len(_ROOT_PREFIX):].lstrip("/") or None
    if os.path.isabs(raw):
        return None
    try:
        return Path(raw).resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return None


def _record_dir(path) -> None:
    if path is None:
        return
    # A scandir of a file's parent is normal plumbing; what matters is a
    # directory whose *contents* the serving path inspects.
    rel = _rel(path)
    if rel:
        WALKED[rel] = WALKED.get(rel, 0) + 1


def _record(event: str, args) -> None:
    if not _RECORDING.is_set():
        return
    if event in _DIR_EVENTS:
        _record_dir(args[0] if args else None)
        return
    if event != "open":
        return
    path, mode, _flags = args
    if path is None:
        return
    # The hook runs on every open in the process, imports of site-packages
    # included -- tens of thousands during a cold fit -- so the cheap string
    # test comes before the syscalls `Path.resolve` makes.
    try:
        raw = os.fsdecode(path).replace("\\", "/")
    except (TypeError, ValueError):
        return
    if raw.lower().startswith(_ROOT_PREFIX):
        rel = raw[len(_ROOT_PREFIX):].lstrip("/")
        if not rel:
            return
    elif os.path.isabs(raw):
        return                   # outside the repo: site-packages, the venv, /proc
    else:
        try:
            rel = Path(raw).resolve().relative_to(ROOT).as_posix()
        except (ValueError, OSError):
            return
    # `mode` is None for os.open and io.FileIO; those carry the intent in flags,
    # which the audit event does not break out, so treat an unknown mode as a
    # read and let the write set be the conservative one.
    writing = bool(mode) and any(c in str(mode) for c in "wxa+")
    table = WRITES if writing else READS
    table[rel] = table.get(rel, 0) + 1


sys.addaudithook(_record)


# ── The crawl ─────────────────────────────────────────────────────────────────

def _crawl(base: str, log) -> tuple[set[str], list[str]]:
    """Call every route the app declares. Returns (routes hit, failures)."""
    import requests

    from api.main import app

    session = requests.Session()
    hit: set[str] = set()
    failures: list[str] = []

    def call(method: str, path: str, route: str, **kw) -> dict | list | None:
        url = f"{base}{path}"
        try:
            res = session.request(method, url, timeout=600, **kw)
        except Exception as exc:                       # noqa: BLE001
            failures.append(f"{method} {path}: {exc}")
            return None
        log(f"  {res.status_code}  {method} {path}")
        if res.status_code >= 500:
            failures.append(f"{method} {path}: HTTP {res.status_code}")
            return None
        hit.add(route)
        try:
            return res.json()
        except ValueError:
            return None

    def slugs(payload) -> list[str]:
        """The slugs of a catalogue-shaped answer, list or {"items": [...]}."""
        rows = payload if isinstance(payload, list) else []
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list):
                    rows = value
                    break
        return [r.get("slug") for r in rows if isinstance(r, dict) and r.get("slug")]

    # The flat routes, in the order a visitor warms them.
    call("GET", "/health", "/health")
    catalogue = call("GET", "/catalogue", "/catalogue")
    call("GET", "/leagues", "/leagues")
    uefa = call("GET", "/competitions", "/competitions")
    call("GET", "/track-record", "/track-record")
    call("GET", "/fixtures/day", "/fixtures/day")
    call("GET", "/fixtures/calendar", "/fixtures/calendar")
    call("GET", "/fixtures/upcoming?days=21", "/fixtures/upcoming")

    # Every domestic competition the catalogue declares, because the per-league
    # pages are where the foundation slices and the award files get read. The
    # fixture list needs its competition: /fixtures has no default.
    fixtures: list[dict] = []
    for slug in slugs(catalogue):
        call("GET", f"/league/{slug}", "/league/{competition}")
        call("GET", f"/awards/{slug}", "/awards/{competition}")
        round_ = call("GET", f"/fixtures?competition={slug}", "/fixtures") or {}
        fixtures.extend(round_.get("fixtures", []) if isinstance(round_, dict) else [])

    # The UEFA competitions are a different model on a different slug space.
    for slug in slugs(uefa):
        call("GET", f"/competition/{slug}", "/competition/{slug}")

    # Match pages: a played one and an upcoming one read different things (the
    # logged prediction and the events feed against the live engine).
    played = [f for f in fixtures if f.get("played") and f.get("analysis", True)][-3:]
    upcoming = [f for f in fixtures if not f.get("played") and f.get("analysis", True)][:3]
    for fx in played + upcoming:
        call("GET", f"/match/{fx['competition']}/{fx['slug']}", "/match/{competition}/{fixture}")

    # SquadLab: the pool, one deal, the historic tournament, and a full season
    # with an eleven built from the pool (the heaviest call in the app).
    pool = call("GET", "/squadlab/pool", "/squadlab/pool") or {}
    players = pool.get("players", {})
    call("POST", "/squadlab/deal", "/squadlab/deal",
         json={"seed": 1, "position": "Forward", "index": 0, "reroll": 0, "taken": []})
    call("GET", "/squadlab/historic", "/squadlab/historic")
    # `player` is the CARD id, not the man's name (api/squadlab.py `_card_entry`),
    # and the eleven it posts back needs exactly one goalkeeper.
    slots = pool.get("slots") or {"Goalkeeper": 1, "Defender": 4, "Midfielder": 3, "Forward": 3}
    squad: list[str] = []
    for position, n in slots.items():
        ids = [p.get("player") for p in players.get(position, []) if p.get("player")]
        squad.extend(ids[:n])
    if len(squad) == 11:
        call("POST", "/squadlab/season", "/squadlab/season",
             json={"squad": squad, "bench": [], "seed": 7})
    else:
        failures.append(f"POST /squadlab/season: only {len(squad)} card ids in the pool")

    declared = {
        getattr(r, "path") for r in app.routes
        if getattr(r, "path", "").startswith(("/health", "/catalogue", "/fixtures",
                                              "/match", "/track-record", "/league",
                                              "/competition", "/awards", "/squadlab"))
    }
    for route in sorted(declared - hit):
        failures.append(f"route never exercised: {route}")
    return hit, failures


# ── Reporting ─────────────────────────────────────────────────────────────────

def _size(rel: str) -> int:
    try:
        return (ROOT / rel).stat().st_size
    except OSError:
        return 0


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765, help="port for the probe server")
    ap.add_argument("--out", type=Path, default=None,
                    help="directory to write the manifest and the file lists into")
    ap.add_argument("--quiet", action="store_true", help="only the summary")
    ap.add_argument("--cold", action="store_true",
                    help="measure with no fit caches, which is what a server has")
    args = ap.parse_args()

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    import uvicorn

    if args.cold:
        # A server's volume starts with no fits on it, and a fit loaded from
        # cache never opens the files that BUILT it: measured warm, this script
        # misses `understat_shots.csv` (121 MB) entirely, and the props model
        # that a bundle without it produces is a working model with the penalty
        # information silently absent. So point the cache somewhere empty --
        # never deleting the real one, which costs two minutes to rebuild -- and
        # let the fits actually run. ~10 minutes against ~1.
        #
        # The mechanism is a parallel session's (branch
        # `claude/settlement-audit`, which measured 439 files / 356 MB this way);
        # it belongs here because a measurement taken warm publishes an
        # incomplete volume.
        import tempfile

        from api import engine as _engine

        cold_dir = Path(tempfile.mkdtemp(prefix="mundialytics-cold-cache-"))
        _engine.CACHE_DIR = cold_dir
        log(f"cold: fit cache redirected to {cold_dir} (the real one is untouched)")

    _RECORDING.set()
    log("importing the app (module-level reads count too)...")
    from api.main import app

    config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    t0 = time.perf_counter()
    thread.start()

    # The lifespan loads the models before the socket is bound, which is the
    # whole reason the container healthcheck needs a start_period; here it just
    # means waiting.
    import requests
    base = f"http://127.0.0.1:{args.port}"
    deadline = time.time() + 900
    while time.time() < deadline:
        try:
            body = requests.get(f"{base}/health", timeout=5).json()
            if body.get("ready"):
                break
        except Exception:                              # noqa: BLE001
            pass
        time.sleep(1.0)
    else:
        print("the API never became ready", file=sys.stderr)
        return 1
    boot = time.perf_counter() - t0
    log(f"ready in {boot:.1f}s; crawling\n")

    hit, failures = _crawl(base, log)
    server.should_exit = True
    thread.join(timeout=30)
    _RECORDING.clear()

    reads = {k: v for k, v in READS.items() if (ROOT / k).is_file()}
    writes = {k: v for k, v in WRITES.items()}
    bundle = {k: _size(k) for k in reads}
    total = sum(bundle.values())

    print("\n" + "=" * 78)
    print(f"{len(hit)} routes exercised, {len(reads)} files read, {len(writes)} written")
    print(f"serving bundle: {_human(total)}")
    print("=" * 78)

    # Group on the first two path components, which is the granularity a
    # transfer or a mount is decided at.
    by_dir: dict[str, list[str]] = {}
    for rel in reads:
        parts = rel.split("/")
        key = "/".join(parts[:2]) if len(parts) > 1 else parts[0]
        by_dir.setdefault(key, []).append(rel)

    print(f"\n{'group':<34} {'files':>6} {'read':>10}   of the tree on disk")
    for key in sorted(by_dir, key=lambda k: -sum(bundle[r] for r in by_dir[k])):
        files = by_dir[key]
        read_bytes = sum(bundle[r] for r in files)
        tree = _tree_size(ROOT / key)
        share = f"{_human(tree)}" if tree else "-"
        print(f"{key:<34} {len(files):>6} {_human(read_bytes):>10}   {share}")

    print("\nthe ten largest files the serving path reads:")
    for rel, size in sorted(bundle.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {_human(size):>9}  {rel}  (x{reads[rel]})")

    if writes:
        print("\nwritten while serving -- so the mount cannot be read-only:")
        for rel, n in sorted(writes.items()):
            print(f"  x{n:<4} {rel}")

    # Directories whose contents are inspected without every file being opened.
    # A disk memo keyed on those contents hides its own inputs from the read set
    # (see the note on _DIR_EVENTS), so this list is where to look for a bundle
    # that serves today and stops serving the moment a memo expires.
    walked = {d: n for d, n in WALKED.items()
              if d.startswith("data/") and not set(by_dir) & {d}}
    if walked:
        print("\nlisted (not every file opened) -- check these before trusting the bundle:")
        for rel, n in sorted(walked.items(), key=lambda kv: -_tree_size(ROOT / kv[0])):
            tree = _tree_size(ROOT / rel)
            opened = sum(1 for r in reads if r.startswith(rel + "/"))
            total = sum(1 for _ in (ROOT / rel).glob("*")) if (ROOT / rel).is_dir() else 0
            print(f"  x{n:<3} {rel:<44} {opened}/{total} opened, {_human(tree)} on disk")

    if failures:
        print("\nFAILURES (the read set is incomplete without these):")
        for f in failures:
            print(f"  {f}")

    if args.out:
        out = args.out if args.out.is_absolute() else ROOT / args.out
        out.mkdir(parents=True, exist_ok=True)
        (out / "reads.txt").write_text(
            "\n".join(sorted(reads)) + "\n", encoding="utf-8")
        (out / "writes.txt").write_text(
            "\n".join(sorted(writes)) + "\n", encoding="utf-8")
        (out / "manifest.json").write_text(json.dumps({
            "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "boot_seconds": round(boot, 1),
            "routes": sorted(hit),
            "failures": failures,
            "total_bytes": total,
            "files": {k: bundle[k] for k in sorted(bundle)},
            "writes": dict(sorted(writes.items())),
            "walked": dict(sorted(WALKED.items())),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}/reads.txt, writes.txt, manifest.json")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
