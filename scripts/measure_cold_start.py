"""
Cold-start stopwatch for the HTTP API (and the Python load path behind it).

WHY A SCRIPT. The cold-start work (docs/COLD_START_DIAGNOSIS.md) is a
before/after table, and a table is only evidence if both halves were measured
the same way. Timing by hand with a browser open mixes the process boot, the
model load, a live ESPN fetch and Next.js compilation into one number nobody
can take apart afterwards.

What it measures, per endpoint:

    boot    process spawn -> the port accepts a connection
    first   the first request on that fresh process
    cold    boot + first: what a visitor waits for, end to end
    warm    the same request again, same process

`boot` is the interesting column once a lifespan warm-up exists: uvicorn runs
the lifespan *before* it binds the socket, so work moved into startup leaves
`first` and lands in `boot` -- the total barely moves, but it is now paid by
whoever started the server, not by whoever opened the page.

Modes:
    python scripts/measure_cold_start.py imports   # the load path, no server
    python scripts/measure_cold_start.py cold      # one fresh process per endpoint
    python scripts/measure_cold_start.py warm      # one process, every endpoint twice
    python scripts/measure_cold_start.py fanout    # _predict calls behind one round
    python scripts/measure_cold_start.py web       # every route, next dev
    python scripts/measure_cold_start.py web --prod   # ...and next build && next start

`--json out.json` writes the raw numbers so two runs can be diffed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8010
BASE = f"http://127.0.0.1:{PORT}"
WEB_PORT = 3100
PYTHON = ROOT / ".venv/Scripts/python.exe"
if not PYTHON.exists():                     # a POSIX venv, or none at all
    PYTHON = Path(sys.executable)


# ── the endpoints, in the order a visitor meets them ──────────────────────────
def endpoints() -> list[tuple[str, str]]:
    today = date.today().isoformat()
    return [
        ("health", "/health"),
        ("catalogue", "/catalogue"),
        ("fixtures/day", f"/fixtures/day?day={today}"),
        ("fixtures/calendar", "/fixtures/calendar?days_back=14&days_forward=21"),
        # the matchday screen's favourites strip, fetched from the browser
        ("fixtures/upcoming", "/fixtures/upcoming?days=21"),
        ("fixtures (round)", "/fixtures?competition=laliga"),
        ("match", None),                    # resolved from the round, see below
        ("league", "/league/laliga"),
        ("awards", "/awards/laliga"),
        ("track-record", "/track-record"),
        ("leagues", "/leagues"),
        ("competitions", "/competitions"),
        ("squadlab/pool", "/squadlab/pool"),
    ]


def _get(path: str, timeout: float = 600.0, base: str = BASE) -> tuple[float, int, bytes]:
    """One GET. Returns (seconds, status, body); a 4xx/5xx is a result, not an error."""
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(f"{base}{path}", timeout=timeout) as fh:
            body = fh.read()
            return time.perf_counter() - t0, fh.status, body
    except urllib.error.HTTPError as exc:
        return time.perf_counter() - t0, exc.code, exc.read()
    except Exception as exc:                # a timeout or a dropped connection
        return time.perf_counter() - t0, 0, str(exc).encode()


class Proc:
    """A server process, timed from spawn to the moment its port accepts."""

    def __init__(self, cmd: list[str], port: int, cwd: Path = ROOT,
                 log: Path | None = None, env: dict | None = None,
                 ready_limit: float = 900.0):
        self.cmd, self.port, self.cwd, self.log = cmd, port, cwd, log
        self.env, self.ready_limit = env, ready_limit
        self.proc: subprocess.Popen | None = None
        self.boot = 0.0

    def __enter__(self) -> "Proc":
        self._out = open(self.log, "w", encoding="utf-8") if self.log else subprocess.DEVNULL
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            self.cmd, cwd=self.cwd, stdout=self._out, stderr=subprocess.STDOUT,
            env={**os.environ, **(self.env or {})},
        )
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.cmd[0]} exited with {self.proc.returncode}; "
                                   f"see {self.log}")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.02)
            if time.perf_counter() - t0 > self.ready_limit:
                raise RuntimeError(f"{self.cmd[0]} never bound port {self.port}")
        self.boot = time.perf_counter() - t0
        return self

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            # npm spawns node as a child, so terminating the launcher leaves the
            # server holding the port; the whole tree has to go.
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                               capture_output=True)
            else:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._out is not subprocess.DEVNULL:
            self._out.close()
        _wait_port_free(self.port)


def Server(log: Path | None = None) -> Proc:
    """The API, on PORT."""
    return Proc([str(PYTHON), "-m", "uvicorn", "api.main:app", "--port", str(PORT),
                 "--log-level", "warning"], PORT, log=log)


def _wait_port_free(port: int = PORT, limit: float = 30.0) -> None:
    """Windows keeps the listener briefly after a kill; the next run needs it."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < limit:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                time.sleep(0.2)
        except OSError:
            return


def _match_path() -> str | None:
    """A real upcoming fixture's URL, read off the round endpoint."""
    _, status, body = _get("/fixtures?competition=laliga")
    if status != 200:
        return None
    data = json.loads(body)
    for fx in data.get("fixtures", []):
        if not fx.get("played"):
            return f"/match/laliga/{fx['slug']}"
    fxs = data.get("fixtures", [])
    return f"/match/laliga/{fxs[0]['slug']}" if fxs else None


# ── modes ─────────────────────────────────────────────────────────────────────
def mode_imports() -> list[dict]:
    """The Python load path alone: imports, the frame, the two joblib files."""
    code = r"""
import sys, time
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
out, t0 = [], time.perf_counter()
def mark(name):
    global t0
    now = time.perf_counter(); out.append((name, now - t0)); t0 = now
import pandas; mark("import pandas")
import fastapi, uvicorn; mark("import fastapi+uvicorn")
from mundialytics.statistical_core import prediction_engine; mark("import prediction_engine")
from api.engine import club_engine, props_models; mark("import api.engine")
club_engine(); mark("club_engine() [joblib]")
props_models(); mark("props_models() [joblib]")
import json; print("@@" + json.dumps(out))
"""
    res = subprocess.run([str(PYTHON), "-c", code], cwd=ROOT, capture_output=True, text=True)
    line = next((l for l in res.stdout.splitlines() if l.startswith("@@")), None)
    if line is None:
        raise RuntimeError(res.stdout + res.stderr)
    rows = [{"step": n, "seconds": round(s, 3)} for n, s in json.loads(line[2:])]
    rows.append({"step": "TOTAL", "seconds": round(sum(r["seconds"] for r in rows), 3)})
    return rows


def mode_cold(logdir: Path, only: str | None = None) -> list[dict]:
    """One fresh uvicorn per endpoint: the number a first visitor actually waits."""
    rows, match_path = [], None
    for name, path in endpoints():
        if only and only not in name:
            continue
        if path is None:
            path = match_path
            if path is None:
                continue
        with Server(log=logdir / f"cold_{name.replace('/', '_')}.log") as srv:
            first, status, body = _get(path)
            warm, _, _ = _get(path)
            if name == "fixtures (round)" and match_path is None:
                match_path = _match_path()
        rows.append({
            "endpoint": name, "path": path, "status": status,
            "boot": round(srv.boot, 2), "first": round(first, 2),
            "cold": round(srv.boot + first, 2), "warm_same_proc": round(warm, 2),
            "bytes": len(body),
        })
        print(f"  {name:<20} boot {srv.boot:6.2f}s  first {first:7.2f}s  "
              f"cold {srv.boot + first:7.2f}s  again {warm:6.3f}s  [{status}]")
    return rows


def mode_warm(logdir: Path) -> list[dict]:
    """One process, every endpoint twice: the per-endpoint work, model load aside."""
    rows = []
    with Server(log=logdir / "warm.log") as srv:
        _get("/health")                      # pay the model load once, up front
        match_path = _match_path()
        print(f"  (boot {srv.boot:.2f}s, warmed)")
        for name, path in endpoints():
            if path is None:
                path = match_path
                if path is None:
                    continue
            first, status, body = _get(path)
            again = [_get(path)[0] for _ in range(3)]
            rows.append({
                "endpoint": name, "path": path, "status": status,
                "warm_first": round(first, 3),
                "warm_cached": round(statistics.median(again), 3),
                "bytes": len(body),
            })
            print(f"  {name:<20} first {first:7.2f}s  cached {statistics.median(again):6.3f}s  "
                  f"[{status}] {len(body) / 1024:.0f} KB")
    return rows


def mode_fanout() -> list[dict]:
    """How much of a round's render is `_predict`, called once per fixture."""
    code = r"""
import sys, time, json
from pathlib import Path
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT / "src"))
import api.main as m
from api import catalogue as cat
m.club_engine()                       # the load is not what we are timing
_, _, df = m.club_engine()
meta = cat.COMPETITIONS["laliga"]; season = cat.current_season(df, meta["id"])
cal = cat.trusted_calendar(meta["id"], season, df)
import pandas as pd
today = pd.Timestamp(__import__("datetime").date.today())
# the NEXT round that is still wholly unplayed: a round half-settled prices only
# its remaining fixtures, which would understate the fan-out it is here to size
pending = cal.loc[cal["date"] >= today]
md = int(pending.groupby("matchday").size().idxmax()) if len(pending) else int(cal["matchday"].max())
rows = cal[cal["matchday"] == md]
calls = []
real = m._predict
def spy(h, a, c):
    t = time.perf_counter(); out = real(h, a, c); calls.append(time.perf_counter() - t); return out
m._predict = spy
t0 = time.perf_counter()
payloads = [m._fixture_payload(r, "laliga", season, with_probs=True) for r in rows.itertuples()]
total = time.perf_counter() - t0
priced = sum(1 for p in payloads if p["probabilities"])
print("@@" + json.dumps({"matchday": md, "fixtures": len(payloads), "priced": priced,
                         "predict_calls": len(calls), "predict_total": sum(calls),
                         "predict_first": calls[0] if calls else 0,
                         "predict_rest_median": (sorted(calls[1:])[len(calls[1:]) // 2] if len(calls) > 1 else 0),
                         "payload_total": total}))
"""
    res = subprocess.run([str(PYTHON), "-c", code], cwd=ROOT, capture_output=True, text=True)
    line = next((l for l in res.stdout.splitlines() if l.startswith("@@")), None)
    if line is None:
        raise RuntimeError(res.stdout + res.stderr)
    d = json.loads(line[2:])
    print(f"  matchday {d['matchday']}: {d['fixtures']} fixtures, {d['priced']} priced, "
          f"{d['predict_calls']} _predict calls")
    print(f"  first _predict {d['predict_first'] * 1000:.0f} ms "
          f"(the ESPN calendar fetch), the rest {d['predict_rest_median'] * 1000:.1f} ms median")
    print(f"  all of them {d['predict_total']:.2f}s of a {d['payload_total']:.2f}s render")
    return [d]


WEB_ROUTES = [
    ("/", "landing"),
    ("/matchday", "matchday"),
    ("/leagues", "leagues"),
    ("/leagues/laliga", "league detail"),
    ("/competitions", "competitions"),
    ("/results", "results"),
    ("/awards", "awards"),
    ("/squadlab", "squadlab"),
    ("/es", "landing (es)"),
]


def _npm() -> str:
    exe = shutil.which("npm") or shutil.which("npm.cmd")
    if not exe:
        raise RuntimeError("npm is not on PATH")
    return exe


def mode_web(logdir: Path, prod: bool) -> list[dict]:
    """Every route, first visit and second, against a running API.

    The API is started first and warmed: a web number that includes the model
    load measures the API, which the other modes already do properly. What is
    left is Next.js — compilation in dev, a prerendered answer in prod.
    """
    web = ROOT / "web"
    rows: list[dict] = []
    build_s = None
    # NEXT_PUBLIC_* is read at runtime in dev and INLINED at build time in prod, so
    # the measurement API's port has to be in the environment of both commands --
    # without it every page renders its "the model is warming up" box in 1 s and
    # the table measures nothing.
    env = {"NEXT_TELEMETRY_DISABLED": "1", "NEXT_PUBLIC_API_URL": BASE}
    cmd = ([_npm(), "run", "start", "--", "--port", str(WEB_PORT)] if prod
           else [_npm(), "run", "dev", "--", "--port", str(WEB_PORT)])
    base = f"http://127.0.0.1:{WEB_PORT}"
    with Server(log=logdir / "web_api.log") as api_srv:
        _get("/health")                      # the API is warm; we are timing Next
        print(f"  (api boot {api_srv.boot:.2f}s + warm)")
        if prod:
            # The build has to run with the API UP. Every data page is SSG
            # (generateStaticParams over the locales), so `next build`
            # prerenders it against whatever the API answers at build time --
            # and with the API down it bakes the "the model is warming up" box
            # into the static HTML, where it stays until the ISR window lapses.
            t0 = time.perf_counter()
            # next build prints box-drawing characters; the console codepage
            # cannot decode them and a strict decoder loses the whole log
            res = subprocess.run([_npm(), "run", "build"], cwd=web, capture_output=True,
                                 text=True, encoding="utf-8", errors="replace",
                                 env={**os.environ, **env})
            build_s = time.perf_counter() - t0
            (logdir / "web_build.log").write_text((res.stdout or "") + (res.stderr or ""),
                                                  encoding="utf-8")
            print(f"  next build: {build_s:.1f}s (exit {res.returncode})")
            if res.returncode != 0:
                raise RuntimeError(f"next build failed; see {logdir / 'web_build.log'}")
        with Proc(cmd, WEB_PORT, cwd=web, log=logdir / f"web_{'prod' if prod else 'dev'}.log",
                  env=env, ready_limit=300) as web_srv:
            print(f"  (web boot {web_srv.boot:.2f}s)")
            for path, name in WEB_ROUTES:
                first, status, body = _get(path, base=base)
                again = [_get(path, base=base)[0] for _ in range(3)]
                rows.append({
                    "route": path, "name": name, "status": status,
                    "first": round(first, 2),
                    "second": round(statistics.median(again), 3),
                    "bytes": len(body),
                })
                print(f"  {path:<20} first {first:7.2f}s  second {statistics.median(again):6.3f}s  "
                      f"[{status}] {len(body) / 1024:.0f} KB")
            rows.append({"route": "(server boot)", "name": "npm start" if prod else "npm dev",
                         "first": round(web_srv.boot, 2), "second": None, "status": 0,
                         "bytes": 0})
    if build_s is not None:
        rows.append({"route": "(next build)", "name": "build", "first": round(build_s, 1),
                     "second": None, "status": 0, "bytes": 0})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["imports", "cold", "warm", "fanout", "web", "all"])
    ap.add_argument("--json", type=Path, help="write the raw numbers here")
    ap.add_argument("--only", help="cold mode: just the endpoints whose name contains this")
    ap.add_argument("--prod", action="store_true",
                    help="web mode: next build && next start instead of next dev")
    args = ap.parse_args()

    logdir = ROOT / "data/processed/logs/cold_start"
    logdir.mkdir(parents=True, exist_ok=True)
    modes = ["imports", "fanout", "warm", "cold"] if args.mode == "all" else [args.mode]
    out: dict[str, list[dict]] = {}
    for mode in modes:
        print(f"\n== {mode} ==")
        if mode == "imports":
            out[mode] = mode_imports()
            for r in out[mode]:
                print(f"  {r['step']:<28} {r['seconds']:6.2f}s")
        elif mode == "cold":
            out[mode] = mode_cold(logdir, args.only)
        elif mode == "warm":
            out[mode] = mode_warm(logdir)
        elif mode == "web":
            out[mode] = mode_web(logdir, args.prod)
        else:
            out[mode] = mode_fanout()
    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
