"""
Which files does the serving path actually open?

Task A2 needs this before a single Dockerfile: the image must not carry the data
(the daily refresh would have to rebuild it), and the volume must not carry the
2.2 GB of research material the server never reads. "We think it reads these" is
not good enough for a bundle you then delete things from.

So this measures instead of guessing. An audit hook records every `open` under
the repository, the app is driven over ASGI in this process — lifespan included,
because that is where the models are fitted since v0.64.0 — and every GET
endpoint is called once.

Driving ASGI by hand rather than through `fastapi.testclient` keeps this free of
httpx, which the project does not install.

Run with the venv that has fastapi:  .venv/Scripts/python scripts/measure_serving_bundle.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/serving_bundle.json"

opened: dict[str, int] = {}


def _hook(event: str, args) -> None:
    if event != "open":
        return
    path = args[0]
    if not isinstance(path, (str, bytes, os.PathLike)):
        return
    try:
        rel = Path(os.fspath(path)).resolve().relative_to(ROOT)
    except Exception:
        return                      # outside the repo: site-packages, temp…
    key = str(rel).replace("\\", "/")
    if key.split("/")[0] in {"data", "models", "outputs", "web"}:
        opened[key] = opened.get(key, 0) + 1


sys.addaudithook(_hook)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

COLD = "--cold" in sys.argv
if COLD:
    # A warm run measures the wrong thing. The fits load from
    # data/processed/cache, so the files that BUILT them never get opened and
    # the bundle comes out missing exactly what a server with an empty volume
    # needs on its first boot. Point the cache at a temporary directory — the
    # module reads the constant at call time — and the fit runs for real.
    # The real cache is left untouched.
    import tempfile

    import api.engine as _engine

    _tmp = Path(tempfile.mkdtemp(prefix="bundle_cold_"))
    _engine.CACHE_DIR = _tmp
    OUT = ROOT / "docs/serving_bundle_cold.json"
    print(f"EN FRIO · cache desviada a {_tmp}", flush=True)

from api.main import app  # noqa: E402

# One call each. Parameterised routes get a real value from the catalogue rather
# than a guess, so the handler does the work instead of returning a 404.
STATIC = ["/health", "/catalogue", "/competitions", "/leagues",
          "/fixtures/upcoming", "/fixtures", "/track-record",
          "/fixtures/day", "/fixtures/calendar", "/squadlab/pool",
          "/squadlab/historic"]


async def call(method: str, path: str, query: str = "") -> tuple[int, bytes]:
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
             "http_version": "1.1", "method": method, "path": path,
             "raw_path": path.encode(), "query_string": query.encode(),
             "root_path": "", "scheme": "http",
             "headers": [(b"host", b"measure")],
             "client": ("127.0.0.1", 0), "server": ("measure", 80)}
    status, body = {}, []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        if msg["type"] == "http.response.start":
            status["code"] = msg["status"]
        elif msg["type"] == "http.response.body":
            body.append(msg.get("body", b""))

    await app(scope, receive, send)
    return status.get("code", 0), b"".join(body)


async def lifespan():
    """Start the app the way uvicorn does, so the fit happens here."""
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put({"type": "lifespan.startup"})
    seen: list[dict] = []

    async def receive():
        return await queue.get()

    async def send(msg):
        seen.append(msg)

    task = asyncio.create_task(
        app({"type": "lifespan", "asgi": {"version": "3.0"}}, receive, send))
    while not any(m["type"].startswith("lifespan.startup.") for m in seen):
        if task.done():
            break
        await asyncio.sleep(0.2)
    return task, queue


async def main() -> int:
    t0 = time.time()
    print("arrancando la app (el ajuste del modelo va en el lifespan)…",
          flush=True)
    task, queue = await lifespan()
    print(f"  listo en {time.time() - t0:.1f}s · "
          f"{len(opened)} ficheros abiertos\n", flush=True)

    paths = list(STATIC)
    # a real competition and a real fixture, read from the catalogue
    code, body = await call("GET", "/catalogue")
    try:
        cat = json.loads(body)
        comps = [c.get("slug") or c.get("id") for c in (cat if isinstance(cat, list)
                 else cat.get("competitions", []))]
        comps = [c for c in comps if c][:3]
        for c in comps:
            paths += [f"/league/{c}", f"/competition/{c}", f"/awards/{c}"]
    except Exception as exc:
        print(f"  (catalogo no parseable: {exc})", flush=True)

    for p in paths:
        before = len(opened)
        t = time.time()
        try:
            code, _ = await call("GET", p)
        except Exception as exc:
            print(f"  {p:<34} EXCEPCION {type(exc).__name__}", flush=True)
            continue
        print(f"  {p:<34} {code}  {time.time() - t:5.1f}s  "
              f"+{len(opened) - before} ficheros", flush=True)

    await queue.put({"type": "lifespan.shutdown"})
    try:
        await asyncio.wait_for(task, timeout=30)
    except Exception:
        pass

    rows = []
    for key, times in sorted(opened.items()):
        f = ROOT / key
        size = f.stat().st_size if f.exists() else 0
        rows.append({"path": key, "bytes": size, "opens": times})
    total = sum(r["bytes"] for r in rows)
    OUT.write_text(json.dumps(
        {"files": rows, "total_bytes": total, "count": len(rows)},
        indent=2), encoding="utf-8")

    print(f"\n=== PAQUETE DE SERVICIO ===")
    print(f"{len(rows)} ficheros · {total / 1e6:.0f} MB")
    print("\nlos 15 mayores:")
    for r in sorted(rows, key=lambda r: -r["bytes"])[:15]:
        print(f"  {r['bytes'] / 1e6:8.1f} MB  x{r['opens']:<3} {r['path']}")
    print(f"\nEscrito: {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
