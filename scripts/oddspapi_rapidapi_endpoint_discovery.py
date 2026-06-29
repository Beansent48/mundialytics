#!/usr/bin/env python3
"""Discover the real OddsPapi RapidAPI route prefix without using the project client.

Why this exists:
- Direct OddsPapi docs use https://api.oddspapi.io/v4/...
- RapidAPI providers sometimes expose a different path prefix.
- This script tests a small, explicit list of sports endpoints and writes the working prefix to the external config.

It uses a hard request budget and only tests the cheapest catalog endpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.providers.api_config import provider_runtime

CANDIDATE_SPORTS_PATHS = [
    "/sports",
    "/v4/sports",
    "/en/sports",
    "/api/sports",
    "/api/v4/sports",
    "/api/en/sports",
]

ENDPOINT_SUFFIXES = {
    "sports": "/sports",
    "bookmakers": "/bookmakers",
    "markets": "/markets",
    "tournaments": "/tournaments",
    "fixtures": "/fixtures",
    "fixture_odds": "/odds",
    "fixture_main_odds": "/odds-by-tournaments",
    "fixture_historical_odds": "/historical-odds",
    "fixture_clv": "/clv",
}


def _resolve(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def _masked(v: str | None) -> str:
    if not v:
        return ""
    return v[:4] + "..." + "*" * min(8, max(4, len(v) - 4))


def _prefix_from_sports_path(path: str) -> str:
    if not path.endswith("/sports"):
        return ""
    prefix = path[: -len("/sports")]
    return prefix.rstrip("/")


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _apply_config(config_path: Path, base_url: str, host: str, prefix: str) -> None:
    data = _load_yaml(config_path)
    p = data.setdefault("providers", {}).setdefault("oddspapi", {})
    rapidapi = p.setdefault("rapidapi", {})
    rapidapi["base_url"] = base_url.rstrip("/")
    rapidapi["host"] = host
    endpoints = p.setdefault("endpoints", {})
    for key, suffix in ENDPOINT_SUFFIXES.items():
        endpoints[key] = (prefix + suffix) if prefix else suffix
    _save_yaml(config_path, data)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Discover OddsPapi RapidAPI endpoint prefix and optionally update external config.")
    ap.add_argument("--config", default="config/mundialytics_api_config.local.yaml")
    ap.add_argument("--base-url", default=None, help="Override RapidAPI base URL from the RapidAPI code snippet.")
    ap.add_argument("--host", default=None, help="Override X-RapidAPI-Host from the RapidAPI code snippet.")
    ap.add_argument("--paths", default=",".join(CANDIDATE_SPORTS_PATHS), help="Comma-separated candidate sports paths to test.")
    ap.add_argument("--max-api-calls", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--out-dir", default="outputs/oddspapi_endpoint_discovery_current")
    ap.add_argument("--apply", action="store_true", help="Update config with the working prefix if a 200 response is found.")
    args = ap.parse_args(argv)

    config_path = _resolve(args.config)
    assert config_path is not None
    out_dir = _resolve(args.out_dir) or (ROOT / "outputs/oddspapi_endpoint_discovery_current")
    out_dir.mkdir(parents=True, exist_ok=True)

    rt = provider_runtime("oddspapi", config_path, required=False)
    base_url = (args.base_url or rt.base_url or "https://odds-api1.p.rapidapi.com").rstrip("/")
    host = args.host or rt.host or "odds-api1.p.rapidapi.com"
    api_key = os.getenv("RAPIDAPI_ODDSPAPI_KEY") or os.getenv("RAPIDAPI_KEY") or rt.api_key
    if not api_key:
        raise SystemExit("Missing RAPIDAPI_KEY / RAPIDAPI_ODDSPAPI_KEY.")

    headers = {
        "Accept": "application/json",
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": host,
        "User-Agent": "MundialyticsBettingEngine/0.46 endpoint-discovery",
    }
    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    results: list[dict[str, Any]] = []
    winner: dict[str, Any] | None = None

    print(f"RapidAPI host: {host}")
    print(f"Base URL: {base_url}")
    print(f"Key: {_masked(api_key)}")
    print(f"Testing up to {min(args.max_api_calls, len(paths))} candidate sports paths...")

    for idx, path in enumerate(paths[: args.max_api_calls], start=1):
        url = base_url + (path if path.startswith("/") else "/" + path)
        try:
            r = requests.get(url, params={"language": "en"}, headers=headers, timeout=args.timeout)
            preview = r.text[:300]
            item = {
                "idx": idx,
                "url": url,
                "path": path,
                "status_code": r.status_code,
                "content_type": r.headers.get("Content-Type"),
                "preview": preview,
            }
            if r.status_code == 200:
                try:
                    payload = r.json()
                    n = len(payload) if isinstance(payload, list) else len(payload.keys()) if isinstance(payload, dict) else None
                    item["json_shape"] = type(payload).__name__
                    item["json_len"] = n
                except Exception:
                    pass
                winner = item
                results.append(item)
                print(f"OK 200  {path}")
                break
            else:
                print(f"{r.status_code:<4}  {path}")
            results.append(item)
        except Exception as e:
            item = {"idx": idx, "url": url, "path": path, "error": repr(e)}
            results.append(item)
            print(f"ERR   {path}  {e!r}")

    summary = {
        "version": "v0.46.2_oddspapi_rapidapi_endpoint_discovery",
        "base_url": base_url,
        "host": host,
        "tested": results,
        "winner": winner,
        "applied": False,
    }
    if winner and args.apply:
        prefix = _prefix_from_sports_path(str(winner["path"]))
        _apply_config(config_path, base_url, host, prefix)
        summary["applied"] = True
        summary["applied_prefix"] = prefix
        summary["config_path"] = str(config_path)
        print(f"Applied config. Endpoint prefix: '{prefix or '/'}'")
    elif not winner:
        print("No candidate returned 200. Open RapidAPI Playground and copy the exact cURL path for a catalog endpoint.")

    (out_dir / "oddspapi_rapidapi_endpoint_discovery.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_dir / 'oddspapi_rapidapi_endpoint_discovery.json'}")
    return 0 if winner else 2


if __name__ == "__main__":
    raise SystemExit(main())
