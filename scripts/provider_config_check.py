#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.providers.api_config import discover_config_path, load_api_config, masked, provider_runtime


def _resolve(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    p = Path(path_text)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate external API provider config without exposing keys.")
    parser.add_argument("--config", default=None, help="External local config path. Defaults to MUNDIALYTICS_API_CONFIG or config/mundialytics_api_config.local.yaml")
    parser.add_argument("--out-dir", default="outputs/provider_config_check_current")
    args = parser.parse_args(argv)

    cfg_path = discover_config_path(_resolve(args.config))
    if cfg_path is None:
        raise SystemExit("No config found. Put mundialytics_api_config.local.yaml in project root/config or set MUNDIALYTICS_API_CONFIG.")
    cfg = load_api_config(cfg_path, required=True)
    providers = cfg.get("providers", {}) if isinstance(cfg, dict) else {}
    rows = []
    for name in sorted(providers):
        rt = provider_runtime(name, cfg_path, required=False)
        rows.append({
            "provider": name,
            "enabled": providers[name].get("enabled", True) if isinstance(providers.get(name), dict) else True,
            "mode": rt.mode,
            "base_url": rt.base_url,
            "host": rt.host,
            "key_env": rt.key_env,
            "key_loaded_masked": masked(rt.api_key),
            "monthly_budget": rt.monthly_budget,
            "max_calls_per_run": rt.max_calls_per_run,
            "cache_dir": rt.cache_dir,
            "ledger_path": rt.ledger_path,
            "endpoint_keys": ",".join(sorted(rt.endpoints.keys())) if rt.endpoints else "",
        })
    out_dir = _resolve(args.out_dir) or (ROOT / "outputs/provider_config_check_current")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "provider_config_summary.json").write_text(json.dumps({"config_path": str(cfg_path), "providers": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"config_path": str(cfg_path), "providers": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
