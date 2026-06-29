from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_ENV = "MUNDIALYTICS_API_CONFIG"
DEFAULT_LOCAL_CONFIGS = [
    PROJECT_ROOT / "config" / "mundialytics_api_config.local.yaml",
    PROJECT_ROOT / "config" / "api_providers.local.yaml",
    PROJECT_ROOT / "mundialytics_api_config.local.yaml",
]


class ProviderConfigError(RuntimeError):
    pass


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ProviderConfigError(f"Provider config must be a YAML object: {path}")
    return data


def discover_config_path(path: str | Path | None = None) -> Path | None:
    if path:
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p
    env_path = os.getenv(DEFAULT_CONFIG_ENV)
    if env_path:
        p = Path(env_path)
        return p if p.is_absolute() else PROJECT_ROOT / p
    for p in DEFAULT_LOCAL_CONFIGS:
        if p.exists():
            return p
    return None


def load_api_config(path: str | Path | None = None, *, required: bool = False) -> dict[str, Any]:
    cfg_path = discover_config_path(path)
    if cfg_path is None:
        if required:
            raise ProviderConfigError(
                "No provider config found. Set MUNDIALYTICS_API_CONFIG or create config/mundialytics_api_config.local.yaml."
            )
        return {}
    return _read_yaml(cfg_path)


def get_provider_config(provider_name: str, path: str | Path | None = None, *, required: bool = False) -> dict[str, Any]:
    cfg = load_api_config(path, required=required)
    providers = cfg.get("providers", {}) if isinstance(cfg, dict) else {}
    provider = providers.get(provider_name, {}) if isinstance(providers, dict) else {}
    if required and not provider:
        raise ProviderConfigError(f"Provider '{provider_name}' not found in API config.")
    return provider if isinstance(provider, dict) else {}


def _env_or_value(config: dict[str, Any], *, value_key: str = "key", env_key: str = "key_env", fallback_envs: list[str] | None = None) -> str | None:
    direct_value = config.get(value_key)
    if direct_value and "TU_" not in str(direct_value).upper() and "YOUR_" not in str(direct_value).upper():
        return str(direct_value)
    named_env = config.get(env_key)
    if named_env and os.getenv(str(named_env)):
        return os.getenv(str(named_env))
    for env_name in fallback_envs or []:
        if os.getenv(env_name):
            return os.getenv(env_name)
    return None


def masked(value: Any, *, keep: int = 4) -> str:
    if value is None or value == "":
        return ""
    s = str(value)
    if len(s) <= keep:
        return "*" * len(s)
    return s[:keep] + "..." + "*" * max(4, min(12, len(s) - keep))


@dataclass(frozen=True)
class RapidApiProviderRuntime:
    provider_name: str
    mode: str
    base_url: str | None
    host: str | None
    api_key: str | None
    key_env: str | None
    cache_dir: str | None
    ledger_path: str | None
    monthly_budget: int | None
    max_calls_per_run: int | None
    min_interval_sec: float
    endpoints: dict[str, Any]
    raw: dict[str, Any]


def provider_runtime(provider_name: str, path: str | Path | None = None, *, required: bool = False) -> RapidApiProviderRuntime:
    provider = get_provider_config(provider_name, path, required=required)
    mode = str(provider.get("mode") or "rapidapi").lower().strip()
    rapidapi = provider.get("rapidapi", {}) if isinstance(provider.get("rapidapi", {}), dict) else {}
    direct = provider.get("direct", {}) if isinstance(provider.get("direct", {}), dict) else {}
    source = rapidapi if mode == "rapidapi" else direct
    limits = provider.get("limits", {}) if isinstance(provider.get("limits", {}), dict) else {}
    storage = provider.get("storage", {}) if isinstance(provider.get("storage", {}), dict) else {}
    endpoints = provider.get("endpoints", {}) if isinstance(provider.get("endpoints", {}), dict) else {}
    fallback_envs = [
        f"RAPIDAPI_{provider_name.upper()}_KEY",
        f"{provider_name.upper()}_API_KEY",
        "RAPIDAPI_KEY",
    ]
    api_key = _env_or_value(source, fallback_envs=fallback_envs)
    return RapidApiProviderRuntime(
        provider_name=provider_name,
        mode=mode,
        base_url=source.get("base_url"),
        host=source.get("host"),
        api_key=api_key,
        key_env=source.get("key_env"),
        cache_dir=storage.get("cache_dir"),
        ledger_path=storage.get("ledger_path"),
        monthly_budget=int(limits["monthly_budget"]) if str(limits.get("monthly_budget", "")).strip().isdigit() else None,
        max_calls_per_run=int(limits["max_calls_per_run"]) if str(limits.get("max_calls_per_run", "")).strip().isdigit() else None,
        min_interval_sec=float(limits.get("min_interval_sec", 0.25) or 0.25),
        endpoints=endpoints,
        raw=provider,
    )


def endpoint_spec(runtime: RapidApiProviderRuntime, endpoint_key: str) -> dict[str, Any]:
    spec = runtime.endpoints.get(endpoint_key)
    if spec is None:
        raise ProviderConfigError(f"Endpoint key '{endpoint_key}' is not configured for provider '{runtime.provider_name}'.")
    if isinstance(spec, str):
        return {"path": spec, "params": {}}
    if not isinstance(spec, dict):
        raise ProviderConfigError(f"Endpoint key '{endpoint_key}' must be a string or object.")
    return spec


def render_template(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, str):
        try:
            return value.format(**variables)
        except KeyError:
            return value
    if isinstance(value, dict):
        return {k: render_template(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [render_template(v, variables) for v in value]
    return value
