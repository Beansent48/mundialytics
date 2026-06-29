from __future__ import annotations

from pathlib import Path

from mundialytics.providers.api_config import endpoint_spec, masked, provider_runtime


def test_provider_runtime_reads_external_config(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAPIDAPI_KEY", "abc123456789")
    cfg = tmp_path / "providers.yaml"
    cfg.write_text(
        """
providers:
  oddspapi:
    mode: rapidapi
    rapidapi:
      key_env: RAPIDAPI_KEY
      host: odds-api1.p.rapidapi.com
      base_url: https://odds-api1.p.rapidapi.com/en
    limits:
      monthly_budget: 250
      max_calls_per_run: 9
    storage:
      cache_dir: data/raw/oddspapi/cache
      ledger_path: data/raw/oddspapi/request_ledger.jsonl
    endpoints:
      fixtures: /fixtures
""",
        encoding="utf-8",
    )
    rt = provider_runtime("oddspapi", cfg, required=True)
    assert rt.api_key == "abc123456789"
    assert rt.host == "odds-api1.p.rapidapi.com"
    assert rt.monthly_budget == 250
    assert rt.max_calls_per_run == 9
    assert endpoint_spec(rt, "fixtures")["path"] == "/fixtures"


def test_masked_does_not_expose_full_key():
    assert masked("abcdef123456").startswith("abcd...")
    assert "ef123456" not in masked("abcdef123456")
