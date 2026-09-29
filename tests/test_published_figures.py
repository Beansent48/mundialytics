"""Every published benchmark figure comes from one versioned artifact.

The site's RPS was a constant in api/main.py and another in web/lib/stats.ts;
both still said 0.2025 while the deployed engine, scored by the same script on
the same matches, was at 0.2008 (the README had it right). Now the script
writes web/data/benchmark_vs_bet365.json, the API and the landing page read
it, and this checks the README against it.

Run:  .venv/Scripts/python.exe -m pytest tests/test_published_figures.py -q
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "web/data/benchmark_vs_bet365.json"


def _bench() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_has_what_the_pages_read():
    b = _bench()
    assert {r["key"] for r in b["rows"]} == {"market", "engine", "baseRates", "uniform"}
    assert b["sample"] > 5000
    assert b["provenance"]["predictionsSha256"]
    rps = {r["key"]: r["rps"] for r in b["rows"]}
    assert rps["market"] < rps["engine"] < rps["baseRates"] < rps["uniform"]


def test_readme_quotes_the_artifact():
    b = _bench()
    engine = next(r["rps"] for r in b["rows"] if r["key"] == "engine")
    market = next(r["rps"] for r in b["rows"] if r["key"] == "market")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"**{engine:.4f}**" in readme, "README engine RPS differs from the artifact"
    assert f"| {market:.4f} |" in readme, "README market RPS differs from the artifact"
    assert f"{b['sample']:,}" in readme


def test_no_hand_typed_rps_left_in_the_serving_code():
    for rel in ("api/main.py", "web/lib/stats.ts"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert not re.search(r"rps\"?:\s*0\.\d{4}", src), f"{rel} hard-codes an RPS"
