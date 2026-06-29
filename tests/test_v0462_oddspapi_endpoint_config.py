from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.data.adapters.oddspapi import OddsPapiClient


def test_oddspapi_client_uses_configured_endpoints():
    c = OddsPapiClient(mode="rapidapi", rapidapi_key="x", rapidapi_host="h", endpoints={"sports": "/custom/sports"})
    assert c._endpoint("sports", "/v4/sports") == "/custom/sports"
    assert c._endpoint("markets", "/v4/markets") == "/v4/markets"


def test_oddspapi_client_accepts_object_endpoint_spec():
    c = OddsPapiClient(mode="rapidapi", rapidapi_key="x", rapidapi_host="h", endpoints={"sports": {"path": "/object/sports"}})
    assert c._endpoint("sports", "/v4/sports") == "/object/sports"
