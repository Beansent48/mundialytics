"""Who may call the API from a browser (api/main.py `cors_config`).

The development default allows any `localhost:*` origin, which is right for a
dev server that does not always get port 3000 and wrong for anything public.
docs/DEPLOYMENT_PLAN.md flagged it as a thing that must not ship, so the
container sets MUNDIALYTICS_CORS_ORIGINS to an empty value and the deployed
site is one origin (Caddy serves the app and hands /engine/* to the API), where
no cross-origin request is made at all.

The subtlety worth a test: **empty is not unset**. An empty variable means "no
cross-origin caller is allowed", an absent one means "development". Collapsing
the two is how the localhost regex would end up in production anyway.

Run:  .venv/Scripts/python.exe -m pytest tests/test_cors_origins.py -q
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from api.main import cors_config  # noqa: E402


def test_unset_is_development():
    origins, regex = cors_config(None)
    assert "http://localhost:3000" in origins
    assert regex is not None


def test_empty_allows_nobody_and_drops_the_localhost_regex():
    origins, regex = cors_config("")
    assert origins == []
    assert regex is None


def test_an_explicit_list_is_taken_literally():
    origins, regex = cors_config("https://mundialytics.example, https://www.mundialytics.example")
    assert origins == ["https://mundialytics.example", "https://www.mundialytics.example"]
    assert regex is None
