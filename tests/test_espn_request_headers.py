"""ESPN is asked without a browser User-Agent, and nothing may put one back.

On 2026-09-17 ESPN's edge started refusing browser-shaped User-Agents: a request
claiming to be Chrome while carrying Python's TLS fingerprint reads as a bot and
gets a hard 403. The repo sent "Mozilla/5.0", so the fixture calendar, the player
stats and the current squads all stopped arriving at once -- and the European
page went on serving a cached bracket that looked perfectly healthy, because the
fetch swallowed the error.

Measured that day, three tries each: no User-Agent and "python-urllib/3.14"
succeeded on every endpoint; "Mozilla/5.0", a full Chrome string and a plain
"Mundialytics/0.50" failed on every one. So the rule is not "send a polite UA",
it is "send none", and these tests pin it.
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import pytest

from mundialytics.providers.espn_fixtures import espn_json

ROOT = Path(__file__).resolve().parents[1]

# Every module that talks to ESPN. Kept explicit: a new one added without a
# thought for the header is exactly the regression this file exists to catch.
ESPN_CALLERS = [
    "src/mundialytics/providers/espn_fixtures.py",
    "scripts/build_current_squads.py",
    "scripts/fetch_espn_match_events.py",
]


class _FakeResponse:
    def __init__(self) -> None:
        self.read_calls = 0

    def read(self) -> bytes:
        self.read_calls += 1
        return b'{"events": []}'

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


def test_the_request_carries_no_browser_user_agent(monkeypatch) -> None:
    """urllib adds its own "Python-urllib/x.y"; anything else is ours to explain."""
    seen: list[urllib.request.Request] = []

    def fake_urlopen(req, timeout=None):
        seen.append(req)
        return _FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    espn_json("https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard")

    assert len(seen) == 1
    ua = seen[0].get_header("User-agent")
    assert ua is None, f"espn_json set a User-Agent explicitly: {ua!r}"


def test_no_espn_caller_hardcodes_a_user_agent() -> None:
    """The header must not creep back into one script at a time."""
    offenders = {}
    for rel in ESPN_CALLERS:
        text = (ROOT / rel).read_text(encoding="utf-8")
        # the word appears in prose explaining the rule; only code sets a header
        hits = re.findall(r'"User-Agent"\s*:\s*"[^"]+"', text)
        if hits:
            offenders[rel] = hits
    assert not offenders, f"ESPN callers setting a User-Agent: {offenders}"


def test_every_espn_caller_goes_through_the_shared_helper() -> None:
    """One place to change means the next block is one edit, not a hunt."""
    missing = [rel for rel in ESPN_CALLERS[1:]
               if "espn_json" not in (ROOT / rel).read_text(encoding="utf-8")]
    assert not missing, f"ESPN callers not using espn_json: {missing}"


def test_a_failing_call_still_raises(monkeypatch) -> None:
    """Callers decide how to degrade; the helper must not hide a dead host."""
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    with pytest.raises(urllib.error.HTTPError):
        espn_json("https://site.api.espn.com/x", tries=2)
