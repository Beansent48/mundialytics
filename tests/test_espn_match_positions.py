"""Every match roster records where each player stood.

build_current_squads settles the one boundary ESPN's roster endpoint gets wrong
-- a goalkeeper filed as a defender -- by reading positions off the MATCH
rosters. From 2026-09-07 the fetcher stopped writing that column: every match
since carried no position, so the check ran on August alone, and it turned a
forward (Mamadou Diop) into a keeper on that thin evidence. On a clean checkout
the column was missing entirely and the squad build raised KeyError.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_current_squads as squads  # noqa: E402
import fetch_espn_match_events as fetch  # noqa: E402


def _summary(abbrev: str) -> dict:
    return {"rosters": [{
        "team": {"displayName": "Some Club"},
        "roster": [{
            "athlete": {"displayName": "A Keeper"},
            "position": {"abbreviation": abbrev},
            "starter": True,
            "stats": [{"name": "appearances", "value": 1.0}],
        }],
    }]}


def test_a_match_roster_row_carries_the_position() -> None:
    game = {"event_id": "1", "competition": "LaLiga", "date": "2026-09-20"}
    rows = fetch.roster_rows(_summary("G"), game, canon=lambda n: n.lower())
    assert rows and rows[0]["position"] == "G"


def test_a_missing_position_is_empty_not_absent() -> None:
    """ESPN omits it now and then; the column must still be there."""
    game = {"event_id": "1", "competition": "LaLiga", "date": "2026-09-20"}
    payload = _summary("G")
    del payload["rosters"][0]["roster"][0]["position"]
    rows = fetch.roster_rows(payload, game, canon=lambda n: n.lower())
    assert rows[0]["position"] == ""


def test_the_squad_build_survives_a_file_without_positions(tmp_path) -> None:
    """No position column means no evidence, not a crash in the squad build."""
    src = tmp_path / "espn_player_match_current.csv"
    pd.DataFrame({"player": ["A Keeper"], "team": ["x"]}).to_csv(src, index=False)
    assert squads.match_positions(src) == {}


def test_positions_are_read_when_they_are_there(tmp_path) -> None:
    src = tmp_path / "espn_player_match_current.csv"
    pd.DataFrame({"player": ["A Keeper"] * 3, "position": ["G", "G", "SUB"]}).to_csv(
        src, index=False)
    assert squads.match_positions(src) == {"A Keeper": "Goalkeeper"}
