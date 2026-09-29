"""Text xG from ESPN commentary (src/mundialytics/enrichment/text_xg.py).

Sofascore, the last live xG source, answers 403 since 2026-09-26. This model reads the
Opta-standard commentary line of every attempt; trained on 2024/25 and applied to 2025/26
it kept all but +0.0003 RPS of Understat's value (scripts/espn_xg/). These tests pin the
parser on real line shapes, the side resolution, and the per-match sums the foundation reads.

Run:  .venv/Scripts/python.exe -m pytest tests/test_text_xg.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.enrichment.text_xg import (  # noqa: E402
    PENALTY_XG, TextXG, parse_shot, per_match, shots_from_commentary, side_of)

LINES = [
    "Attempt saved. Evanilson (Bournemouth) left footed shot from the left side of the six yard box is saved "
    "in the bottom left corner by Alisson Becker (Liverpool). Assisted by Marcus Tavernier with a cross.",
    "Attempt missed. Ryan Christie (Bournemouth) header from a difficult angle on the left is too high. "
    "Assisted by Alex Scott with a cross following a corner.",
    "Attempt blocked. Cody Gakpo (Liverpool) right footed shot from outside the box is blocked. Assisted by Florian Wirtz.",
    "Goal! Liverpool 1, Bournemouth 0. Hugo Ekitiké (Liverpool) right footed shot from the centre of the box "
    "to the bottom right corner. Assisted by Alexis Mac Allister.",
    "Attempt missed. Dominik Szoboszlai (Liverpool) right footed shot from outside the box is close, but misses "
    "the top left corner from a direct free kick.",
    "Goal! Liverpool 2, Bournemouth 0. Mohamed Salah (Liverpool) converts the penalty with a left footed shot "
    "to the bottom right corner.",
    "Own Goal by Adam Smith, Bournemouth.  Liverpool 3, Bournemouth 0.",
    "Corner, Bournemouth. Conceded by Ibrahima Konaté.",
    "Substitution, Liverpool. Wataru Endo replaces Jeremie Frimpong.",
]


def test_parses_the_opta_line_shapes():
    s = [parse_shot(x) for x in LINES]
    assert s[0]["zone"] == "the left side of the six yard box" and s[0]["body"] == "foot"
    assert s[0]["outcome"] == "saved" and s[0]["assist"] == "cross" and s[0]["team"] == "Bournemouth"
    assert s[1]["zone"] == "a difficult angle" and s[1]["body"] == "header" and s[1]["situation"] == "corner"
    assert s[2]["outcome"] == "blocked" and s[2]["assist"] == "pass"
    assert s[3]["goal"] and s[3]["team"] == "Liverpool" and s[3]["zone"] == "the centre of the box"
    assert s[4]["situation"] == "free_kick"
    assert s[5]["penalty"] and s[5]["goal"]
    assert s[6] is None and s[7] is None and s[8] is None     # own goals and non-attempts


def test_sides_resolve_despite_different_spellings():
    assert side_of("Bournemouth", "AFC Bournemouth", "Liverpool") == "home"
    assert side_of("Brighton and Hove Albion", "Fulham", "Brighton & Hove Albion") == "away"
    assert side_of("Real Madrid", "Arsenal", "Chelsea") == "?"
    s = shots_from_commentary(LINES, "Liverpool", "AFC Bournemouth")
    assert len(s) == 6 and set(s["side"]) == {"home", "away"}


def test_model_orders_chances_and_fixes_penalties():
    rng = np.random.default_rng(0)
    rows = []
    for zone, p in [("very close range", 0.6), ("the centre of the box", 0.15), ("outside the box", 0.03)]:
        for _ in range(400):
            rows.append({"team": "A", "outcome": "x", "goal": rng.random() < p, "zone": zone, "body": "foot",
                         "situation": "open_play", "assist": "pass", "penalty": False})
    shots = pd.DataFrame(rows)
    m = TextXG().fit(shots)
    q = pd.DataFrame([{**rows[0], "zone": z} for z in ("very close range", "the centre of the box", "outside the box")]
                     + [{**rows[0], "penalty": True}])
    p = m.predict(q)
    assert p[0] > p[1] > p[2]
    assert p[3] == PENALTY_XG


def test_per_match_sums_follow_the_understat_convention():
    s = shots_from_commentary(LINES, "Liverpool", "AFC Bournemouth").assign(event_id="1")
    s["xg"] = [0.3, 0.1, 0.05, 0.4, 0.06, PENALTY_XG]
    t = per_match(s).loc["1"]
    assert t["home_xg"] == pytest.approx(0.05 + 0.4 + 0.06 + PENALTY_XG)
    assert t["home_npxg"] == pytest.approx(0.05 + 0.4 + 0.06)        # penalties only in xg
    assert t["home_xg_sp"] == pytest.approx(0.06)                    # the direct free kick
    assert t["away_xg_sp"] == pytest.approx(0.1)                     # the header from a corner
    assert t["home_n"] == 4 and t["away_n"] == 2 and t["home_g"] == 2
