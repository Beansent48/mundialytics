"""The shape of the season refresh: step names, and what runs how often.

Sibling of test_update_season_steps.py, which covers how a step REPORTS a
failure. This one covers two decisions about the step list itself, and the
pipeline got both wrong once.

Step numbers must be unique. PR #5 sat open for six weeks carrying a ratings
chain numbered 7g-7l, and by the time anyone read it master had reused 7g for
squad values and 7h for referees. A duplicate is not cosmetic: last_steps.json
reports failures by name, so two steps sharing one would report each other's.

And that ratings chain must stay on the weekly --full run. It costs up to 100
minutes, and PR #5 had it inside the daily block, which would have paid that
every morning to refresh data that moves about once a season.

Run:  .venv/Scripts/python.exe -m pytest tests/test_update_season_shape.py -q
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "scripts" / "update_season.py").read_text(encoding="utf-8")

# The player layer is built from these, in this order, once a week.
CHAIN = ["download_fbref_kaggle", "build_player_defense_fbref",
         "build_player_attack_fbref", "audit_player_data",
         "build_unified_player_ratings", "build_squadlab_cards"]


def _labels() -> list[str]:
    return re.findall(r'run_step\(\s*"([^"]+)"', SRC)


def test_the_refresh_still_has_its_steps():
    # Guards the regex: if run_step is ever called differently, every other
    # test here would pass on an empty list.
    assert len(_labels()) > 20


def test_every_step_has_its_own_label():
    labels = _labels()
    repeated = sorted({lab for lab in labels if labels.count(lab) > 1})
    assert not repeated, f"etiquetas repetidas: {repeated}"


def test_two_steps_never_share_a_number():
    numbers = [lab.split("/", 1)[0] for lab in _labels()]
    repeated = sorted({n for n in numbers if numbers.count(n) > 1})
    assert not repeated, f"numeros de paso repetidos: {repeated}"


def _scripts_behind_full() -> set[str]:
    """Every string run inside an `if args.full:` block."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.If) and "args.full" in ast.unparse(node.test):
            found |= {n.value for n in ast.walk(node)
                      if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    return found


def test_the_ratings_chain_is_weekly_not_daily():
    behind_full = _scripts_behind_full()
    for script in CHAIN:
        assert any(script in s for s in behind_full), f"{script} no cuelga de --full"


def test_the_daily_steps_are_not_behind_full():
    # Without this, the test above would pass just as happily if
    # _scripts_behind_full() returned every string in the file.
    behind_full = _scripts_behind_full()
    for daily in ("fetch_espn_referees", "build_squad_values",
                  "build_local_clubelo", "log_upcoming_round"):
        assert not any(daily in s for s in behind_full), f"{daily} no es semanal"


def test_the_chain_still_points_at_scripts_that_exist():
    for script in CHAIN:
        assert (ROOT / "scripts" / f"{script}.py").is_file(), script


def test_the_summary_names_what_went_stale_unnoticed():
    # The squads refreshed daily while the cards built on them sat 34 days old,
    # and nothing in the run's summary said so.
    assert "current_squads.csv" in SRC and "squadlab_cards.csv" in SRC
