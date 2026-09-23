"""Calendar club name -> ClubElo name, for the European layer.

A wrong match is worse than no match: the club then plays the whole season
with somebody else's rating and nothing looks broken. "Iberia 1999" (Georgia)
was handed Ironi Tiberias (Israel) by substring matching, and Slavia Praha,
whom ClubElo never gave us, silently took the Champions League down to 35
teams and eight fewer fixtures.

Run:  .venv/Scripts/python.exe -m pytest tests/test_club_resolver.py -q
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mundialytics.ratings.clubelo_local import load_manual_seeds  # noqa: E402
from mundialytics.statistical_core.competition.european import make_resolver  # noqa: E402

ELO_NAMES = ["Ironi Tiberias", "Roma", "Celta", "OFI", "Inter", "Slavia Sofia",
             "Sturm Graz", "Atletico"]


def test_rated_name_inside_listed_name_still_resolves():
    r = make_resolver(ELO_NAMES)
    assert r("AS Roma") == "Roma"
    assert r("Celta Vigo") == "Celta"
    assert r("OFI CRETE") == "OFI"
    assert r("SK Sturm Graz") == "Sturm Graz"


def test_short_listed_name_never_lands_inside_a_different_club():
    r = make_resolver(ELO_NAMES)
    assert r("Iberia 1999") is None


def test_ambiguous_substring_resolves_to_nothing():
    r = make_resolver(["Slavia Sofia", "Slavia Praha"])
    assert r("Slavia") is None


def test_manual_seed_file_is_well_formed():
    seeds = load_manual_seeds(ROOT)
    assert {"Slavia Praha", "Torreense", "Iberia 1999"} <= set(seeds)
    assert all(900 < v < 2200 for v in seeds.values())
    df = pd.read_csv(ROOT / "data/curated/clubelo_manual_seeds.csv")
    # every value must say how it was chosen
    assert df[["rule", "peer"]].notna().all().all()


def test_short_name_resolves_to_a_longer_name_by_whole_words():
    # SquadLab maps the field's ClubElo names onto ESPN's longer squad names
    r = make_resolver(["Bayern Munich", "FC Porto", "AEK Athens", "Ironi Tiberias"])
    assert r("Bayern") == "Bayern Munich"
    assert r("Porto") == "FC Porto"
    assert r("AEK") == "AEK Athens"
    assert r("Iberia") is None
