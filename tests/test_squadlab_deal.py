"""The draft deals cards, instead of showing everyone the same five.

The draft used to be deterministic twice over: the pool sampled its primes and
icons with a constant MD5 seed, and the client then shuffled the top 18 of that
with a seed made from the slot's own name ("Forward_0" -> the same hand, for
everybody, for ever). The card tiers meant nothing, because nothing ever rolled
against them -- picking the top card in each slot handed you an eleven of icons.

DraftPool had done this properly all along and was never called. These tests
hold the two properties that have to survive together: a hand must not change
under the reader, and it must differ between drafts.
"""
from __future__ import annotations

import collections

import pytest

from api import squadlab as sl
from mundialytics.statistical_core.squadlab.cards import KIND_P, load_cards

pytestmark = pytest.mark.skipif(load_cards().empty, reason="no card catalogue")

POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]


def ids(deal: dict) -> list[str]:
    return [c["player"] for c in deal["candidates"]]


def test_a_hand_is_fixed_by_its_draft_slot_and_reroll() -> None:
    """Re-render, reload, second tab: the same five cards come back.

    This is what the client's own shuffle used to buy, and it is the reason the
    server can stay stateless -- the answer is reproducible from the request.
    """
    a = sl.deal_slot(4242, "Forward", 0, 0, [])
    b = sl.deal_slot(4242, "Forward", 0, 0, [])
    assert ids(a) == ids(b)


def test_two_drafts_are_dealt_differently() -> None:
    """The bug as reported: everybody saw the same cards in the same hole."""
    hands = {tuple(ids(sl.deal_slot(s, "Forward", 0, 0, []))) for s in range(12)}
    assert len(hands) > 1, "every draft dealt the same five forwards"


def test_a_reroll_deals_a_new_hand() -> None:
    first = ids(sl.deal_slot(7, "Midfielder", 0, 0, []))
    for roll in (1, 2, 3):
        assert ids(sl.deal_slot(7, "Midfielder", 0, roll, [])) != first


def test_two_slots_of_one_position_differ() -> None:
    """Defender #1 and Defender #4 must not be offered the same five."""
    a = ids(sl.deal_slot(11, "Defender", 0, 0, []))
    b = ids(sl.deal_slot(11, "Defender", 3, 0, []))
    assert a != b


def test_drafting_a_man_removes_all_of_his_cards() -> None:
    """Uniqueness is by PERSON. Taking Messi takes his icon, his prime and his
    current card at once, or an eleven can field three of him."""
    cards = load_cards()
    people = cards.groupby("player")["card_id"].apply(list)
    multi = [(who, cid) for who, cid in people.items() if len(cid) > 1]
    assert multi, "catalogue has nobody with two cards; test is meaningless"

    who, card_ids = multi[0]
    position = str(cards.loc[cards["player"] == who, "position"].iloc[0])
    for seed in range(40):
        got = sl.deal_slot(seed, position, 0, 0, [card_ids[0]])
        assert not (set(ids(got)) & set(card_ids)), f"{who} dealt after being drafted"


def test_a_full_hand_is_always_dealt() -> None:
    """An empty bucket falls down the ladder rather than dealing four cards:
    a short hand just looks like a bug to whoever is drafting."""
    for position in POSITIONS:
        for seed in range(25):
            assert len(sl.deal_slot(seed, position, 0, 0, [])["candidates"]) == 5


def test_every_card_dealt_plays_the_slot_it_was_dealt_for() -> None:
    for position in POSITIONS:
        for c in sl.deal_slot(99, position, 0, 0, [])["candidates"]:
            assert c["position"] == position


def test_the_rarities_are_the_ones_the_catalogue_declares() -> None:
    """Primes near 10% and icons near 2% -- that is what makes them worth
    chasing. The old path floated every special to the top instead."""
    kinds: collections.Counter = collections.Counter()
    for seed in range(150):
        for position in POSITIONS:
            for c in sl.deal_slot(seed, position, 0, 0, [])["candidates"]:
                kinds[c["kind"]] += 1

    n = sum(kinds.values())
    for kind, expected in KIND_P.items():
        got = kinds[kind] / n
        assert abs(got - expected) < 0.04, f"{kind}: {got:.3f} vs {expected}"


def test_the_reroll_cap_is_published_with_the_hand() -> None:
    """The client must not hardcode it; the server owns the number."""
    assert sl.deal_slot(1, "Forward", 0, 0, [])["maxRerolls"] == sl.MAX_REROLLS
    assert sl.MAX_REROLLS >= 1


def test_an_unknown_card_id_in_taken_is_ignored() -> None:
    """An old tab posting a stale id must not break the draft."""
    got = sl.deal_slot(5, "Forward", 0, 0, ["not-a-card", ""])
    assert len(got["candidates"]) == 5
