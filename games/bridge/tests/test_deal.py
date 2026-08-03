import random

import pytest

from engine.cards import Deck, standard_52
from games.bridge.deal import CARDS_PER_SEAT, FULL_DEAL, SEATS, deal_hands


def test_every_seat_gets_thirteen_cards() -> None:
    hands = deal_hands(rng=random.Random(1))
    assert len(hands) == SEATS
    assert all(len(hand) == CARDS_PER_SEAT for hand in hands)


def test_a_deal_uses_the_whole_deck_exactly_once() -> None:
    # Unlike poker, bridge deals every card, so a dealing bug that duplicated or
    # dropped one shows up here rather than hiding in an undealt remainder.
    hands = deal_hands(rng=random.Random(1))
    dealt = [card for hand in hands for card in hand]
    assert len(dealt) == FULL_DEAL
    assert set(dealt) == set(standard_52())


def test_a_deal_is_reproducible_from_a_seeded_rng() -> None:
    assert deal_hands(rng=random.Random(7)) == deal_hands(rng=random.Random(7))


def test_a_stacked_deck_deals_each_seat_the_next_block_of_cards() -> None:
    # The property scripted deals rely on: seat 0 takes the top thirteen cards, seat 3
    # the bottom thirteen, so a test can lay out exactly the hands it wants to play.
    cards = standard_52()
    hands = deal_hands(deck=Deck(cards))
    assert hands[0] == tuple(cards[:CARDS_PER_SEAT])
    assert hands[3] == tuple(cards[-CARDS_PER_SEAT:])


def test_dealing_from_a_short_deck_raises() -> None:
    with pytest.raises(ValueError):
        deal_hands(deck=Deck(standard_52()[:-1]))
