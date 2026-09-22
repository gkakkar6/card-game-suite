import random

from games.court_piece.deal import CARDS_PER_SEAT, FULL_DEAL, SEATS, deal_hands


def test_deals_thirteen_cards_to_each_of_four_seats() -> None:
    hands = deal_hands(rng=random.Random(1))
    assert len(hands) == SEATS
    assert all(len(hand) == CARDS_PER_SEAT for hand in hands)


def test_every_card_is_dealt_exactly_once() -> None:
    hands = deal_hands(rng=random.Random(2))
    all_cards = [card for hand in hands for card in hand]
    assert len(all_cards) == FULL_DEAL
    assert len(set(all_cards)) == FULL_DEAL


def test_deal_is_reproducible_with_the_same_seed() -> None:
    first = deal_hands(rng=random.Random(42))
    second = deal_hands(rng=random.Random(42))
    assert first == second
