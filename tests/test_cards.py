import random

from engine.cards import Card, Deck, Rank, Suit, deal, shuffle, standard_52


def test_standard_52_has_52_unique_cards() -> None:
    deck = standard_52()
    assert len(deck) == 52
    assert len(set(deck)) == 52


def test_card_str() -> None:
    assert str(Card(Rank.ACE, Suit.SPADES)) == "AS"
    assert str(Card(Rank.TEN, Suit.HEARTS)) == "TH"


def test_deal_removes_cards_from_source() -> None:
    cards = standard_52()
    hand = deal(cards, 2)
    assert len(hand) == 2
    assert len(cards) == 50
    assert set(hand).isdisjoint(cards)


def test_deal_too_many_raises() -> None:
    cards = standard_52()[:3]
    try:
        deal(cards, 4)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_shuffle_is_deterministic_with_seeded_rng() -> None:
    a = standard_52()
    b = standard_52()
    shuffle(a, random.Random(42))
    shuffle(b, random.Random(42))
    assert a == b


def test_deck_deal_and_len() -> None:
    deck = Deck()
    assert len(deck) == 52
    hand = deck.deal(5)
    assert len(hand) == 5
    assert len(deck) == 47
