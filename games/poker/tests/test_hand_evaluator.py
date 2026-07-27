from engine.cards import Card, Rank, Suit
from games.poker.hand_evaluator import HandCategory, best_hand, evaluate_five


def C(rank: Rank, suit: Suit) -> Card:
    return Card(rank, suit)


def test_pair_of_aces_beats_pair_of_kings() -> None:
    aa = [
        C(Rank.ACE, Suit.CLUBS), C(Rank.ACE, Suit.DIAMONDS),
        C(Rank.TWO, Suit.HEARTS), C(Rank.SEVEN, Suit.SPADES), C(Rank.NINE, Suit.CLUBS),
    ]
    kk = [
        C(Rank.KING, Suit.CLUBS), C(Rank.KING, Suit.DIAMONDS),
        C(Rank.TWO, Suit.CLUBS), C(Rank.SEVEN, Suit.HEARTS), C(Rank.NINE, Suit.DIAMONDS),
    ]
    assert evaluate_five(aa) > evaluate_five(kk)
    assert evaluate_five(aa)[0] == HandCategory.PAIR


def test_flush_beats_straight() -> None:
    flush = [
        C(Rank.TWO, Suit.SPADES), C(Rank.FIVE, Suit.SPADES),
        C(Rank.SEVEN, Suit.SPADES), C(Rank.NINE, Suit.SPADES), C(Rank.KING, Suit.SPADES),
    ]
    straight = [
        C(Rank.FOUR, Suit.CLUBS), C(Rank.FIVE, Suit.DIAMONDS),
        C(Rank.SIX, Suit.HEARTS), C(Rank.SEVEN, Suit.SPADES), C(Rank.EIGHT, Suit.CLUBS),
    ]
    assert evaluate_five(flush)[0] == HandCategory.FLUSH
    assert evaluate_five(straight)[0] == HandCategory.STRAIGHT
    assert evaluate_five(flush) > evaluate_five(straight)


def test_full_house_beats_flush() -> None:
    full_house = [
        C(Rank.THREE, Suit.CLUBS), C(Rank.THREE, Suit.DIAMONDS), C(Rank.THREE, Suit.HEARTS),
        C(Rank.NINE, Suit.SPADES), C(Rank.NINE, Suit.CLUBS),
    ]
    flush = [
        C(Rank.TWO, Suit.SPADES), C(Rank.FIVE, Suit.SPADES),
        C(Rank.SEVEN, Suit.SPADES), C(Rank.NINE, Suit.SPADES), C(Rank.KING, Suit.SPADES),
    ]
    assert evaluate_five(full_house)[0] == HandCategory.FULL_HOUSE
    assert evaluate_five(full_house) > evaluate_five(flush)


def test_wheel_straight_ace_plays_low() -> None:
    wheel = [
        C(Rank.ACE, Suit.CLUBS), C(Rank.TWO, Suit.DIAMONDS),
        C(Rank.THREE, Suit.HEARTS), C(Rank.FOUR, Suit.SPADES), C(Rank.FIVE, Suit.CLUBS),
    ]
    rank = evaluate_five(wheel)
    assert rank[0] == HandCategory.STRAIGHT
    assert rank[1] == Rank.FIVE.value  # ace counts low, five is the wheel's high card


def test_straight_flush_beats_four_of_a_kind() -> None:
    straight_flush = [
        C(Rank.FIVE, Suit.HEARTS), C(Rank.SIX, Suit.HEARTS),
        C(Rank.SEVEN, Suit.HEARTS), C(Rank.EIGHT, Suit.HEARTS), C(Rank.NINE, Suit.HEARTS),
    ]
    quads = [
        C(Rank.KING, Suit.CLUBS), C(Rank.KING, Suit.DIAMONDS),
        C(Rank.KING, Suit.HEARTS), C(Rank.KING, Suit.SPADES), C(Rank.TWO, Suit.CLUBS),
    ]
    assert evaluate_five(straight_flush)[0] == HandCategory.STRAIGHT_FLUSH
    assert evaluate_five(quads)[0] == HandCategory.FOUR_OF_A_KIND
    assert evaluate_five(straight_flush) > evaluate_five(quads)


def test_best_hand_picks_best_five_of_seven() -> None:
    seven_cards = [
        C(Rank.ACE, Suit.CLUBS), C(Rank.ACE, Suit.DIAMONDS),  # hole cards: pocket aces
        C(Rank.ACE, Suit.HEARTS), C(Rank.TWO, Suit.SPADES), C(Rank.SEVEN, Suit.CLUBS),
        C(Rank.NINE, Suit.DIAMONDS), C(Rank.KING, Suit.HEARTS),  # board
    ]
    assert best_hand(seven_cards)[0] == HandCategory.THREE_OF_A_KIND

    seven_with_flush = [
        C(Rank.TWO, Suit.SPADES), C(Rank.NINE, Suit.SPADES),  # hole cards
        C(Rank.FIVE, Suit.SPADES), C(Rank.SEVEN, Suit.SPADES),
        C(Rank.KING, Suit.SPADES), C(Rank.KING, Suit.CLUBS), C(Rank.TWO, Suit.HEARTS),  # board
    ]
    assert best_hand(seven_with_flush)[0] == HandCategory.FLUSH


def test_evaluate_five_requires_exactly_five_cards() -> None:
    try:
        evaluate_five([C(Rank.ACE, Suit.CLUBS)])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
