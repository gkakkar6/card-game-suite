from engine.cards import Suit
from games.court_piece.hand_evaluation import (
    best_trump_candidate,
    suit_honors,
    suit_lengths,
    suit_score,
)
from games.court_piece.tests.test_rules import hand


def test_suit_lengths_counts_every_suit_including_zero() -> None:
    lengths = suit_lengths(hand("2H", "3H", "4S"))
    assert lengths[Suit.HEARTS] == 2
    assert lengths[Suit.SPADES] == 1
    assert lengths[Suit.CLUBS] == 0
    assert lengths[Suit.DIAMONDS] == 0


def test_suit_honors_only_counts_within_the_named_suit() -> None:
    assert suit_honors(hand("AH", "KS"), Suit.HEARTS) == 4  # ace of hearts only
    assert suit_honors(hand("AH", "KS"), Suit.SPADES) == 3  # king of spades only


def test_a_long_low_suit_can_outscore_a_short_high_one() -> None:
    long_low = hand("2H", "3H", "4H", "5H")
    short_high = hand("AS", "2C")
    assert suit_score(long_low, Suit.HEARTS) > suit_score(short_high, Suit.SPADES)


def test_best_trump_candidate_picks_the_highest_scoring_suit() -> None:
    preview = hand("AH", "KH", "QH", "2C", "3D")
    assert best_trump_candidate(preview) == Suit.HEARTS


def test_best_trump_candidate_always_returns_a_suit_even_with_nothing_good() -> None:
    # The call is mandatory - there is no "pass," so this must still return something
    # even from a genuinely weak preview.
    preview = hand("2C", "3D", "4H", "5S", "6C")
    assert best_trump_candidate(preview) in list(Suit)
