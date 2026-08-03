import pytest

from engine.cards import Card, Rank, Suit
from engine.trick_taking.resolution import legal_plays, trick_winner


def C(rank: Rank, suit: Suit) -> Card:
    return Card(rank, suit)


def test_highest_card_of_the_led_suit_wins_when_nobody_trumps() -> None:
    # Hearts led and everyone follows, so the trump suit never comes into it: the king
    # is simply the highest heart played.
    trick = [
        (0, C(Rank.FOUR, Suit.HEARTS)),
        (1, C(Rank.KING, Suit.HEARTS)),
        (2, C(Rank.TWO, Suit.HEARTS)),
        (3, C(Rank.QUEEN, Suit.HEARTS)),
    ]
    assert trick_winner(trick, Suit.HEARTS, Suit.SPADES) == 1


def test_lowest_trump_beats_the_highest_card_of_the_led_suit() -> None:
    # The whole point of a trump suit, and the case a "highest card wins" shortcut gets
    # wrong: seat 2 is void in hearts and ruffs with the two of spades, the lowest card
    # in the trick, which still beats the ace of the suit actually led.
    trick = [
        (0, C(Rank.ACE, Suit.HEARTS)),
        (1, C(Rank.KING, Suit.HEARTS)),
        (2, C(Rank.TWO, Suit.SPADES)),
        (3, C(Rank.QUEEN, Suit.HEARTS)),
    ]
    assert trick_winner(trick, Suit.HEARTS, Suit.SPADES) == 2


def test_highest_trump_wins_when_more_than_one_player_ruffs() -> None:
    # Two players are void in hearts and both trump; the higher trump takes it, not the
    # one played later.
    trick = [
        (0, C(Rank.ACE, Suit.HEARTS)),
        (1, C(Rank.NINE, Suit.SPADES)),
        (2, C(Rank.FIVE, Suit.SPADES)),
        (3, C(Rank.THREE, Suit.HEARTS)),
    ]
    assert trick_winner(trick, Suit.HEARTS, Suit.SPADES) == 1


def test_a_trump_suit_nobody_played_changes_nothing() -> None:
    # Spades are trump but no spade was played, so this has to resolve exactly as a
    # trumpless trick would - and the off-suit ace of diamonds cannot win, high as it
    # is, because a discard does not contest the trick.
    trick = [
        (0, C(Rank.NINE, Suit.HEARTS)),
        (1, C(Rank.ACE, Suit.DIAMONDS)),
        (2, C(Rank.TEN, Suit.HEARTS)),
        (3, C(Rank.THREE, Suit.CLUBS)),
    ]
    assert trick_winner(trick, Suit.HEARTS, Suit.SPADES) == 2
    assert trick_winner(trick, Suit.HEARTS, None) == 2  # same answer with no trump at all


def test_trump_led_is_won_by_the_highest_trump() -> None:
    # When trump is itself the suit led, "highest trump" and "highest of the led suit"
    # pick the same cards: the jack beats the ten and the three, and the heart discard
    # is not in the running.
    trick = [
        (0, C(Rank.TEN, Suit.SPADES)),
        (1, C(Rank.JACK, Suit.SPADES)),
        (2, C(Rank.ACE, Suit.HEARTS)),
        (3, C(Rank.THREE, Suit.SPADES)),
    ]
    assert trick_winner(trick, Suit.SPADES, Suit.SPADES) == 1


def test_trick_winner_rejects_an_empty_trick() -> None:
    with pytest.raises(ValueError):
        trick_winner([], Suit.HEARTS, Suit.SPADES)


def test_a_hand_holding_the_led_suit_must_follow_it() -> None:
    hand = [
        C(Rank.ACE, Suit.HEARTS),
        C(Rank.KING, Suit.SPADES),
        C(Rank.TWO, Suit.HEARTS),
    ]
    # only the hearts, and in the order they sit in the hand
    assert legal_plays(hand, Suit.HEARTS) == [hand[0], hand[2]]


def test_a_hand_void_in_the_led_suit_may_play_anything() -> None:
    # No hearts at all, so nothing is compulsory - trumping and discarding are both
    # available, which is the choice that makes ruffing a decision rather than a rule.
    hand = [
        C(Rank.KING, Suit.SPADES),
        C(Rank.TWO, Suit.CLUBS),
        C(Rank.NINE, Suit.DIAMONDS),
    ]
    assert legal_plays(hand, Suit.HEARTS) == hand


def test_leading_allows_any_card() -> None:
    hand = [
        C(Rank.ACE, Suit.HEARTS),
        C(Rank.KING, Suit.SPADES),
        C(Rank.TWO, Suit.CLUBS),
    ]
    assert legal_plays(hand, None) == hand  # led_suit is None for the player on lead
