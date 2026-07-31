import random
import time

import pytest

from engine.cards import Card, Rank, Suit
from games.poker.equity import Equity, enumeration_size, hand_equity

# The turn/unknown-opponent enumeration is the slowest case in this file. The bound is a
# regression guard against an accidental blow-up in cost, not a benchmark - it sits well
# above the ~2s this actually takes so it won't flake on a slower CI runner.
TURN_ENUMERATION_TIME_LIMIT_SECONDS = 10.0


def C(rank: Rank, suit: Suit) -> Card:
    return Card(rank, suit)


def test_equity_combines_wins_and_half_of_ties() -> None:
    result = Equity(wins=6, ties=2, losses=2, exact=True)
    assert result.trials == 10
    assert result.equity == 0.7  # (6 + 0.5 * 2) / 10
    assert result.win_probability == 0.6
    assert result.tie_probability == 0.2
    assert result.loss_probability == 0.2


def test_equity_of_empty_result_is_zero() -> None:
    result = Equity(wins=0, ties=0, losses=0, exact=True)
    assert result.trials == 0
    assert result.equity == 0.0


def test_river_enumerates_every_opponent_hand_exactly() -> None:
    # Player holds two aces and the board holds the other two, so the player has quad
    # aces and the opponent cannot hold an ace at all. No straight flush is possible
    # (only two hearts in play), so the player wins against all C(45, 2) = 990 hands.
    result = hand_equity(
        [C(Rank.ACE, Suit.CLUBS), C(Rank.ACE, Suit.DIAMONDS)],
        [
            C(Rank.ACE, Suit.HEARTS), C(Rank.ACE, Suit.SPADES), C(Rank.KING, Suit.CLUBS),
            C(Rank.KING, Suit.DIAMONDS), C(Rank.QUEEN, Suit.HEARTS),
        ],
    )
    assert result.exact
    assert result.trials == 990
    assert result.wins == 990
    assert result.equity == 1.0


def test_river_board_that_plays_itself_ties_every_time() -> None:
    # A royal flush on the board is the best hand for both players no matter what
    # anyone holds, so all 990 enumerated showdowns are ties and equity is exactly 0.5.
    result = hand_equity(
        [C(Rank.TWO, Suit.CLUBS), C(Rank.THREE, Suit.DIAMONDS)],
        [
            C(Rank.ACE, Suit.SPADES), C(Rank.KING, Suit.SPADES), C(Rank.QUEEN, Suit.SPADES),
            C(Rank.JACK, Suit.SPADES), C(Rank.TEN, Suit.SPADES),
        ],
    )
    assert result.exact
    assert result.trials == 990
    assert result.ties == 990
    assert result.equity == 0.5


def test_river_against_a_known_opponent_is_a_single_showdown() -> None:
    # Nothing is unknown: both hands and the whole board are fixed, so there is exactly
    # one showdown to score. Trip aces beat the opponent's pair of kings.
    result = hand_equity(
        [C(Rank.ACE, Suit.CLUBS), C(Rank.ACE, Suit.DIAMONDS)],
        [
            C(Rank.ACE, Suit.HEARTS), C(Rank.SEVEN, Suit.SPADES), C(Rank.TWO, Suit.CLUBS),
            C(Rank.THREE, Suit.DIAMONDS), C(Rank.NINE, Suit.HEARTS),
        ],
        opponent_hole=[C(Rank.KING, Suit.CLUBS), C(Rank.KING, Suit.DIAMONDS)],
    )
    assert result.exact
    assert result.trials == 1
    assert result.wins == 1
    assert result.equity == 1.0


def test_flop_against_an_unknown_opponent_is_sampled() -> None:
    # Two unknown board cards plus two unknown opponent cards is far too many
    # combinations to enumerate, so this must take the Monte Carlo path.
    result = hand_equity(
        [C(Rank.ACE, Suit.CLUBS), C(Rank.KING, Suit.CLUBS)],
        [C(Rank.TWO, Suit.HEARTS), C(Rank.SEVEN, Suit.SPADES), C(Rank.NINE, Suit.DIAMONDS)],
        iterations=300,
        rng=random.Random(31),
    )
    assert not result.exact
    assert result.trials == 300
    # a real mix of outcomes, not a degenerate all-win or all-loss run
    assert 0.0 < result.equity < 1.0
    assert result.wins > 0
    assert result.losses > 0


def test_turn_against_an_unknown_opponent_enumerates_every_combination() -> None:
    # 46 unseen cards: each can be the river, leaving 45 from which the opponent's two
    # cards are drawn. So 46 x C(45, 2) = 46 x 990 = 45,540 exact showdowns.
    started = time.perf_counter()
    result = hand_equity(
        [C(Rank.ACE, Suit.CLUBS), C(Rank.KING, Suit.CLUBS)],
        [
            C(Rank.TWO, Suit.HEARTS), C(Rank.SEVEN, Suit.SPADES),
            C(Rank.NINE, Suit.DIAMONDS), C(Rank.QUEEN, Suit.HEARTS),
        ],
    )
    elapsed = time.perf_counter() - started

    assert result.exact
    assert result.trials == 46 * 990 == 45_540
    assert 0.0 < result.equity < 1.0
    assert elapsed < TURN_ENUMERATION_TIME_LIMIT_SECONDS


def test_turn_enumerates_every_river_card_exactly() -> None:
    # Trip aces against pocket kings with one card to come. The opponent can reach at
    # most trip kings (only one of the two remaining kings can land), so the player
    # wins on all 44 possible river cards.
    result = hand_equity(
        [C(Rank.ACE, Suit.CLUBS), C(Rank.ACE, Suit.DIAMONDS)],
        [
            C(Rank.ACE, Suit.HEARTS), C(Rank.SEVEN, Suit.SPADES),
            C(Rank.TWO, Suit.CLUBS), C(Rank.THREE, Suit.DIAMONDS),
        ],
        opponent_hole=[C(Rank.KING, Suit.CLUBS), C(Rank.KING, Suit.DIAMONDS)],
    )
    assert result.exact
    assert result.trials == 44
    assert result.wins == 44


def test_aces_beat_kings_preflop_about_eighty_percent() -> None:
    # The standard reference number for AA against KK heads-up all-in preflop.
    result = hand_equity(
        [C(Rank.ACE, Suit.CLUBS), C(Rank.ACE, Suit.DIAMONDS)],
        opponent_hole=[C(Rank.KING, Suit.HEARTS), C(Rank.KING, Suit.SPADES)],
        iterations=10_000,
        rng=random.Random(2024),
    )
    assert not result.exact
    assert result.trials == 10_000
    assert 0.79 < result.equity < 0.86


def test_monte_carlo_is_reproducible_with_seeded_rng() -> None:
    hole = [C(Rank.SEVEN, Suit.CLUBS), C(Rank.TWO, Suit.DIAMONDS)]
    first = hand_equity(hole, iterations=500, rng=random.Random(7))
    second = hand_equity(hole, iterations=500, rng=random.Random(7))
    assert (first.wins, first.ties, first.losses) == (second.wins, second.ties, second.losses)


def test_enumeration_size_matches_the_combinatorics() -> None:
    # Known opponent: only the board has to be completed.
    assert enumeration_size(5, opponent_known=True) == 1  # nothing left to deal
    assert enumeration_size(4, opponent_known=True) == 44  # each possible river
    assert enumeration_size(3, opponent_known=True) == 990  # C(45, 2) turn+river pairs
    assert enumeration_size(0, opponent_known=True) == 1_712_304  # C(48, 5) whole boards

    # Unknown opponent: every board completion is multiplied by every opponent hand
    # drawable from the cards it leaves behind.
    assert enumeration_size(5, opponent_known=False) == 990
    assert enumeration_size(4, opponent_known=False) == 46 * 990
    assert enumeration_size(3, opponent_known=False) == 1081 * 990


def test_method_is_chosen_by_cost_not_by_street() -> None:
    # A flop against a known hand is 990 combinations - cheaper than a river against an
    # unknown one - so the street alone can't decide the method. What matters is how
    # much work enumerating would actually be.
    hole = [C(Rank.ACE, Suit.CLUBS), C(Rank.KING, Suit.CLUBS)]
    flop = [C(Rank.TWO, Suit.HEARTS), C(Rank.SEVEN, Suit.SPADES), C(Rank.NINE, Suit.DIAMONDS)]
    opponent = [C(Rank.FOUR, Suit.CLUBS), C(Rank.FIVE, Suit.DIAMONDS)]

    # ~1.7M combinations even though the opponent is known: too many, so sample.
    preflop = hand_equity(hole, iterations=50, rng=random.Random(1), opponent_hole=opponent)
    assert not preflop.exact
    assert preflop.trials == 50

    # same flop, unknown opponent: over a million combinations, so sample.
    flop_unknown = hand_equity(hole, flop, iterations=50, rng=random.Random(1))
    assert not flop_unknown.exact

    on_turn = hand_equity(hole, [*flop, C(Rank.QUEEN, Suit.HEARTS)], opponent_hole=opponent)
    assert on_turn.exact
    assert on_turn.trials == 44  # every remaining river card


def test_flop_against_a_known_opponent_is_enumerated_exactly() -> None:
    # Both hands are known and only the turn and river are missing, so every one of the
    # C(45, 2) = 990 turn/river pairs can be scored exactly - no sampling needed, and
    # the seeded rng below is therefore ignored entirely.
    result = hand_equity(
        [C(Rank.ACE, Suit.CLUBS), C(Rank.KING, Suit.CLUBS)],
        [C(Rank.TWO, Suit.HEARTS), C(Rank.SEVEN, Suit.SPADES), C(Rank.NINE, Suit.DIAMONDS)],
        opponent_hole=[C(Rank.FOUR, Suit.CLUBS), C(Rank.FIVE, Suit.DIAMONDS)],
        rng=random.Random(1),
    )
    assert result.exact
    assert result.trials == 990
    assert 0.0 < result.equity < 1.0


def test_probabilities_sum_to_one() -> None:
    result = hand_equity(
        [C(Rank.ACE, Suit.CLUBS), C(Rank.KING, Suit.CLUBS)],
        iterations=200,
        rng=random.Random(11),
    )
    total = result.win_probability + result.tie_probability + result.loss_probability
    assert abs(total - 1.0) < 1e-9


def test_invalid_inputs_are_rejected() -> None:
    ace = C(Rank.ACE, Suit.CLUBS)
    king = C(Rank.KING, Suit.CLUBS)

    with pytest.raises(ValueError):
        hand_equity([ace])  # only one hole card
    with pytest.raises(ValueError):
        hand_equity([ace, king], [ace])  # board of one card
    with pytest.raises(ValueError):
        hand_equity([ace, king], opponent_hole=[ace])  # opponent short a card
    with pytest.raises(ValueError):
        hand_equity([ace, ace])  # the same card twice in one hand
    with pytest.raises(ValueError):
        hand_equity([ace, king], opponent_hole=[ace, C(Rank.TWO, Suit.HEARTS)])  # shared card


def test_opponent_range_is_reserved_but_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        hand_equity(
            [C(Rank.ACE, Suit.CLUBS), C(Rank.KING, Suit.CLUBS)],
            opponent_range=[(C(Rank.QUEEN, Suit.HEARTS), C(Rank.QUEEN, Suit.SPADES))],
        )
