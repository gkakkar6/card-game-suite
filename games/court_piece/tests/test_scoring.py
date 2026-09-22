from engine.cards import Suit
from games.court_piece.rules import CompletedTrick, TrumpCall
from games.court_piece.scoring import (
    CALLER_MULTIPLIER,
    KOT_VALUE,
    NON_CALLER_MULTIPLIER,
    PIECE_VALUE,
    score_hand,
    score_running_hand,
)
from games.court_piece.tests.test_rules import card

CALL = TrumpCall(trump=Suit.SPADES, caller=0)  # side {0, 2} is the caller's side


def _trick(winner: int) -> CompletedTrick:
    # The actual cards played don't matter to scoring - only who won.
    cards = ((0, card("2H")), (1, card("3H")), (2, card("4H")), (3, card("5H")))
    return CompletedTrick(cards=cards, winner=winner)


def test_neither_side_reaching_seven_scores_nothing() -> None:
    # An incomplete hand (fewer than 13 tricks played) - nobody's reached 7 yet.
    incomplete = tuple(_trick(0) for _ in range(6))
    result = score_hand(CALL, incomplete)
    assert result.winner is None
    assert result.courts == 0.0


def test_callers_side_winning_a_plain_piece() -> None:
    # Caller's side (0, 2) wins exactly 7, spread out - no straight run of 7 from the start.
    winners = [0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0]  # caller's side: 0,2 appear 7 times
    completed = tuple(_trick(w) for w in winners)
    result = score_hand(CALL, completed)
    assert result.winner == 0
    assert result.tricks_won == 7
    assert not result.is_kot
    assert result.courts == PIECE_VALUE * CALLER_MULTIPLIER


def test_non_callers_side_winning_a_plain_piece_is_worth_more() -> None:
    winners = [1, 0, 3, 0, 1, 0, 3, 0, 1, 0, 3, 0, 1]  # non-caller side (1, 3): 7 tricks
    completed = tuple(_trick(w) for w in winners)
    result = score_hand(CALL, completed)
    assert result.winner == 1
    assert not result.is_kot
    assert result.courts == PIECE_VALUE * NON_CALLER_MULTIPLIER


def test_winning_all_thirteen_tricks_is_a_kot() -> None:
    completed = tuple(_trick(0) for _ in range(13))
    result = score_hand(CALL, completed)
    assert result.is_kot
    assert result.tricks_won == 13
    assert result.courts == KOT_VALUE * CALLER_MULTIPLIER


def test_winning_the_first_seven_straight_is_a_kot_even_if_the_hand_isnt_a_shutout() -> None:
    # Caller's side takes tricks 1-7 straight, then loses the rest - still a kot,
    # because it's the *first seven straight* rule, not just "won a majority early."
    winners = [0] * 7 + [1] * 6
    completed = tuple(_trick(w) for w in winners)
    result = score_hand(CALL, completed)
    assert result.is_kot
    assert result.winner == 0
    assert result.courts == KOT_VALUE * CALLER_MULTIPLIER


def test_winning_seven_of_the_first_eight_is_not_a_kot() -> None:
    # One early loss breaks the "straight from trick 1" requirement, even though
    # caller's side still comfortably wins the hand overall.
    winners = [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # caller's side wins 7, but not straight
    completed = tuple(_trick(w) for w in winners)
    result = score_hand(CALL, completed)
    assert result.winner == 0
    assert not result.is_kot
    assert result.courts == PIECE_VALUE * CALLER_MULTIPLIER


def test_only_the_winning_side_ever_scores() -> None:
    winners = [0] * 7 + [1] * 6
    completed = tuple(_trick(w) for w in winners)
    result = score_hand(CALL, completed)
    # The losing side's score is never represented as a negative number here - the
    # caller (session.py) simply never credits anyone but result.winner's side.
    assert result.winner in (0, 2)


# ---------------------------------------------------------------------------
# Running trump: same kot/piece rule, but flat - no caller/non-caller multiplier
# at all, since nobody deliberately chose trump.
# ---------------------------------------------------------------------------


def test_running_trump_plain_win_has_no_multiplier() -> None:
    winners = [0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0]  # side {0, 2}: 7 tricks
    completed = tuple(_trick(w) for w in winners)
    result = score_running_hand(completed)
    assert result.winner in (0, 2)
    assert not result.is_kot
    assert result.courts == PIECE_VALUE  # not multiplied by anything


def test_running_trump_kot_is_worth_a_full_court_no_multiplier() -> None:
    completed = tuple(_trick(0) for _ in range(13))
    result = score_running_hand(completed)
    assert result.is_kot
    assert result.courts == KOT_VALUE  # not multiplied by anything


def test_running_trump_neither_side_reaching_seven_scores_nothing() -> None:
    incomplete = tuple(_trick(0) for _ in range(6))
    result = score_running_hand(incomplete)
    assert result.winner is None
    assert result.courts == 0.0


def test_running_and_fixed_trump_agree_on_who_won_and_how_for_the_same_history() -> None:
    # The two scoring functions should only ever differ by the multiplier, not by
    # which side won or whether it was a kot - both delegate to the same
    # _winning_side()/_is_kot() logic.
    winners = [0] * 7 + [1] * 6
    completed = tuple(_trick(w) for w in winners)
    fixed = score_hand(CALL, completed)
    running = score_running_hand(completed)
    assert fixed.winner == running.winner
    assert fixed.is_kot == running.is_kot
    assert fixed.tricks_won == running.tricks_won
