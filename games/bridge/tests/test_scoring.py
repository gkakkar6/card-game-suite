from engine.cards import Suit
from games.bridge.rules import Contract
from games.bridge.scoring import score_contract

# ---------------------------------------------------------------------------
# Made exactly, hand-verifiable trick values
# ---------------------------------------------------------------------------


def test_3nt_made_exactly_is_40_plus_30_plus_30() -> None:
    contract = Contract(trump=None, declarer=0, target=9)
    result = score_contract(contract, tricks_won=9)
    assert result.made
    assert result.margin == 0
    assert result.points == 40 + 30 + 30


def test_1nt_made_exactly_is_just_the_first_notrump_trick() -> None:
    contract = Contract(trump=None, declarer=0, target=7)
    result = score_contract(contract, tricks_won=7)
    assert result.points == 40


def test_4_spades_made_exactly_is_4_times_the_major_trick_value() -> None:
    contract = Contract(trump=Suit.SPADES, declarer=0, target=10)
    result = score_contract(contract, tricks_won=10)
    assert result.points == 30 * 4


def test_3_clubs_made_exactly_is_3_times_the_minor_trick_value() -> None:
    contract = Contract(trump=Suit.CLUBS, declarer=0, target=9)
    result = score_contract(contract, tricks_won=9)
    assert result.points == 20 * 3


# ---------------------------------------------------------------------------
# Overtricks
# ---------------------------------------------------------------------------


def test_an_overtrick_in_a_suit_contract_is_worth_the_suits_own_trick_value() -> None:
    contract = Contract(trump=Suit.HEARTS, declarer=0, target=8)  # 2H
    result = score_contract(contract, tricks_won=9)
    assert result.made
    assert result.margin == 1
    assert result.points == 30 * 2 + 30


def test_an_overtrick_in_notrump_never_gets_the_first_trick_bonus() -> None:
    contract = Contract(trump=None, declarer=0, target=9)  # 3NT
    result = score_contract(contract, tricks_won=10)
    assert result.margin == 1
    assert result.points == (40 + 30 + 30) + 30  # the overtrick is a plain 30, not 40


def test_several_overtricks_stack_at_the_same_per_trick_value() -> None:
    contract = Contract(trump=Suit.SPADES, declarer=0, target=8)  # 2S
    result = score_contract(contract, tricks_won=11)
    assert result.margin == 3
    assert result.points == 30 * 2 + 30 * 3


# ---------------------------------------------------------------------------
# Undertricks
# ---------------------------------------------------------------------------


def test_going_down_one_costs_a_single_undertrick_penalty() -> None:
    contract = Contract(trump=None, declarer=0, target=9)  # 3NT
    result = score_contract(contract, tricks_won=8)
    assert not result.made
    assert result.margin == -1
    assert result.points == -50


def test_going_down_several_scales_the_penalty_by_how_many_tricks_short() -> None:
    contract = Contract(trump=Suit.SPADES, declarer=0, target=10)  # 4S
    result = score_contract(contract, tricks_won=6)
    assert result.margin == -4
    assert result.points == -50 * 4


def test_going_down_exactly_one_trick_short_of_the_lowest_possible_contract() -> None:
    contract = Contract(trump=Suit.CLUBS, declarer=0, target=7)  # 1C
    result = score_contract(contract, tricks_won=6)
    assert result.margin == -1
    assert result.points == -50


# ---------------------------------------------------------------------------
# Passed-out hands are a session-level concern, not a scoring.py one - see
# games/bridge/tests/test_session.py for that case. score_contract() itself only
# ever receives a real, played contract.
# ---------------------------------------------------------------------------
