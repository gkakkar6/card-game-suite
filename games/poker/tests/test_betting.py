import pytest

from games.poker.betting import ActionType, BettingAction, BettingRound


def test_check_check_settles_round() -> None:
    round_ = BettingRound(stacks=[100, 100])
    assert not round_.is_settled()
    round_.apply(BettingAction(ActionType.CHECK))
    assert not round_.is_settled()
    round_.apply(BettingAction(ActionType.CHECK))
    assert round_.is_settled()
    assert round_.pot == 0


def test_bet_call_settles_round_and_moves_chips_to_pot() -> None:
    round_ = BettingRound(stacks=[100, 100])
    round_.apply(BettingAction(ActionType.BET, amount=10))
    assert not round_.is_settled()
    round_.apply(BettingAction(ActionType.CALL))
    assert round_.is_settled()
    assert round_.pot == 20
    assert round_.stacks == [90, 90]


def test_fold_ends_round_immediately() -> None:
    round_ = BettingRound(stacks=[100, 100])
    round_.apply(BettingAction(ActionType.BET, amount=10))
    round_.apply(BettingAction(ActionType.FOLD))
    assert round_.is_settled()
    assert round_.folded == {1}


def test_raise_reopens_action_for_earlier_players() -> None:
    round_ = BettingRound(stacks=[100, 100, 100])
    round_.apply(BettingAction(ActionType.BET, amount=10))
    round_.apply(BettingAction(ActionType.RAISE, amount=30))
    assert not round_.is_settled()
    # player 0 must act again since the raise reopened the round
    assert round_.current_player == 2
    round_.apply(BettingAction(ActionType.CALL))
    assert not round_.is_settled()
    round_.apply(BettingAction(ActionType.CALL))
    assert round_.is_settled()
    assert round_.pot == 90


def test_check_is_illegal_facing_a_bet() -> None:
    round_ = BettingRound(stacks=[100, 100])
    round_.apply(BettingAction(ActionType.BET, amount=10))
    try:
        round_.apply(BettingAction(ActionType.CHECK))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_raise_below_minimum_is_illegal() -> None:
    round_ = BettingRound(stacks=[100, 100], min_bet=10)
    round_.apply(BettingAction(ActionType.BET, amount=10))
    try:
        round_.apply(BettingAction(ActionType.RAISE, amount=15))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_all_in_short_raise_is_legal_even_below_minimum() -> None:
    round_ = BettingRound(stacks=[100, 15], min_bet=10)
    round_.apply(BettingAction(ActionType.BET, amount=10))
    # player 1 only has 15 total behind, so raising all-in to 15 is a "short" raise below min-bet
    round_.apply(BettingAction(ActionType.RAISE, amount=15))
    assert round_.stacks[1] == 0
    assert 1 in round_.all_in


def _raise_war(max_raises: int | None) -> BettingRound:
    """Open for 10, then trade three raises. Returns the round with player 0 to act."""
    round_ = BettingRound(stacks=[200, 200], min_bet=10, max_raises=max_raises)
    round_.apply(BettingAction(ActionType.BET, amount=10))
    round_.apply(BettingAction(ActionType.RAISE, amount=20))
    round_.apply(BettingAction(ActionType.RAISE, amount=30))
    round_.apply(BettingAction(ActionType.RAISE, amount=40))
    return round_


def test_raise_cap_removes_raise_once_reached() -> None:
    round_ = _raise_war(max_raises=3)
    assert round_.raises_made == 3  # the opening bet is not a raise
    assert round_.current_player == 0
    # player 0 has both the chips and the room to raise, so only the cap stops them
    assert round_.stacks[0] > round_.current_bet - round_.bets[0]
    assert round_.legal_actions() == [ActionType.FOLD, ActionType.CALL]

    with pytest.raises(ValueError):
        round_.apply(BettingAction(ActionType.RAISE, amount=50))


def test_uncapped_round_keeps_allowing_raises() -> None:
    round_ = _raise_war(max_raises=None)
    assert round_.raises_made == 3
    assert ActionType.RAISE in round_.legal_actions()
    round_.apply(BettingAction(ActionType.RAISE, amount=50))
    assert round_.raises_made == 4


def test_raise_cap_of_one_allows_a_single_raise() -> None:
    round_ = BettingRound(stacks=[200, 200], min_bet=10, max_raises=1)
    round_.apply(BettingAction(ActionType.BET, amount=10))
    assert ActionType.RAISE in round_.legal_actions()
    round_.apply(BettingAction(ActionType.RAISE, amount=20))
    assert round_.legal_actions() == [ActionType.FOLD, ActionType.CALL]


def test_calling_more_than_the_stack_goes_all_in_for_less() -> None:
    # A short stack facing a bet it cannot cover must end at exactly zero. Chip
    # conservation alone does not catch this: subtracting 50 from a 5-chip stack and
    # adding 50 to the pot still balances, it just leaves the stack at -45.
    round_ = BettingRound(stacks=[200, 5], min_bet=2)
    round_.apply(BettingAction(ActionType.BET, amount=50))
    round_.apply(BettingAction(ActionType.CALL))

    assert round_.stacks[1] == 0
    assert round_.bets[1] == 5  # only what they actually had
    assert round_.pot == 55  # the 50 bet plus their 5, not 100
    assert 1 in round_.all_in
    assert round_.is_settled()  # nothing left for them to decide


def test_action_skips_a_player_left_all_in_by_their_blind() -> None:
    # A one-chip stack posting a small blind of one is already all-in before the round
    # starts, so it has no decision to make and the action has to begin elsewhere.
    round_ = BettingRound(stacks=[0, 1], first_to_act=0, min_bet=2, initial_bets=[1, 2])
    assert 0 in round_.all_in
    assert round_.current_player == 1
    assert ActionType.CHECK in round_.legal_actions()


def test_no_stack_can_go_negative_across_a_round() -> None:
    round_ = BettingRound(stacks=[100, 3, 7], min_bet=2)
    round_.apply(BettingAction(ActionType.BET, amount=40))
    round_.apply(BettingAction(ActionType.CALL))
    round_.apply(BettingAction(ActionType.CALL))

    assert all(stack >= 0 for stack in round_.stacks)
    assert round_.stacks[1] == 0
    assert round_.stacks[2] == 0
    assert round_.pot == 40 + 3 + 7


def test_initial_bets_seed_pot_for_posted_blinds() -> None:
    round_ = BettingRound(stacks=[99, 98], first_to_act=0, initial_bets=[1, 2])
    assert round_.pot == 3
    assert round_.current_bet == 2


def test_short_all_in_does_not_reopen_action_for_players_who_already_called() -> None:
    # A and B both call a full 10 bet, then C shoves all-in for only 15 total - a raise
    # of just 5, well below the min-raise of 20. That's legal (C can only raise what they
    # have), but it must NOT let A and B re-raise: they already matched the previous bet,
    # so they may only call the extra 5 or fold.
    round_ = BettingRound(stacks=[100, 100, 15], min_bet=10)
    round_.apply(BettingAction(ActionType.BET, amount=10))  # A
    round_.apply(BettingAction(ActionType.CALL))  # B
    round_.apply(BettingAction(ActionType.RAISE, amount=15))  # C, short all-in

    assert round_.current_player == 0
    assert round_.legal_actions() == [ActionType.FOLD, ActionType.CALL]
    round_.apply(BettingAction(ActionType.CALL))  # A: call the extra 5, can't re-raise

    assert round_.current_player == 1
    assert round_.legal_actions() == [ActionType.FOLD, ActionType.CALL]
    round_.apply(BettingAction(ActionType.CALL))  # B: call the extra 5, can't re-raise

    assert round_.is_settled()
    assert round_.pot == 45


def test_raise_illegal_when_no_other_player_can_respond() -> None:
    # B shoves their entire stack as the opening bet; A is the only other player and
    # isn't all-in, but once A calls or raises there's nobody left who could still
    # respond to a further raise - so RAISE shouldn't be offered to A at all.
    round_ = BettingRound(stacks=[100, 20], first_to_act=1)
    round_.apply(BettingAction(ActionType.BET, amount=20))  # B, all-in

    assert round_.current_player == 0
    assert 1 in round_.all_in
    assert round_.legal_actions() == [ActionType.FOLD, ActionType.CALL]
