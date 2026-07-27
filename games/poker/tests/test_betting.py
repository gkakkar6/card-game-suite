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
