from engine.cards import Card, Rank, Suit
from games.court_piece.rules import (
    CourtPiece,
    CourtPieceState,
    TrumpCall,
    opening_leader,
    partner,
)

_RANKS = {rank.symbol: rank for rank in Rank}
_SUITS = {suit.symbol: suit for suit in Suit}


def card(text: str) -> Card:
    """"KH" -> the king of hearts, so a scripted hand reads like real cards."""
    return Card(_RANKS[text[0]], _SUITS[text[1]])


def hand(*texts: str) -> tuple[Card, ...]:
    return tuple(card(text) for text in texts)


CALLER = 0
CALL = TrumpCall(trump=Suit.SPADES, caller=CALLER)


def test_partner_is_the_opposite_seat() -> None:
    assert partner(0) == 2
    assert partner(2) == 0
    assert partner(1) == 3
    assert partner(3) == 1


def test_opening_leader_is_the_caller_themselves() -> None:
    assert opening_leader(0) == 0
    assert opening_leader(3) == 3


def test_current_player_is_always_the_seat_on_turn() -> None:
    # No dummy redirection at all - the real difference from bridge.
    game = CourtPiece()
    state = CourtPieceState(call=CALL, hands=(hand("2H"), hand("3H"), hand("4H"), hand("5H")))
    for _ in range(4):
        assert game.current_player(state) == state.to_play
        state = game.apply(state, state.hands[state.to_play][0])


def test_must_follow_suit_when_able() -> None:
    game = CourtPiece()
    state = CourtPieceState(
        call=CALL,
        hands=(hand("2H", "3S"), hand(), hand(), hand()),
        trick=((3, card("4H")),),
    )
    assert game.legal_actions(state) == [card("2H")]


def test_void_in_led_suit_may_play_anything() -> None:
    game = CourtPiece()
    state = CourtPieceState(
        call=CALL,
        hands=(hand("2C", "3S"), hand(), hand(), hand()),
        trick=((3, card("4H")),),
    )
    assert set(game.legal_actions(state)) == {card("2C"), card("3S")}


def test_trump_wins_over_a_higher_card_of_the_led_suit() -> None:
    game = CourtPiece()
    state = CourtPieceState(
        call=CALL,
        hands=(hand("2S"), hand(), hand(), hand()),
        trick=((1, card("AH")), (2, card("KH")), (3, card("2H"))),
    )
    result = game.apply(state, card("2S"))
    assert result.completed[-1].winner == 0  # the ruff, not the ace


def test_payoff_credits_both_partners_equally() -> None:
    game = CourtPiece()
    from games.court_piece.rules import CompletedTrick

    # One completed trick won by seat 2 (caller's partner) - both of seat 2's side
    # should be credited, not just seat 2 itself.
    trick_cards = ((0, card("2H")), (1, card("3H")), (2, card("AH")), (3, card("4H")))
    won_by_partner = CourtPieceState(
        call=CALL,
        hands=(hand(), hand(), hand(), hand()),
        completed=(CompletedTrick(cards=trick_cards, winner=2),),
    )
    payoff = game.payoff(won_by_partner)
    assert payoff[0] == 1.0
    assert payoff[2] == 1.0
    assert payoff[1] == 0.0
    assert payoff[3] == 0.0


def test_information_set_shows_only_the_players_own_hand() -> None:
    game = CourtPiece()
    state = CourtPieceState(
        call=CALL, hands=(hand("2H"), hand("3H"), hand("4H"), hand("5H"))
    )
    info = game.information_set(state, 0)
    assert set(info.hands) == {0}
    assert info.hands[0] == hand("2H")


def test_is_terminal_after_thirteen_completed_tricks() -> None:
    game = CourtPiece()
    from games.court_piece.rules import CompletedTrick

    trick_cards = ((0, card("2H")), (1, card("3H")), (2, card("4H")), (3, card("5H")))
    one_trick = CompletedTrick(cards=trick_cards, winner=0)
    empty_hands = (hand(), hand(), hand(), hand())
    full = CourtPieceState(call=CALL, hands=empty_hands, completed=(one_trick,) * 13)
    almost_full = CourtPieceState(call=CALL, hands=empty_hands, completed=(one_trick,) * 12)
    assert game.is_terminal(full)
    assert not game.is_terminal(almost_full)


# ---------------------------------------------------------------------------
# Running trump: trump starts undetermined, and the first player who can't follow
# suit sets it to whatever they play - CourtPiece.apply()'s own new mechanic.
# ---------------------------------------------------------------------------

RUNNING_CALL = TrumpCall(trump=None, caller=0)


def test_a_void_play_sets_trump_to_that_cards_own_suit() -> None:
    game = CourtPiece()
    # Seat 1 led a heart; seat 2 holds no hearts at all and plays a spade.
    state = CourtPieceState(
        call=RUNNING_CALL,
        hands=(hand(), hand(), hand("3S", "4C"), hand()),
        trick=((1, card("2H")),),
    )
    result = game.apply(state, card("3S"))
    assert result.call.trump == Suit.SPADES


def test_leading_never_sets_trump_even_if_it_is_still_undetermined() -> None:
    # Nothing is led yet, so nobody can be "void" - the leader is always free to play
    # anything, and that alone must never set trump. RUNNING_CALL's caller is seat 0,
    # so opening_leader() puts seat 0 on lead with no trick or history yet.
    game = CourtPiece()
    state = CourtPieceState(call=RUNNING_CALL, hands=(hand("3S"), hand(), hand(), hand()))
    assert state.to_play == 0
    result = game.apply(state, card("3S"))
    assert result.call.trump is None


def test_following_suit_never_sets_trump_even_if_it_is_still_undetermined() -> None:
    # Seat 1 leads a heart; seat 2 also holds a heart and follows normally - not void,
    # so trump must stay undetermined regardless of what card they play.
    game = CourtPiece()
    state = CourtPieceState(
        call=RUNNING_CALL,
        hands=(hand(), hand(), hand("9H", "4C"), hand()),
        trick=((1, card("2H")),),
    )
    result = game.apply(state, card("9H"))
    assert result.call.trump is None


def test_trump_stays_fixed_once_set_even_through_a_later_void() -> None:
    # Trump is already spades; a later trick's void play must never change it again.
    already_running = TrumpCall(trump=Suit.SPADES, caller=0)
    game = CourtPiece()
    state = CourtPieceState(
        call=already_running,
        hands=(hand(), hand(), hand("4D", "5C"), hand()),
        trick=((1, card("2H")),),  # seat 2 is void in hearts again, holds no hearts
    )
    result = game.apply(state, card("4D"))
    assert result.call.trump == Suit.SPADES


def test_the_trump_setting_card_can_win_its_own_trick_outright() -> None:
    # Seat 1 leads the ace of hearts (unbeatable in hearts); seat 2 is void and plays
    # a spade, setting trump to spades right now - and since nobody has played a
    # trump before, that spade wins the trick outright once the trick completes,
    # not the ace. Seats 0 and 3 follow with low hearts to complete the trick.
    game = CourtPiece()
    state = CourtPieceState(
        call=RUNNING_CALL,
        hands=(hand("2H"), hand(), hand("3S"), hand("4H")),
        trick=((1, card("AH")),),
    )
    assert state.to_play == 2
    state = game.apply(state, card("3S"))
    assert state.to_play == 3
    state = game.apply(state, card("4H"))
    assert state.to_play == 0
    state = game.apply(state, card("2H"))
    assert state.call.trump == Suit.SPADES
    assert state.completed[-1].winner == 2  # the spade, not the ace of hearts
