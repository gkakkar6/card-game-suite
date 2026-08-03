import random

import pytest

from engine.cards import Card, Rank, Suit
from engine.game import Game
from games.bridge.deal import CARDS_PER_SEAT, SEATS, deal_hands
from games.bridge.rules import (
    TRICKS,
    Bridge,
    BridgeState,
    Contract,
    new_deal,
    opening_leader,
    partner,
)

_RANKS = {rank.symbol: rank for rank in Rank}
_SUITS = {suit.symbol: suit for suit in Suit}


def card(text: str) -> Card:
    """"KH" -> the king of hearts, so the scripted deal below reads like a bridge hand
    rather than a wall of constructor calls."""
    return Card(_RANKS[text[0]], _SUITS[text[1]])


DECLARER = 0
DUMMY = partner(DECLARER)  # seat 2
TRUMP = Suit.SPADES
CONTRACT = Contract(trump=TRUMP, declarer=DECLARER, target=9)

# One complete deal, worked out by hand. Spades are trump, seat 0 is declarer, so seat
# 2 is the dummy and seat 1 - on declarer's left - makes the opening lead.
#
# Each line is one trick: its four cards in the order they are played, starting with
# whoever won the previous trick, then the seat that wins it. Every winner is decided
# by the rule alone (highest trump, else highest card of the suit led), which is what
# makes the expected sequence checkable line by line without running anything.
#
# The four hands are derived from this script rather than written out separately (see
# `scripted_hands`), which makes the script the single source of truth for both. If the
# engine ever seated a player wrongly, or let the wrong player lead, a scripted card
# would land on a hand that doesn't hold it and apply() would reject the play outright.
SCRIPT: tuple[tuple[tuple[str, ...], int], ...] = (
    (("QD", "5D", "3D", "2D"), 1),  # E leads, all follow, the queen is high
    (("JD", "6D", "4D", "KD"), 0),  # N follows with the king and takes it
    (("AD", "TD", "2H", "2S"), 3),  # S is void and discards; W ruffs the ace with a two
    (("JC", "9C", "9H", "AC"), 2),  # E is void in clubs and discards; the ace wins
    (("KC", "TC", "8C", "9S"), 1),  # E ruffs the king of clubs instead of discarding
    (("JS", "6S", "3S", "QS"), 0),  # trump led: highest trump wins, same rule
    (("AH", "QH", "8H", "6C"), 0),  # W is void in hearts but discards rather than ruffs
    (("KH", "JH", "7H", "5C"), 0),
    (("5H", "TH", "6H", "4C"), 1),  # N leads low and E's ten takes it
    (("9D", "7S", "4S", "7C"), 2),  # two players ruff; the seven of spades is higher
    (("8S", "5S", "KS", "TS"), 0),
    (("AS", "8D", "3H", "3C"), 0),  # last trump out, the other three all discard
    (("4H", "7D", "QC", "2C"), 0),  # nobody can follow and nobody has a trump left
)

# Declarer's side takes 9 of the 13 tricks, exactly its target - a coincidence worth
# nothing to payoff() in Phase 1, which reports raw trick counts either way.
DECLARER_TRICKS = 9
DEFENCE_TRICKS = TRICKS - DECLARER_TRICKS


def scripted_hands() -> tuple[tuple[Card, ...], ...]:
    """The four hands SCRIPT plays out, collected from the script itself."""
    hands: list[list[Card]] = [[] for _ in range(SEATS)]
    leader = opening_leader(DECLARER)
    for plays, winner in SCRIPT:
        for offset, text in enumerate(plays):
            hands[(leader + offset) % SEATS].append(card(text))
        leader = winner
    return tuple(tuple(hand) for hand in hands)


def scripted_state() -> BridgeState:
    """The scripted deal, dealt but with nothing played yet."""
    return BridgeState(contract=CONTRACT, hands=scripted_hands())


def play_script(game: Bridge, state: BridgeState) -> BridgeState:
    """Play SCRIPT out in full. Any illegal or misdealt card raises from apply()."""
    for plays, _winner in SCRIPT:
        for text in plays:
            state = game.apply(state, card(text))
    return state


# ---------------------------------------------------------------------------
# The scripted deal is a real deal
# ---------------------------------------------------------------------------


def test_the_scripted_deal_is_a_complete_legal_deal() -> None:
    # Guards the test data itself: if the script ever drifted into playing a card twice
    # or leaving one out, every assertion built on it below would be meaningless.
    hands = scripted_hands()
    assert [len(hand) for hand in hands] == [CARDS_PER_SEAT] * SEATS
    dealt = [c for hand in hands for c in hand]
    assert len(set(dealt)) == SEATS * CARDS_PER_SEAT


# ---------------------------------------------------------------------------
# Turn order and the opening lead
# ---------------------------------------------------------------------------


def test_the_opening_lead_is_the_player_on_declarers_left() -> None:
    hands = deal_hands(rng=random.Random(0))
    for declarer in range(SEATS):
        state = BridgeState(contract=Contract(TRUMP, declarer, target=7), hands=hands)
        assert state.leader == (declarer + 1) % SEATS
        assert state.to_play == state.leader  # nobody has played to the trick yet


def test_the_opening_lead_wraps_from_the_last_seat_to_the_first() -> None:
    # The seat arithmetic that is easiest to get wrong: declarer in the last seat leads
    # from seat 0, not from a seat 4 that doesn't exist.
    assert opening_leader(3) == 0
    state = BridgeState(
        contract=Contract(TRUMP, declarer=3, target=7), hands=deal_hands(rng=random.Random(0))
    )
    assert state.leader == 0
    assert Bridge().current_player(state) == 0  # an opponent, deciding for themselves


def test_play_runs_clockwise_from_whoever_is_on_lead() -> None:
    game = Bridge()
    state = scripted_state()
    leader = state.leader
    for offset, text in enumerate(SCRIPT[0][0]):
        assert state.to_play == (leader + offset) % SEATS
        state = game.apply(state, card(text))


# ---------------------------------------------------------------------------
# Legality: following suit
# ---------------------------------------------------------------------------


def test_a_hand_holding_the_led_suit_must_follow_it() -> None:
    # Declarer sits in seat 3, so seat 0 leads. Seat 1 holds a heart and therefore has
    # exactly one legal card, whatever else is in the hand.
    game = Bridge()
    state = BridgeState(
        contract=Contract(trump=TRUMP, declarer=3, target=7),
        hands=(
            (card("2H"),),
            (card("AH"), card("KS")),
            (card("3H"),),
            (card("4H"),),
        ),
    )
    state = game.apply(state, card("2H"))

    assert state.to_play == 1
    assert game.legal_actions(state) == [card("AH")]
    with pytest.raises(ValueError):
        game.apply(state, card("KS"))  # a revoke, rejected rather than trusted


def test_a_hand_void_in_the_led_suit_may_trump_or_discard() -> None:
    game = Bridge()
    state = BridgeState(
        contract=Contract(trump=TRUMP, declarer=3, target=7),
        hands=(
            (card("2H"),),
            (card("2S"), card("KC")),  # no heart: both the trump and the club are legal
            (card("3H"),),
            (card("4H"),),
        ),
    )
    state = game.apply(state, card("2H"))
    assert game.legal_actions(state) == [card("2S"), card("KC")]


def test_a_completed_trick_credits_its_winner_and_passes_the_lead() -> None:
    game = Bridge()
    state = BridgeState(
        contract=Contract(trump=TRUMP, declarer=3, target=7),
        hands=((card("2H"),), (card("2S"),), (card("AH"),), (card("KH"),)),
    )
    for text in ("2H", "2S", "AH", "KH"):
        state = game.apply(state, card(text))

    assert state.trick == ()  # the trick in progress is cleared once resolved
    assert len(state.completed) == 1
    assert state.completed[0].winner == 1  # the two of spades ruffs the ace of hearts
    assert state.completed[0].led_suit is Suit.HEARTS
    assert state.leader == 1  # the winner leads the next trick
    assert game.payoff(state) == {0: 0.0, 1: 1.0, 2: 0.0, 3: 1.0}


def test_apply_leaves_the_state_it_was_given_untouched() -> None:
    # apply() returns a new state rather than mutating, which is what will let the
    # double-dummy search walk the game tree without undoing moves.
    before = scripted_state()
    after = Bridge().apply(before, card(SCRIPT[0][0][0]))

    assert before.hands == scripted_hands()
    assert before.trick == ()
    assert len(after.trick) == 1
    assert len(after.hands[1]) == CARDS_PER_SEAT - 1


# ---------------------------------------------------------------------------
# A full thirteen-trick hand
# ---------------------------------------------------------------------------


def test_the_scripted_deal_produces_the_expected_trick_winners_in_sequence() -> None:
    game = Bridge()
    state = play_script(game, scripted_state())

    assert [trick.winner for trick in state.completed] == [winner for _plays, winner in SCRIPT]
    assert [tuple(str(c) for _seat, c in trick.cards) for trick in state.completed] == [
        plays for plays, _winner in SCRIPT
    ]


def test_the_deal_is_terminal_only_once_all_thirteen_tricks_are_played() -> None:
    game = Bridge()
    state = scripted_state()
    for plays, _winner in SCRIPT[:-1]:
        for text in plays:
            state = game.apply(state, card(text))
        assert not game.is_terminal(state)

    last, _winner = SCRIPT[-1]
    for text in last[:-1]:
        state = game.apply(state, card(text))
        assert not game.is_terminal(state)  # the thirteenth trick is still in progress

    state = game.apply(state, card(last[-1]))
    assert game.is_terminal(state)
    assert state.hands == ((), (), (), ())
    assert game.legal_actions(state) == []


def test_payoff_reports_raw_trick_counts_for_each_partnership() -> None:
    game = Bridge()
    payoff = game.payoff(play_script(game, scripted_state()))

    assert payoff == {0: 9.0, 1: 4.0, 2: 9.0, 3: 4.0}
    # counted straight off the script, so the numbers above aren't taken on trust
    assert payoff[DECLARER] == sum(1 for _plays, w in SCRIPT if w in (DECLARER, DUMMY))
    assert payoff[DECLARER] == DECLARER_TRICKS
    assert payoff[DECLARER] == payoff[partner(DECLARER)]  # partners share one number
    assert payoff[1] == payoff[3] == DEFENCE_TRICKS
    assert payoff[DECLARER] + payoff[1] == TRICKS  # every trick went to exactly one side


def test_the_contract_is_carried_through_play_and_not_scored() -> None:
    # Declarer's side needed 9 and took exactly 9, but payoff() above reports the trick
    # count either way: contract scoring is a deliberately deferred later layer.
    game = Bridge()
    state = play_script(game, scripted_state())
    assert state.contract == CONTRACT
    assert state.contract.target == DECLARER_TRICKS


# ---------------------------------------------------------------------------
# Dummy exposure: who decides, and who sees what
# ---------------------------------------------------------------------------


def test_current_player_is_declarer_for_declarers_own_and_the_dummys_cards() -> None:
    # The rule most easily got subtly wrong: declarer plays the dummy's cards, so the
    # dummy never appears as the player making a decision - not once in the whole deal -
    # while each opponent always decides for themselves.
    game = Bridge()
    state = scripted_state()
    decisions: list[tuple[int, int]] = []  # (hand played from, seat deciding)
    for plays, _winner in SCRIPT:
        for text in plays:
            decisions.append((state.to_play, game.current_player(state)))
            state = game.apply(state, card(text))

    assert len(decisions) == SEATS * TRICKS
    for to_play, decider in decisions:
        expected = DECLARER if to_play in (DECLARER, DUMMY) else to_play
        assert decider == expected
    assert all(decider != DUMMY for _to_play, decider in decisions)
    # and this isn't vacuous - the dummy's cards really are being played all deal
    assert sum(1 for to_play, _decider in decisions if to_play == DUMMY) == TRICKS


def test_declarer_chooses_from_both_hands_across_a_trick() -> None:
    # The other half of the same rule: on the two turns declarer decides, the cards
    # offered come from whichever hand is actually on turn, not always declarer's own.
    game = Bridge()
    state = scripted_state()
    offered: dict[int, list[Card]] = {}
    for text in SCRIPT[0][0]:
        if game.current_player(state) == DECLARER:
            offered[state.to_play] = game.legal_actions(state)
        state = game.apply(state, card(text))

    assert sorted(offered) == [DECLARER, DUMMY]
    assert set(offered[DECLARER]) <= set(scripted_hands()[DECLARER])
    assert set(offered[DUMMY]) <= set(scripted_hands()[DUMMY])


def test_information_set_shows_each_seat_its_own_hand_and_the_dummys() -> None:
    game = Bridge()
    state = scripted_state()
    hands = scripted_hands()

    # Spelled out seat by seat rather than looped, since the point is exactly which
    # hands each of the four can and cannot see.
    assert set(game.information_set(state, 0).hands) == {0, DUMMY}  # declarer
    assert set(game.information_set(state, 1).hands) == {1, DUMMY}  # an opponent
    assert set(game.information_set(state, 2).hands) == {DUMMY}  # the dummy's own view
    assert set(game.information_set(state, 3).hands) == {3, DUMMY}  # the other opponent

    for player in range(SEATS):
        view = game.information_set(state, player)
        assert view.player == player
        assert view.hands[player] == hands[player]
        assert view.hands[DUMMY] == hands[DUMMY]  # face up for everyone once exposed

    # Neither opponent ever sees the other's hand, and declarer sees neither of them.
    assert 3 not in game.information_set(state, 1).hands
    assert 1 not in game.information_set(state, 3).hands
    assert {1, 3}.isdisjoint(game.information_set(state, DECLARER).hands)


def test_information_set_carries_the_public_record_of_the_deal() -> None:
    # Trick history is public: what a persona is later allowed to remember of it is a
    # separate concern (ARCHITECTURE.md §3), not something withheld here.
    game = Bridge()
    state = scripted_state()
    for text in SCRIPT[0][0] + SCRIPT[1][0][:2]:
        state = game.apply(state, card(text))

    for player in range(SEATS):
        view = game.information_set(state, player)
        assert view.contract == CONTRACT
        assert view.completed == state.completed
        assert view.trick == state.trick
    assert len(state.completed) == 1
    assert len(state.trick) == 2


# ---------------------------------------------------------------------------
# The Game protocol
# ---------------------------------------------------------------------------


def test_bridge_satisfies_the_game_protocol() -> None:
    # Annotated deliberately: mypy checks this assignment, so a method drifting from
    # engine/game.py's contract fails type checking rather than waiting for a test to
    # notice it (ARCHITECTURE.md §8).
    game: Game[BridgeState, Card] = Bridge()
    state = new_deal(CONTRACT, rng=random.Random(0))

    assert game.current_player(state) in range(SEATS)
    assert len(game.legal_actions(state)) == CARDS_PER_SEAT  # on lead, anything goes
    assert not game.is_terminal(state)
    assert game.payoff(state) == dict.fromkeys(range(SEATS), 0.0)
    assert game.information_set(state, 0) is not None


def test_new_deal_deals_every_seat_a_full_hand() -> None:
    state = new_deal(CONTRACT, rng=random.Random(3))
    assert [len(hand) for hand in state.hands] == [CARDS_PER_SEAT] * SEATS
    assert state.trick == ()
    assert state.completed == ()


def test_a_contract_outside_the_table_or_the_deal_is_rejected() -> None:
    with pytest.raises(ValueError):
        Contract(trump=TRUMP, declarer=4, target=7)
    with pytest.raises(ValueError):
        Contract(trump=TRUMP, declarer=0, target=14)


def test_a_state_needs_one_hand_per_seat() -> None:
    with pytest.raises(ValueError):
        BridgeState(contract=CONTRACT, hands=((), (), ()))
