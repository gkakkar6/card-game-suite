import random

import pytest

from engine.cards import Card, Deck, Rank, Suit
from games.poker.betting import ActionType, BettingAction
from games.poker.hand import Blinds, DecisionView, play_hand

CALL = BettingAction(ActionType.CALL)
CHECK = BettingAction(ActionType.CHECK)
FOLD = BettingAction(ActionType.FOLD)


def C(rank: Rank, suit: Suit) -> Card:
    return Card(rank, suit)


class ScriptedStrategy:
    """Plays a fixed list of actions in order, recording what it was shown."""

    def __init__(self, *actions: BettingAction) -> None:
        self.actions = list(actions)
        self.views: list[DecisionView] = []

    def __call__(self, view: DecisionView) -> BettingAction:
        self.views.append(view)
        if not self.actions:
            raise AssertionError(f"player {view.player} ran out of scripted actions")
        return self.actions.pop(0)


def stacked_deck(
    player_cards: list[list[Card]], board: list[Card], burn: list[Card] | None = None
) -> Deck:
    """A deck that deals exactly these hole cards and board, in hand.py's deal order."""
    cards = [card for hand in player_cards for card in hand] + board
    return Deck(cards + (burn or []))


# Aces against kings on a board that helps neither, so the winner is unambiguous.
ACES = [C(Rank.ACE, Suit.CLUBS), C(Rank.ACE, Suit.DIAMONDS)]
KINGS = [C(Rank.KING, Suit.CLUBS), C(Rank.KING, Suit.DIAMONDS)]
DRY_BOARD = [
    C(Rank.TWO, Suit.HEARTS), C(Rank.SEVEN, Suit.SPADES), C(Rank.NINE, Suit.DIAMONDS),
    C(Rank.QUEEN, Suit.HEARTS), C(Rank.THREE, Suit.CLUBS),
]


def test_scripted_hand_runs_every_street_to_showdown() -> None:
    # Small blind calls, big blind checks, then both check down all three later streets.
    small_blind = ScriptedStrategy(CALL, CHECK, CHECK, CHECK)
    big_blind = ScriptedStrategy(CHECK, CHECK, CHECK, CHECK)
    result = play_hand(
        [small_blind, big_blind], deck=stacked_deck([ACES, KINGS], DRY_BOARD)
    )

    assert result.went_to_showdown
    assert result.board == tuple(DRY_BOARD)  # flop, turn and river all dealt
    assert [view.street for view in small_blind.views] == ["preflop", "flop", "turn", "river"]
    assert result.pot == 4  # two big blinds
    assert result.winners == (0,)
    assert result.net(0) == 2
    assert result.net(1) == -2
    assert sum(result.final_stacks) == sum(result.starting_stacks)  # chips conserved


def test_showdown_awards_the_pot_to_the_better_hand() -> None:
    # Same board, but now the aces sit in the big blind seat instead.
    result = play_hand(
        [ScriptedStrategy(CALL, CHECK, CHECK, CHECK), ScriptedStrategy(CHECK, CHECK, CHECK, CHECK)],
        deck=stacked_deck([KINGS, ACES], DRY_BOARD),
    )
    assert result.winners == (1,)
    assert result.net(1) == 2
    assert result.net(0) == -2


def test_folding_ends_the_hand_without_dealing_further_streets() -> None:
    result = play_hand(
        [ScriptedStrategy(FOLD), ScriptedStrategy()],
        deck=stacked_deck([ACES, KINGS], DRY_BOARD),
    )

    assert not result.went_to_showdown
    assert result.board == ()  # folded before the flop, so nothing else was dealt
    assert result.winners == (1,)
    assert result.pot == 3  # the two blinds
    assert result.net(1) == 1  # wins the small blind
    assert result.net(0) == -1  # loses the small blind they posted


def test_tie_splits_the_pot() -> None:
    # A royal flush on the board is the best hand for both players, whatever they hold.
    board_plays_itself = [
        C(Rank.ACE, Suit.SPADES), C(Rank.KING, Suit.SPADES), C(Rank.QUEEN, Suit.SPADES),
        C(Rank.JACK, Suit.SPADES), C(Rank.TEN, Suit.SPADES),
    ]
    result = play_hand(
        [ScriptedStrategy(CALL, CHECK, CHECK, CHECK), ScriptedStrategy(CHECK, CHECK, CHECK, CHECK)],
        deck=stacked_deck(
            [[C(Rank.TWO, Suit.CLUBS), C(Rank.THREE, Suit.DIAMONDS)],
             [C(Rank.FOUR, Suit.CLUBS), C(Rank.FIVE, Suit.DIAMONDS)]],
            board_plays_itself,
        ),
    )

    assert result.winners == (0, 1)
    assert result.pot == 4
    assert result.net(0) == 0  # each gets their two chips back
    assert result.net(1) == 0


def test_raising_moves_chips_and_is_awarded_at_showdown() -> None:
    # Small blind raises to 10 preflop, big blind calls, then both check it down.
    small_blind = ScriptedStrategy(
        BettingAction(ActionType.RAISE, amount=10), CHECK, CHECK, CHECK
    )
    big_blind = ScriptedStrategy(CALL, CHECK, CHECK, CHECK)
    result = play_hand([small_blind, big_blind], deck=stacked_deck([ACES, KINGS], DRY_BOARD))

    assert result.pot == 20
    assert result.winners == (0,)
    assert result.net(0) == 10
    assert result.net(1) == -10


def test_strategies_never_see_an_opponents_hole_cards() -> None:
    small_blind = ScriptedStrategy(CALL, CHECK, CHECK, CHECK)
    big_blind = ScriptedStrategy(CHECK, CHECK, CHECK, CHECK)
    play_hand([small_blind, big_blind], deck=stacked_deck([ACES, KINGS], DRY_BOARD))

    assert all(view.hole_cards == tuple(ACES) for view in small_blind.views)
    assert all(view.hole_cards == tuple(KINGS) for view in big_blind.views)


def test_pot_in_the_view_accumulates_across_streets() -> None:
    small_blind = ScriptedStrategy(CALL, CHECK, CHECK, CHECK)
    big_blind = ScriptedStrategy(CHECK, CHECK, CHECK, CHECK)
    play_hand([small_blind, big_blind], deck=stacked_deck([ACES, KINGS], DRY_BOARD))

    # preflop the small blind sees 3 (both blinds), and 4 on every later street
    assert [view.pot for view in small_blind.views] == [3, 4, 4, 4]


def test_blinds_are_a_named_structure_and_configurable() -> None:
    blinds = Blinds(small=5, big=10)
    assert blinds.forced_bets(2) == [5, 10]
    assert blinds.forced_bets(4) == [5, 10, 0, 0]

    result = play_hand(
        [ScriptedStrategy(CALL, CHECK, CHECK, CHECK), ScriptedStrategy(CHECK, CHECK, CHECK, CHECK)],
        blinds=blinds,
        deck=stacked_deck([ACES, KINGS], DRY_BOARD),
    )
    assert result.pot == 20


def test_starting_stacks_are_configurable() -> None:
    result = play_hand(
        [ScriptedStrategy(FOLD), ScriptedStrategy()],
        starting_stack=50,
        deck=stacked_deck([ACES, KINGS], DRY_BOARD),
    )
    assert result.starting_stacks == (50, 50)
    assert result.final_stacks == (49, 51)


def test_a_hand_needs_at_least_two_players() -> None:
    with pytest.raises(ValueError):
        play_hand([ScriptedStrategy()])


def _check_or_call(view: DecisionView) -> BettingAction:
    return CHECK if ActionType.CHECK in view.legal_actions else CALL


def _always_aggressive(view: DecisionView) -> BettingAction:
    # A bet has to exceed the current bet, which is not zero when the big blind is
    # already posted, so open to one bet above whatever is out there - but never past
    # what this player actually has, which for a short stack means all-in for less.
    own_bet = view.current_bet - view.to_call
    target = min(view.current_bet + view.min_bet, own_bet + view.stack)
    if target <= view.current_bet:
        return _check_or_call(view)  # too short to raise at all
    if ActionType.RAISE in view.legal_actions:
        return BettingAction(ActionType.RAISE, amount=target)
    if ActionType.BET in view.legal_actions:
        return BettingAction(ActionType.BET, amount=target)
    return _check_or_call(view)


def _folds_to_any_bet(view: DecisionView) -> BettingAction:
    return FOLD if view.to_call > 0 else CHECK


@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_random_hands_conserve_chips(num_players: int) -> None:
    # Real shuffled decks rather than stacked ones, so this covers deal order, seat
    # order and street progression together. Chips can move between players but must
    # never be created or destroyed.
    strategies = [_check_or_call, _always_aggressive, _folds_to_any_bet, _check_or_call]
    rng = random.Random(99)
    for _ in range(150):
        result = play_hand(strategies[:num_players], rng=rng)
        assert sum(result.final_stacks) == sum(result.starting_stacks)
        # Conservation alone would still pass with a stack driven negative, since the
        # chips subtracted from it land in the pot either way, so check the floor too.
        assert all(stack >= 0 for stack in result.final_stacks)
        assert result.pot >= 3  # at least both blinds
        assert len(result.winners) >= 1
        if result.went_to_showdown:
            assert len(result.board) == 5


@pytest.mark.parametrize("num_players", [2, 3])
def test_short_stacks_never_go_negative(num_players: int) -> None:
    # Uneven, very short stacks against an opponent who bets every chance: exactly the
    # setup where a call larger than the stack would otherwise overdraw it.
    rng = random.Random(7)
    for _ in range(150):
        stacks = [rng.randint(1, 12) for _ in range(num_players)]
        result = play_hand(
            [_always_aggressive, _check_or_call, _check_or_call][:num_players],
            starting_stacks=stacks,
            rng=rng,
        )
        assert all(stack >= 0 for stack in result.final_stacks)
        assert sum(result.final_stacks) == sum(stacks)
