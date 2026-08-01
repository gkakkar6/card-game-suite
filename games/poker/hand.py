"""Plays one complete hand of hold'em, from the deal through to the showdown.

Orchestration only (ARCHITECTURE.md §5, poker week 3): this module posts the forced
bets, deals each street, runs the betting rounds in order and awards the pot. It never
decides what a player does - every decision comes from the `Strategy` callables passed
in, which keeps scripted test hands and real decision logic interchangeable.

Side pots are handled: a player can only win chips up to their own total contribution
from each opponent, so a short stack going all-in for less than others keep betting
doesn't win money it never had a chance to risk. Each pot (main and side) is split
evenly between winners on a tie, with any odd chip going to the first winner by seat.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from engine.cards import Card, Deck
from games.poker.betting import ActionType, BettingAction, BettingRound
from games.poker.hand_evaluator import HandRank, best_hand

DEFAULT_SMALL_BLIND = 1
DEFAULT_BIG_BLIND = 2
DEFAULT_STARTING_STACK = 200
HOLE_CARDS_PER_PLAYER = 2
MIN_PLAYERS = 2

SMALL_BLIND_SEAT = 0
BIG_BLIND_SEAT = 1

# Raises are capped preflop and on the river, uncapped on the flop and turn, so a
# scripted or badly-tuned bot can't spiral into an unbounded raising war.
PREFLOP_MAX_RAISES = 3
RIVER_MAX_RAISES = 3


@dataclass(frozen=True)
class Blinds:
    """The forced bets posted before any cards are dealt.

    Only blinds exist today. Keeping them behind a named structure rather than inline
    means an ante (or any other forced bet) can be added later by extending
    `forced_bets` alone, without touching how a hand is played.
    """

    small: int = DEFAULT_SMALL_BLIND
    big: int = DEFAULT_BIG_BLIND

    def forced_bets(self, num_players: int) -> list[int]:
        """What each seat is forced to put in before the deal, in seat order."""
        bets = [0] * num_players
        bets[SMALL_BLIND_SEAT] = self.small
        bets[BIG_BLIND_SEAT] = self.big
        return bets


@dataclass(frozen=True)
class Street:
    name: str
    cards_dealt: int
    max_raises: int | None


STREETS = (
    Street("preflop", 0, PREFLOP_MAX_RAISES),
    Street("flop", 3, None),
    Street("turn", 1, None),
    Street("river", 1, RIVER_MAX_RAISES),
)


@dataclass(frozen=True)
class DecisionView:
    """Everything a strategy is allowed to see when choosing its action.

    Deliberately excludes other players' hole cards, so a strategy physically cannot
    read information a real player wouldn't have.
    """

    player: int
    street: str
    hole_cards: tuple[Card, ...]
    board: tuple[Card, ...]
    pot: int  # includes chips committed on earlier streets, not just this one
    to_call: int
    stack: int
    min_bet: int
    current_bet: int
    legal_actions: tuple[ActionType, ...]


class Strategy(Protocol):
    """Chooses an action for one decision. Scripted in tests, value-driven in play."""

    def __call__(self, view: DecisionView) -> BettingAction: ...


@dataclass(frozen=True)
class HandResult:
    board: tuple[Card, ...]
    hole_cards: tuple[tuple[Card, ...], ...]
    pot: int
    winners: tuple[int, ...]
    contenders: tuple[int, ...]  # players who did not fold, i.e. reached showdown
    starting_stacks: tuple[int, ...]
    final_stacks: tuple[int, ...]
    went_to_showdown: bool

    def net(self, player: int) -> int:
        """What the player won or lost on this hand."""
        return self.final_stacks[player] - self.starting_stacks[player]


def _first_actor(seats: Sequence[int], preferred: int) -> int:
    """Index into `seats` of the first seat at or after `preferred`, wrapping around."""
    for index, seat in enumerate(seats):
        if seat >= preferred:
            return index
    return 0  # everyone left sits before `preferred`, so action wraps to the lowest seat


def _seat_order(num_players: int) -> tuple[int, int]:
    """Seats that act first preflop and postflop.

    Heads-up the small blind is also the button, so it acts first preflop and last
    afterwards. With three or more the button sits just before the small blind.
    """
    button = SMALL_BLIND_SEAT if num_players == MIN_PLAYERS else num_players - 1
    return (BIG_BLIND_SEAT + 1) % num_players, (button + 1) % num_players


def _run_betting(
    street: Street,
    strategies: Sequence[Strategy],
    stacks: list[int],
    folded: set[int],
    hole_cards: Sequence[Sequence[Card]],
    board: Sequence[Card],
    pot_so_far: int,
    blinds: Blinds,
    first_seat: int,
    forced_bets: list[int] | None,
) -> int:
    """Run one street's betting. Mutates `stacks` and `folded`, returns chips added."""
    seats = [seat for seat in range(len(stacks)) if seat not in folded]
    if len(seats) < MIN_PLAYERS:
        return 0

    round_ = BettingRound(
        stacks=[stacks[seat] for seat in seats],
        first_to_act=_first_actor(seats, first_seat),
        min_bet=blinds.big,
        initial_bets=[forced_bets[seat] for seat in seats] if forced_bets else None,
        max_raises=street.max_raises,
    )
    while not round_.is_settled():
        index = round_.current_player
        seat = seats[index]
        view = DecisionView(
            player=seat,
            street=street.name,
            hole_cards=tuple(hole_cards[seat]),
            board=tuple(board),
            pot=pot_so_far + round_.pot,
            to_call=round_.current_bet - round_.bets[index],
            stack=round_.stacks[index],
            min_bet=round_.min_bet,
            current_bet=round_.current_bet,
            legal_actions=tuple(round_.legal_actions()),
        )
        round_.apply(strategies[seat](view))

    for index, seat in enumerate(seats):
        stacks[seat] = round_.stacks[index]
        if index in round_.folded:
            folded.add(seat)
    return round_.pot


def _showdown_winners(
    contenders: Sequence[int], hole_cards: Sequence[Sequence[Card]], board: Sequence[Card]
) -> tuple[int, ...]:
    ranks: dict[int, HandRank] = {
        seat: best_hand([*hole_cards[seat], *board]) for seat in contenders
    }
    best = max(ranks.values())
    return tuple(seat for seat in contenders if ranks[seat] == best)


def _award(pot: int, winners: Sequence[int], stacks: list[int]) -> None:
    share, odd_chip = divmod(pot, len(winners))
    for seat in winners:
        stacks[seat] += share
    stacks[min(winners)] += odd_chip  # odd chip goes to the first winner by seat


def _award_side_pots(
    contributions: Sequence[int],
    contenders: Sequence[int],
    hole_cards: Sequence[Sequence[Card]],
    board: Sequence[Card],
    stacks: list[int],
) -> tuple[int, ...]:
    """Award the pot in side-pot layers, keyed by each player's total contribution
    this hand (blinds plus every street). A short all-in can only win chips up to its
    own contribution level; money above that level forms a separate layer that only
    the players who put in that much can contest - so a short stack can never win
    more than it risked.

    Layers are built from every player's contribution, folded or not (their money is
    still real chips in the pot and has to go to someone), but only `contenders`
    (non-folded players) are eligible to win any given layer.

    Returns every seat that won at least one layer, in seat order - the natural
    generalisation of the single-pot case, where this reduces to exactly one layer.
    """
    levels = sorted(set(contributions))
    winners: set[int] = set()
    previous = 0
    for level in levels:
        if level == previous:
            continue
        layer_payers = sum(1 for contribution in contributions if contribution >= level)
        layer_size = (level - previous) * layer_payers
        eligible = [seat for seat in contenders if contributions[seat] >= level]
        if not eligible:
            # Should be unreachable: whoever set the contribution level that caused
            # everyone below it to fold or fall short is themselves a contender who
            # reaches at least that level, so every layer has at least one eligible
            # player. Guarded anyway rather than silently dropping chips.
            raise AssertionError(f"no eligible winners for a side-pot layer at {level}")
        layer_winners = _showdown_winners(eligible, hole_cards, board)
        _award(layer_size, layer_winners, stacks)
        winners.update(layer_winners)
        previous = level
    return tuple(sorted(winners))


def play_hand(
    strategies: Sequence[Strategy],
    *,
    blinds: Blinds | None = None,
    starting_stacks: Sequence[int] | None = None,
    starting_stack: int = DEFAULT_STARTING_STACK,
    deck: Deck | None = None,
    rng: random.Random | None = None,
) -> HandResult:
    """Play one hand and return how it finished.

    Pass `deck` to control exactly which cards come out (tests do this); otherwise a
    fresh deck is shuffled, seeded by `rng` when reproducibility matters.
    """
    if len(strategies) < MIN_PLAYERS:
        raise ValueError(f"a hand needs at least {MIN_PLAYERS} players")
    blinds = blinds if blinds is not None else Blinds()
    if starting_stacks is not None:
        stacks = list(starting_stacks)
    else:
        stacks = [starting_stack] * len(strategies)
    if len(stacks) != len(strategies):
        raise ValueError("one starting stack per player")
    initial_stacks = tuple(stacks)

    if deck is None:
        deck = Deck()
        deck.shuffle(rng)
    hole_cards = [deck.deal(HOLE_CARDS_PER_PLAYER) for _ in strategies]

    # blinds come out of the stacks first; betting.py expects them already deducted
    forced_bets = blinds.forced_bets(len(strategies))
    forced_bets = [min(forced, stacks[seat]) for seat, forced in enumerate(forced_bets)]
    for seat, forced in enumerate(forced_bets):
        stacks[seat] -= forced

    preflop_seat, postflop_seat = _seat_order(len(strategies))
    board: list[Card] = []
    folded: set[int] = set()
    pot = 0
    contributions = list(forced_bets)  # each player's total chips in, across the hand

    for street in STREETS:
        if len(strategies) - len(folded) < MIN_PLAYERS:
            break  # everyone else folded, no reason to deal any further
        board.extend(deck.deal(street.cards_dealt))
        stacks_before_street = list(stacks)
        pot += _run_betting(
            street,
            strategies,
            stacks,
            folded,
            hole_cards,
            board,
            pot,
            blinds,
            preflop_seat if street.name == "preflop" else postflop_seat,
            forced_bets if street.name == "preflop" else None,
        )
        for seat in range(len(strategies)):
            contributions[seat] += stacks_before_street[seat] - stacks[seat]

    contenders = [seat for seat in range(len(strategies)) if seat not in folded]
    went_to_showdown = len(contenders) > 1
    if went_to_showdown:
        winners = _award_side_pots(contributions, contenders, hole_cards, board, stacks)
    else:
        winners = tuple(contenders)
        _award(pot, winners, stacks)

    return HandResult(
        board=tuple(board),
        hole_cards=tuple(tuple(cards) for cards in hole_cards),
        pot=pot,
        winners=winners,
        contenders=tuple(contenders),
        starting_stacks=initial_stacks,
        final_stacks=tuple(stacks),
        went_to_showdown=went_to_showdown,
    )
