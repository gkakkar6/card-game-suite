"""Hand equity for hold'em: how often a player's hole cards win by showdown.

Street-aware method (ARCHITECTURE.md §5, poker week 2):
- turn/river (at most one unknown board card): exact enumeration, no sampling
- flop/preflop (too many completions to enumerate): Monte Carlo estimate

Heads-up only for now, and the opponent's unknown cards are drawn uniformly from
the remaining deck. Weighted ranges and multiway pots are stretch goals per §5.
"""

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations

from engine.cards import Card, standard_52
from games.poker.hand_evaluator import best_hand

HOLE_CARDS = 2
FULL_BOARD = 5
VALID_BOARD_SIZES = (0, 3, 4, 5)  # preflop, flop, turn, river
# once four board cards are known there are few enough completions to enumerate exactly
EXACT_ENUMERATION_MIN_BOARD = 4
DEFAULT_ITERATIONS = 10_000

# Placeholder for weighted/non-uniform opponent ranges. The parameter exists so callers
# and signatures don't have to change when ranges land; the weighting logic itself is
# deliberately not built yet (week 2 scope is uniform-random opponent cards).
OpponentRange = Sequence[tuple[Card, Card]]

WIN = 1
TIE = 0
LOSS = -1


@dataclass(frozen=True)
class Equity:
    """Showdown outcome counts for a player, and the equity they imply."""

    wins: int
    ties: int
    losses: int
    exact: bool  # True if enumerated exhaustively, False if Monte Carlo estimated

    @property
    def trials(self) -> int:
        return self.wins + self.ties + self.losses

    @property
    def equity(self) -> float:
        """Share of the pot this hand wins on average: win% + 0.5 x tie%."""
        if self.trials == 0:
            return 0.0
        return (self.wins + 0.5 * self.ties) / self.trials

    @property
    def win_probability(self) -> float:
        return self.wins / self.trials if self.trials else 0.0

    @property
    def tie_probability(self) -> float:
        return self.ties / self.trials if self.trials else 0.0

    @property
    def loss_probability(self) -> float:
        return self.losses / self.trials if self.trials else 0.0


def _remaining_deck(known: Sequence[Card]) -> list[Card]:
    """Every card not already visible to the player."""
    seen = set(known)
    return [card for card in standard_52() if card not in seen]


def _showdown(
    player_hole: Sequence[Card], opponent_hole: Sequence[Card], board: Sequence[Card]
) -> int:
    """WIN, TIE or LOSS from the player's point of view on a complete board."""
    player_rank = best_hand([*player_hole, *board])
    opponent_rank = best_hand([*opponent_hole, *board])
    if player_rank > opponent_rank:
        return WIN
    if player_rank < opponent_rank:
        return LOSS
    return TIE


def _enumerate_outcomes(
    player_hole: Sequence[Card],
    board: Sequence[Card],
    opponent_hole: Sequence[Card] | None,
    deck: Sequence[Card],
) -> Iterator[int]:
    """Every possible completion of the board (and opponent hand, if unknown)."""
    missing = FULL_BOARD - len(board)
    for extra in combinations(deck, missing):
        full_board = [*board, *extra]
        if opponent_hole is not None:
            yield _showdown(player_hole, opponent_hole, full_board)
            continue
        # opponent's cards come from whatever the board didn't consume
        undealt = [card for card in deck if card not in extra]
        player_rank = best_hand([*player_hole, *full_board])
        for opponent_cards in combinations(undealt, HOLE_CARDS):
            opponent_rank = best_hand([*opponent_cards, *full_board])
            if player_rank > opponent_rank:
                yield WIN
            elif player_rank < opponent_rank:
                yield LOSS
            else:
                yield TIE


def _simulate_outcomes(
    player_hole: Sequence[Card],
    board: Sequence[Card],
    opponent_hole: Sequence[Card] | None,
    deck: Sequence[Card],
    iterations: int,
    rng: random.Random,
) -> Iterator[int]:
    """Random completions of the board (and opponent hand, if unknown)."""
    missing = FULL_BOARD - len(board)
    # one sample covers the opponent's hole cards and the rest of the board at once,
    # which keeps the draw uniform over all disjoint (opponent, board) combinations
    needed = missing + (0 if opponent_hole is not None else HOLE_CARDS)
    for _ in range(iterations):
        drawn = rng.sample(list(deck), needed)
        if opponent_hole is None:
            opponent_cards: Sequence[Card] = drawn[:HOLE_CARDS]
            extra = drawn[HOLE_CARDS:]
        else:
            opponent_cards = opponent_hole
            extra = drawn
        yield _showdown(player_hole, opponent_cards, [*board, *extra])


def _tally(outcomes: Iterator[int], exact: bool) -> Equity:
    wins = ties = losses = 0
    for outcome in outcomes:
        if outcome == WIN:
            wins += 1
        elif outcome == TIE:
            ties += 1
        else:
            losses += 1
    return Equity(wins=wins, ties=ties, losses=losses, exact=exact)


def _validate(
    player_hole: Sequence[Card], board: Sequence[Card], opponent_hole: Sequence[Card] | None
) -> None:
    if len(player_hole) != HOLE_CARDS:
        raise ValueError(f"a player holds exactly {HOLE_CARDS} cards, got {len(player_hole)}")
    if len(board) not in VALID_BOARD_SIZES:
        raise ValueError(f"board must have one of {VALID_BOARD_SIZES} cards, got {len(board)}")
    if opponent_hole is not None and len(opponent_hole) != HOLE_CARDS:
        raise ValueError(f"an opponent holds exactly {HOLE_CARDS} cards, got {len(opponent_hole)}")
    known = [*player_hole, *board, *(opponent_hole or ())]
    if len(set(known)) != len(known):
        raise ValueError("the same card cannot appear twice")


def hand_equity(
    player_hole: Sequence[Card],
    board: Sequence[Card] = (),
    *,
    opponent_hole: Sequence[Card] | None = None,
    opponent_range: OpponentRange | None = None,
    iterations: int = DEFAULT_ITERATIONS,
    rng: random.Random | None = None,
) -> Equity:
    """Equity for `player_hole` against one opponent, given the board so far.

    Enumerates exactly on the turn and river; estimates by Monte Carlo on the flop and
    preflop, where `iterations` and `rng` apply. Pass `opponent_hole` to run against a
    specific holding, otherwise the opponent's cards are uniformly random.
    """
    if opponent_range is not None:
        raise NotImplementedError(
            "weighted opponent ranges are not implemented yet; opponent cards are uniform"
        )
    _validate(player_hole, board, opponent_hole)

    deck = _remaining_deck([*player_hole, *board, *(opponent_hole or ())])
    if len(board) >= EXACT_ENUMERATION_MIN_BOARD:
        return _tally(_enumerate_outcomes(player_hole, board, opponent_hole, deck), exact=True)
    outcomes = _simulate_outcomes(
        player_hole, board, opponent_hole, deck, iterations, rng or random.Random()
    )
    return _tally(outcomes, exact=False)
