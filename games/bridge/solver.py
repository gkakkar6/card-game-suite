"""Double-dummy solver: with all four hands face up, how many tricks can declarer's
side force against a defence that sees just as much?

Minimax over the cards that are left, with alpha-beta pruning, a transposition table
and equivalent-card reduction (ARCHITECTURE.md §5, bridge Phase 2). One consistent
viewpoint throughout: every value is a declarer's-side trick count, so declarer's own
turns maximize it and an opponent's turns minimize it. `Bridge.current_player()` says
which of the two a decision is - it returns declarer's seat for the dummy's cards as
well as declarer's own, and an opponent's seat otherwise - so the test is direct.

Alpha-beta is a correctness precondition here rather than a later speed extra: a full
deal's tree is far too large to search without it, so even checking the answer would
take impractically long. It changes nothing about the result, only which provably
irrelevant branches get skipped. Every optimization below can be switched off
independently, and the tests check that each combination gives identical answers to
plain unpruned minimax.

**Why this sits in games/bridge/ rather than engine/trick_taking/.** The search itself
is game-agnostic and Court Piece will want it unchanged. What is not game-agnostic is
what makes it fast: which trick a move just won (the `Game` protocol only exposes a
running total through `payoff()`, which walks the trick history), and which part of a
state determines future play (the transposition key below). Both would have to be
injected as hooks designed from a single example. Lifting this into
engine/trick_taking/ once Court Piece exists is a mechanical change made against two
real implementations, which is a better moment to design that seam than now.
"""

from dataclasses import dataclass

from engine.cards import Card
from engine.trick_taking.equivalence import play_groups
from engine.trick_taking.resolution import Seat
from games.bridge.rules import Bridge, BridgeState

INFINITY = float("inf")

# What determines play from here on, at the start of a trick: who leads and what
# everyone still holds. Deliberately *not* the completed-trick history - the whole point
# of the table is that the same remaining cards get reached by many different move
# orders, and those histories all differ. See `_Search.future_tricks` for why the values
# stored against this key are future tricks rather than totals.
_Key = tuple[Seat, tuple[tuple[Card, ...], ...]]


@dataclass(frozen=True)
class _Entry:
    """What is known about a position's value: somewhere in [low, high].

    Two bounds rather than one value, because a search that cut off only learns which
    side of its window the answer is on. Keeping both matters more than it sounds: the
    narrow-window searches below ask a position about one threshold at a time, so the
    same position gets asked repeatedly from different sides. With a single slot each
    answer overwrites the last, and the work gets redone; with both, the bounds close in
    on each other and the position is soon answered outright.
    """

    low: float
    high: float


@dataclass(frozen=True)
class Solution:
    """What the search found at one decision point."""

    values: dict[Card, float]  # declarer's-side tricks for the whole deal, per legal card
    best: Card  # the card the player to move would choose
    tricks: float  # what that card is worth, i.e. the value of the position
    nodes: int  # positions visited - the number Phase 2b's optimizations move


class _Search:
    """One run of the search.

    Holds the options, the transposition table and the node count in one place, so the
    recursive step doesn't have to thread six arguments through every call.
    """

    def __init__(
        self,
        game: Bridge,
        state: BridgeState,
        *,
        prune: bool,
        transpositions: bool,
        equivalence: bool,
        narrow: bool,
    ) -> None:
        self.game = game
        self.declarer = state.contract.declarer
        self.dummy = state.dummy
        self.prune = prune
        self.equivalence = equivalence
        self.narrow = narrow and prune  # a one-wide window does nothing without pruning
        # One table per search, so the contract is fixed for every entry in it and does
        # not need to be part of the key.
        self.table: dict[_Key, _Entry] | None = {} if transpositions else None
        self.nodes = 0

    def groups(self, state: BridgeState) -> list[list[Card]]:
        """Legal plays, grouped so only one card from each group has to be tried.

        Ordered highest card first. Alpha-beta cuts off as soon as it has seen a move
        good enough to make the rest irrelevant, so how quickly it finishes depends on
        how early the good moves are tried - and in a trick-taking game the high card is
        the obvious first guess for either side, since both are trying to win tricks.
        This changes nothing about the answer, only the order branches are visited in.
        """
        moves = self.game.legal_actions(state)
        if not self.equivalence or len(moves) < 2:
            groups = [[card] for card in moves]
        else:
            in_play = [card for hand in state.hands for card in hand]
            in_play.extend(card for _seat, card in state.trick)
            groups = play_groups(moves, in_play)
        groups.sort(key=lambda group: group[0].rank.value, reverse=True)
        return groups

    def step(self, state: BridgeState, card: Card) -> tuple[BridgeState, float]:
        """Play `card`, and report the trick it just won for declarer's side, if any."""
        child = self.game.apply(state, card)
        if len(child.completed) == len(state.completed):
            return child, 0.0
        return child, 1.0 if child.completed[-1].winner in (self.declarer, self.dummy) else 0.0

    def future_tricks(self, state: BridgeState, alpha: float, beta: float) -> float:
        """Tricks declarer's side wins from `state` on, with best play from both sides.

        Counted from here on rather than as a running total, deliberately: the same
        remaining cards are reached by many different move orders, and the future is the
        only part those orders have in common. A total would make every entry in the
        transposition table unique to the history that produced it, which is exactly the
        sharing the table exists to get.
        """
        self.nodes += 1
        # Nobody holds a card, so there is nothing left to win. This rather than
        # `game.is_terminal()` because an endgame built from part of a deal runs out
        # before thirteen tricks have been completed; for a full deal the two coincide.
        if not any(state.hands):
            return 0.0

        # Only positions at the start of a trick are keyed. They are where different move
        # orders actually converge, and hashing four hands every ply rather than every
        # fourth costs more than the extra hits are worth - hashing the key was the
        # single largest line in the profile before this. With the trick empty by
        # definition, it drops out of the key too.
        table = self.table if not state.trick else None
        key = (state.to_play, state.hands)
        if table is not None:
            entry = table.get(key)
            if entry is not None:
                if entry.low == entry.high:
                    return entry.low  # pinned down already
                # A bound answers the question outright when it falls outside the
                # window; otherwise it still narrows what is left to search.
                if entry.low >= beta:
                    return entry.low
                if entry.high <= alpha:
                    return entry.high
                alpha = max(alpha, entry.low)
                beta = min(beta, entry.high)

        # The window actually searched, which is what decides whether the answer below
        # comes back exact or only as a bound.
        window_low, window_high = alpha, beta
        maximizing = self.game.current_player(state) == self.declarer
        best = -INFINITY if maximizing else INFINITY

        for group in self.groups(state):
            child, gained = self.step(state, group[0])
            # The child counts its own future, which is this node's future less the
            # trick just won, so the window has to shift by the same amount - otherwise
            # the child's cut-offs would be measured against the wrong number.
            value = gained + self.future_tricks(child, alpha - gained, beta - gained)
            if maximizing:
                best = max(best, value)
                if self.prune:
                    alpha = max(alpha, best)
                    if alpha >= beta:
                        break
            else:
                best = min(best, value)
                if self.prune:
                    beta = min(beta, best)
                    if beta <= alpha:
                        break

        if table is not None:
            previous = table.get(key)
            low = previous.low if previous is not None else -INFINITY
            high = previous.high if previous is not None else INFINITY
            if best <= window_low:
                high = min(high, best)  # cut off low: the value is at most this
            elif best >= window_high:
                low = max(low, best)  # cut off high: the value is at least this
            else:
                low = high = best  # searched the whole window, so this is the answer
            table[key] = _Entry(low, high)
        return best

    def value(self, state: BridgeState, most: int) -> float:
        """The exact future value, given that it cannot exceed `most`.

        With `narrow` set this asks a series of yes/no questions - "is this worth at
        least k tricks?" - instead of one open one, and binary-searches k. Each question
        gives the search a window one trick wide, which is the narrowest it can be while
        still having an answer, so alpha-beta can cut off almost everywhere; four or so
        of those beat a single open search comfortably. Trick counts being whole numbers
        is what makes it work: there is nothing between "at least k" and "at most k-1".

        The answer is the same either way. Each question is still exact alpha-beta, and
        the transposition table carries the work of one question into the next.
        """
        if not self.narrow:
            return self.future_tricks(state, -INFINITY, INFINITY)

        low, high = 0, most
        while low < high:
            target = (low + high + 1) // 2
            if self.future_tricks(state, target - 1, target) >= target:
                low = target
            else:
                high = target - 1
        return float(low)


def solve(
    game: Bridge,
    state: BridgeState,
    *,
    prune: bool = True,
    transpositions: bool = True,
    equivalence: bool = True,
    narrow: bool = True,
) -> Solution:
    """Solve `state` exactly, and report what each legal card is worth.

    Works from any position, not just a fresh deal - mid-trick included - since that is
    how Phase 3's sampling and an eventual `action_values()` will call it.

    Values are whole-deal declarer's-side trick counts: tricks already in the bag plus
    what perfect play wins from here, so they compare directly against the contract's
    target. `best` is the card the player to move would actually choose, which means the
    highest value on declarer's turn and the lowest on an opponent's.

    Each root card is searched with a full window rather than the narrowing one
    alpha-beta uses between siblings. Narrowing would leave every card searched after
    the best one holding a bound instead of a value, and a value per action is the point
    of this function. Only the root ply gives that up - everything below prunes normally,
    and the transposition table carries most of the shared work from one root card to the
    next.
    """
    search = _Search(
        game,
        state,
        prune=prune,
        transpositions=transpositions,
        equivalence=equivalence,
        narrow=narrow,
    )
    groups = search.groups(state)
    if not groups:
        raise ValueError("nothing to solve: no cards left to play")

    won = game.payoff(state)[state.contract.declarer]
    remaining = max(len(hand) for hand in state.hands)  # nothing can be worth more
    values: dict[Card, float] = {}
    for group in groups:
        child, gained = search.step(state, group[0])
        total = won + gained + search.value(child, remaining)
        for card in group:
            # Cards left out of the search are interchangeable with the one that stood
            # in for them, which is exactly what makes them worth the same.
            values[card] = total

    if game.current_player(state) == state.contract.declarer:
        best = max(values, key=lambda card: values[card])
    else:
        best = min(values, key=lambda card: values[card])
    return Solution(values=values, best=best, tricks=values[best], nodes=search.nodes)
