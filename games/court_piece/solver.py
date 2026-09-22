"""Double-dummy solver: with all four hands face up, how many tricks can the
`state.call.caller`-side force against a defence that sees just as much?

A direct copy of games/bridge/solver.py's algorithm - minimax over the cards left,
with alpha-beta pruning, a transposition table and equivalent-card reduction. The
search itself is genuinely game-agnostic (bridge's own module docstring says as
much); what changes here is only the one place bridge's version reads a `dummy`
property off the state. Court Piece has no dummy at all, so the caller's partner is
just `partner(caller)` computed directly, the same partnership math rules.py already
uses everywhere else.

One consistent viewpoint throughout: every value is the caller-side trick count, so
the caller's own turns (and their partner's) maximize it and an opponent's turns
minimize it - exactly bridge's convention, with "declarer" renamed to "caller" to
match Court Piece's own vocabulary (there is no bid to declare here).

Works unchanged for both trump variants. "Caller" here is just a fixed anchor seat
for that convention to stay consistent across one whole search, not necessarily
"whoever called trump" - see TrumpCall's own docstring (games/court_piece/rules.py)
for why running trump's caller field means "whoever leads first this hand" instead,
and why that's still the right seat to anchor this search on. The one thing that
genuinely differs for running trump is already handled below: trump can become
concrete *during* a search (the first void play sets it), so the transposition key
includes it - see `_Key`'s own comment for why that's not optional.
"""

from dataclasses import dataclass

from engine.cards import Card, Suit
from engine.trick_taking.equivalence import play_groups
from engine.trick_taking.resolution import Seat
from games.court_piece.rules import CourtPiece, CourtPieceState, partner

INFINITY = float("inf")

# What determines play from here on, at the start of a trick: who leads, what trump
# currently is, and what everyone still holds. Bridge's own _Key has no trump
# component, since bridge's trump is fixed for the whole search and would just be a
# constant added to every key - safe to omit there. Running trump's trump is NOT
# fixed for the whole search: two different move orders can play the exact same set
# of cards (arriving at identical `hands`) while having triggered the first-void rule
# on different tricks along the way, setting a different trump each time - a real
# case, not a hypothetical one, checked directly in
# games/court_piece/tests/test_solver.py. Omitting trump from the key would silently
# conflate those two genuinely different positions. Including it costs nothing for
# fixed trump, where it is one more constant component every entry already shares.
_Key = tuple[Seat, Suit | None, tuple[tuple[Card, ...], ...]]


@dataclass(frozen=True)
class _Entry:
    """What is known about a position's value: somewhere in [low, high]. See
    games/bridge/solver.py's _Entry for why two bounds instead of one."""

    low: float
    high: float


@dataclass(frozen=True)
class Solution:
    """What the search found at one decision point."""

    values: dict[Card, float]  # caller's-side tricks for the whole deal, per legal card
    best: Card  # the card the player to move would choose
    tricks: float  # what that card is worth, i.e. the value of the position
    nodes: int  # positions visited


class _Search:
    """One run of the search - see games/bridge/solver.py's _Search for the full
    rationale of each option; unchanged here beyond dropping the dummy reference."""

    def __init__(
        self,
        game: CourtPiece,
        state: CourtPieceState,
        *,
        prune: bool,
        transpositions: bool,
        equivalence: bool,
        narrow: bool,
    ) -> None:
        self.game = game
        self.caller = state.call.caller
        self.caller_partner = partner(self.caller)
        self.prune = prune
        self.equivalence = equivalence
        self.narrow = narrow and prune
        # One table per search, so the call is fixed for every entry in it and does
        # not need to be part of the key.
        self.table: dict[_Key, _Entry] | None = {} if transpositions else None
        self.nodes = 0

    def groups(self, state: CourtPieceState) -> list[list[Card]]:
        """Legal plays, grouped so only one card from each group has to be tried -
        identical reasoning to games/bridge/solver.py's own groups()."""
        moves = self.game.legal_actions(state)
        if not self.equivalence or len(moves) < 2:
            groups = [[card] for card in moves]
        else:
            in_play = [card for hand in state.hands for card in hand]
            in_play.extend(card for _seat, card in state.trick)
            groups = play_groups(moves, in_play)
        groups.sort(key=lambda group: group[0].rank.value, reverse=True)
        return groups

    def step(self, state: CourtPieceState, card: Card) -> tuple[CourtPieceState, float]:
        """Play `card`, and report the trick it just won for the caller's side, if any."""
        child = self.game.apply(state, card)
        if len(child.completed) == len(state.completed):
            return child, 0.0
        won_by_caller_side = child.completed[-1].winner in (self.caller, self.caller_partner)
        return child, 1.0 if won_by_caller_side else 0.0

    def future_tricks(self, state: CourtPieceState, alpha: float, beta: float) -> float:
        """Tricks the caller's side wins from `state` on, with best play from both
        sides - identical reasoning to games/bridge/solver.py's own future_tricks()."""
        self.nodes += 1
        if not any(state.hands):
            return 0.0

        table = self.table if not state.trick else None
        key = (state.to_play, state.call.trump, state.hands)
        if table is not None:
            entry = table.get(key)
            if entry is not None:
                if entry.low == entry.high:
                    return entry.low
                if entry.low >= beta:
                    return entry.low
                if entry.high <= alpha:
                    return entry.high
                alpha = max(alpha, entry.low)
                beta = min(beta, entry.high)

        window_low, window_high = alpha, beta
        maximizing = self.game.current_player(state) in (self.caller, self.caller_partner)
        best = -INFINITY if maximizing else INFINITY

        for group in self.groups(state):
            child, gained = self.step(state, group[0])
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
                high = min(high, best)
            elif best >= window_high:
                low = max(low, best)
            else:
                low = high = best
            table[key] = _Entry(low, high)
        return best

    def value(self, state: CourtPieceState, most: int) -> float:
        """The exact future value, given that it cannot exceed `most` - see
        games/bridge/solver.py's own value() for why narrow-window binary search
        works here (trick counts are whole numbers)."""
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
    game: CourtPiece,
    state: CourtPieceState,
    *,
    prune: bool = True,
    transpositions: bool = True,
    equivalence: bool = True,
    narrow: bool = True,
) -> Solution:
    """Solve `state` exactly, and report what each legal card is worth.

    Works from any position, not just a fresh deal - mid-trick included, the same as
    games/bridge/solver.py's solve(). Values are whole-deal caller's-side trick
    counts: tricks already in the bag plus what perfect play wins from here.
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

    payoff = game.payoff(state)
    won = payoff[state.call.caller]
    remaining = max(len(hand) for hand in state.hands)
    values: dict[Card, float] = {}
    for group in groups:
        child, gained = search.step(state, group[0])
        total = won + gained + search.value(child, remaining)
        for card in group:
            values[card] = total

    maximizing = game.current_player(state) in (state.call.caller, partner(state.call.caller))
    if maximizing:
        best = max(values, key=lambda card: values[card])
    else:
        best = min(values, key=lambda card: values[card])
    return Solution(values=values, best=best, tricks=values[best], nodes=search.nodes)
