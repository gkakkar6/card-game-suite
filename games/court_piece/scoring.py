"""Court Piece scoring: what one played hand is worth, in courts.

Nothing like games/bridge/scoring.py's point-scale contract scoring transfers here -
that whole module is genuinely bridge-specific (bid targets, per-suit trick values,
over/undertricks). Court Piece's own rule, settled in conversation rather than
sourced from a generic reference, since it's how this actually gets played at the
table:

**Only the winning side ever scores anything** - the losing side's score is always
zero, never negative, for a given hand. What a win is worth starts from **how** it
was won: a "piece" (7 or more tricks, any other way) is worth `PIECE_VALUE` of a
court; a "kot" - all 13 tricks, or the first 7 won consecutively from the very first
trick - is worth a full court (`KOT_VALUE` = 1.0). `score_running_hand()` stops
there.

**Fixed trump only** (`score_hand()`) stacks a second, independent multiplier on top:
**who** won it. The trump-caller's own side winning is worth `CALLER_MULTIPLIER`
(less impressive - they had the information advantage of choosing trump), the
non-calling side winning it off them is worth `NON_CALLER_MULTIPLIER` (more
impressive - twice as much, a clean multiplier rather than bridge's uneven 1/3-vs-2/4
scale). Running trump scores flat, with no such multiplier - there is no caller to
apply it against, since nobody deliberately chose trump.

Session-level match length (first-to-N-courts, or open-ended free play) is
session.py's concern, not this module's - this module only ever scores one hand.
"""

from dataclasses import dataclass

from engine.trick_taking.resolution import Seat
from games.court_piece.rules import CompletedTrick, TrumpCall, partner

PIECE_VALUE = 0.1  # a plain win: 7+ tricks, not won as a straight run from trick 1
KOT_VALUE = 1.0  # all 13 tricks, or the first 7 straight from trick 1

WIN_THRESHOLD = 7  # of 13 tricks - the majority either side needs to win the hand
STRAIGHT_KOT_TRICKS = 7  # tricks won consecutively from trick 1 that also count as kot

CALLER_MULTIPLIER = 1.0  # the trump-caller's own side winning - less impressive
NON_CALLER_MULTIPLIER = 2.0  # the non-calling side winning it off them


@dataclass(frozen=True)
class HandScore:
    """One played hand's outcome. `winner` is a representative seat of the winning
    side (both seats of a side are worth the same, the same convention
    CourtPiece.payoff() uses for raw trick counts) - `None` only if neither side
    reached `WIN_THRESHOLD`, which real, complete play never produces (13 tricks
    always gives one side at least 7). `is_kot` names *how* it was won; `courts` is
    the actual number this hand is worth, already multiplier-applied where one
    applies."""

    winner: Seat | None
    tricks_won: int
    is_kot: bool
    courts: float


def _is_kot(completed: tuple[CompletedTrick, ...], winner: Seat) -> bool:
    """All 13 tricks, or the winning side took the first `STRAIGHT_KOT_TRICKS`
    tricks with no interruption - either counts, and both really do require looking
    at the ordered sequence, not just the final count, which is exactly why
    CourtPieceState.completed is kept in play order rather than only a tally."""
    winner_side = {winner, partner(winner)}
    if all(trick.winner in winner_side for trick in completed):
        return True  # every trick, not just a majority
    straight = completed[:STRAIGHT_KOT_TRICKS]
    return len(straight) == STRAIGHT_KOT_TRICKS and all(
        trick.winner in winner_side for trick in straight
    )


def _winning_side(completed: tuple[CompletedTrick, ...]) -> tuple[frozenset[Seat] | None, int]:
    """Which of the two fixed partnerships - {0, 2} or {1, 3} - reached
    `WIN_THRESHOLD`, and how many tricks they took. `None` if neither did (only
    possible from an incomplete trick history - shared by both scoring functions
    below, since which side won is exactly the same question either way)."""
    counts: dict[Seat, int] = {}
    for trick in completed:
        counts[trick.winner] = counts.get(trick.winner, 0) + 1

    side_a: frozenset[Seat] = frozenset({0, 2})
    side_b: frozenset[Seat] = frozenset({1, 3})
    tricks_a = sum(count for seat, count in counts.items() if seat in side_a)
    tricks_b = len(completed) - tricks_a

    if tricks_a >= WIN_THRESHOLD:
        return side_a, tricks_a
    if tricks_b >= WIN_THRESHOLD:
        return side_b, tricks_b
    return None, max(tricks_a, tricks_b)


def score_hand(call: TrumpCall, completed: tuple[CompletedTrick, ...]) -> HandScore:
    """Score one complete fixed-trump Court Piece hand from its full, ordered trick
    history - the caller/non-caller multiplier applies here, and only here."""
    side, tricks_won = _winning_side(completed)
    if side is None:
        return HandScore(winner=None, tricks_won=tricks_won, is_kot=False, courts=0.0)

    caller_side = call.caller in side
    winner = call.caller if caller_side else min(side)
    is_kot = _is_kot(completed, winner)
    base = KOT_VALUE if is_kot else PIECE_VALUE
    multiplier = CALLER_MULTIPLIER if caller_side else NON_CALLER_MULTIPLIER
    return HandScore(winner=winner, tricks_won=tricks_won, is_kot=is_kot, courts=base * multiplier)


def score_running_hand(completed: tuple[CompletedTrick, ...]) -> HandScore:
    """Score one complete running-trump Court Piece hand - the same kot/piece
    distinction as fixed trump, but no caller/non-caller multiplier: nobody
    deliberately chose trump, so there is no information advantage to score against."""
    side, tricks_won = _winning_side(completed)
    if side is None:
        return HandScore(winner=None, tricks_won=tricks_won, is_kot=False, courts=0.0)

    winner = min(side)  # an arbitrary but consistent representative - no caller to prefer
    is_kot = _is_kot(completed, winner)
    courts = KOT_VALUE if is_kot else PIECE_VALUE
    return HandScore(winner=winner, tricks_won=tricks_won, is_kot=is_kot, courts=courts)
