"""PIMC (perfect-information Monte Carlo): estimate a decision's value by sampling
complete deals consistent with what's actually known, solving each one exactly with
games/court_piece/solver.py, and averaging - the same technique as
games/bridge/pimc.py, generalized from bridge's two unseen hands to Court Piece's
three, since there is no dummy here to make one of them public.

Not having a dummy means one more unknown hand per decision - a larger space of
consistent deals to sample from than bridge's. That turned out not to change the
wall-clock depth gate itself (games/court_piece/action_values.py's DEPTH_GATE
measures out to the same 6 as bridge's - raw solve() cost per sample doesn't depend
on how many hands were unseen before sampling, see that module's docstring); what it
may still cost is sample *adequacy*, a decision-quality question left open, not a
timing one.

Receding-horizon by design, same as bridge's: every call samples and solves
completely fresh from what's genuinely known at that exact decision point.
"""

import random
import time
from dataclasses import replace

from engine.cards import Card
from games.court_piece.rules import CourtPiece, CourtPieceState
from games.court_piece.solver import solve

DEFAULT_TIME_BUDGET = 5.0  # seconds, a safety net for an unlucky batch
DEFAULT_MAX_SAMPLES = 30  # samples per decision, once real solving is attempted


def sample_unseen(
    game: CourtPiece, state: CourtPieceState, player: int, rng: random.Random
) -> CourtPieceState:
    """One uniformly random completion of what `player` cannot see.

    `player`'s own hand stays exactly as it is. The other three hands - partner and
    both opponents, none of them public the way bridge's dummy is - are pooled
    together and redealt at random, into hands of the sizes they actually have. Hand
    size is always public (everyone can count how many cards are left in a hand from
    the play so far); only which specific unseen card sits in which of the three
    unseen hands is what's actually hidden.
    """
    info = game.information_set(state, player)
    unseen_seats = [seat for seat in range(len(state.hands)) if seat not in info.hands]
    if len(unseen_seats) != 3:
        raise AssertionError(f"expected exactly 3 unseen hands, got {len(unseen_seats)}")

    pool = [card for seat in unseen_seats for card in state.hands[seat]]
    rng.shuffle(pool)

    hands = list(state.hands)
    cursor = 0
    for seat in unseen_seats:
        size = len(state.hands[seat])
        hands[seat] = tuple(pool[cursor : cursor + size])
        cursor += size
    return replace(state, hands=tuple(hands))


def pimc_values(
    game: CourtPiece,
    state: CourtPieceState,
    *,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    time_budget: float = DEFAULT_TIME_BUDGET,
    rng: random.Random | None = None,
) -> dict[Card, float] | None:
    """Caller's-side trick value per legal card, averaged over up to `max_samples`
    random completions each solved exactly - or None if zero samples finished inside
    `time_budget` seconds, which the caller falls back from. Identical shape to
    games/bridge/pimc.py's pimc_values(), just pointed at Court Piece's own solver
    and 3-way sample_unseen().
    """
    rng = rng if rng is not None else random.Random()
    player = game.current_player(state)
    deadline = time.monotonic() + time_budget

    totals: dict[Card, float] = {}
    completed = 0
    for _ in range(max_samples):
        if time.monotonic() >= deadline:
            break
        sampled = sample_unseen(game, state, player, rng)
        solution = solve(game, sampled)
        for card, value in solution.values.items():
            totals[card] = totals.get(card, 0.0) + value
        completed += 1

    if completed == 0:
        return None
    return {card: total / completed for card, total in totals.items()}
