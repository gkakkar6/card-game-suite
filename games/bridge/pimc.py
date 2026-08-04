"""PIMC (perfect-information Monte Carlo): estimate a decision's value by sampling
complete deals consistent with what's actually known, solving each one exactly with
games/bridge/solver.py, and averaging (ARCHITECTURE.md §5, bridge Phase 3).

Double-dummy solving is exact only because it assumes zero uncertainty - all four
hands known. This is what turns that into a real decision under genuine uncertainty:
sample many complete hands consistent with what's actually been observed, solve each
one exactly, combine.

Receding-horizon by design: every call samples and solves completely fresh, using only
what's genuinely known at that exact decision point - nothing from an earlier decision
is reused or extended. Each card actually played is real new evidence that narrows what
is still possible, so refitting from scratch is the correct thing to do here, not just a
simplification - and it means every individual solve is a complete, exact solve of
whatever is genuinely left, never artificially depth-limited, so no heuristic evaluator
is needed for this to work.

games/bridge/action_values.py is the actual decision-time entry point: it decides
*whether* to call into this module at all (the depth gate) and what to fall back to if
this comes back with nothing. This module only knows how to sample and average; it has
no notion of a depth gate or a fallback.
"""

import random
import time
from dataclasses import replace

from engine.cards import Card
from games.bridge.rules import Bridge, BridgeState
from games.bridge.solver import solve

# Starting points to tune against real measurements, not fixed requirements - see
# DECISIONS.md for the numbers these were checked against.
DEFAULT_TIME_BUDGET = 5.0  # seconds, a safety net for an unlucky batch, not the routine case
DEFAULT_MAX_SAMPLES = 30  # samples per decision, once real solving is actually attempted


def sample_unseen(
    game: Bridge, state: BridgeState, player: int, rng: random.Random
) -> BridgeState:
    """One uniformly random completion of what `player` cannot see.

    `player`'s own hand and the dummy's stay exactly as they are - information_set()
    says those are genuinely known. The other two hands are pooled together and
    redealt at random, into hands of the sizes they actually have. Hand size is
    always public in bridge (everyone can count how many cards are left in a hand
    from the play so far); only which specific unseen card sits in which of the two
    unseen hands is what's actually hidden, and that's exactly what gets scrambled.
    """
    info = game.information_set(state, player)
    unseen_seats = [seat for seat in range(len(state.hands)) if seat not in info.hands]
    if len(unseen_seats) != 2:
        raise AssertionError(f"expected exactly 2 unseen hands, got {len(unseen_seats)}")

    pool = [card for seat in unseen_seats for card in state.hands[seat]]
    rng.shuffle(pool)

    hands = list(state.hands)
    cut = len(state.hands[unseen_seats[0]])
    hands[unseen_seats[0]] = tuple(pool[:cut])
    hands[unseen_seats[1]] = tuple(pool[cut:])
    return replace(state, hands=tuple(hands))


def pimc_values(
    game: Bridge,
    state: BridgeState,
    *,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    time_budget: float = DEFAULT_TIME_BUDGET,
    rng: random.Random | None = None,
) -> dict[Card, float] | None:
    """Declarer's-side trick value per legal card, averaged over up to `max_samples`
    random completions each solved exactly - or None if zero samples finished inside
    `time_budget` seconds, which the caller falls back from.

    The budget is checked before starting each new sample, not partway through one -
    solve() isn't built to be interrupted mid-search, and a position that's already
    inside the depth gate is one the benchmarks say should solve quickly, so a sample
    that's already running is expected to finish rather than needing to be cut off.
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
