"""Bridge's decision-time value estimate - mirrors the role of
games/poker/action_values.py: the single entry point that turns a state into a value
per legal action, so a policy has something to choose between (ARCHITECTURE.md §5,
bridge Phase 3).

A depth gate, not a blind time budget, decides how to answer, informed by real Phase 2
benchmarks (scripts/benchmark_solver.py) rather than guessed:

  - `DEPTH_GATE` tricks remaining or fewer: games/bridge/pimc.py samples several
    complete deals consistent with what's actually known, solves each one exactly,
    and averages. A time budget underneath that is a safety net for an unlucky batch,
    not the routine case.
  - More than `DEPTH_GATE` tricks remaining: skip straight to the myopic single-trick
    heuristic in games/bridge/trick_odds.py - a real solve at that depth isn't
    attempted at all, not merely capped by a budget that would rarely finish anyway.
  - The same heuristic is also what's used if literally zero samples finish inside
    the PIMC budget even at `DEPTH_GATE` tricks or fewer - meant to stay rare, but
    handled rather than left to error or hang.

One honest, real limitation, worth stating plainly: the two paths return values on
different scales. PIMC's values are whole-deal declarer's-side trick counts (the same
units solve() itself uses); the fallback's are single-trick win probabilities, 0 to 1.
Nothing here converts between them - a caller only ever sees one or the other for a
given decision, never both mixed together, so comparing actions *within* one call is
still sound. Making the two scales consistent with each other is a real problem for
whatever normalises these values before persona bias/temperature gets applied
(ARCHITECTURE.md §3's normalisation rule) - deliberately not solved here, since that
layer doesn't exist yet.
"""

import random

from engine.cards import Card
from games.bridge.pimc import DEFAULT_MAX_SAMPLES, DEFAULT_TIME_BUDGET, pimc_values
from games.bridge.rules import Bridge, BridgeState
from games.bridge.trick_odds import trick_win_probabilities

# Tricks remaining, at or below which a real solve is attempted at all. Checked against
# real measured PIMC batch behaviour, not just single-solve cost - see DECISIONS.md for
# the numbers and why this needed lowering from the plan's original starting value of 9.
# Half a deal, and every decision re-samples and re-solves fresh anyway (receding
# horizon), so a conservative gate costs little - a couple more early decisions fall
# through to the fallback heuristic, which is itself a real, principled estimate, not
# a guess.
DEPTH_GATE = 6


def action_values(
    game: Bridge,
    state: BridgeState,
    *,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    time_budget: float = DEFAULT_TIME_BUDGET,
    rng: random.Random | None = None,
) -> dict[Card, float]:
    """The value mapping a policy would choose between - matching poker's
    action_values() shape (dict[Action, float]), one value per legal card here.
    """
    tricks_remaining = max(len(hand) for hand in state.hands)
    if tricks_remaining <= DEPTH_GATE:
        estimate = pimc_values(
            game, state, max_samples=max_samples, time_budget=time_budget, rng=rng
        )
        if estimate is not None:
            return estimate
    return trick_win_probabilities(game, state)
