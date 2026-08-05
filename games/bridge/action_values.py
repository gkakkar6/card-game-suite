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
still sound. `analyse()` is what makes the two scales usable together despite this: it
reports which scale a given call's values actually landed on, so games/bridge/
personas.py can normalise correctly either way before applying persona bias/
temperature (ARCHITECTURE.md §3's normalisation rule) - `action_values()` itself still
returns only the values, unchanged, for anything that just wants the numbers.
"""

import random
from dataclasses import dataclass

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


@dataclass(frozen=True)
class ActionAnalysis:
    """Everything derived from one action_values() call - the values themselves,
    plus the scale a persona should divide by before applying bias/temperature to
    them (ARCHITECTURE.md §3's normalisation rule).

    Exists because a caller outside this module has no reliable way to tell, from the
    values dict alone, which of the two scales it landed on: PIMC's whole-deal
    declarer's-side trick totals, or the fallback's single-trick win probabilities
    (0-1, already its own natural scale, so its scale is just 1.0). Re-deriving which
    path fired by calling this a second time would double the cost and, worse, could
    land on a *different* branch than the first call did - PIMC's own zero-samples
    fallback depends partly on wall-clock time, not purely on the state, so the same
    call made twice is not guaranteed to take the same path. One call, one answer.
    """

    values: dict[Card, float]
    scale: float  # divide values by this before bias/temperature - never zero


def analyse(
    game: Bridge,
    state: BridgeState,
    *,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    time_budget: float = DEFAULT_TIME_BUDGET,
    rng: random.Random | None = None,
) -> ActionAnalysis:
    """Score every legal card, and report which scale the scores are actually on.

    PIMC's values can range beyond a plain 0-1 span (they include tricks already
    banked, not just what is still undecided), so the natural scale for them is
    `tricks_remaining` - the most any decision at this depth could actually swing by -
    matching poker's own "divide by what's at stake" convention (games/poker/
    personas.py) rather than a fixed constant that would mean something different at
    every depth.
    """
    tricks_remaining = max(len(hand) for hand in state.hands)
    if tricks_remaining <= DEPTH_GATE:
        estimate = pimc_values(
            game, state, max_samples=max_samples, time_budget=time_budget, rng=rng
        )
        if estimate is not None:
            return ActionAnalysis(values=estimate, scale=max(tricks_remaining, 1))
    return ActionAnalysis(values=trick_win_probabilities(game, state), scale=1.0)


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
    return analyse(
        game, state, max_samples=max_samples, time_budget=time_budget, rng=rng
    ).values
