"""Court Piece's decision-time value estimate - the single entry point that turns a
state into a value per legal action, mirroring games/bridge/action_values.py's role
and depth-gate design exactly, including the gate's actual value.

Before this was measured, the assumption here was that Court Piece needed a tighter
gate than bridge's 6, since PIMC samples three unseen hands per decision instead of
bridge's two - a larger space of consistent deals at the same trick count.
`scripts/benchmark_court_piece_solver.py` (mirroring bridge's own
scripts/benchmark_solver.py) shows that assumption doesn't actually hold for *this*
number: raw solve() cost at a given trick count is essentially identical to bridge's
own measured numbers, because it's the same algorithm operating on an already-fully-
specified sampled deal - solving doesn't care how many hands were conceptually unseen
before PIMC sampled them, only how many tricks remain. At the benchmarked numbers, a
30-sample PIMC batch fits comfortably inside the 5s budget at 6 tricks remaining
(~0.10s/solve, ~3.1s total) but not at 7 (~0.32s/solve, ~9.7s total) - the exact same
cutoff bridge's own benchmark landed on, for the exact same reason.

What genuinely differs from bridge, and this benchmark doesn't measure, is sample
*adequacy*: whether 30 samples represents a 3-unseen-hand permutation space as well as
it represents bridge's 2-hand one. That's a decision-quality question, not a
wall-clock one, and is a real, separate thing to check later (e.g. against actual
played-hand outcomes) - not a reason to lower this gate below what timing supports.
"""

import random
from dataclasses import dataclass

from engine.cards import Card
from games.court_piece.pimc import DEFAULT_MAX_SAMPLES, DEFAULT_TIME_BUDGET, pimc_values
from games.court_piece.rules import CourtPiece, CourtPieceState
from games.court_piece.trick_odds import trick_win_probabilities

# Tricks remaining, at or below which a real solve is attempted at all. Measured
# against scripts/benchmark_court_piece_solver.py, not guessed - see the module
# docstring for why this landed on the same value as bridge's own DEPTH_GATE.
DEPTH_GATE = 6


@dataclass(frozen=True)
class ActionAnalysis:
    """Everything derived from one action_values() call - the values themselves,
    plus the scale a persona should divide by before applying bias/temperature
    (ARCHITECTURE.md §3's normalisation rule). Identical shape to bridge's own
    ActionAnalysis, for the same reason: PIMC and the fallback land on different
    scales, and a caller needs to know which one it got."""

    values: dict[Card, float]
    scale: float  # divide values by this before bias/temperature - never zero


def analyse(
    game: CourtPiece,
    state: CourtPieceState,
    *,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    time_budget: float = DEFAULT_TIME_BUDGET,
    rng: random.Random | None = None,
) -> ActionAnalysis:
    """Score every legal card, and report which scale the scores are actually on."""
    tricks_remaining = max(len(hand) for hand in state.hands)
    if tricks_remaining <= DEPTH_GATE:
        estimate = pimc_values(
            game, state, max_samples=max_samples, time_budget=time_budget, rng=rng
        )
        if estimate is not None:
            return ActionAnalysis(values=estimate, scale=max(tricks_remaining, 1))
    return ActionAnalysis(values=trick_win_probabilities(game, state), scale=1.0)


def action_values(
    game: CourtPiece,
    state: CourtPieceState,
    *,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    time_budget: float = DEFAULT_TIME_BUDGET,
    rng: random.Random | None = None,
) -> dict[Card, float]:
    """The value mapping a policy would choose between - one value per legal card."""
    return analyse(
        game, state, max_samples=max_samples, time_budget=time_budget, rng=rng
    ).values
