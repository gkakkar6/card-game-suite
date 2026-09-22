"""The fixed-trump declare-phase decision: which suit to call as trump, from a
5-card preview, with no partner input at all (that silence is a deliberate fairness
rule of the real game, not an omission here).

Shaped the same way bridge's bid_values()/action_values() and Court Piece's own
action_values() are: a value per legal choice (games/court_piece/hand_evaluation.py's
suit_score(), for every suit - all four are always "legal" since the call is
mandatory), not a function that returns one chosen suit directly. That's what lets a
persona's bias/temperature sit on top later the same way it already does for card
play, rather than the decision being hardcoded to "always pick the best score."
"""

import random
from collections.abc import Callable, Sequence
from typing import Protocol

from engine.cards import Card, Suit
from engine.personas.quantal import choose
from games.court_piece.hand_evaluation import suit_score

DeclareBias = Callable[[dict[Suit, float]], dict[Suit, float]]


def _no_declare_bias(values: dict[Suit, float]) -> dict[Suit, float]:
    return {}


def declare_values(preview: Sequence[Card]) -> dict[Suit, float]:
    """Every suit's trump-candidate score from a 5-card (or any partial-hand) preview."""
    return {suit: suit_score(preview, suit) for suit in Suit}


class DeclareStrategy(Protocol):
    """Chooses the trump suit for one seat's declare-phase decision."""

    def __call__(self, preview: Sequence[Card]) -> Suit: ...


class QuantalDeclareStrategy:
    """Scores every suit with declare_values(), then lets temperature/bias choose -
    the same two-knob mechanism as every other decision in this project
    (engine/personas/quantal.py), not a bespoke rule for this one domain."""

    def __init__(
        self,
        *,
        temperature: float,
        bias: DeclareBias = _no_declare_bias,
        rng: random.Random | None = None,
    ) -> None:
        self.temperature = temperature
        self.bias = bias
        self.rng = rng if rng is not None else random.Random()

    def __call__(self, preview: Sequence[Card]) -> Suit:
        values = declare_values(preview)
        return choose(values, self.temperature, self.bias(values), self.rng)
