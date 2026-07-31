"""Quantal-response policy: turn action values into a choice between them.

Two independent knobs (ARCHITECTURE.md §3):

Temperature is noise. Near zero the best-valued action is taken almost always; as it
rises the choice flattens towards uniform random. It scales how *decisively* a player
acts, without changing which action they think is best.

Bias is a systematic shift, not noise - a per-action offset added to the value before
the choice is made. This is what separates a maniac from a nervous beginner: not more
randomness, but a consistent lean in a particular direction. Because it is applied to
the value, a large enough bias will choose an action the game honestly scored as bad.

Game-agnostic on purpose: values are keyed by whatever action type the game uses, so
nothing here knows about poker, bridge or Court Piece.
"""

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field

# Below this, temperature is treated as zero and the best action is taken outright.
# Dividing by a temperature this small overflows exp() long before it changes a choice.
MIN_TEMPERATURE = 1e-9


@dataclass(frozen=True)
class Persona[Action]:
    """A named playing style: how noisy it is, and which way it leans."""

    name: str
    temperature: float
    bias: Mapping[Action, float] = field(default_factory=dict)


def shift[Action](
    values: Mapping[Action, float], bias: Mapping[Action, float]
) -> dict[Action, float]:
    """Apply the persona's systematic lean to each action's value."""
    return {action: value + bias.get(action, 0.0) for action, value in values.items()}


def policy[Action](
    values: Mapping[Action, float],
    temperature: float,
    bias: Mapping[Action, float] | None = None,
) -> dict[Action, float]:
    """Probability of taking each action: softmax over value / temperature."""
    if not values:
        raise ValueError("need at least one action to choose between")
    shifted = shift(values, bias) if bias else dict(values)

    best = max(shifted.values())
    if temperature <= MIN_TEMPERATURE:
        # Deterministic: split evenly between actions tied for the best value.
        top = [action for action, value in shifted.items() if value == best]
        return {action: (1.0 / len(top) if action in top else 0.0) for action in shifted}

    # Subtracting the best value first keeps exp() in range; it cancels out once
    # the weights are normalised, so the distribution is unchanged.
    weights = {action: math.exp((value - best) / temperature) for action, value in shifted.items()}
    total = sum(weights.values())
    return {action: weight / total for action, weight in weights.items()}


def choose[Action](
    values: Mapping[Action, float],
    temperature: float,
    bias: Mapping[Action, float] | None = None,
    rng: random.Random | None = None,
) -> Action:
    """Sample one action from the quantal policy."""
    distribution = policy(values, temperature, bias)
    actions = list(distribution)
    weights = [distribution[action] for action in actions]
    return (rng or random.Random()).choices(actions, weights=weights, k=1)[0]


def choose_for[Action](
    persona: Persona[Action],
    values: Mapping[Action, float],
    rng: random.Random | None = None,
) -> Action:
    """Sample one action using a persona's own temperature and bias."""
    return choose(values, persona.temperature, persona.bias, rng)
