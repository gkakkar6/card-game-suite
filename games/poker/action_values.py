"""What each legal action is worth, in chips, at one decision point.

This is poker's implementation of the action-value contract the persona layer consumes
(engine/game.py, ARCHITECTURE.md §3): given a state, score every legal action so a
policy can choose between them. Fold is fixed at 0.0 as the reference point, so a
positive value means "better than folding" and a negative one means "worse".

Two real limitations, both deliberate (ARCHITECTURE.md §5):

Single-street EV. Every value assumes the hand goes to showdown right now, from the
current board, with no further betting. It does not model how the hand might develop
over later streets - so implied odds, drawing hands and future bets are all invisible
to it.

No fold-equity model. Bet and raise values assume the opponent always calls, because
there is no model of how often an opponent folds to a given size. That is honest but
one-sided: it means this produces a "value bets only" style, never equilibrium-balanced
bluffing in the GTO sense. A persona's aggression bias can still choose an objectively
bad bet on purpose (§3), which is a real describable style - but it is not balanced
bluffing, and shouldn't be described as such.
"""

import random
from dataclasses import dataclass

from games.poker.betting import ActionType, BettingAction
from games.poker.equity import hand_equity
from games.poker.hand import DecisionView

# Sizing defaults. All three are parameters rather than constants baked into the
# formula, so a persona or an experiment can move them without touching this logic.
DEFAULT_REFERENCE = 0.5  # opening, "beats a coinflip"; facing a bet, pot odds replace it
DEFAULT_MIN_SIZE = 0.33  # smallest bet as a fraction of the pot
DEFAULT_MAX_SIZE = 1.5  # largest bet as a fraction of the pot

# Decision-time equity is tuned for speed, not precision: action_values() runs on every
# decision of every simulated hand, so the evaluation harness would crawl at
# equity.py's own defaults. Keeping this below the turn's 45,540 combinations forces
# the turn onto sampling while leaving the river (990) exact and fast. Individual
# estimates are noisier, which is fine - the harness aggregates over thousands of hands.
DECISION_MAX_EXACT_ENUMERATION = 5_000
DECISION_ITERATIONS = 500


@dataclass(frozen=True)
class Sizing:
    """How bet and raise sizes scale with equity above a reference point."""

    reference: float = DEFAULT_REFERENCE
    min_size: float = DEFAULT_MIN_SIZE
    max_size: float = DEFAULT_MAX_SIZE


@dataclass(frozen=True)
class ActionAnalysis:
    """Everything derived from one equity calculation at a decision point."""

    equity: float
    values: dict[ActionType, float]
    raise_to: int  # total this player would put in when betting or raising


def _reference_equity(view: DecisionView, sizing: Sizing) -> float:
    """The equity a hand has to beat before it is worth sizing up.

    Opening, that is the configured reference. Facing a bet, the pot odds break-even
    is the honest bar instead - the equity at which calling is exactly break-even.
    """
    if view.to_call == 0:
        return sizing.reference
    return view.to_call / (view.pot + view.to_call)


def size_fraction(equity: float, reference: float, sizing: Sizing) -> float:
    """Bet size as a fraction of the pot, scaling with equity above `reference`.

    Clamped at zero from below so a hand worse than the reference still gets the
    minimum size rather than a negative one - a bluff needs a sensible size even
    though its value is honestly negative.
    """
    headroom = 1.0 - reference
    strength = 0.0 if headroom <= 0 else max(0.0, (equity - reference) / headroom)
    return sizing.min_size + strength * (sizing.max_size - sizing.min_size)


def _raise_target(view: DecisionView, equity: float, sizing: Sizing) -> int:
    """Total chips this player would have in the pot after betting or raising."""
    own_bet = view.current_bet - view.to_call
    fraction = size_fraction(equity, _reference_equity(view, sizing), sizing)
    wanted = view.current_bet + round(fraction * view.pot)
    smallest = view.current_bet + view.min_bet  # betting.py's minimum-raise legality
    largest = own_bet + view.stack  # everything this player has left
    return min(max(wanted, smallest), largest)


def _call_value(view: DecisionView, equity: float) -> float:
    """Win the pot as it stands with probability `equity`, having paid to_call."""
    return equity * (view.pot + view.to_call) - view.to_call


def _aggressive_value(view: DecisionView, equity: float, raise_to: int) -> float:
    """Value of betting or raising to `raise_to`, assuming the opponent always calls."""
    own_bet = view.current_bet - view.to_call
    added = raise_to - own_bet  # what this player puts in now
    called = raise_to - view.current_bet  # what the opponent matches
    return equity * (view.pot + added + called) - added


def analyse(
    view: DecisionView,
    *,
    sizing: Sizing | None = None,
    iterations: int = DECISION_ITERATIONS,
    max_exact_enumeration: int = DECISION_MAX_EXACT_ENUMERATION,
    rng: random.Random | None = None,
    equity: float | None = None,
) -> ActionAnalysis:
    """Score every legal action from a single equity calculation.

    Pass `equity` to reuse an already-computed number - the same hole cards and board
    give the same equity however many times a player acts on that street.
    """
    sizing = sizing if sizing is not None else Sizing()
    if equity is None:
        equity = hand_equity(
            view.hole_cards,
            view.board,
            iterations=iterations,
            max_exact_enumeration=max_exact_enumeration,
            rng=rng,
        ).equity

    raise_to = _raise_target(view, equity, sizing)
    values: dict[ActionType, float] = {}
    for action in view.legal_actions:
        if action is ActionType.FOLD:
            values[action] = 0.0
        elif action in (ActionType.CHECK, ActionType.CALL):
            values[action] = _call_value(view, equity)
        else:  # BET or RAISE, always scored when chip-legal, never hidden
            values[action] = _aggressive_value(view, equity, raise_to)
    return ActionAnalysis(equity=equity, values=values, raise_to=raise_to)


def action_values(
    view: DecisionView,
    *,
    sizing: Sizing | None = None,
    iterations: int = DECISION_ITERATIONS,
    max_exact_enumeration: int = DECISION_MAX_EXACT_ENUMERATION,
    rng: random.Random | None = None,
) -> dict[ActionType, float]:
    """The action-value mapping the persona layer consumes."""
    return analyse(
        view,
        sizing=sizing,
        iterations=iterations,
        max_exact_enumeration=max_exact_enumeration,
        rng=rng,
    ).values


def action_for(action: ActionType, analysis: ActionAnalysis) -> BettingAction:
    """Turn a chosen action type into a playable action, sized when aggressive."""
    if action in (ActionType.BET, ActionType.RAISE):
        return BettingAction(action, amount=analysis.raise_to)
    return BettingAction(action)
