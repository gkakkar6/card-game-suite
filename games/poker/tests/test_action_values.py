import random

import pytest

from engine.cards import Card, Rank, Suit
from games.poker import equity as equity_module
from games.poker.action_values import (
    DECISION_ITERATIONS,
    DECISION_MAX_EXACT_ENUMERATION,
    Sizing,
    action_for,
    action_values,
    analyse,
    size_fraction,
)
from games.poker.betting import ActionType
from games.poker.hand import DecisionView


def C(rank: Rank, suit: Suit) -> Card:
    return Card(rank, suit)


ACES = (C(Rank.ACE, Suit.CLUBS), C(Rank.ACE, Suit.DIAMONDS))
RAGS = (C(Rank.SEVEN, Suit.CLUBS), C(Rank.TWO, Suit.DIAMONDS))


def view(
    *,
    hole: tuple[Card, ...] = ACES,
    board: tuple[Card, ...] = (),
    pot: int = 10,
    to_call: int = 0,
    stack: int = 200,
    min_bet: int = 2,
    current_bet: int = 0,
    legal: tuple[ActionType, ...] = (ActionType.FOLD, ActionType.CHECK, ActionType.BET),
) -> DecisionView:
    return DecisionView(
        player=0,
        street="flop" if board else "preflop",
        hole_cards=hole,
        board=board,
        pot=pot,
        to_call=to_call,
        stack=stack,
        min_bet=min_bet,
        current_bet=current_bet,
        legal_actions=legal,
    )


def test_folding_is_always_the_zero_reference() -> None:
    values = action_values(view(), rng=random.Random(1))
    assert values[ActionType.FOLD] == 0.0


def test_only_legal_actions_are_scored() -> None:
    legal = (ActionType.FOLD, ActionType.CALL, ActionType.RAISE)
    values = action_values(
        view(to_call=5, current_bet=5, legal=legal), rng=random.Random(1)
    )
    assert set(values) == set(legal)


def test_calling_with_a_strong_hand_is_worth_more_than_folding() -> None:
    values = action_values(
        view(hole=ACES, to_call=5, current_bet=5,
             legal=(ActionType.FOLD, ActionType.CALL, ActionType.RAISE)),
        rng=random.Random(2),
    )
    assert values[ActionType.CALL] > values[ActionType.FOLD]


def test_calling_a_big_bet_with_a_weak_hand_is_worse_than_folding() -> None:
    # Facing 60 into a 10 pot with 7-2: the pot odds are nowhere near the equity, so
    # calling has to score below the zero that folding is fixed at.
    values = action_values(
        view(hole=RAGS, pot=70, to_call=60, current_bet=60,
             legal=(ActionType.FOLD, ActionType.CALL)),
        rng=random.Random(3),
    )
    assert values[ActionType.CALL] < 0.0


def test_a_weak_bet_is_scored_honestly_rather_than_hidden() -> None:
    # The point of scoring bets whenever they are chip-legal: a bad bet still appears,
    # carrying a negative value, so a persona's bias can choose it knowingly.
    values = action_values(view(hole=RAGS, board=(
        C(Rank.ACE, Suit.SPADES), C(Rank.KING, Suit.SPADES), C(Rank.QUEEN, Suit.HEARTS),
    )), rng=random.Random(4))
    assert ActionType.BET in values
    assert values[ActionType.BET] < 0.0


def test_sizing_grows_with_equity_between_the_configured_bounds() -> None:
    sizing = Sizing(reference=0.5, min_size=0.33, max_size=1.5)
    assert size_fraction(0.5, 0.5, sizing) == pytest.approx(0.33)  # at the reference
    assert size_fraction(0.2, 0.5, sizing) == pytest.approx(0.33)  # below it, clamped
    assert size_fraction(1.0, 0.5, sizing) == pytest.approx(1.5)  # certain to win
    assert size_fraction(0.75, 0.5, sizing) == pytest.approx(0.915)  # halfway up


def test_sizing_bounds_are_parameters_not_constants() -> None:
    tight = Sizing(reference=0.5, min_size=0.1, max_size=0.4)
    assert size_fraction(0.5, 0.5, tight) == pytest.approx(0.1)
    assert size_fraction(1.0, 0.5, tight) == pytest.approx(0.4)


def test_facing_a_bet_uses_pot_odds_as_the_reference() -> None:
    # Calling 10 into a pot of 30 needs 25% equity to break even, a lower bar than the
    # 0.5 used when opening, so the same hand sizes up more when facing a bet.
    strong = (C(Rank.ACE, Suit.SPADES), C(Rank.KING, Suit.SPADES))
    board = (C(Rank.ACE, Suit.HEARTS), C(Rank.KING, Suit.HEARTS), C(Rank.TWO, Suit.CLUBS))
    opening = analyse(view(hole=strong, board=board, pot=30), rng=random.Random(5))
    facing = analyse(
        view(hole=strong, board=board, pot=30, to_call=10, current_bet=10,
             legal=(ActionType.FOLD, ActionType.CALL, ActionType.RAISE)),
        rng=random.Random(5),
    )
    assert facing.raise_to > opening.raise_to


def test_raise_is_never_below_the_minimum_or_above_the_stack() -> None:
    # A tiny pot would size below the minimum raise, and a short stack cannot cover a
    # big one; both clamps have to hold for the action to be legal at all.
    tiny = analyse(view(hole=ACES, pot=2, min_bet=10, current_bet=4, to_call=4,
                        legal=(ActionType.FOLD, ActionType.CALL, ActionType.RAISE)),
                   rng=random.Random(6))
    assert tiny.raise_to >= 4 + 10

    short = analyse(view(hole=ACES, pot=500, stack=15, current_bet=0),
                    rng=random.Random(7))
    assert short.raise_to <= 15


def test_action_for_sizes_aggressive_actions_and_leaves_others_bare() -> None:
    analysis = analyse(view(hole=ACES), rng=random.Random(8))
    bet = action_for(ActionType.BET, analysis)
    assert bet.type is ActionType.BET
    assert bet.amount == analysis.raise_to

    assert action_for(ActionType.FOLD, analysis).amount == 0
    assert action_for(ActionType.CHECK, analysis).amount == 0


def test_decision_time_equity_settings_are_actually_used(monkeypatch: pytest.MonkeyPatch) -> None:
    # The whole point of the decision-time overrides is speed; if they silently fell
    # back to equity.py's precise defaults, evaluation runs would crawl and nothing
    # would fail loudly. Capture what reaches hand_equity and check.
    recorded: dict[str, object] = {}
    original = equity_module.hand_equity

    def spy(*args: object, **kwargs: object) -> object:
        recorded.update(kwargs)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("games.poker.action_values.hand_equity", spy)
    action_values(view(hole=ACES), rng=random.Random(9))

    assert recorded["iterations"] == DECISION_ITERATIONS
    assert recorded["max_exact_enumeration"] == DECISION_MAX_EXACT_ENUMERATION
    # cheaper than the module defaults, which stay precise for tests and analysis
    assert DECISION_ITERATIONS < equity_module.DEFAULT_ITERATIONS
    assert DECISION_MAX_EXACT_ENUMERATION < equity_module.MAX_EXACT_ENUMERATION


def test_decision_threshold_keeps_the_river_exact_but_samples_the_turn() -> None:
    # The threshold is chosen to sit between these two, which is what makes a decision
    # cheap on every street without giving up the exact river.
    assert equity_module.enumeration_size(5, False) <= DECISION_MAX_EXACT_ENUMERATION
    assert equity_module.enumeration_size(4, False) > DECISION_MAX_EXACT_ENUMERATION


def test_a_precomputed_equity_is_reused_instead_of_recomputed() -> None:
    forced = analyse(view(hole=RAGS), equity=1.0)
    assert forced.equity == 1.0
    # certain to win, so betting must beat checking
    assert forced.values[ActionType.BET] > forced.values[ActionType.CHECK]
