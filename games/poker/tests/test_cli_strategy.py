import pytest

from engine.cards import Card, Rank, Suit
from games.poker.betting import ActionType
from games.poker.cli_strategy import format_view, parse_action
from games.poker.hand import DecisionView


def view(
    legal: tuple[ActionType, ...] = (ActionType.FOLD, ActionType.CHECK, ActionType.BET),
    current_bet: int = 0,
    min_bet: int = 2,
) -> DecisionView:
    return DecisionView(
        player=0,
        street="flop",
        hole_cards=(Card(Rank.ACE, Suit.CLUBS), Card(Rank.KING, Suit.CLUBS)),
        board=(),
        pot=10,
        to_call=0,
        stack=100,
        min_bet=min_bet,
        current_bet=current_bet,
        legal_actions=legal,
    )


def test_parses_fold_check_call() -> None:
    fold_or_call = view(legal=(ActionType.FOLD, ActionType.CALL))
    assert parse_action("fold", fold_or_call).type is ActionType.FOLD
    assert parse_action("call", fold_or_call).type is ActionType.CALL
    assert parse_action("check", view()).type is ActionType.CHECK


def test_parses_bet_and_raise_with_amount() -> None:
    action = parse_action("bet 20", view())
    assert action.type is ActionType.BET
    assert action.amount == 20

    action = parse_action(
        "raise 50", view(legal=(ActionType.FOLD, ActionType.CALL, ActionType.RAISE))
    )
    assert action.type is ActionType.RAISE
    assert action.amount == 50


def test_is_case_insensitive_and_trims_whitespace() -> None:
    action = parse_action("  BET   20  ", view())
    assert action.type is ActionType.BET
    assert action.amount == 20


def test_rejects_unrecognised_words() -> None:
    with pytest.raises(ValueError):
        parse_action("shuffle", view())


def test_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        parse_action("", view())
    with pytest.raises(ValueError):
        parse_action("   ", view())


def test_rejects_actions_not_currently_legal() -> None:
    with pytest.raises(ValueError):
        parse_action("check", view(legal=(ActionType.FOLD, ActionType.CALL)))


def test_bet_without_an_amount_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_action("bet", view())


def test_bet_with_a_non_numeric_amount_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_action("bet a lot", view())


def test_bet_with_a_non_positive_amount_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_action("bet 0", view())
    with pytest.raises(ValueError):
        parse_action("bet -5", view())


def test_format_view_includes_the_essentials_and_nothing_computed() -> None:
    text = format_view(view(legal=(ActionType.FOLD, ActionType.CHECK, ActionType.BET)))
    assert "AC" in text and "KC" in text  # hole cards
    assert "pot: 10" in text
    assert "to call: 0" in text
    assert "your stack: 100" in text
    assert "fold" in text and "check" in text and "bet" in text
    # the hard boundary: no equity/EV vocabulary anywhere in the rendered text
    for forbidden in ("equity", "EV", "value"):
        assert forbidden not in text.lower()


def test_cli_strategy_module_never_imports_equity_or_action_values() -> None:
    # A hard boundary, checked against the actual import statements rather than
    # against what happens to be bound in the module's namespace at runtime (which an
    # indirect or renamed import could still satisfy without pulling in equity/EV
    # code) or a plain substring search (which trips on this file's own docstring
    # mentioning the module names it forbids).
    import ast
    import inspect

    import games.poker.cli_strategy as module

    tree = ast.parse(inspect.getsource(module))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(name.startswith("games.poker.equity") for name in imported_modules)
    assert not any(name.startswith("games.poker.action_values") for name in imported_modules)
