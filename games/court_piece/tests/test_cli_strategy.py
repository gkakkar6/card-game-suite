import pytest

from engine.cards import Suit
from games.court_piece.cli_strategy import (
    format_call,
    format_declare_view,
    format_play_view,
    format_trump,
    parse_card,
    parse_suit,
)
from games.court_piece.rules import CourtPiece, CourtPieceState, TrumpCall
from games.court_piece.tests.test_rules import card

CALL = TrumpCall(trump=Suit.SPADES, caller=0)

PREVIEW = [card("AS"), card("KS"), card("2S"), card("QH"), card("2H")]

# ---------------------------------------------------------------------------
# parse_suit()
# ---------------------------------------------------------------------------


def test_parses_a_suit_symbol() -> None:
    assert parse_suit("H") == Suit.HEARTS
    assert parse_suit("s") == Suit.SPADES


def test_parses_a_full_suit_name() -> None:
    assert parse_suit("hearts") == Suit.HEARTS
    assert parse_suit("Spades") == Suit.SPADES


def test_rejects_an_unrecognised_suit() -> None:
    with pytest.raises(ValueError):
        parse_suit("X")
    with pytest.raises(ValueError):
        parse_suit("")


# ---------------------------------------------------------------------------
# parse_card()
# ---------------------------------------------------------------------------


def test_parses_a_legal_card() -> None:
    legal = [card("AS"), card("2H")]
    assert parse_card("AS", legal) == card("AS")
    assert parse_card("as", legal) == card("AS")


def test_rejects_a_card_not_in_the_legal_set() -> None:
    with pytest.raises(ValueError):
        parse_card("KH", [card("AS"), card("2H")])


def test_rejects_unparseable_card_text() -> None:
    with pytest.raises(ValueError):
        parse_card("shuffle", [card("AS")])
    with pytest.raises(ValueError):
        parse_card("A", [card("AS")])


# ---------------------------------------------------------------------------
# Formatting shows only real game information, nothing computed
# ---------------------------------------------------------------------------


def test_format_declare_view_shows_only_the_preview_cards() -> None:
    text = format_declare_view(PREVIEW)
    assert "AS" in text and "KS" in text and "QH" in text
    assert "call one of" in text.lower()


def test_format_call_names_trump_and_caller() -> None:
    text = format_call(TrumpCall(trump=Suit.HEARTS, caller=2))
    assert "H" in text
    assert "seat 2" in text


def test_format_play_view_shows_your_hand_and_legal_cards() -> None:
    # CALL's caller is seat 0, so opening_leader() puts seat 1 on lead with an empty
    # trick and no completed history - the cards have to sit with seat 1, not seat 0.
    state = CourtPieceState(call=CALL, hands=((), (card("2C"), card("3C")), (), ()))
    game = CourtPiece()
    assert state.to_play == 1
    text = format_play_view(game, state)
    assert "your hand" in text
    assert "2C" in text and "3C" in text


def test_format_play_view_never_mentions_computed_values() -> None:
    state = CourtPieceState(call=CALL, hands=((), (card("2C"), card("3C")), (), ()))
    text = format_play_view(CourtPiece(), state)
    for forbidden in ("probability", "equity", "value", "pimc"):
        assert forbidden not in text.lower()


def test_format_trump_shows_the_suit_once_set() -> None:
    assert format_trump(Suit.HEARTS) == "H"


def test_format_trump_shows_not_yet_set_for_running_trump_before_it_fires() -> None:
    assert format_trump(None) == "not yet set"


def test_format_play_view_shows_trump_not_yet_set_for_running_trump() -> None:
    running_call = TrumpCall(trump=None, caller=0)
    state = CourtPieceState(call=running_call, hands=((), (card("2C"), card("3C")), (), ()))
    text = format_play_view(CourtPiece(), state)
    assert "not yet set" in text


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------


def test_cli_strategy_module_never_imports_computed_value_modules() -> None:
    # A hard boundary, checked against the actual import statements - a human player
    # should never be shown a computed value, matching games/bridge/cli_strategy.py's
    # identical discipline.
    import ast
    import inspect

    import games.court_piece.cli_strategy as module

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
    forbidden_prefixes = (
        "games.court_piece.action_values",
        "games.court_piece.pimc",
        "games.court_piece.trick_odds",
        "games.court_piece.hand_evaluation",
    )
    for prefix in forbidden_prefixes:
        assert not any(name.startswith(prefix) for name in imported_modules)
