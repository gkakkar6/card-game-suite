import pytest

from engine.cards import Suit
from games.bridge.bidding import Bid, new_auction
from games.bridge.cli_strategy import (
    format_auction,
    format_bidding_view,
    format_play_view,
    parse_call,
    parse_card,
)
from games.bridge.rules import Bridge, BridgeState, Contract
from games.bridge.tests.test_rules import card

HAND = [
    card("AS"), card("KS"), card("2S"),
    card("QH"), card("2H"),
    card("KD"), card("2D"),
    card("JC"), card("TC"), card("2C"), card("3C"), card("4C"), card("5C"),
]

# ---------------------------------------------------------------------------
# parse_call()
# ---------------------------------------------------------------------------


def test_parses_pass() -> None:
    assert parse_call("pass", new_auction(dealer=0)) is None
    assert parse_call("PASS", new_auction(dealer=0)) is None
    assert parse_call("p", new_auction(dealer=0)) is None


def test_parses_a_suit_bid() -> None:
    assert parse_call("1H", new_auction(dealer=0)) == Bid(1, Suit.HEARTS)
    assert parse_call("3s", new_auction(dealer=0)) == Bid(3, Suit.SPADES)


def test_parses_a_notrump_bid() -> None:
    assert parse_call("1NT", new_auction(dealer=0)) == Bid(1, None)
    assert parse_call("2N", new_auction(dealer=0)) == Bid(2, None)


def test_is_case_insensitive_and_trims_whitespace() -> None:
    assert parse_call("  1h  ", new_auction(dealer=0)) == Bid(1, Suit.HEARTS)


def test_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        parse_call("", new_auction(dealer=0))
    with pytest.raises(ValueError):
        parse_call("   ", new_auction(dealer=0))


def test_rejects_an_unrecognised_strain() -> None:
    with pytest.raises(ValueError):
        parse_call("1X", new_auction(dealer=0))


def test_rejects_a_missing_level() -> None:
    with pytest.raises(ValueError):
        parse_call("H", new_auction(dealer=0))


def test_rejects_an_out_of_range_level() -> None:
    with pytest.raises(ValueError):
        parse_call("9H", new_auction(dealer=0))


def test_rejects_a_call_that_does_not_outrank_the_current_high() -> None:
    auction = new_auction(dealer=0).apply(Bid(3, Suit.HEARTS))
    with pytest.raises(ValueError):
        parse_call("2S", auction)


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


def test_format_auction_lists_each_call_by_seat() -> None:
    auction = new_auction(dealer=1).apply(Bid(1, Suit.SPADES)).apply(None)
    text = format_auction(auction)
    assert "seat 1: 1S" in text
    assert "seat 2: Pass" in text


def test_format_bidding_view_includes_hand_and_legal_calls() -> None:
    text = format_bidding_view(HAND, new_auction(dealer=0))
    assert "AS" in text and "KS" in text
    assert "Pass" in text
    assert "1H" in text  # a legal opening bid


def test_format_play_view_labels_the_dummys_hand_distinctly() -> None:
    contract = Contract(trump=Suit.SPADES, declarer=0, target=9)
    state = BridgeState(
        contract=contract,
        hands=((), (), (card("8H"), card("3S")), (card("2H"),)),
        trick=((0, card("KH")), (1, card("2H"))),
    )
    game = Bridge()
    assert state.to_play == 2  # dummy
    text = format_play_view(game, state)
    assert "dummy's hand" in text
    assert "8H" in text


def test_format_play_view_shows_own_hand_when_not_the_dummy() -> None:
    contract = Contract(trump=None, declarer=0, target=7)
    state = BridgeState(contract=contract, hands=((card("2C"), card("3C")), (), (), ()))
    game = Bridge()
    text = format_play_view(game, state)
    assert "your hand" in text


def test_format_play_view_never_mentions_computed_values() -> None:
    contract = Contract(trump=None, declarer=0, target=7)
    state = BridgeState(contract=contract, hands=((card("2C"), card("3C")), (), (), ()))
    text = format_play_view(Bridge(), state)
    for forbidden in ("probability", "equity", "value", "pimc"):
        assert forbidden not in text.lower()


# ---------------------------------------------------------------------------
# Import boundary
# ---------------------------------------------------------------------------


def test_cli_strategy_module_never_imports_computed_value_modules() -> None:
    # A hard boundary, checked against the actual import statements rather than
    # against what happens to be bound in the module's namespace at runtime (which an
    # indirect or renamed import could still satisfy without pulling in the computed-
    # value modules) or a plain substring search (which trips on this file's own
    # docstring naming the modules it avoids).
    import ast
    import inspect

    import games.bridge.cli_strategy as module

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
    assert not any(name.startswith("games.bridge.action_values") for name in imported_modules)
    assert not any(name.startswith("games.bridge.pimc") for name in imported_modules)
    assert not any(name.startswith("games.bridge.trick_odds") for name in imported_modules)

    # bid_values() specifically, not the whole bidding module - Bid/Auction/Call are
    # exactly what a human's own decisions are made of and stay fair game.
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "games.bridge.bidding"
        for alias in node.names
    }
    assert "bid_values" not in imported_names
    assert "choose_bid" not in imported_names
