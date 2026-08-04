import random

import pytest

from engine.cards import Card, Deck, Suit
from games.bridge import action_values as action_values_module
from games.bridge.action_values import DEPTH_GATE, action_values
from games.bridge.rules import Bridge, BridgeState, Contract, new_deal
from games.bridge.tests.test_rules import card

CONTRACT = Contract(trump=Suit.SPADES, declarer=0, target=7)


def _endgame(tricks: int, rng: random.Random) -> BridgeState:
    """A position with `tricks` cards in each hand, off a shuffled deck."""
    deck = Deck()
    deck.shuffle(rng)
    hands = tuple(tuple(deck.deal(tricks)) for _ in range(4))
    return BridgeState(contract=CONTRACT, hands=hands)


def test_more_than_the_depth_gate_never_calls_pimc_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def spy(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(action_values_module, "pimc_values", spy)

    game = Bridge()
    state = _endgame(DEPTH_GATE + 1, random.Random(1))
    action_values(game, state)

    assert calls == 0


def test_the_depth_gate_boundary_itself_does_attempt_pimc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def spy(game: object, state: object, **kwargs: object) -> dict[object, float] | None:
        nonlocal calls
        calls += 1
        return {card("AH"): 1.0}

    monkeypatch.setattr(action_values_module, "pimc_values", spy)

    game = Bridge()
    state = _endgame(DEPTH_GATE, random.Random(1))
    action_values(game, state)

    assert calls == 1


def test_fewer_than_the_depth_gate_also_attempts_pimc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def spy(game: object, state: object, **kwargs: object) -> dict[object, float] | None:
        nonlocal calls
        calls += 1
        return {card("AH"): 1.0}

    monkeypatch.setattr(action_values_module, "pimc_values", spy)

    game = Bridge()
    state = _endgame(DEPTH_GATE - 1, random.Random(1))
    action_values(game, state)

    assert calls == 1


def test_a_pimc_result_is_returned_as_is_when_it_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = {card("AH"): 7.0, card("KH"): 3.0}
    monkeypatch.setattr(action_values_module, "pimc_values", lambda *a, **k: fake)

    game = Bridge()
    state = _endgame(DEPTH_GATE, random.Random(1))
    assert action_values(game, state) == fake


def test_pimc_returning_none_falls_back_to_the_trick_heuristic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(action_values_module, "pimc_values", lambda *a, **k: None)
    fallback_calls = 0
    fake_fallback = {card("AH"): 0.5}

    def fake_heuristic(game: object, state: object) -> dict[Card, float]:
        nonlocal fallback_calls
        fallback_calls += 1
        return fake_fallback

    monkeypatch.setattr(action_values_module, "trick_win_probabilities", fake_heuristic)

    game = Bridge()
    state = _endgame(DEPTH_GATE, random.Random(1))
    result = action_values(game, state)

    assert fallback_calls == 1
    assert result == fake_fallback


def test_beyond_the_depth_gate_goes_straight_to_the_trick_heuristic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_calls = 0
    fake_fallback = {card("AH"): 0.5}

    def fake_heuristic(game: object, state: object) -> dict[Card, float]:
        nonlocal fallback_calls
        fallback_calls += 1
        return fake_fallback

    monkeypatch.setattr(action_values_module, "trick_win_probabilities", fake_heuristic)

    game = Bridge()
    state = _endgame(DEPTH_GATE + 1, random.Random(1))
    result = action_values(game, state)

    assert fallback_calls == 1
    assert result == fake_fallback


def test_a_real_fresh_deal_returns_a_value_for_every_legal_card() -> None:
    # End-to-end sanity check with the real machinery, no mocking - a fresh 13-trick
    # deal is far beyond the depth gate, so this exercises the real fallback heuristic
    # path, not PIMC (which is separately tested against real solves elsewhere).
    game = Bridge()
    state = new_deal(CONTRACT, rng=random.Random(3))
    result = action_values(game, state)

    assert set(result) == set(game.legal_actions(state))
    assert all(0.0 <= value <= 1.0 for value in result.values())  # the heuristic path


def test_a_real_pimc_decision_returns_a_value_for_every_legal_card() -> None:
    # Same end-to-end check, but shallow enough to actually exercise real PIMC.
    game = Bridge()
    state = _endgame(3, random.Random(5))
    result = action_values(game, state, max_samples=3, rng=random.Random(6))

    assert set(result) == set(game.legal_actions(state))
