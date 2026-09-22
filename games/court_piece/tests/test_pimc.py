import random

import pytest

from engine.cards import Suit
from games.court_piece.pimc import pimc_values, sample_unseen
from games.court_piece.rules import (
    CourtPiece,
    CourtPieceInformationSet,
    CourtPieceState,
    TrumpCall,
)
from games.court_piece.tests.test_rules import hand

CALL = TrumpCall(trump=Suit.SPADES, caller=0)


def _state() -> CourtPieceState:
    return CourtPieceState(
        call=CALL,
        hands=(hand("AH", "2H"), hand("3H", "4H"), hand("5H", "6H"), hand("7H", "8H")),
    )


def test_sample_unseen_keeps_the_players_own_hand_fixed() -> None:
    game = CourtPiece()
    state = _state()
    sampled = sample_unseen(game, state, player=0, rng=random.Random(3))
    assert sampled.hands[0] == state.hands[0]


def test_sample_unseen_preserves_each_unseen_hands_size() -> None:
    game = CourtPiece()
    state = _state()
    sampled = sample_unseen(game, state, player=0, rng=random.Random(3))
    for seat in (1, 2, 3):
        assert len(sampled.hands[seat]) == len(state.hands[seat])


def test_sample_unseen_redeals_the_same_pool_of_cards() -> None:
    game = CourtPiece()
    state = _state()
    sampled = sample_unseen(game, state, player=0, rng=random.Random(3))
    original_pool = {card for seat in (1, 2, 3) for card in state.hands[seat]}
    sampled_pool = {card for seat in (1, 2, 3) for card in sampled.hands[seat]}
    assert sampled_pool == original_pool


def test_sample_unseen_requires_exactly_three_unseen_hands() -> None:
    state = _state()

    class _FakeGame(CourtPiece):
        def information_set(
            self, state: CourtPieceState, player: int
        ) -> CourtPieceInformationSet:
            info = super().information_set(state, player)
            # Pretend a second hand is visible, to break the 3-unseen-hands invariant.
            hands = {**info.hands, 1: state.hands[1]}
            return CourtPieceInformationSet(
                player=info.player,
                hands=hands,
                call=info.call,
                trick=info.trick,
                completed=info.completed,
            )

    with pytest.raises(AssertionError):
        sample_unseen(_FakeGame(), state, player=0, rng=random.Random(1))


def test_pimc_values_returns_a_value_for_every_legal_card() -> None:
    game = CourtPiece()
    state = _state()
    values = pimc_values(game, state, max_samples=3, rng=random.Random(5))
    assert values is not None
    assert set(values) == set(game.legal_actions(state))
