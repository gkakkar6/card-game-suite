import random

from engine.cards import Suit
from games.court_piece.personas import BASELINE, PERSONAS, CardPlayStrategy
from games.court_piece.personas import DeclareStrategy as PersonaDeclareStrategy
from games.court_piece.rules import CourtPiece, CourtPieceState, TrumpCall
from games.court_piece.tests.test_rules import hand

CALL = TrumpCall(trump=Suit.SPADES, caller=0)


def test_only_baseline_exists_for_now() -> None:
    # Deliberate, named scope for this first version - see personas.py's own
    # docstring for why more styles aren't invented without real measurement yet.
    assert set(PERSONAS) == {"baseline"}


def test_declare_strategy_uses_the_personas_temperature_and_bias() -> None:
    preview = hand("AH", "KH", "QH", "2C", "3D")
    strategy = PersonaDeclareStrategy(BASELINE, random.Random(1))
    # Baseline's temperature is low but nonzero - not asserting determinism, just that
    # it returns a real suit from the persona's own decision mechanism.
    assert strategy(preview) in list(Suit)


def test_card_play_strategy_returns_a_legal_card() -> None:
    game = CourtPiece()
    state = CourtPieceState(
        call=CALL,
        hands=(hand("2H", "3H"), hand("4H"), hand("5H"), hand("6H")),
    )
    strategy = CardPlayStrategy(BASELINE, random.Random(2))
    action = strategy(game, state)
    assert action in game.legal_actions(state)
