import random

from engine.cards import Suit
from games.court_piece.declare import QuantalDeclareStrategy, declare_values
from games.court_piece.hand_evaluation import suit_score
from games.court_piece.tests.test_rules import hand


def test_declare_values_matches_suit_score_for_every_suit() -> None:
    preview = hand("AH", "KH", "2C", "3D", "4S")
    values = declare_values(preview)
    assert set(values) == set(Suit)
    for suit in Suit:
        assert values[suit] == suit_score(preview, suit)


def test_zero_temperature_always_calls_the_best_scoring_suit() -> None:
    preview = hand("AH", "KH", "QH", "2C", "3D")
    strategy = QuantalDeclareStrategy(temperature=0.0, rng=random.Random(1))
    assert strategy(preview) == Suit.HEARTS


def test_always_returns_a_suit_even_from_a_weak_preview() -> None:
    # The call is mandatory - there's no "pass" - so this must produce something even
    # from a genuinely weak hand.
    preview = hand("2C", "3D", "4H", "5S", "6C")
    strategy = QuantalDeclareStrategy(temperature=0.0, rng=random.Random(1))
    assert strategy(preview) in list(Suit)


def test_a_bias_can_override_the_raw_score() -> None:
    preview = hand("AH", "KH", "2C", "3D", "4S")

    # Hearts would win on raw score; a strong bias toward clubs should override it at
    # zero temperature, the same mechanism as every other persona bias in this project.
    def bias_toward_clubs(values: dict[Suit, float]) -> dict[Suit, float]:
        return {Suit.CLUBS: 1000.0}

    strategy = QuantalDeclareStrategy(temperature=0.0, bias=bias_toward_clubs, rng=random.Random(1))
    assert strategy(preview) == Suit.CLUBS
