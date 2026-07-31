import random
from enum import Enum, auto

import pytest

from engine.personas.quantal import Persona, choose, choose_for, policy, shift


class Move(Enum):
    PASS = auto()
    PUSH = auto()
    QUIT = auto()


VALUES = {Move.PASS: 1.0, Move.PUSH: 2.0, Move.QUIT: -3.0}


def test_probabilities_form_a_distribution() -> None:
    distribution = policy(VALUES, temperature=1.0)
    assert set(distribution) == set(VALUES)
    assert abs(sum(distribution.values()) - 1.0) < 1e-9
    assert all(probability > 0.0 for probability in distribution.values())


def test_better_valued_actions_are_more_likely() -> None:
    distribution = policy(VALUES, temperature=1.0)
    assert distribution[Move.PUSH] > distribution[Move.PASS] > distribution[Move.QUIT]


def test_low_temperature_concentrates_on_the_best_action() -> None:
    distribution = policy(VALUES, temperature=0.01)
    assert distribution[Move.PUSH] > 0.99


def test_zero_temperature_is_deterministic() -> None:
    distribution = policy(VALUES, temperature=0.0)
    assert distribution == {Move.PASS: 0.0, Move.PUSH: 1.0, Move.QUIT: 0.0}
    assert choose(VALUES, temperature=0.0, rng=random.Random(1)) is Move.PUSH


def test_zero_temperature_splits_evenly_between_tied_actions() -> None:
    tied = {Move.PASS: 5.0, Move.PUSH: 5.0, Move.QUIT: 0.0}
    distribution = policy(tied, temperature=0.0)
    assert distribution == {Move.PASS: 0.5, Move.PUSH: 0.5, Move.QUIT: 0.0}


def test_high_temperature_approaches_uniform() -> None:
    distribution = policy(VALUES, temperature=10_000.0)
    assert all(abs(probability - 1 / 3) < 0.01 for probability in distribution.values())


def test_high_temperature_does_not_overflow_on_large_values() -> None:
    # Values far apart used to be the obvious way to blow up exp(); the shift by the
    # best value before exponentiating is what keeps this finite.
    extreme = {Move.PASS: -1e6, Move.PUSH: 1e6, Move.QUIT: 0.0}
    distribution = policy(extreme, temperature=0.001)
    assert abs(sum(distribution.values()) - 1.0) < 1e-9
    assert distribution[Move.PUSH] == pytest.approx(1.0)


def test_bias_shifts_values_before_the_choice() -> None:
    assert shift(VALUES, {Move.QUIT: 10.0}) == {Move.PASS: 1.0, Move.PUSH: 2.0, Move.QUIT: 7.0}


def test_a_large_enough_bias_overrides_the_values() -> None:
    # QUIT is the worst action by a wide margin, but a big enough lean still takes it.
    biased = policy(VALUES, temperature=0.1, bias={Move.QUIT: 20.0})
    assert biased[Move.QUIT] > 0.99


def test_bias_is_a_lean_not_noise() -> None:
    # Same temperature, opposite biases: the ordering flips while the noise level does
    # not, which is exactly the separation between the two knobs.
    pushy = policy(VALUES, temperature=1.0, bias={Move.PUSH: 5.0})
    shy = policy(VALUES, temperature=1.0, bias={Move.PUSH: -5.0})
    assert pushy[Move.PUSH] > 0.9
    assert shy[Move.PUSH] < 0.1


def test_choice_is_reproducible_with_a_seeded_rng() -> None:
    first = [choose(VALUES, 1.0, rng=random.Random(4)) for _ in range(20)]
    second = [choose(VALUES, 1.0, rng=random.Random(4)) for _ in range(20)]
    assert first == second


def test_choices_follow_the_distribution() -> None:
    rng = random.Random(3)
    draws = [choose(VALUES, temperature=1.0, rng=rng) for _ in range(4_000)]
    share = draws.count(Move.PUSH) / len(draws)
    assert abs(share - policy(VALUES, 1.0)[Move.PUSH]) < 0.03


def test_persona_carries_its_own_temperature_and_bias() -> None:
    nit = Persona[Move](name="nit", temperature=0.05, bias={Move.PUSH: -50.0})
    assert choose_for(nit, VALUES, random.Random(0)) is Move.PASS


def test_empty_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        policy({}, temperature=1.0)
