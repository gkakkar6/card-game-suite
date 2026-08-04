import random

import pytest

from engine.cards import Suit
from games.bridge import pimc as pimc_module
from games.bridge.pimc import pimc_values, sample_unseen
from games.bridge.rules import Bridge, BridgeState, Contract, new_deal
from games.bridge.solver import Solution
from games.bridge.tests.test_rules import card

CONTRACT = Contract(trump=Suit.SPADES, declarer=0, target=7)


def test_sample_unseen_keeps_the_deciding_players_own_hand_and_the_dummys() -> None:
    game = Bridge()
    state = new_deal(CONTRACT, rng=random.Random(1))
    sampled = sample_unseen(game, state, 0, random.Random(2))

    assert sampled.hands[0] == state.hands[0]  # declarer's own, unchanged
    assert sampled.hands[2] == state.hands[2]  # dummy's, unchanged


def test_sample_unseen_gives_the_unseen_hands_their_real_sizes() -> None:
    game = Bridge()
    state = new_deal(CONTRACT, rng=random.Random(1))
    sampled = sample_unseen(game, state, 0, random.Random(2))

    assert len(sampled.hands[1]) == len(state.hands[1])
    assert len(sampled.hands[3]) == len(state.hands[3])


def test_sample_unseen_never_leaks_the_deciding_players_own_cards_into_the_pool() -> None:
    game = Bridge()
    state = new_deal(CONTRACT, rng=random.Random(1))
    sampled = sample_unseen(game, state, 0, random.Random(2))

    unseen_pool = set(sampled.hands[1]) | set(sampled.hands[3])
    assert unseen_pool.isdisjoint(state.hands[0])  # declarer's own hand
    assert unseen_pool.isdisjoint(state.hands[2])  # the dummy's


def test_sample_unseen_uses_the_correct_known_set_for_an_opponent_too() -> None:
    # An opponent's known set is their own hand plus the dummy's - different from
    # declarer's, since declarer's own hand is genuinely unseen to an opponent.
    game = Bridge()
    state = new_deal(CONTRACT, rng=random.Random(1))
    sampled = sample_unseen(game, state, 1, random.Random(2))

    assert sampled.hands[1] == state.hands[1]  # the deciding opponent's own hand
    assert sampled.hands[2] == state.hands[2]  # the dummy's
    unseen_pool = set(sampled.hands[0]) | set(sampled.hands[3])
    assert unseen_pool.isdisjoint(state.hands[1])
    assert unseen_pool.isdisjoint(state.hands[2])
    # declarer's real hand is exactly what got scrambled into the unseen pool
    assert unseen_pool == set(state.hands[0]) | set(state.hands[3])


def test_sample_unseen_matches_each_unseen_seat_to_its_own_actual_size() -> None:
    # All 4 hands are always the same size at any real point in a bridge deal (every
    # trick removes exactly one card from each hand at once), so a bug that swapped
    # which unseen seat gets which cut size would be invisible against a real,
    # normally-reached deal - both sizes are always equal there. Built by hand instead,
    # with genuinely unequal unseen hand sizes, specifically to exercise that this
    # matches each seat to its own real count, not just "some" correct split.
    game = Bridge()
    state = BridgeState(
        contract=CONTRACT,
        hands=(
            (card("AH"),),  # declarer: 1 card
            (card("2H"), card("2D")),  # unseen, size 2
            (card("KH"),),  # dummy: 1 card
            (card("3H"), card("3D"), card("3C")),  # unseen, size 3
        ),
    )
    sampled = sample_unseen(game, state, 0, random.Random(1))
    assert len(sampled.hands[1]) == 2
    assert len(sampled.hands[3]) == 3


def test_sample_unseen_redeals_the_same_pool_of_cards_every_time() -> None:
    # Whatever the actual split between the two unseen hands turns out to be, their
    # combined content must be exactly the true unseen pool - nothing invented, nothing
    # dropped, just reshuffled between the two hands it's genuinely hidden between.
    game = Bridge()
    state = new_deal(CONTRACT, rng=random.Random(4))
    true_pool = set(state.hands[1]) | set(state.hands[3])

    for seed in range(5):
        sampled = sample_unseen(game, state, 0, random.Random(seed))
        assert set(sampled.hands[1]) | set(sampled.hands[3]) == true_pool


def _fake_solution(*values: tuple[str, float], best: str) -> Solution:
    mapping = {card(text): value for text, value in values}
    return Solution(values=mapping, best=card(best), tricks=mapping[card(best)], nodes=0)


def test_pimc_values_averages_across_samples_not_just_takes_the_last_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # solve() is patched to return three fixed, known results in sequence, so the
    # averaging can be checked against numbers worked out by hand rather than trusting
    # a real solve. sample_unseen is bypassed too - the actual sampled deal doesn't
    # matter here, only that pimc_values() combines whatever solve() reports correctly.
    fakes = [
        _fake_solution(("AH", 4.0), ("KH", 2.0), best="AH"),
        _fake_solution(("AH", 6.0), ("KH", 6.0), best="AH"),
        _fake_solution(("AH", 5.0), ("KH", 1.0), best="AH"),
    ]

    def fake_solve(game: object, state: object) -> Solution:
        return fakes.pop(0)

    monkeypatch.setattr(pimc_module, "solve", fake_solve)
    monkeypatch.setattr(pimc_module, "sample_unseen", lambda game, state, player, rng: state)

    game = Bridge()
    state = new_deal(CONTRACT, rng=random.Random(1))
    result = pimc_values(game, state, max_samples=3, rng=random.Random(2))

    assert result == {card("AH"): 5.0, card("KH"): 3.0}  # (4+6+5)/3, (2+6+1)/3


def test_pimc_values_averages_by_how_many_samples_actually_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Distinct from the test above: there, every requested sample happened to finish,
    # so dividing by the number completed and dividing by max_samples give the same
    # answer either way - too weak to catch averaging by the wrong count. Here the
    # clock is faked so the batch is cut short by the time budget after 2 samples,
    # with max_samples=5 requested - the average must be over the 2 that actually
    # ran, not the 5 that were asked for.
    fakes = [
        _fake_solution(("AH", 4.0), best="AH"),
        _fake_solution(("AH", 8.0), best="AH"),
    ]

    def fake_solve(game: object, state: object) -> Solution:
        return fakes.pop(0)

    # One call to establish the deadline, then one per loop iteration's check: under
    # budget for the first two samples, over it before a third can start.
    clock = iter([0.0, 0.0, 0.5, 1.5])

    monkeypatch.setattr(pimc_module, "solve", fake_solve)
    monkeypatch.setattr(pimc_module, "sample_unseen", lambda game, state, player, rng: state)
    monkeypatch.setattr("games.bridge.pimc.time.monotonic", lambda: next(clock))

    game = Bridge()
    state = new_deal(CONTRACT, rng=random.Random(1))
    result = pimc_values(game, state, max_samples=5, time_budget=1.0, rng=random.Random(2))

    assert result == {card("AH"): 6.0}  # (4+8)/2, not /5


def test_pimc_values_stops_at_max_samples_even_with_time_to_spare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def fake_solve(game: object, state: object) -> Solution:
        nonlocal call_count
        call_count += 1
        return _fake_solution(("AH", 1.0), best="AH")

    monkeypatch.setattr(pimc_module, "solve", fake_solve)
    monkeypatch.setattr(pimc_module, "sample_unseen", lambda game, state, player, rng: state)

    game = Bridge()
    state = new_deal(CONTRACT, rng=random.Random(1))
    pimc_values(game, state, max_samples=4, time_budget=60.0, rng=random.Random(2))

    assert call_count == 4


def test_zero_samples_within_budget_falls_through_to_none() -> None:
    # An artificially tiny (in fact zero) time budget forces the very first deadline
    # check to already be in the past, so no sample is even attempted - the case
    # action_values.py has to catch and fall back from, rather than erroring or
    # returning an empty-but-not-None result.
    game = Bridge()
    state = new_deal(CONTRACT, rng=random.Random(1))
    result = pimc_values(game, state, time_budget=0.0, rng=random.Random(2))
    assert result is None


def test_zero_max_samples_also_falls_through_to_none() -> None:
    # A different, more direct way of forcing the same "nothing completed" case,
    # not dependent on wall-clock timing at all.
    game = Bridge()
    state = new_deal(CONTRACT, rng=random.Random(1))
    result = pimc_values(game, state, max_samples=0, rng=random.Random(2))
    assert result is None
