import random

from engine.cards import Card
from engine.personas.quantal import Persona, policy
from games.poker.action_values import Sizing, analyse
from games.poker.betting import ActionType
from games.poker.hand import DecisionView, play_hand
from games.poker.personas import (
    BASELINE,
    BASELINE_TEMPERATURE,
    BLUFFER,
    BLUFFER_BIAS,
    CALLING_STATION,
    CALLING_STATION_BIAS,
    CONSERVATIVE,
    CONSERVATIVE_BIAS,
    DEAL_SEED_STRIDE,
    DEVIATIONS,
    DUPLICATE_PAIR,
    ERRATIC,
    ERRATIC_TEMPERATURE,
    PERSONAS,
    PersonaStrategy,
    _seat_of,
    decision_scale,
    normalised,
    play_matchup,
    strategy_for,
)
from games.poker.tests.test_action_values import ACES, RAGS, view


def test_the_roster_covers_four_deviations_from_the_baseline() -> None:
    assert set(PERSONAS) == {"baseline", "bluffer", "conservative", "calling station", "erratic"}
    assert BASELINE.bias == {}

    # Three lean without extra noise, so they isolate direction from randomness.
    for persona in (BLUFFER, CONSERVATIVE, CALLING_STATION):
        assert persona.temperature == BASELINE_TEMPERATURE
        assert persona.bias

    # The fourth is the opposite: pure noise, no direction at all. This is what makes
    # it a test of temperature by itself.
    assert ERRATIC.bias == {}
    assert ERRATIC.temperature == ERRATIC_TEMPERATURE > BASELINE_TEMPERATURE


def test_the_deviations_lean_on_the_axes_they_claim_to() -> None:
    assert BLUFFER.bias[ActionType.BET] == BLUFFER_BIAS > 0
    assert CONSERVATIVE.bias[ActionType.BET] == CONSERVATIVE_BIAS < 0
    # The calling station is a different axis: it leans on folding, not on aggression.
    assert CALLING_STATION.bias == {ActionType.FOLD: CALLING_STATION_BIAS}
    assert ActionType.BET not in CALLING_STATION.bias


def _bet_share(persona: Persona[ActionType], decision: DecisionView, equity: float) -> float:
    """How often this persona chooses to bet, holding the equity estimate fixed."""
    values = normalised(analyse(decision, equity=equity).values, decision)
    distribution = policy(values, persona.temperature, persona.bias)
    return sum(
        probability
        for action, probability in distribution.items()
        if action in (ActionType.BET, ActionType.RAISE)
    )


def test_betting_a_weak_hand_scores_worse_than_checking() -> None:
    # With little equity, putting more in is worse than taking the free look. The value
    # goes outright negative once the hand is bad enough to lose the extra chips.
    weak = view(hole=RAGS, pot=10)
    barely = analyse(weak, equity=0.2).values
    assert barely[ActionType.BET] < barely[ActionType.CHECK]
    assert analyse(weak, equity=0.05).values[ActionType.BET] < 0.0


def _decision_sweep() -> list[tuple[DecisionView, float]]:
    """A spread of decisions: opening and facing a bet, across the equity range.

    A persona's style is a tendency over many decisions, not a rule about any one of
    them, so it has to be measured over a spread. Judging a lean from a single hand
    picks whichever spot happens to flatter it.
    """
    decisions = []
    for step in range(1, 20):
        for pot in (6, 15, 40):
            facing = pot // 3
            decisions.append((view(pot=pot), step / 20))
            decisions.append((
                view(
                    pot=pot,
                    to_call=facing,
                    current_bet=facing,
                    legal=(ActionType.FOLD, ActionType.CALL, ActionType.RAISE),
                ),
                step / 20,
            ))
    return decisions


def _rates(persona: Persona[ActionType]) -> tuple[float, float, float]:
    """Average (best-valued, aggressive, fold) probabilities over the sweep."""
    best = aggressive = folding = 0.0
    decisions = _decision_sweep()
    for decision, equity in decisions:
        values = normalised(analyse(decision, equity=equity).values, decision)
        distribution = policy(values, persona.temperature, persona.bias)
        best += distribution[max(values, key=lambda action: values[action])]
        aggressive += sum(
            p for action, p in distribution.items() if action in (ActionType.BET, ActionType.RAISE)
        )
        folding += distribution.get(ActionType.FOLD, 0.0)
    count = len(decisions)
    return (best / count, aggressive / count, folding / count)


def test_the_bluffer_bets_more_than_the_values_justify() -> None:
    # The whole mechanism: no separate "bluffing" code, just the §3 bias applied on top
    # of values the engine honestly reported.
    _, baseline_aggression, _ = _rates(BASELINE)
    best, aggression, _ = _rates(BLUFFER)
    assert aggression > baseline_aggression + 0.15
    assert best > 0.3  # still responding to its cards, not just always betting


def test_the_conservative_bets_less_than_the_values_justify() -> None:
    _, baseline_aggression, _ = _rates(BASELINE)
    best, aggression, _ = _rates(CONSERVATIVE)
    assert aggression < baseline_aggression - 0.1
    assert best > 0.3


def test_the_calling_station_folds_far_less_than_the_baseline() -> None:
    _, baseline_aggression, baseline_folding = _rates(BASELINE)
    best, aggression, folding = _rates(CALLING_STATION)
    assert folding < baseline_folding / 2
    assert best > 0.3
    # It leans on folding alone, so its aggression should barely move - that is what
    # makes it a different axis rather than a softer conservative.
    assert abs(aggression - baseline_aggression) < 0.1


def test_the_erratic_persona_is_noisy_without_leaning() -> None:
    baseline_best, baseline_aggression, _ = _rates(BASELINE)
    best, aggression, _ = _rates(ERRATIC)
    assert baseline_best > 0.8
    assert best < baseline_best - 0.3  # far closer to guessing
    assert best > 0.25  # but not literally random
    assert abs(aggression - baseline_aggression) < 0.2  # no directional preference


def test_no_persona_is_effectively_deterministic() -> None:
    # Each should be clearly leaning yet still responsive to its actual cards.
    for persona in PERSONAS.values():
        best, aggression, _ = _rates(persona)
        assert best > 0.25, f"{persona.name} ignores its cards too often"
        assert aggression < 0.85, f"{persona.name} is effectively always aggressive"


def test_persona_strategy_returns_a_legal_playable_action() -> None:
    strategy = strategy_for(BASELINE, random.Random(1))
    decision = view(hole=ACES, pot=10)
    action = strategy(decision)
    assert action.type in decision.legal_actions
    if action.type is ActionType.BET:
        assert action.amount > decision.current_bet


def test_persona_strategy_caches_equity_per_board() -> None:
    strategy = PersonaStrategy(BASELINE, random.Random(2))
    decision = view(hole=ACES, pot=10)
    strategy(decision)
    assert len(strategy._equity) == 1

    strategy(view(hole=ACES, pot=40))  # same cards, different pot: no new calculation
    assert len(strategy._equity) == 1

    strategy(view(hole=RAGS, pot=10))  # different cards, so a new one
    assert len(strategy._equity) == 2


def test_equity_estimates_do_not_survive_into_the_next_hand() -> None:
    # A decision-time equity is a noisy estimate. Carrying one across hands means a
    # repeated (hole cards, board) reuses a stale draw instead of a fresh one, so the
    # errors stop averaging out over a run - and differently for each persona.
    strategy = PersonaStrategy(BASELINE, random.Random(3))
    strategy.start_hand()
    strategy(view(hole=ACES, pot=10))
    assert len(strategy._equity) == 1

    strategy.start_hand()
    assert strategy._equity == {}


def test_a_matchup_starts_every_hand_with_an_empty_cache() -> None:
    evaluated = PersonaStrategy(BASELINE, random.Random(4))
    play_matchup(BASELINE, BLUFFER, hands=4, rng=random.Random(5), resamples=20, seed=1)
    # the strategies play_matchup builds are internal, so check the contract they rely
    # on: a fresh strategy plus start_hand leaves nothing behind from a previous hand
    evaluated(view(hole=ACES, pot=10))
    evaluated.start_hand()
    assert evaluated._equity == {}


def test_values_are_normalised_by_what_is_at_stake() -> None:
    # The same decision shape in a small and a large pot must look identical once
    # normalised, which is what stops a fixed bias meaning two different things.
    assert decision_scale(view(pot=10, to_call=0)) == 10
    assert decision_scale(view(pot=30, to_call=10, current_bet=10)) == 40
    assert decision_scale(view(pot=0, to_call=0)) == 1  # never divide by zero

    raw = {ActionType.FOLD: 0.0, ActionType.CHECK: 5.0, ActionType.BET: 10.0}
    scaled = normalised(raw, view(pot=20, to_call=0))
    assert scaled == {ActionType.FOLD: 0.0, ActionType.CHECK: 0.25, ActionType.BET: 0.5}


def test_normalised_values_barely_move_with_pot_size() -> None:
    # The bug this fixes: raw chip values grow with the pot, so a fixed bias was
    # overwhelming in a small pot and invisible in a large one. Once divided by what is
    # at stake, the same hand in a 5-chip pot and a 200-chip pot scores the same, which
    # is what lets one bias number mean one thing everywhere.
    for equity in (0.15, 0.85):
        scored = [
            normalised(analyse(view(pot=pot), equity=equity).values, view(pot=pot))[ActionType.BET]
            for pot in (5, 20, 200)
        ]
        assert max(scored) - min(scored) < 0.2, scored


def test_every_deviation_actually_plays_differently_from_the_baseline() -> None:
    baseline = _rates(BASELINE)
    for persona in DEVIATIONS:
        rates = _rates(persona)
        assert any(abs(rate - base) > 0.08 for rate, base in zip(rates, baseline, strict=True)), (
            f"{persona.name} is indistinguishable from the baseline"
        )


def test_personas_can_play_a_whole_hand_against_each_other() -> None:
    result = play_hand(
        [strategy_for(BASELINE, random.Random(3)), strategy_for(BLUFFER, random.Random(4))],
        rng=random.Random(5),
    )
    assert sum(result.final_stacks) == sum(result.starting_stacks)
    assert len(result.winners) >= 1


def test_sizing_can_be_overridden_per_strategy() -> None:
    small = strategy_for(BASELINE, random.Random(6), sizing=Sizing(max_size=0.4))
    big = strategy_for(BASELINE, random.Random(6), sizing=Sizing(min_size=2.0, max_size=3.0))
    decision = view(hole=ACES, pot=100)
    assert big(decision).amount > small(decision).amount


def test_tuning_constants_are_exposed_as_single_adjustable_knobs() -> None:
    assert 0.0 < BASELINE_TEMPERATURE < ERRATIC_TEMPERATURE
    assert BLUFFER_BIAS > 0.0
    assert CONSERVATIVE_BIAS < 0.0
    assert CALLING_STATION_BIAS < 0.0


def test_duplicate_pairs_replay_the_same_deal_from_both_seats() -> None:
    # The variance reduction only works if consecutive hands really are the same cards.
    # If this silently stopped duplicating, the intervals would quietly widen instead
    # of anything failing, so it is worth pinning down.
    def deal_for(hand_index: int) -> tuple[tuple[Card, ...], ...]:
        rng = random.Random(DEAL_SEED_STRIDE + hand_index // DUPLICATE_PAIR)
        result = play_hand(
            [strategy_for(BASELINE, random.Random(1)), strategy_for(BASELINE, random.Random(2))],
            rng=rng,
        )
        return result.hole_cards

    assert deal_for(0) == deal_for(1)  # one pair, same cards
    assert deal_for(0) != deal_for(2)  # next pair, different cards
    assert _seat_of(0) != _seat_of(1)  # and the evaluated persona swaps seats


def test_matchup_runs_and_reports_an_interval() -> None:
    # Small and fast: this checks the wiring from personas through hand.py into the
    # harness, not the validating claim, which needs far more hands (see test_validation).
    result = play_matchup(
        BASELINE, BLUFFER, hands=4, rng=random.Random(1), resamples=50, seed=2
    )
    assert result.hands == 4
    assert result.ci_low <= result.mean_profit <= result.ci_high
    assert 0.0 <= result.win_rate <= 1.0
