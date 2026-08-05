import random
from collections.abc import Sequence

import pytest

from engine.cards import Card, Deck, Suit
from engine.personas.quantal import policy
from engine.trick_taking.equivalence import play_groups
from games.bridge.action_values import analyse
from games.bridge.bidding import (
    SIGNATURE_TIER,
    SIGNOFF_TIER,
    Auction,
    Bid,
    bid_values,
    new_auction,
)
from games.bridge.personas import (
    AGGRESSIVE,
    BAITER,
    BASELINE,
    CONSERVATIVE,
    PERSONAS,
    SELFISH,
    UNAWARE,
    BiddingStrategy,
    BridgePersona,
    CardPlayStrategy,
    jittered,
)
from games.bridge.rules import Bridge, BridgeState, Contract, new_deal
from games.bridge.tests.test_rules import card

pytestmark = pytest.mark.slow

GAME = Bridge()
CONTRACT = Contract(trump=Suit.SPADES, declarer=0, target=7)


def test_the_roster_covers_six_personas() -> None:
    assert set(PERSONAS) == {
        "baseline", "aggressive", "conservative", "selfish", "baiter", "unaware",
    }
    assert BASELINE.bidding_bias({}) == {}
    assert BASELINE.play_bias(GAME, new_deal(CONTRACT, rng=random.Random(1)), {}) == {}
    assert UNAWARE.unaware  # sanity: exactly one persona sets this
    assert not any(p.unaware for p in PERSONAS.values() if p is not UNAWARE)


# ---------------------------------------------------------------------------
# Bidding calibration
# ---------------------------------------------------------------------------


def _bidding_sweep(rng: random.Random, n: int = 80) -> list[tuple[Sequence[Card], Auction]]:
    """A spread of bidding decisions: opening, response, and rebid stages - a
    persona's lean is a tendency over many decisions, not a rule about any single
    one, so it has to be measured over a spread (same reasoning as poker's own
    calibration sweep)."""
    decisions: list[tuple[Sequence[Card], Auction]] = []
    for _ in range(n):
        hands = new_deal(CONTRACT, rng=rng).hands
        decisions.append((hands[0], new_auction(dealer=0)))
        opening = bid_values(hands[0], new_auction(dealer=0))
        best_open = max(opening, key=lambda c: opening[c])
        if best_open is not None:
            auction = new_auction(dealer=0).apply(best_open).apply(None)
            decisions.append((hands[2], auction))
            response = bid_values(hands[2], auction)
            best_resp = max(response, key=lambda c: response[c])
            if best_resp is not None:
                decisions.append((hands[0], auction.apply(best_resp).apply(None)))
    return decisions


def _bidding_rates(
    persona: BridgePersona, decisions: Sequence[tuple[Sequence[Card], Auction]]
) -> tuple[float, float, float]:
    """Average (P(a best-tied call), P(>=SIGNATURE_TIER), P(<=SIGNOFF_TIER))."""
    best_sum = above_sum = below_sum = 0.0
    count = 0
    for hand, auction in decisions:
        values = bid_values(hand, auction)
        if len(values) < 2:
            continue
        bias = persona.bidding_bias(values)
        dist = policy(values, persona.bidding_temperature, bias)
        top = max(values.values())
        best_sum += sum(p for c, p in dist.items() if values[c] == top)
        above_sum += sum(p for c, p in dist.items() if values[c] >= SIGNATURE_TIER)
        below_sum += sum(p for c, p in dist.items() if values[c] <= SIGNOFF_TIER)
        count += 1
    return best_sum / count, above_sum / count, below_sum / count


def test_conservative_bidding_leans_toward_pass_and_signoff_tier_calls() -> None:
    # Random hands don't reliably reach a genuinely close call, so this is measured
    # over a real spread rather than one hand-picked example (matching poker's own
    # calibration discipline). See DECISIONS.md for why aggressive's bidding lean
    # doesn't show up the same way in this kind of aggregate sweep - a real,
    # documented finding, not an oversight.
    decisions = _bidding_sweep(random.Random(42))
    baseline_best, _, baseline_below = _bidding_rates(BASELINE, decisions)
    best, above, below = _bidding_rates(CONSERVATIVE, decisions)
    assert below > baseline_below + 0.15
    assert below < 0.85  # sanity ceiling: not choosing its lean past ~80-85%
    assert best > 0.25  # sanity floor: still responsive to its hand sometimes


def test_aggressive_bidding_prefers_the_higher_tier_call_at_a_genuine_close_decision() -> None:
    # Verified directly at a real multi-way decision, since an aggregate random
    # sweep is dominated by opening-stage decisions where bid_values()'s gaps are
    # too large for any reasonable bias to matter (see DECISIONS.md) - this is the
    # kind of decision aggressive's bias actually has real, measurable leverage
    # over: a new-suit response competing against a plain raise and 1NT.
    hand = [
        card("JS"), card("QS"), card("2S"), card("3S"),
        card("KH"), card("2H"), card("3H"),
        card("QD"), card("2D"), card("3D"),
        card("2C"), card("3C"), card("4C"),
    ]
    auction = new_auction(dealer=0).apply(Bid(1, Suit.HEARTS)).apply(None)
    values = bid_values(hand, auction)
    baseline_dist = policy(values, BASELINE.bidding_temperature, BASELINE.bidding_bias(values))
    aggressive_bias = AGGRESSIVE.bidding_bias(values)
    aggressive_dist = policy(values, AGGRESSIVE.bidding_temperature, aggressive_bias)
    top = max(values.values())
    baseline_top_rate = sum(p for c, p in baseline_dist.items() if values[c] == top)
    aggressive_top_rate = sum(p for c, p in aggressive_dist.items() if values[c] == top)
    assert aggressive_top_rate > baseline_top_rate


# ---------------------------------------------------------------------------
# Card-play calibration
# ---------------------------------------------------------------------------


def _play_sweep(
    rng: random.Random, tricks_list: Sequence[int], n_per: int = 6
) -> list[BridgeState]:
    states = []
    for tricks in tricks_list:
        for _ in range(n_per):
            deck = Deck()
            deck.shuffle(rng)
            hands = tuple(tuple(deck.deal(tricks)) for _ in range(4))
            states.append(BridgeState(contract=CONTRACT, hands=hands))
    return states


def _play_rates(
    persona: BridgePersona, states: Sequence[BridgeState], rng: random.Random
) -> tuple[float, float, float]:
    """Average (P(a best-tied card), P(above-mean), P(at-or-below-mean))."""
    best_sum = above_sum = below_sum = 0.0
    count = 0
    for state in states:
        result = analyse(GAME, state, rng=rng)
        values = {c: v / result.scale for c, v in result.values.items()}
        if len(values) < 2:
            continue
        bias = persona.play_bias(GAME, state, values)
        dist = policy(values, persona.play_temperature, bias)
        top = max(values.values())
        best_sum += sum(p for c, p in dist.items() if values[c] == top)
        mean = sum(values.values()) / len(values)
        above_sum += sum(p for c, p in dist.items() if values[c] > mean)
        below_sum += sum(p for c, p in dist.items() if values[c] <= mean)
        count += 1
    return best_sum / count, above_sum / count, below_sum / count


def test_aggressive_play_leans_above_the_decisions_own_mean_value() -> None:
    states = _play_sweep(random.Random(43), [2, 3, 4, 5, 6, 8, 10])
    rng = random.Random(44)
    baseline_best, baseline_above, _ = _play_rates(BASELINE, states, rng)
    best, above, _ = _play_rates(AGGRESSIVE, states, rng)
    assert above > baseline_above + 0.1
    assert above < 0.85
    assert best > 0.25


def test_conservative_play_leans_at_or_below_the_decisions_own_mean_value() -> None:
    states = _play_sweep(random.Random(43), [2, 3, 4, 5, 6, 8, 10])
    rng = random.Random(44)
    baseline_best, _, baseline_below = _play_rates(BASELINE, states, rng)
    best, _, below = _play_rates(CONSERVATIVE, states, rng)
    assert below > baseline_below + 0.1
    assert below < 0.85
    assert best > 0.25


def test_no_persona_is_effectively_deterministic_in_card_play() -> None:
    states = _play_sweep(random.Random(43), [2, 3, 4, 5, 6, 8, 10])
    for name in ("baseline", "aggressive", "conservative"):
        best, _, _ = _play_rates(PERSONAS[name], states, random.Random(44))
        assert best > 0.25, f"{name} ignores its cards too often"


# ---------------------------------------------------------------------------
# selfish: the overtake-avoidance scenario, hand-constructed and checkable by hand
# ---------------------------------------------------------------------------


def _selfish_scenario() -> BridgeState:
    """declarer already led 9H, an opponent followed with 2H, and dummy holds KH
    (overtakes) and 8H (doesn't) - both a certain win either way for the side, so
    action_values() should score them exactly tied. Filler cards in the other
    suits keep this beyond DEPTH_GATE (routing to the myopic fallback, which is
    genuinely single-trick-only and ignores them entirely) - worked out by hand,
    same discipline as trick_odds.py's own tests.
    """
    filler_declarer = [card(t) for t in ["2C", "3C", "4C", "5C", "6C", "7C"]]
    filler_opp1 = [card(t) for t in ["2D", "3D", "4D", "5D", "6D", "7D"]]
    filler_dummy = [card(t) for t in ["2S", "3S", "4S", "5S", "6S"]]
    filler_opp3 = [card(t) for t in ["2C", "3D", "4S", "5C", "6D", "7S"]]
    return BridgeState(
        contract=Contract(trump=None, declarer=0, target=7),
        hands=(
            tuple(filler_declarer),
            tuple(filler_opp1),
            (card("KH"), card("8H"), *filler_dummy),
            (card("3H"), *filler_opp3),
        ),
        trick=((0, card("9H")), (1, card("2H"))),
    )


def test_selfish_values_are_genuinely_tied_at_the_hand_constructed_scenario() -> None:
    # Confirms the premise before checking the persona's behaviour on top of it -
    # if this weren't a real tie, selfish's bias wouldn't be "free" at all.
    state = _selfish_scenario()
    result = analyse(GAME, state, rng=None)
    values = {c: v / result.scale for c, v in result.values.items()}
    assert values[card("KH")] == values[card("8H")] == 1.0


def test_baseline_splits_evenly_between_two_genuinely_tied_cards() -> None:
    state = _selfish_scenario()
    result = analyse(GAME, state, rng=random.Random(1))
    values = {c: v / result.scale for c, v in result.values.items()}
    bias = BASELINE.play_bias(GAME, state, values)
    dist = policy(values, BASELINE.play_temperature, bias)
    assert abs(dist[card("KH")] - dist[card("8H")]) < 0.05


def test_selfish_avoids_overtaking_declarers_own_card_when_it_costs_nothing() -> None:
    state = _selfish_scenario()
    result = analyse(GAME, state, rng=random.Random(1))
    values = {c: v / result.scale for c, v in result.values.items()}
    bias = SELFISH.play_bias(GAME, state, values)
    assert bias.get(card("8H"), 0.0) > 0.0  # the non-overtaking card is favoured
    assert card("KH") not in bias  # overtaking gets no bias itself

    dist = policy(values, SELFISH.play_temperature, bias)
    assert dist[card("8H")] > 0.9  # near-certain, since it is a genuine free choice


def test_selfish_bias_requires_an_exact_tie_not_merely_a_close_value() -> None:
    # The "genuinely free" framing only holds for an EXACT tie - anything even a
    # hair apart is a real difference selfish has no business touching. Hand-crafted
    # values (not analyse()'s real output) isolate the tie-check itself; caught a
    # real gap during mutation testing where a rounded/fuzzy comparison slipped
    # through every other selfish test untouched.
    state = _selfish_scenario()
    values = {card("KH"): 1.0, card("8H"): 1.0 - 1e-9}
    assert SELFISH.play_bias(GAME, state, values) == {}


def test_selfish_bias_does_not_apply_outside_the_narrow_dummy_scenario() -> None:
    # Declarer's own decision, not dummy's - selfish should be indistinguishable
    # from baseline here, since the mechanism is deliberately narrow.
    state = new_deal(CONTRACT, rng=random.Random(5))
    result = analyse(GAME, state, rng=random.Random(6))
    values = {c: v / result.scale for c, v in result.values.items()}
    assert SELFISH.play_bias(GAME, state, values) == {}


# ---------------------------------------------------------------------------
# baiter: only ever picks among genuinely tied-value options
# ---------------------------------------------------------------------------


def _baiter_tied_scenario() -> BridgeState:
    """dummy's last two cards, 8H and 9H, are adjacent in what's left of hearts -
    nothing else of that suit is anywhere in play to sit between them, so
    play_groups() must treat them as one group. Note this is a *different* notion
    of "tied" than selfish's: _selfish_scenario()'s KH/8H are tied by trick_odds.py
    value (both a certain win) but NOT by play_groups()'s stricter rank-adjacency
    rule, since 9H sits on the table between them in suit order - equivalence.py
    correctly refuses to call them interchangeable. The two mechanisms deliberately
    don't share a scenario.
    """
    return BridgeState(
        contract=Contract(trump=None, declarer=0, target=7),
        hands=((), (), (card("8H"), card("9H")), (card("4C"),)),
        trick=((0, card("2C")), (1, card("3C"))),
    )


def test_baiter_biases_the_lowest_card_of_a_provably_tied_group() -> None:
    state = _baiter_tied_scenario()
    assert state.to_play == 2
    bias = BAITER.play_bias(GAME, state, {})  # baiter's bias ignores values entirely
    assert bias.get(card("8H"), 0.0) > 0.0
    assert card("9H") not in bias


def test_baiter_never_biases_toward_a_card_outside_a_tied_group() -> None:
    # Across a real spread of random endgames, every card baiter's bias favours
    # must belong to a group equivalence.py has actually proven tied - this is the
    # whole point of it being "free," so it must never cost real value.
    rng = random.Random(7)
    for _ in range(20):
        deck = Deck()
        deck.shuffle(rng)
        hands = tuple(tuple(deck.deal(4)) for _ in range(4))
        state = BridgeState(contract=CONTRACT, hands=hands)
        legal = GAME.legal_actions(state)
        in_play = [c for hand in state.hands for c in hand]
        groups = play_groups(legal, in_play)
        tied_cards = {c for group in groups if len(group) >= 2 for c in group}
        bias = BAITER.play_bias(GAME, state, {})
        assert set(bias).issubset(tied_cards)


def test_baiter_favours_only_the_lowest_card_within_each_tied_group() -> None:
    rng = random.Random(8)
    found_a_multi_card_group = False
    for _ in range(20):
        deck = Deck()
        deck.shuffle(rng)
        hands = tuple(tuple(deck.deal(4)) for _ in range(4))
        state = BridgeState(contract=CONTRACT, hands=hands)
        legal = GAME.legal_actions(state)
        in_play = [c for hand in state.hands for c in hand]
        groups = play_groups(legal, in_play)
        bias = BAITER.play_bias(GAME, state, {})
        for group in groups:
            if len(group) < 2:
                continue
            found_a_multi_card_group = True
            lowest = min(group, key=lambda c: c.rank.value)
            assert bias.get(lowest, 0.0) > 0.0
            for other in group:
                if other != lowest:
                    assert other not in bias
    assert found_a_multi_card_group  # otherwise this test proves nothing


# ---------------------------------------------------------------------------
# unaware: always the myopic fallback, never PIMC, regardless of depth
# ---------------------------------------------------------------------------


def test_unaware_never_invokes_pimc_even_deep_in_the_hand(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def spy(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("analyse() should never be called for an unaware persona")

    monkeypatch.setattr("games.bridge.personas.analyse", spy)

    # A shallow (well within DEPTH_GATE) position - exactly where a normal persona
    # would call analyse() and hit PIMC.
    deck = Deck()
    deck.shuffle(random.Random(9))
    hands = tuple(tuple(deck.deal(3)) for _ in range(4))
    state = BridgeState(contract=CONTRACT, hands=hands)

    strategy = CardPlayStrategy(UNAWARE, random.Random(10))
    card_chosen = strategy(GAME, state)
    assert card_chosen in GAME.legal_actions(state)
    assert calls == 0


def test_unaware_bidding_uses_the_conservative_config() -> None:
    assert UNAWARE.bidding_temperature == CONSERVATIVE.bidding_temperature
    assert UNAWARE.bidding_bias is CONSERVATIVE.bidding_bias


# ---------------------------------------------------------------------------
# Jitter
# ---------------------------------------------------------------------------


def test_jitter_stays_within_the_bounded_spread() -> None:
    rng = random.Random(11)
    for _ in range(200):
        result = jittered(BASELINE, rng, spread=0.15)
        low = BASELINE.bidding_temperature * 0.85
        high = BASELINE.bidding_temperature * 1.15
        assert low <= result.bidding_temperature <= high
        low_play = BASELINE.play_temperature * 0.85
        high_play = BASELINE.play_temperature * 1.15
        assert low_play <= result.play_temperature <= high_play


def test_jitter_at_the_loosest_edge_of_the_range() -> None:
    # rng.uniform(-spread, spread) returning exactly +spread is the loosest the
    # jitter can go - forced directly rather than hoping a seed happens to land
    # there, matching the "test the edges, not just the center" instruction.
    class FixedRng(random.Random):
        def uniform(self, a: float, b: float) -> float:
            return b  # always the loosest end

    result = jittered(BASELINE, FixedRng(), spread=0.15)
    assert result.bidding_temperature == BASELINE.bidding_temperature * 1.15
    assert result.play_temperature == BASELINE.play_temperature * 1.15


def test_jitter_at_the_tightest_edge_of_the_range() -> None:
    class FixedRng(random.Random):
        def uniform(self, a: float, b: float) -> float:
            return a  # always the tightest end

    result = jittered(BASELINE, FixedRng(), spread=0.15)
    assert result.bidding_temperature == BASELINE.bidding_temperature * 0.85
    assert result.play_temperature == BASELINE.play_temperature * 0.85


def test_jitter_scales_bias_magnitude_consistently_with_temperature() -> None:
    class FixedRng(random.Random):
        def uniform(self, a: float, b: float) -> float:
            return b

    result = jittered(AGGRESSIVE, FixedRng(), spread=0.15)
    values = bid_values(
        [card("AS"), card("2S"), card("3S"), card("4S"), card("5S"), card("6S")]
        + [card("2H"), card("3H"), card("2D"), card("3D"), card("2C"), card("3C"), card("4C")],
        new_auction(dealer=0),
    )
    base_bias = AGGRESSIVE.bidding_bias(values)
    jittered_bias = result.bidding_bias(values)
    for call, amount in base_bias.items():
        assert jittered_bias[call] == amount * 1.15


def test_jitter_is_reproducible_with_a_seeded_rng() -> None:
    a = jittered(AGGRESSIVE, random.Random(99))
    b = jittered(AGGRESSIVE, random.Random(99))
    assert a.bidding_temperature == b.bidding_temperature
    assert a.play_temperature == b.play_temperature


def test_jitter_is_drawn_once_per_session_not_reused_across_different_personas() -> None:
    # Two different personas jittered from the SAME rng state draw independent
    # factors (the rng advances), not the same factor reused - confirms jitter is
    # a real per-instantiation draw, not an accidental constant.
    rng = random.Random(12)
    first = jittered(BASELINE, rng)
    second = jittered(BASELINE, rng)
    assert first.bidding_temperature != second.bidding_temperature


# ---------------------------------------------------------------------------
# Strategies return legal choices
# ---------------------------------------------------------------------------


def test_bidding_strategy_returns_a_legal_call() -> None:
    hand = new_deal(CONTRACT, rng=random.Random(13)).hands[0]
    strategy = BiddingStrategy(BASELINE, random.Random(14))
    call = strategy(hand, new_auction(dealer=0))
    assert call in new_auction(dealer=0).legal_calls()


def test_card_play_strategy_returns_a_legal_card() -> None:
    state = new_deal(CONTRACT, rng=random.Random(15))
    strategy = CardPlayStrategy(BASELINE, random.Random(16))
    played = strategy(GAME, state)
    assert played in GAME.legal_actions(state)
