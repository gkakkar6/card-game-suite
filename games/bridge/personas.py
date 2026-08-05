"""Bridge personas: named playing styles for both bidding and card play, built on
the shared quantal engine (ARCHITECTURE.md §3).

One combined profile per seat, not independent bidding/play dials - the simplest
complete version, matching what was settled during planning. Six named styles:
baseline (near-optimal, both domains), aggressive (biased toward higher-tier bids and
higher-value committal card play), conservative (the mirror), unaware (a genuinely
different kind of imperfection - no bias or temperature at all, just always the
myopic single-trick fallback for card play, never PIMC, regardless of tricks
remaining), selfish (biased toward declarer's own hand getting credit for a trick
over dummy's, in the one narrow case that costs the partnership nothing - card play
only for now), baiter (biased toward the less-committal-looking card among options
`equivalence.py` has already proven tied in value - card play only for now).

**Why bridge's bias can't be a static per-action map the way poker's is.** Poker's
`ActionType` is five fixed values that recur identically at every decision, so a
persona's bias is naturally "add this much to BET, forever." Bridge's actions are
`Call`s and `Card`s - a different, decision-specific set every time - so "bias toward
higher-tier bids" can't be a fixed lookup table; it has to be computed fresh each
decision from that decision's own scored values. Every bias below is a small function
of the values, not a static dict, still fed into `engine.personas.quantal.choose()`
the same way poker's static ones are - the quantal mechanism itself doesn't care where
the bias dict came from.

**Bidding needs no per-decision normalisation; card play does, and why.**
`bid_values()` already scores every call on a fixed 0-5 tier scale (`UNSCORED` to
`PREFERRED_TIER`, games/bridge/bidding.py) - the same currency at every decision
regardless of the hand or the auction, unlike poker's chip values or bridge card
play's trick counts, both of which scale with what's actually at stake. Card play's
`action_values.analyse()` reports a `scale` for exactly this reason (ARCHITECTURE.md
§3's normalisation rule) - divide by it before bias or temperature ever sees a value.
"""

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from engine.cards import Card, Suit
from engine.personas.quantal import choose
from engine.trick_taking.equivalence import play_groups
from engine.trick_taking.resolution import trick_winner
from games.bridge.action_values import analyse
from games.bridge.bidding import SIGNATURE_TIER, SIGNOFF_TIER, Auction, Call, bid_values
from games.bridge.rules import Bridge, BridgeState
from games.bridge.trick_odds import trick_win_probabilities

BiddingBias = Callable[[dict[Call, float]], dict[Call, float]]
PlayBias = Callable[[Bridge, BridgeState, dict[Card, float]], dict[Card, float]]


def _no_bidding_bias(values: dict[Call, float]) -> dict[Call, float]:
    return {}


def _no_play_bias(game: Bridge, state: BridgeState, values: dict[Card, float]) -> dict[Card, float]:
    return {}


@dataclass(frozen=True)
class BridgePersona:
    """One named style, covering both domains - a single combined profile per seat,
    not independent bidding/play dials (ARCHITECTURE.md §3)."""

    name: str
    bidding_temperature: float
    bidding_bias: BiddingBias
    play_temperature: float
    play_bias: PlayBias
    # A different kind of imperfection from bias/temperature: which information a
    # persona's card play works from in the first place, not what it does with that
    # information (ARCHITECTURE.md §3's trick-history-recall note is the general
    # version of this idea - "unaware" is the first concrete instance of it). True
    # for exactly one persona, and overrides play_temperature/play_bias entirely when
    # set: card play always uses trick_odds.py's fallback, never PIMC.
    unaware: bool = False


# ---------------------------------------------------------------------------
# Bidding bias: threshold on the tier a call already scored, not proportional to it.
#
# A bias *proportional* to the raw value would just rescale every value by the same
# factor - mathematically identical to dividing the temperature by that factor, not a
# genuinely different direction at all (ARCHITECTURE.md §3's actual point: bias and
# temperature have to be two different mechanisms, or one of them is doing nothing).
# A threshold avoids this: which calls get the bonus depends on which *tier* they
# landed in, not smoothly on their exact score, so it can't collapse into a
# temperature change.
# ---------------------------------------------------------------------------


def _tiered_bidding_bias(amount: float) -> BiddingBias:
    """Push high-tier and low-tier calls apart by `amount`, in opposite directions.

    A one-sided version of this (bonus only on high-tier calls, say) turned out to be
    close to invisible when measured: Pass is scored on *every* decision, so it is
    already the only real competition a genuinely strong call has, and a strong call
    typically already beats Pass by several tiers before any bias is added - boosting
    an already-dominant option further barely moves how often it gets chosen. Pass is
    what makes the bias measurable, since it is the one call always present to push
    against - so both ends of the tier scale need to move, not just one, for either
    "aggressive" or "conservative" to be a real, checkable lean rather than a
    bias that happens to do nothing in practice. `amount` is positive for aggressive
    (pushes toward high tiers, away from Pass) and negative for conservative (the
    mirror) - one function, one sign flip, rather than two separate ones to keep
    in sync.
    """

    def bias(values: dict[Call, float]) -> dict[Call, float]:
        result: dict[Call, float] = {}
        for call, value in values.items():
            if value >= SIGNATURE_TIER:
                result[call] = amount
            elif value <= SIGNOFF_TIER:
                result[call] = -amount
        return result

    return bias


# ---------------------------------------------------------------------------
# Card-play bias: same threshold idea, against each decision's own mean value as the
# reference point - card play has no fixed tiers the way bidding does, so "committal"
# has to be judged relative to what the other legal cards at this same decision are
# worth, the same way action_values.py's own _reference_equity() picks a
# decision-relative bar rather than a fixed number.
# ---------------------------------------------------------------------------


def _cards_above_the_mean(amount: float) -> PlayBias:
    def bias(game: Bridge, state: BridgeState, values: dict[Card, float]) -> dict[Card, float]:
        if not values:
            return {}
        mean = sum(values.values()) / len(values)
        return {card: amount for card, value in values.items() if value > mean}

    return bias


def _cards_at_or_below_the_mean(amount: float) -> PlayBias:
    def bias(game: Bridge, state: BridgeState, values: dict[Card, float]) -> dict[Card, float]:
        if not values:
            return {}
        mean = sum(values.values()) / len(values)
        return {card: amount for card, value in values.items() if value <= mean}

    return bias


# ---------------------------------------------------------------------------
# Selfish: biased toward declarer's own hand getting credit for a trick over
# dummy's - the one bridge archetype poker has no concept of at all (partnership).
#
# The mechanism is deliberately narrow, not a general "declarer plays for themselves"
# rule: it only ever applies on dummy's own turn, when declarer has already played to
# this same trick, and only among cards *already proven tied* by action_values() -
# overtaking declarer's own card or not costs the partnership nothing in that exact
# case (Phase 3's same-side fix is what makes "tied" the honest word here rather than
# a coincidence). Outside that narrow window, selfish plays exactly like baseline.
# ---------------------------------------------------------------------------


def _overtakes(candidate: Card, declarer_card: Card, led_suit: Suit, trump: Suit | None) -> bool:
    """Does `candidate` beat declarer's own already-played card in this trick?

    Same two-card comparison trick_odds.py's own `_beats()` uses - small enough (and
    private there) that duplicating it here reads more plainly than exporting a
    one-line helper across modules for this alone.
    """
    trick = ((0, declarer_card), (1, candidate))
    return trick_winner(trick, led_suit, trump) == 1


def _selfish_bias(amount: float) -> PlayBias:
    def bias(game: Bridge, state: BridgeState, values: dict[Card, float]) -> dict[Card, float]:
        if state.to_play != state.dummy or game.current_player(state) != state.contract.declarer:
            return {}
        declarer_card = next(
            (card for seat, card in state.trick if seat == state.contract.declarer), None
        )
        if declarer_card is None:
            return {}
        led_suit = state.led_suit
        assert led_suit is not None  # declarer_card being found means the trick isn't empty
        trump = state.contract.trump

        by_value: dict[float, list[Card]] = {}
        for card, value in values.items():
            by_value.setdefault(value, []).append(card)

        result: dict[Card, float] = {}
        for tied_cards in by_value.values():
            if len(tied_cards) < 2:
                continue
            overtaking = [c for c in tied_cards if _overtakes(c, declarer_card, led_suit, trump)]
            staying_under = [c for c in tied_cards if c not in overtaking]
            if overtaking and staying_under:
                for card in staying_under:
                    result[card] = amount
        return result

    return bias


# ---------------------------------------------------------------------------
# Baiter: biased toward the less-committal-looking card within a group
# equivalence.py has already proven genuinely tied - never a real cost, since the
# alternatives are proven equal.
#
# Convention: the *lowest* card of a tied group looks least committal, whether
# leading or following. Leading low is the standard passive signal (not trying to
# establish command of the suit). Following with the lowest card that still does the
# job - winning the trick as cheaply as possible rather than "spending" a higher card
# - is the same idea from the other side of the trick, and it is a real, ordinary
# bridge principle independent of this persona (play the cheapest card that achieves
# the same result). Both cases point the same way, so one convention covers both
# rather than needing a separate rule for leading versus following.
# ---------------------------------------------------------------------------


def _baiter_bias(amount: float) -> PlayBias:
    def bias(game: Bridge, state: BridgeState, values: dict[Card, float]) -> dict[Card, float]:
        # Genuinely doesn't use values - the whole mechanism is structural (which
        # cards are provably tied), so the candidate list comes straight from the
        # game/state, not from whatever the caller happened to pass as values.
        legal = game.legal_actions(state)
        in_play = [card for hand in state.hands for card in hand]
        in_play.extend(card for _seat, card in state.trick)
        groups = play_groups(legal, in_play)

        result: dict[Card, float] = {}
        for group in groups:
            if len(group) < 2:
                continue
            lowest = min(group, key=lambda card: card.rank.value)
            result[lowest] = amount
        return result

    return bias


# ---------------------------------------------------------------------------
# The six personas. Every temperature and bias below was measured, not guessed - see
# DECISIONS.md for the real numbers and the sample they were checked against, the
# same discipline poker's five personas were built with.
# ---------------------------------------------------------------------------

BASELINE_BIDDING_TEMPERATURE = 0.3
BASELINE_PLAY_TEMPERATURE = 0.05

AGGRESSIVE_BID_BIAS = 1.5
CONSERVATIVE_BID_BIAS = -1.5

AGGRESSIVE_PLAY_BIAS = 0.15
CONSERVATIVE_PLAY_BIAS = 0.15

SELFISH_BIAS = 0.5
BAITER_BIAS = 0.5

BASELINE = BridgePersona(
    name="baseline",
    bidding_temperature=BASELINE_BIDDING_TEMPERATURE,
    bidding_bias=_no_bidding_bias,
    play_temperature=BASELINE_PLAY_TEMPERATURE,
    play_bias=_no_play_bias,
)

AGGRESSIVE = BridgePersona(
    name="aggressive",
    bidding_temperature=BASELINE_BIDDING_TEMPERATURE,
    bidding_bias=_tiered_bidding_bias(AGGRESSIVE_BID_BIAS),
    play_temperature=BASELINE_PLAY_TEMPERATURE,
    play_bias=_cards_above_the_mean(AGGRESSIVE_PLAY_BIAS),
)

CONSERVATIVE = BridgePersona(
    name="conservative",
    bidding_temperature=BASELINE_BIDDING_TEMPERATURE,
    bidding_bias=_tiered_bidding_bias(CONSERVATIVE_BID_BIAS),
    play_temperature=BASELINE_PLAY_TEMPERATURE,
    play_bias=_cards_at_or_below_the_mean(CONSERVATIVE_PLAY_BIAS),
)

SELFISH = BridgePersona(
    name="selfish",
    # No natural bidding equivalent yet (ARCHITECTURE.md §3), so bidding falls back
    # to baseline rather than being left undefined.
    bidding_temperature=BASELINE_BIDDING_TEMPERATURE,
    bidding_bias=_no_bidding_bias,
    play_temperature=BASELINE_PLAY_TEMPERATURE,
    play_bias=_selfish_bias(SELFISH_BIAS),
)

BAITER = BridgePersona(
    name="baiter",
    bidding_temperature=BASELINE_BIDDING_TEMPERATURE,
    bidding_bias=_no_bidding_bias,
    play_temperature=BASELINE_PLAY_TEMPERATURE,
    play_bias=_baiter_bias(BAITER_BIAS),
)

UNAWARE = BridgePersona(
    name="unaware",
    # No shallow-fallback equivalent exists for bidding the way trick_odds.py's
    # myopic heuristic does for card play, so bidding uses the conservative config -
    # the safest reasonable default, not a special "unaware bidding" mechanism.
    bidding_temperature=CONSERVATIVE.bidding_temperature,
    bidding_bias=CONSERVATIVE.bidding_bias,
    play_temperature=BASELINE_PLAY_TEMPERATURE,
    play_bias=_no_play_bias,
    unaware=True,
)

PERSONAS: dict[str, BridgePersona] = {
    persona.name: persona
    for persona in (BASELINE, AGGRESSIVE, CONSERVATIVE, SELFISH, BAITER, UNAWARE)
}


# ---------------------------------------------------------------------------
# Session-level jitter: one randomly-shifted version of a persona's temperature/bias
# magnitude, drawn once when a persona is instantiated for a session and reused for
# every decision in it - not re-drawn per hand or per decision, which would compound
# with the existing per-decision randomness from temperature itself in a way that
# would be hard to reason about or test (ARCHITECTURE.md §3).
# ---------------------------------------------------------------------------

DEFAULT_JITTER = 0.15  # +/- 15% of the base value - small, explicitly bounded


def jittered(
    persona: BridgePersona, rng: random.Random, spread: float = DEFAULT_JITTER
) -> BridgePersona:
    """One persona instance with temperature and bias magnitude nudged by the same
    single random factor, drawn once - "one randomly-jittered version," not a
    separate independent nudge per number. `spread` bounds the draw to within
    `spread` of the base value in either direction (0.15 default means +/-15%).

    Bias *functions* can't be jittered by scaling their output at call time without
    either redrawing the factor on every call (defeating the point of sampling once)
    or storing extra state - instead, the jitter is baked into a new closure around
    the original bias function, at construction time, once.
    """
    scale = 1.0 + rng.uniform(-spread, spread)
    return BridgePersona(
        name=persona.name,
        bidding_temperature=persona.bidding_temperature * scale,
        bidding_bias=_scaled_bidding_bias(persona.bidding_bias, scale),
        play_temperature=persona.play_temperature * scale,
        play_bias=_scaled_play_bias(persona.play_bias, scale),
        unaware=persona.unaware,
    )


def _scaled_bidding_bias(bias: BiddingBias, scale: float) -> BiddingBias:
    def scaled_bias(values: dict[Call, float]) -> dict[Call, float]:
        return {call: amount * scale for call, amount in bias(values).items()}

    return scaled_bias


def _scaled_play_bias(bias: PlayBias, scale: float) -> PlayBias:
    def scaled_bias(
        game: Bridge, state: BridgeState, values: dict[Card, float]
    ) -> dict[Card, float]:
        return {card: amount * scale for card, amount in bias(game, state, values).items()}

    return scaled_bias


# ---------------------------------------------------------------------------
# Strategies: turn a persona into an actual chooser, for each domain.
# ---------------------------------------------------------------------------


class BiddingStrategy:
    """Chooses a call by scoring the legal ones with bid_values(), then letting the
    persona pick - no normalisation needed, bid_values() is already on a fixed scale
    (see the module docstring)."""

    def __init__(self, persona: BridgePersona, rng: random.Random | None = None) -> None:
        self.persona = persona
        self.rng = rng if rng is not None else random.Random()

    def __call__(self, hand: Sequence[Card], auction: Auction) -> Call:
        values = bid_values(hand, auction)
        bias = self.persona.bidding_bias(values)
        return choose(values, self.persona.bidding_temperature, bias, self.rng)


class CardPlayStrategy:
    """Chooses a card by scoring the legal ones - via action_values.analyse() and its
    reported scale, unless the persona is "unaware", in which case card play always
    uses trick_odds.py's myopic fallback directly and never attempts PIMC at all,
    regardless of how many tricks remain."""

    def __init__(self, persona: BridgePersona, rng: random.Random | None = None) -> None:
        self.persona = persona
        self.rng = rng if rng is not None else random.Random()

    def __call__(self, game: Bridge, state: BridgeState) -> Card:
        if self.persona.unaware:
            values = trick_win_probabilities(game, state)
        else:
            result = analyse(game, state, rng=self.rng)
            values = {card: value / result.scale for card, value in result.values.items()}
        bias = self.persona.play_bias(game, state, values)
        return choose(values, self.persona.play_temperature, bias, self.rng)
