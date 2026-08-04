"""The auction: bidding for a bridge deal, ending in a Contract that feeds directly
into games/bridge/rules.py (ARCHITECTURE.md §5, bridge Phase 4).

SAYC (Standard American Yellow Card), settled scope. V1 assumes an **uncontested
auction** - the opposing side always passes, no interference, no doubles - the same
discipline every earlier phase used: build the honest, complete version of what
remains once something real is assumed away, rather than a half version of everything.

In scope: HCP counting and balanced-shape detection (games/bridge/hand_evaluation.py);
opening bids (1-of-a-suit ~12-21 HCP longest suit, 1NT 15-17 balanced, 2NT 20-21
balanced, the strong 2C opening at 22+, pass below opening strength); natural
constructive bidding (raises with support, new suits showing values, natural - not
Stayman or Jacoby transfer - responses to 1NT, opener's rebid); correct auction
termination.

Explicitly deferred, named rather than silently missing: named conventions (Stayman,
Jacoby transfers, Blackwood), competitive bidding (interference, doubles), preemptive
openings, and the response structure to a strong 2C opening beyond the opening bid
itself.

**Architected for future bias/temperature, not bolted on later.** `bid_values(hand,
auction) -> dict[Call, float]` is a fit score per legal call, not a function that
returns one chosen bid directly - the same shape as poker's and bridge card-play's
`action_values()`. V1's actual behaviour is just picking the best-scoring legal call,
but the values exist for a future persona layer to bias between, and the thresholds
below (minimum HCP to open, minimum support to raise, and so on) are real named
constants a persona could later shift, not numbers buried inline in conditionals.

**Testing is necessarily different in character from Phases 1-3.** There is no
double-dummy-style objectively computable answer here - what a bid means is
convention, not a computed fact. Tests check against hand-constructed hands with
known, textbook-standard auctions, sourced from an authoritative SAYC reference -
closer in spirit to how the NLP keyword matching was validated against curated real
examples than to how the solver was validated against combinatorics.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from engine.cards import Card, Suit
from engine.trick_taking.resolution import Seat
from games.bridge.deal import SEATS
from games.bridge.hand_evaluation import hcp, is_balanced, suit_lengths
from games.bridge.rules import Contract, partner

MIN_LEVEL = 1
MAX_LEVEL = 7

# One above Suit.SPADES.value (3) - notrump outranks every suit at the same level,
# matching Contract's own trump=None-means-notrump convention extended into a total
# order over calls, the same way hand_evaluator.py's HandRank tuples compare correctly
# just by being ordered right.
NOTRUMP_RANK = 4


@dataclass(frozen=True)
class Bid:
    """One call that names a contract: level 1-7, strain a suit or notrump (None -
    reusing Contract's own convention rather than inventing a parallel one)."""

    level: int
    strain: Suit | None

    def __post_init__(self) -> None:
        if not MIN_LEVEL <= self.level <= MAX_LEVEL:
            raise ValueError(f"level must be {MIN_LEVEL}-{MAX_LEVEL}, got {self.level}")

    @property
    def rank(self) -> tuple[int, int]:
        """(level, strain rank) - compares correctly via tuple ordering, the same
        trick hand_evaluator.py's HandRank tuples use. Level dominates entirely,
        since it is compared first; only a tie on level falls through to strain."""
        strain_rank = NOTRUMP_RANK if self.strain is None else self.strain.value
        return (self.level, strain_rank)

    def __str__(self) -> str:
        strain = "NT" if self.strain is None else self.strain.symbol
        return f"{self.level}{strain}"


# A call is a Bid, or Pass - represented as None, the same convention Bid.strain (and
# Contract.trump before it) already uses None for "the notrump case", extended here to
# "the no-bid case". Avoids a second type for what is otherwise a two-line concept.
Call = Bid | None


def _outranks(candidate: Bid, current_high: Bid | None) -> bool:
    return current_high is None or candidate.rank > current_high.rank


def all_bids_above(current_high: Bid | None) -> list[Bid]:
    """Every legal bid given the current highest bid (or none yet) - level 1-7 across
    every strain, filtered to only what actually outranks what is already on the
    table. Enumerated explicitly, matching legal_actions() elsewhere in this repo
    returning the actual list rather than just a legality predicate."""
    return [
        bid
        for level in range(MIN_LEVEL, MAX_LEVEL + 1)
        for strain in (*Suit, None)
        if _outranks(bid := Bid(level, strain), current_high)
    ]


@dataclass(frozen=True)
class Auction:
    """The bidding for one deal: who calls first, and every call made since, in
    order. Immutable, matching BridgeState - apply() returns a new Auction."""

    dealer: Seat
    calls: tuple[Call, ...] = ()

    def seat_at(self, index: int) -> Seat:
        return (self.dealer + index) % SEATS

    @property
    def to_call(self) -> Seat:
        return self.seat_at(len(self.calls))

    @property
    def _last_bid_index(self) -> int | None:
        """Index of the most recent real bid. Legal calls only ever raise the
        contract or pass, so the highest bid on the table is always the last
        non-pass call - no need to search for a maximum."""
        for index in range(len(self.calls) - 1, -1, -1):
            if self.calls[index] is not None:
                return index
        return None

    @property
    def highest_bid(self) -> Bid | None:
        index = self._last_bid_index
        return None if index is None else self.calls[index]

    @property
    def last_bidder(self) -> Seat | None:
        index = self._last_bid_index
        return None if index is None else self.seat_at(index)

    def is_passed_out(self) -> bool:
        """Four passes with no bid at all - a real, distinct outcome: no contract,
        no play, not an error case."""
        return len(self.calls) == SEATS and self.highest_bid is None

    def is_over(self) -> bool:
        """Three consecutive passes after at least one real bid, or passed out."""
        if self.is_passed_out():
            return True
        if len(self.calls) < SEATS or self.highest_bid is None:
            return False
        return all(call is None for call in self.calls[-3:])

    def legal_calls(self) -> list[Call]:
        """Pass, plus every bid that outranks the current highest - empty once the
        auction is over, the same way legal_actions() empties out at a terminal
        BridgeState."""
        if self.is_over():
            return []
        calls: list[Call] = [None]
        calls.extend(all_bids_above(self.highest_bid))
        return calls

    def apply(self, call: Call) -> "Auction":
        if call not in self.legal_calls():
            raise ValueError(f"{call} is not a legal call for seat {self.to_call}")
        return replace(self, calls=(*self.calls, call))

    def contract(self) -> Contract | None:
        """The Contract this auction produced, or None for a passed-out hand.

        Declarer is whoever, on the side that won the auction, FIRST named the
        strain that was finally agreed - not necessarily whoever made the final bid.
        A side can easily end up declaring a strain one partner introduced several
        rounds before the other partner's later, higher bid in the same strain
        actually won the auction.
        """
        if not self.is_over():
            raise ValueError("the auction is not over yet")
        if self.is_passed_out():
            return None
        final = self.highest_bid
        bidder = self.last_bidder
        assert final is not None and bidder is not None  # is_passed_out() excluded this
        winning_side = {bidder, partner(bidder)}
        declarer = next(
            self.seat_at(index)
            for index, call in enumerate(self.calls)
            if call is not None
            and call.strain == final.strain
            and self.seat_at(index) in winning_side
        )
        return Contract(trump=final.strain, declarer=declarer, target=final.level + 6)


def new_auction(dealer: Seat) -> Auction:
    return Auction(dealer=dealer)


# ---------------------------------------------------------------------------
# bid_values(): a fit score per legal call (ARCHITECTURE.md §5).
#
# Every threshold below is a real, named constant rather than a number buried in a
# condition, so a future persona could shift them without touching the logic that
# uses them - the same discipline as poker's action_values.py sizing constants.
#
# Scores are grouped into a handful of fixed tiers rather than one continuous scale:
# a call that matches its textbook criteria outright scores within its tier: a higher
# tier always beats every lower one (SIGNATURE beats FIT beats SIGNOFF beats PASS
# beats UNSCORED), and a small tie-break score inside a tier only orders calls that
# would otherwise be equally valid. This keeps the priority order between different
# *kinds* of call (e.g. showing a new suit before making a plain raise) an explicit,
# readable fact about the tiers, not an emergent property of tuned numbers.
# ---------------------------------------------------------------------------

# Tier boundaries. UNSCORED covers every legal call V1 has no real judgement about
# (jumps, preempts, anything beyond the auction stages actually modelled below) - it
# must never outscore PASS, so an unmodelled call is never chosen over passing.
UNSCORED = 0.0
PASS_TIER = 1.0
SIGNOFF_TIER = 2.0  # 1NT/2NT responses, notrump rebids - descriptive but not showing extra shape
FIT_TIER = 3.0  # raises: a known trump fit is on the table
SIGNATURE_TIER = 4.0  # a new suit, or an opening bid - the most information-dense call available
PREFERRED_TIER = 5.0  # beats every call above when more than one would otherwise qualify:
# notrump over a same-ranged suit opening, a limit raise over a plain raise, and so on -
# a real priority between two calls that are both individually legitimate, not a tie-break
# nudge of convenience.

OPEN_MIN_HCP = 12
OPEN_MAX_HCP = 21
STRONG_TWO_CLUB_MIN_HCP = 22  # conventional and artificial - shows strength, not real clubs

NOTRUMP_1_MIN_HCP = 15
NOTRUMP_1_MAX_HCP = 17
NOTRUMP_2_MIN_HCP = 20
NOTRUMP_2_MAX_HCP = 21

RESPONSE_MIN_HCP = 6  # minimum to respond at all, rather than pass partner's opening
NEW_SUIT_AT_TWO_MIN_HCP = 10  # promised when a new suit can only be shown at the 2-level
NOTRUMP_RESPONSE_MIN_HCP = 6
NOTRUMP_RESPONSE_MAX_HCP = 9
SIMPLE_RAISE_MIN_SUPPORT = 3  # cards in partner's suit
SIMPLE_RAISE_MIN_HCP = 6
SIMPLE_RAISE_MAX_HCP = 9
LIMIT_RAISE_MIN_HCP = 10
LIMIT_RAISE_MAX_HCP = 11  # ACBL SAYC booklet: "limit raise (10-11 dummy points...)"

NATURAL_1NT_RESPONSE_LONG_SUIT = 5  # length needed to sign off in a suit over 1NT
NATURAL_1NT_INVITE_MIN_HCP = 8
NATURAL_1NT_INVITE_MAX_HCP = 9
NATURAL_1NT_GAME_MIN_HCP = 10

REBID_MINIMUM_MAX_HCP = 15  # opener's own strength tiers, from the opening's 12-21 range
REBID_MEDIUM_MAX_HCP = 18
REBID_RAISE_AGAIN_MIN_HCP = 16  # opener needs at least this much to bid on over a raise
REBID_NOTRUMP_MIN_HCP = 18  # opener needs extra strength to invite notrump after 1NT
OPENERS_REBID_SUIT_MIN_LENGTH = 6  # rebidding your own suit promises extra length, not just 5


def _cheapest(legal: Sequence[Call], strain: Suit | None) -> Bid | None:
    """The lowest-level legal bid in `strain`, or None if none is legal right now.

    Several branches below want "the plain bid in this strain", not "any bid in this
    strain" - without this, a raise and a jump raise of the same suit (very different
    bids, meaning very different things) would both match a strain-only condition and
    score identically, leaving which one gets chosen to incidental dict-ordering
    rather than a real rule.
    """
    candidates = [call for call in legal if call is not None and call.strain == strain]
    return min(candidates, key=lambda bid: bid.level, default=None)


def _fallback_values(legal_calls: Sequence[Call]) -> dict[Call, float]:
    """Every legal call scored UNSCORED except Pass - the honest response for any
    auction stage V1 has no real judgement about (an opener's rebid after a notrump
    opening, anything beyond a single round of bidding each way), rather than
    guessing at a call the way a made-up score would."""
    return {call: (PASS_TIER if call is None else UNSCORED) for call in legal_calls}


def _own_and_partner_bids(auction: Auction, seat: Seat) -> tuple[list[Bid], list[Bid]]:
    """This seat's own real bids so far, and their partner's - Pass calls filtered
    out, since V1's uncontested auction means every real bid belongs to one side or
    the other and Pass carries no information worth branching on here."""
    mine: list[Bid] = []
    theirs: list[Bid] = []
    partner_seat = partner(seat)
    for index, call in enumerate(auction.calls):
        if call is None:
            continue
        acting_seat = auction.seat_at(index)
        if acting_seat == seat:
            mine.append(call)
        elif acting_seat == partner_seat:
            theirs.append(call)
    return mine, theirs


def bid_values(hand: Sequence[Card], auction: Auction) -> dict[Call, float]:
    """A fit score per legal call at `auction.to_call`'s current decision.

    V1 only has real judgement about the first three stages of a natural, uncontested
    auction: an opening decision, the opening side's first response, and opener's
    rebid after that response. Anything past that - or a rebid after a notrump
    opening, whose responses (Stayman, transfers) are out of scope - falls back to
    scoring every call UNSCORED except Pass, honestly rather than guessing.
    """
    legal = auction.legal_calls()
    my_bids, partner_bids = _own_and_partner_bids(auction, auction.to_call)

    if not my_bids and not partner_bids:
        return _opening_values(hand, legal)
    if not my_bids and partner_bids:
        return _response_values(hand, legal, partner_bids[0])
    if my_bids and partner_bids:
        return _rebid_values(hand, legal, my_bids[0], partner_bids[0])
    return _fallback_values(legal)


def choose_bid(hand: Sequence[Card], auction: Auction) -> Call:
    """The single best-scoring legal call - V1's actual behaviour, on top of the
    dict bid_values() returns for a future persona layer to bias between instead."""
    values = bid_values(hand, auction)
    return max(values, key=lambda call: values[call])


# --- Opening ----------------------------------------------------------------------


def _opening_suit(lengths: dict[Suit, int]) -> Suit:
    """Which suit SAYC opens with this shape, applying the sourced tie-break rules in
    order: five-card majors only count once they reach five; two suits tied for
    longest at 4-4 or 3-3 in the minors specifically have their own named rule; any
    other tie (including two 5+ suits, or a tie that mixes a major and a minor) opens
    the higher-ranking of the tied suits, the same general principle the source states
    for 5-5/6-6 ties, applied consistently rather than picking a side on the more
    contested 4-4-4-1 "suit below the singleton" question secondary sources disagree
    on (see DECISIONS.md).
    """
    eligible = [
        suit for suit in Suit if suit in (Suit.CLUBS, Suit.DIAMONDS) or lengths[suit] >= 5
    ]
    longest = max(lengths[suit] for suit in eligible)
    tied = [suit for suit in eligible if lengths[suit] == longest]
    if set(tied) == {Suit.CLUBS, Suit.DIAMONDS}:
        return Suit.DIAMONDS if longest == 4 else Suit.CLUBS if longest == 3 else max(tied)
    return max(tied)


def _opening_values(hand: Sequence[Card], legal: Sequence[Call]) -> dict[Call, float]:
    points = hcp(hand)
    lengths = suit_lengths(hand)
    balanced = is_balanced(hand)
    opening_suit = _opening_suit(lengths)

    scores: dict[Call, float] = {}
    for call in legal:
        if call is None:
            scores[call] = PASS_TIER + (1 if points < OPEN_MIN_HCP else 0)
        elif call == Bid(1, opening_suit) and OPEN_MIN_HCP <= points <= OPEN_MAX_HCP:
            scores[call] = SIGNATURE_TIER
        elif call == Bid(1, None) and balanced and NOTRUMP_1_MIN_HCP <= points <= NOTRUMP_1_MAX_HCP:
            scores[call] = PREFERRED_TIER  # preferred over a same-ranged suit opening
        elif call == Bid(2, None) and balanced and NOTRUMP_2_MIN_HCP <= points <= NOTRUMP_2_MAX_HCP:
            scores[call] = PREFERRED_TIER
        elif call == Bid(2, Suit.CLUBS) and points >= STRONG_TWO_CLUB_MIN_HCP:
            scores[call] = SIGNATURE_TIER
        else:
            scores[call] = UNSCORED
    return scores


# --- Responding ---------------------------------------------------------------------


def _response_values(
    hand: Sequence[Card], legal: Sequence[Call], opening: Bid
) -> dict[Call, float]:
    if opening.level != 1:
        return _fallback_values(legal)  # 2NT/strong-2C responses are out of scope
    if opening.strain is None:
        return _natural_1nt_response_values(hand, legal)
    return _suit_response_values(hand, legal, opening)


def _suit_response_values(
    hand: Sequence[Card], legal: Sequence[Call], opening: Bid
) -> dict[Call, float]:
    points = hcp(hand)
    lengths = suit_lengths(hand)
    assert opening.strain is not None  # dispatcher already excluded a notrump opening
    support = lengths[opening.strain]
    raise_bid = _cheapest(legal, opening.strain)  # the plain raise, if support qualifies
    limit_raise_bid = Bid(raise_bid.level + 1, opening.strain) if raise_bid is not None else None

    scores: dict[Call, float] = {}
    for call in legal:
        if call is None:
            scores[call] = PASS_TIER + (1 if points < RESPONSE_MIN_HCP else 0)
        elif (
            call == limit_raise_bid
            and support >= SIMPLE_RAISE_MIN_SUPPORT
            and LIMIT_RAISE_MIN_HCP <= points <= LIMIT_RAISE_MAX_HCP
        ):
            scores[call] = PREFERRED_TIER  # a jump - more descriptive than the plain raise
        elif (
            call == _cheapest(legal, call.strain)
            and call.strain != opening.strain
            and call.strain is not None
            and lengths[call.strain] >= 4
            and points >= (RESPONSE_MIN_HCP if call.level == 1 else NEW_SUIT_AT_TWO_MIN_HCP)
        ):
            scores[call] = SIGNATURE_TIER
        elif (
            call == raise_bid
            and support >= SIMPLE_RAISE_MIN_SUPPORT
            and SIMPLE_RAISE_MIN_HCP <= points <= SIMPLE_RAISE_MAX_HCP
        ):
            scores[call] = FIT_TIER
        elif (
            call == Bid(1, None)
            and NOTRUMP_RESPONSE_MIN_HCP <= points <= NOTRUMP_RESPONSE_MAX_HCP
        ):
            scores[call] = SIGNOFF_TIER
        else:
            scores[call] = UNSCORED
    return scores


def _natural_1nt_response_values(hand: Sequence[Card], legal: Sequence[Call]) -> dict[Call, float]:
    """Natural responses to 1NT - no Stayman, no Jacoby transfer, both out of scope.
    A long suit signs off directly at the two level rather than asking or transferring;
    everything else is a plain notrump raise, by strength."""
    points = hcp(hand)
    lengths = suit_lengths(hand)
    long_suits = [suit for suit in Suit if lengths[suit] >= NATURAL_1NT_RESPONSE_LONG_SUIT]

    scores: dict[Call, float] = {}
    for call in legal:
        if call is None:
            scores[call] = PASS_TIER + (1 if points < NATURAL_1NT_INVITE_MIN_HCP else 0)
        elif (
            call.level == 2
            and call.strain in long_suits
            and points < NATURAL_1NT_INVITE_MIN_HCP
        ):
            scores[call] = SIGNATURE_TIER
        elif (
            call == Bid(3, None)
            and points >= NATURAL_1NT_GAME_MIN_HCP
        ):
            scores[call] = SIGNOFF_TIER
        elif (
            call == Bid(2, None)
            and NATURAL_1NT_INVITE_MIN_HCP <= points <= NATURAL_1NT_INVITE_MAX_HCP
        ):
            scores[call] = SIGNOFF_TIER
        else:
            scores[call] = UNSCORED
    return scores


# --- Opener's rebid -------------------------------------------------------------------


def _rebid_values(
    hand: Sequence[Card], legal: Sequence[Call], opening: Bid, response: Bid
) -> dict[Call, float]:
    if opening.strain is None:
        return _fallback_values(legal)  # notrump opener's rebid is out of scope
    if response.strain is None:
        return _notrump_response_rebid_values(hand, legal, opening)
    if response.strain == opening.strain:
        return _raise_rebid_values(hand, legal, opening, response)
    return _new_suit_rebid_values(hand, legal, opening, response)


def _raise_rebid_values(
    hand: Sequence[Card], legal: Sequence[Call], opening: Bid, response: Bid
) -> dict[Call, float]:
    """Partner raised opener's own suit - a known fit. A minimum hand passes; extra
    strength bids on again in the same suit, since partner's raise already promised
    some support."""
    points = hcp(hand)
    raise_again = _cheapest(legal, opening.strain)
    scores: dict[Call, float] = {}
    for call in legal:
        if call is None:
            scores[call] = PASS_TIER + (1 if points <= REBID_MINIMUM_MAX_HCP else 0)
        elif call == raise_again and points >= REBID_RAISE_AGAIN_MIN_HCP:
            scores[call] = FIT_TIER
        else:
            scores[call] = UNSCORED
    return scores


def _notrump_response_rebid_values(
    hand: Sequence[Card], legal: Sequence[Call], opening: Bid
) -> dict[Call, float]:
    """Partner responded 1NT: 6-9 points, denies support and denies a biddable
    higher-ranking suit at the one level. A minimum hand passes; extra strength or
    real extra length in the opening suit is worth another bid."""
    points = hcp(hand)
    lengths = suit_lengths(hand)
    assert opening.strain is not None  # dispatcher already excluded a notrump opening
    rebid_own_suit = _cheapest(legal, opening.strain)
    rebid_notrump = _cheapest(legal, None)

    scores: dict[Call, float] = {}
    for call in legal:
        if call is None:
            scores[call] = PASS_TIER + (1 if points <= REBID_MINIMUM_MAX_HCP else 0)
        elif call == rebid_own_suit and lengths[opening.strain] >= OPENERS_REBID_SUIT_MIN_LENGTH:
            scores[call] = SIGNATURE_TIER
        elif call == rebid_notrump and points >= REBID_NOTRUMP_MIN_HCP:
            scores[call] = SIGNOFF_TIER
        else:
            scores[call] = UNSCORED
    return scores


def _new_suit_rebid_values(
    hand: Sequence[Card], legal: Sequence[Call], opening: Bid, response: Bid
) -> dict[Call, float]:
    """Partner bid a new suit, showing 4+ cards there and real values. Opener raises
    with support, rebids a genuinely long own suit, or bids the cheapest notrump with
    a balanced hand and nothing better to show."""
    assert opening.strain is not None  # dispatcher already excluded a notrump opening
    assert response.strain is not None  # dispatcher already excluded a notrump response
    lengths = suit_lengths(hand)
    balanced = is_balanced(hand)
    support = lengths[response.strain]
    raise_partner = _cheapest(legal, response.strain)
    rebid_own_suit = _cheapest(legal, opening.strain)
    rebid_notrump = _cheapest(legal, None)

    scores: dict[Call, float] = {}
    for call in legal:
        if call is None:
            scores[call] = PASS_TIER
        elif call == raise_partner and support >= SIMPLE_RAISE_MIN_SUPPORT:
            scores[call] = FIT_TIER
        elif call == rebid_own_suit and lengths[opening.strain] >= OPENERS_REBID_SUIT_MIN_LENGTH:
            scores[call] = SIGNATURE_TIER
        elif call == rebid_notrump and balanced:
            scores[call] = SIGNOFF_TIER
        else:
            scores[call] = UNSCORED
    return scores
