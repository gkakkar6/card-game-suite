"""Contract scoring: whether declarer's side made its contract, and what that is
worth in points (ARCHITECTURE.md §6, bridge Phase 5) - the real gap Phase 1
deliberately left open, since `Bridge.payoff()` only ever returns raw trick counts.

Non-vulnerable, undoubled scoring only. Sourced against the standard duplicate
bridge scoring table (cross-checked against Wikipedia's "Bridge scoring" article and
the ACBL's published scoring conventions, the same discipline bidding.py used for its
own sourced facts): 20 points/trick for clubs or diamonds, 30 points/trick for hearts
or spades, 40 for notrump's first odd trick and 30 for each notrump trick after that;
50 points per undertrick when neither side is vulnerable and the contract is
undoubled.

Explicitly, honestly out of scope, named rather than silently missing: vulnerability
(every hand here is scored as if neither side is vulnerable), game and slam bonuses,
and doubling-dependent penalties - the last of these isn't buildable yet regardless,
since bidding.py's auction is uncontested-only and never produces a doubled contract.
"""

from dataclasses import dataclass

from engine.cards import Suit
from games.bridge.rules import Contract

BOOK = 6  # tricks every side is credited before "odd tricks" (what target counts) start

MINOR_TRICK_VALUE = 20
MAJOR_TRICK_VALUE = 30
NOTRUMP_FIRST_TRICK_VALUE = 40
NOTRUMP_LATER_TRICK_VALUE = 30

MINOR_SUITS = frozenset({Suit.CLUBS, Suit.DIAMONDS})

UNDERTRICK_PENALTY = 50  # per trick short, non-vulnerable, undoubled


def _overtrick_value(contract: Contract) -> int:
    """Points for one trick beyond the contract - the suit's flat per-trick value,
    or notrump's steady-state value. An overtrick never gets notrump's first-trick
    bonus, since that bonus is already spent on the contract's own first odd trick."""
    if contract.trump is None:
        return NOTRUMP_LATER_TRICK_VALUE
    return MINOR_TRICK_VALUE if contract.trump in MINOR_SUITS else MAJOR_TRICK_VALUE


def _contract_value(contract: Contract) -> int:
    """Points for bidding and making exactly the contract - no overtricks."""
    bid_tricks = contract.target - BOOK
    if contract.trump is None:
        return NOTRUMP_FIRST_TRICK_VALUE + NOTRUMP_LATER_TRICK_VALUE * (bid_tricks - 1)
    return _overtrick_value(contract) * bid_tricks


@dataclass(frozen=True)
class ScoreResult:
    """One played hand's outcome, from declarer's side's perspective.

    `margin` is signed: positive for overtricks, negative for undertricks, zero for
    making the contract exactly. `points` is signed the same way declarer's side
    actually experiences it - positive if made, negative if the contract failed.
    """

    made: bool
    margin: int
    points: int


def score_contract(contract: Contract, tricks_won: int) -> ScoreResult:
    """Score one played hand under `contract`, given how many tricks declarer's
    side actually won.

    A passed-out hand has no contract at all and is never played, so it is never
    scored here - that is a distinct, explicit outcome session.py records directly
    rather than routing through this function with a placeholder contract.
    """
    margin = tricks_won - contract.target
    made = margin >= 0
    if made:
        points = _contract_value(contract) + margin * _overtrick_value(contract)
    else:
        points = -UNDERTRICK_PENALTY * -margin
    return ScoreResult(made=made, margin=margin, points=points)
