"""Reusable hand-strength utilities for the fixed-trump declare-phase
(games/court_piece/declare.py): picking one suit as trump from a 5-card preview,
before the rest of the hand is even dealt.

A genuinely different question from bridge's hand_evaluation.py, not a renamed copy
of it. Bridge's HCP is a general hand-strength proxy for competitive bidding across
all four suits; Court Piece's declare-phase decision is narrower and more concrete -
"which single suit, if any of them, would make the best trump" - so what matters per
candidate suit is two things together, not one number: **quantity** (length - a long
trump suit can be run and used to ruff repeatedly) and **quality** (honors within
that suit specifically - length in low cards is much weaker than the same length
with the ace and king in it).

With only 5 of the deck's 52 cards seen and zero information yet about the other
three hands (no auction, no partner signal - the call is made alone), the only
defensible prior for the other 47 cards is uniform: nothing here does or should model
belief about a specific opponent, unlike a later, evidence-based PIMC decision mid-play.
The scoring below is a static function of the 5 visible cards alone for exactly that
reason - there is nothing else honest to condition on yet.
"""

from collections.abc import Sequence

from engine.cards import Card, Rank, Suit

# Same honor scale as bridge's HONOR_POINTS (games/bridge/hand_evaluation.py) - no
# reason to invent a different one, the relative worth of an ace vs. a jack doesn't
# change between games.
HONOR_POINTS: dict[Rank, int] = {Rank.ACE: 4, Rank.KING: 3, Rank.QUEEN: 2, Rank.JACK: 1}

# Weight on suit length relative to honor points in the trump-candidate score below.
# A real, named parameter rather than an unstated assumption - length matters for
# trump specifically (it's what lets a long suit be run, or ruff repeatedly, once it's
# trump) in a way it doesn't for a general hand-strength count, so honors alone would
# undervalue a long, low suit that still makes a fine trump. Starting point to tune
# once real bot-vs-bot evaluation exists, not asserted as correct from first principles.
LENGTH_WEIGHT = 3.0


def suit_lengths(hand: Sequence[Card]) -> dict[Suit, int]:
    """How many cards of each suit the hand holds, including suits held zero times -
    identical to bridge's own suit_lengths(), reused here as the same primitive."""
    lengths = dict.fromkeys(Suit, 0)
    for card in hand:
        lengths[card.suit] += 1
    return lengths


def suit_honors(hand: Sequence[Card], suit: Suit) -> int:
    """Honor points held within `suit` specifically - the within-suit analogue of
    bridge's whole-hand hcp(), scoped to one suit since that's what a trump call
    actually needs to weigh."""
    return sum(HONOR_POINTS.get(card.rank, 0) for card in hand if card.suit is suit)


def suit_score(hand: Sequence[Card], suit: Suit) -> float:
    """How good a trump candidate `suit` is, from this hand's 5-card preview alone:
    length (how many of it are held) plus honors within it (how much those cards are
    actually worth), combined so a long-but-low suit and a short-but-high one can
    both score well rather than either axis dominating outright.
    """
    lengths = suit_lengths(hand)
    return LENGTH_WEIGHT * lengths[suit] + suit_honors(hand, suit)


def best_trump_candidate(hand: Sequence[Card]) -> Suit:
    """The single best suit to call as trump from `hand` (a 5-card preview, or any
    partial hand) - highest suit_score(), ties broken toward whichever suit sorts
    first in `Suit`'s own declared order, a fixed and reproducible rule rather than
    an arbitrary one.

    Always returns a suit - the call is mandatory in fixed-trump Court Piece (there
    is no "pass"), so there is no not-good-enough-to-call threshold here the way
    bridge's opening-bid HCP minimum works. Whoever's turn it is must call something;
    this is only about which suit is the least-bad (or best) choice among the four.
    """
    return max(Suit, key=lambda suit: (suit_score(hand, suit), -suit.value))
