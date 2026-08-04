"""Reusable hand-strength utilities for bidding (ARCHITECTURE.md §5, bridge Phase 4):
high-card points, balanced-shape detection, and suit length. Nothing here knows about
the auction itself - games/bridge/bidding.py builds bidding decisions on top of these.
"""

from collections.abc import Sequence

from engine.cards import Card, Rank, Suit

# Standard high-card point scale. Named rather than inline so the scale is visible in
# one place - every other hand-strength threshold in bidding.py is stated in these
# points, not a different currency.
HONOR_POINTS: dict[Rank, int] = {Rank.ACE: 4, Rank.KING: 3, Rank.QUEEN: 2, Rank.JACK: 1}


def hcp(hand: Sequence[Card]) -> int:
    """High-card points: 4-3-2-1 for ace-king-queen-jack, nothing for anything else."""
    return sum(HONOR_POINTS.get(card.rank, 0) for card in hand)


def suit_lengths(hand: Sequence[Card]) -> dict[Suit, int]:
    """How many cards of each suit the hand holds, including suits held zero times."""
    lengths = dict.fromkeys(Suit, 0)
    for card in hand:
        lengths[card.suit] += 1
    return lengths


def longest_suits(hand: Sequence[Card]) -> list[Suit]:
    """Every suit tied for longest in the hand, ranked lowest to highest.

    More than one entry means a genuine tie - which suit to actually open on a tie is
    a bidding decision, not a hand-evaluation fact, so that judgement belongs in
    bidding.py, not here.
    """
    lengths = suit_lengths(hand)
    longest = max(lengths.values())
    return [suit for suit in Suit if lengths[suit] == longest]


def is_balanced(hand: Sequence[Card]) -> bool:
    """Standard SAYC balanced shape: no void, no singleton, at most one doubleton.

    The three balanced shapes are 4-3-3-3, 4-4-3-2 and 5-3-3-2 - notably, 5-4-2-2 is
    NOT balanced by this definition, despite having no suit below two cards, because
    two doubletons is one too many.
    """
    lengths = suit_lengths(hand).values()
    if any(length <= 1 for length in lengths):
        return False
    return sum(1 for length in lengths if length == 2) <= 1
