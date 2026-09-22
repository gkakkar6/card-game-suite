"""Dealing a Court Piece deal: the whole deck, thirteen cards to each of the four
seats - mechanically identical to games/bridge/deal.py (same 4x13 block deal), kept as
its own small copy rather than a cross-game import so games/court_piece stays
self-contained the same way games/bridge and games/poker are.
"""

import random

from engine.cards import Card, Deck

SEATS = 4
CARDS_PER_SEAT = 13
FULL_DEAL = SEATS * CARDS_PER_SEAT
DECLARE_PREVIEW = 5  # cards the trump-caller sees before calling


def deal_hands(
    *, deck: Deck | None = None, rng: random.Random | None = None
) -> tuple[tuple[Card, ...], ...]:
    """One 13-card hand per seat, in seat order.

    Pass `deck` to control exactly which cards each seat gets (tests do this);
    otherwise a fresh deck is shuffled, seeded by `rng` when reproducibility matters.
    """
    if deck is None:
        deck = Deck()
        deck.shuffle(rng)
    if len(deck) < FULL_DEAL:
        raise ValueError(f"a Court Piece deal needs {FULL_DEAL} cards, got {len(deck)}")
    return tuple(tuple(deck.deal(CARDS_PER_SEAT)) for _ in range(SEATS))
