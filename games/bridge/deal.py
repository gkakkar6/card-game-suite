"""Dealing a bridge deal: the whole deck, thirteen cards to each of the four seats.

engine/cards/ already does the real work (a deck, a shuffle, taking cards off the top),
so this is only the dealing pattern - which is genuinely different from poker's, where
each player gets two cards and the rest of the deck arrives a street at a time.
"""

import random

from engine.cards import Card, Deck

SEATS = 4
CARDS_PER_SEAT = 13
FULL_DEAL = SEATS * CARDS_PER_SEAT


def deal_hands(
    *, deck: Deck | None = None, rng: random.Random | None = None
) -> tuple[tuple[Card, ...], ...]:
    """One hand per seat, in seat order.

    Pass `deck` to control exactly which cards each seat gets (tests do this);
    otherwise a fresh deck is shuffled, seeded by `rng` when reproducibility matters.

    Each seat takes thirteen consecutive cards off the top rather than one at a time
    round the table. For a shuffled deck the two are equivalent, and dealing in blocks
    makes a stacked deck far easier to write out by hand in a test.
    """
    if deck is None:
        deck = Deck()
        deck.shuffle(rng)
    if len(deck) < FULL_DEAL:
        raise ValueError(f"a bridge deal needs {FULL_DEAL} cards, got {len(deck)}")
    return tuple(tuple(deck.deal(CARDS_PER_SEAT)) for _ in range(SEATS))
