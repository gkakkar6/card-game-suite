import random

from engine.cards.card import Card, Rank, Suit


def standard_52() -> list[Card]:
    return [Card(rank, suit) for suit in Suit for rank in Rank]


def shuffle(cards: list[Card], rng: random.Random | None = None) -> None:
    """Shuffle `cards` in place."""
    (rng if rng is not None else random.Random()).shuffle(cards)


def deal(cards: list[Card], n: int) -> list[Card]:
    """Remove and return the top `n` cards from `cards` (in place)."""
    if n > len(cards):
        raise ValueError(f"cannot deal {n} cards, only {len(cards)} remain")
    dealt = cards[:n]
    del cards[:n]
    return dealt


class Deck:
    def __init__(self, cards: list[Card] | None = None) -> None:
        self.cards: list[Card] = list(cards) if cards is not None else standard_52()

    def shuffle(self, rng: random.Random | None = None) -> None:
        shuffle(self.cards, rng)

    def deal(self, n: int) -> list[Card]:
        return deal(self.cards, n)

    def __len__(self) -> int:
        return len(self.cards)
