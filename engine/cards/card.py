from dataclasses import dataclass
from enum import IntEnum

# one-character display symbols, indexed/keyed by each enum's underlying int value
_SUIT_SYMBOLS: tuple[str, ...] = ("C", "D", "H", "S")
_RANK_SYMBOLS: dict[int, str] = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8",
    9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A",
}


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    @property
    def symbol(self) -> str:
        return _SUIT_SYMBOLS[self.value]

    def __str__(self) -> str:
        return self.symbol


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def symbol(self) -> str:
        return _RANK_SYMBOLS[self.value]

    def __str__(self) -> str:
        return self.symbol


@dataclass(frozen=True, order=True)
class Card:
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{self.rank.symbol}{self.suit.symbol}"
