"""Which of a hand's legal plays are genuinely different from each other.

Two cards of the same suit in the same hand can only ever be told apart by how they
rank against other cards. So if every card between them has already gone with a
completed trick, no trick from here on can distinguish them - a search that tries both
does the same work twice and gets the same answer.

Trick-taking specific but game-agnostic, the same as resolution.py: bridge and Court
Piece both want this and neither wants it done differently, so it is written once here
rather than inside either game.
"""

from collections.abc import Iterable, Sequence
from itertools import pairwise

from engine.cards import Card, Suit


def _ranking(cards: Iterable[Card], suits: set[Suit]) -> dict[Card, int]:
    """Each card's place within what is left of its own suit, lowest first."""
    by_suit: dict[Suit, list[Card]] = {}
    for card in cards:
        if card.suit in suits:
            by_suit.setdefault(card.suit, []).append(card)

    ranking: dict[Card, int] = {}
    for suit_cards in by_suit.values():
        suit_cards.sort(key=lambda card: card.rank.value)
        for index, card in enumerate(suit_cards):
            ranking[card] = index
    return ranking


def play_groups(candidates: Sequence[Card], in_play: Iterable[Card]) -> list[list[Card]]:
    """Group `candidates` so that playing any card of a group gives the same result.

    Cards land in the same group when they are the same suit and sit next to each other
    in what is left of that suit - nothing survives between them to tell them apart, so
    swapping the two would leave every future trick exactly as it was.

    `in_play` is every card that can still affect a trick: all four hands plus whatever
    is already sitting in the trick being played to. A card in the current trick counts,
    since it is still being compared against - the queen and the ten are not
    interchangeable while the jack is on the table. Every candidate must appear in
    `in_play`, which holds whenever the candidates come from a hand.

    Groups come back in the order their first card appears in `candidates`, so a caller
    that took the first of each group gets its own ordering back.
    """
    suits = {card.suit for card in candidates}
    ranking = _ranking(in_play, suits)

    group_of: dict[Card, tuple[Suit, int]] = {}
    by_suit: dict[Suit, list[Card]] = {}
    for card in candidates:
        by_suit.setdefault(card.suit, []).append(card)

    for suit, suit_cards in by_suit.items():
        ordered = sorted(suit_cards, key=lambda card: card.rank.value)
        group = 0
        group_of[ordered[0]] = (suit, group)
        for lower, higher in pairwise(ordered):
            if ranking[higher] - ranking[lower] > 1:
                group += 1  # something between them is still out there
            group_of[higher] = (suit, group)

    groups: dict[tuple[Suit, int], list[Card]] = {}
    for card in candidates:
        groups.setdefault(group_of[card], []).append(card)
    return list(groups.values())
