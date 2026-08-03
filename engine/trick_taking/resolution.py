"""Trick resolution: who wins a completed trick, and what a hand may legally play into
one.

Deliberately generic (ARCHITECTURE.md §5): bridge and Court Piece resolve tricks by
identical rules, so this is written once and shared rather than implemented twice.
Nothing here knows about declarers, dummies, partnerships or contracts - seats are
plain ints, whatever the calling game uses as a player id.

Trump is a suit or None. None covers a trumpless deal (bridge's notrump contracts) and,
later, Court Piece's running-trump variant, where no trump exists at all until the first
player unable to follow suit sets one mid-deal.
"""

from collections.abc import Sequence

from engine.cards import Card, Suit

Seat = int
PlayedCard = tuple[Seat, Card]


def trick_winner(trick: Sequence[PlayedCard], led_suit: Suit, trump: Suit | None) -> Seat:
    """Which seat wins a completed trick, given the suit led and the trump suit.

    The highest trump wins if anyone trumped; otherwise the highest card of the suit
    led. A card of any other suit can never win, however high it is - discarding is
    throwing a card away, not contesting the trick. When trump is itself the suit led
    both rules select the same cards, so no special case is needed for it.
    """
    if not trick:
        raise ValueError("cannot resolve an empty trick")

    trumped = [(seat, card) for seat, card in trick if card.suit is trump]
    contenders = trumped or [(seat, card) for seat, card in trick if card.suit is led_suit]
    if not contenders:
        raise ValueError(f"no card of the led suit ({led_suit}) or of trump was played")

    winner, _card = max(contenders, key=lambda play: play[1].rank)
    return winner


def legal_plays(hand: Sequence[Card], led_suit: Suit | None) -> list[Card]:
    """Which cards of `hand` may be played into the current trick.

    Following suit is compulsory whenever the hand holds the suit led; a hand void in
    it may play anything, trumps included. `led_suit` is None for the player on lead,
    who is free to choose any card.
    """
    if led_suit is None:
        return list(hand)
    following = [card for card in hand if card.suit is led_suit]
    return following or list(hand)
