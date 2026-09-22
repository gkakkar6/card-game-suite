"""Strategies (games/court_piece/session.py's DeclareStrategy and PlayStrategy
protocols) that read a human's decisions from the terminal, instead of computing one -
the same role and the same hard boundary as games/bridge/cli_strategy.py: this file
never imports action_values, pimc, or hand_evaluation's suit_score, so a human player
sees only what a real player would see (their own cards, the trick in progress),
never a computed value.
"""

from collections.abc import Sequence

from engine.cards import Card, Rank, Suit
from games.court_piece.rules import CourtPiece, CourtPieceState, TrumpCall

_RANKS = {rank.symbol: rank for rank in Rank}
_SUITS = {suit.symbol: suit for suit in Suit}


def _format_hand(hand: Sequence[Card]) -> str:
    ordered = sorted(hand, key=lambda card: (card.suit.value, card.rank.value))
    return " ".join(str(card) for card in ordered) or "(empty)"


def format_call(call: TrumpCall) -> str:
    assert call.trump is not None  # fixed trump always has a concrete trump by now
    return f"{call.trump.symbol} (called by seat {call.caller})"


def format_trump(trump: Suit | None) -> str:
    """"H" once set, or a plain "not yet set" for running trump before the first
    void play fixes it - the one display difference the two variants need."""
    return trump.symbol if trump is not None else "not yet set"


def format_declare_view(preview: Sequence[Card]) -> str:
    """A human-readable summary of the declare-phase decision - only the preview
    cards are shown, matching the real rule that the call is made from a partial hand."""
    suits = ", ".join(f"{suit.symbol}={suit.name.title()}" for suit in Suit)
    lines = [
        "--- Declare trump ---",
        f"your first {len(preview)} cards: {_format_hand(preview)}",
        f"call one of: {suits}",
    ]
    return "\n".join(lines)


def parse_suit(text: str) -> Suit:
    """Turn one line of input into a Suit. Raises ValueError with a human-readable
    reason on anything unrecognised - CLIDeclareStrategy catches this and re-prompts."""
    raw = text.strip().upper()
    if raw in _SUITS:
        return _SUITS[raw]
    by_name = {suit.name: suit for suit in Suit}
    if raw in by_name:
        return by_name[raw]
    raise ValueError(f"'{text}' isn't a recognised suit - use C/D/H/S or the full name")


class CLIDeclareStrategy:
    """Prompts the terminal for the trump call, re-prompting on bad input."""

    def __init__(self, prompt_name: str = "You") -> None:
        self.prompt_name = prompt_name

    def __call__(self, preview: Sequence[Card]) -> Suit:
        print()
        print(format_declare_view(preview))
        while True:
            try:
                text = input(f"{self.prompt_name}, call trump> ")
            except EOFError:
                print("(no input available, calling Clubs)")
                return Suit.CLUBS
            try:
                return parse_suit(text)
            except ValueError as exc:
                print(f"  {exc}")


def format_play_view(game: CourtPiece, state: CourtPieceState) -> str:
    """A human-readable summary of one card-play decision."""
    to_play = state.to_play
    trick = ", ".join(f"seat {seat}: {card}" for seat, card in state.trick) or "(you lead)"
    legal = _format_hand(game.legal_actions(state))
    lines = [
        f"--- Trick {len(state.completed) + 1} - trump: {format_trump(state.call.trump)} ---",
        f"trick so far: {trick}",
        f"your hand: {_format_hand(state.hands[to_play])}",
        f"legal cards: {legal}",
    ]
    return "\n".join(lines)


def parse_card(text: str, legal: Sequence[Card]) -> Card:
    """Turn one line of input into a Card. Raises ValueError with a human-readable
    reason on anything unparseable or not in `legal`."""
    raw = text.strip().upper()
    if len(raw) < 2:
        raise ValueError("type a card like '2H', 'TC', or 'AS'")
    rank = _RANKS.get(raw[0])
    suit = _SUITS.get(raw[1:])
    if rank is None or suit is None:
        raise ValueError(f"'{text}' isn't a recognised card")

    card = Card(rank, suit)
    if card not in legal:
        raise ValueError(f"{card} isn't legal right now - try one of: {_format_hand(legal)}")
    return card


class CLIPlayStrategy:
    """Prompts the terminal for one card per play decision, re-prompting on bad input."""

    def __init__(self, prompt_name: str = "You") -> None:
        self.prompt_name = prompt_name

    def __call__(self, game: CourtPiece, state: CourtPieceState) -> Card:
        print()
        print(format_play_view(game, state))
        legal = game.legal_actions(state)
        while True:
            try:
                text = input(f"{self.prompt_name}, your card> ")
            except EOFError:
                print(f"(no input available, playing {legal[0]})")
                return legal[0]
            try:
                return parse_card(text, legal)
            except ValueError as exc:
                print(f"  {exc}")
