"""Strategies (games/bridge/session.py's BidStrategy and PlayStrategy protocols) that
read a human's decisions from the terminal, instead of computing one. Bridge needs
both kinds - poker's cli_strategy.py only ever needed one - since a human at the
table makes two structurally different decisions: what to call during the auction,
and which card to play once it's under way.

Deliberately does not import games.bridge.action_values, games.bridge.pimc,
games.bridge.trick_odds, or bid_values from games.bridge.bidding - a human player
sees only what a real player would see (their own hand, the auction as it actually
happened, the trick in progress), never a computed value. That is a hard boundary,
not just a display choice, matching poker's own cli_strategy.py precedent: this file
has no way to show a computed value even if someone tried to add one carelessly
later, since the modules that compute it are never imported here.
"""

from collections.abc import Sequence

from engine.cards import Card, Rank, Suit
from games.bridge.bidding import Auction, Bid, Call
from games.bridge.rules import Bridge, BridgeState, Contract

_RANKS = {rank.symbol: rank for rank in Rank}
_SUITS = {suit.symbol: suit for suit in Suit}
_NOTRUMP_WORDS = ("NT", "N")


def _format_call(call: Call) -> str:
    return "Pass" if call is None else str(call)


def format_contract(contract: Contract) -> str:
    """"3NT by seat 0" - the same level+strain shorthand a real bid uses, plus who's
    declaring."""
    strain = "NT" if contract.trump is None else contract.trump.symbol
    level = contract.target - 6
    return f"{level}{strain} by seat {contract.declarer}"


def _format_hand(hand: Sequence[Card]) -> str:
    ordered = sorted(hand, key=lambda card: (card.suit.value, card.rank.value))
    return " ".join(str(card) for card in ordered) or "(empty)"


def format_auction(auction: Auction) -> str:
    """The auction so far, one call per line, in the order it actually happened."""
    if not auction.calls:
        return "  (no calls yet)"
    lines = [
        f"  seat {auction.seat_at(index)}: {_format_call(call)}"
        for index, call in enumerate(auction.calls)
    ]
    return "\n".join(lines)


def format_bidding_view(hand: Sequence[Card], auction: Auction) -> str:
    """A human-readable summary of one bidding decision."""
    legal = ", ".join(_format_call(call) for call in auction.legal_calls())
    lines = [
        "--- Auction ---",
        format_auction(auction),
        f"your hand: {_format_hand(hand)}",
        f"legal calls: {legal}",
    ]
    return "\n".join(lines)


def parse_call(text: str, auction: Auction) -> Call:
    """Turn one line of input into a Call.

    Raises ValueError with a human-readable reason on anything unparseable or
    illegal for `auction` - CLIBidStrategy catches this and re-prompts rather than
    letting a typo end the auction.
    """
    raw = text.strip().upper()
    if not raw:
        raise ValueError("type a call, e.g. 'pass', '1H', or '3NT'")

    if raw in ("PASS", "P"):
        call: Call = None
    else:
        if len(raw) < 2 or not raw[0].isdigit():
            raise ValueError("type a call like 'pass', '1H', or '3NT'")
        strain_text = raw[1:]
        if strain_text in _NOTRUMP_WORDS:
            strain = None
        elif strain_text in _SUITS:
            strain = _SUITS[strain_text]
        else:
            raise ValueError(f"'{strain_text}' isn't a recognised strain - use C/D/H/S/NT")
        try:
            call = Bid(int(raw[0]), strain)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

    if call not in auction.legal_calls():
        legal = ", ".join(_format_call(c) for c in auction.legal_calls())
        raise ValueError(f"'{text}' isn't legal right now - try one of: {legal}")
    return call


class CLIBidStrategy:
    """Prompts the terminal for one call per bidding decision, re-prompting on bad
    input."""

    def __init__(self, prompt_name: str = "You") -> None:
        self.prompt_name = prompt_name

    def __call__(self, hand: Sequence[Card], auction: Auction) -> Call:
        print()
        print(format_bidding_view(hand, auction))
        while True:
            try:
                text = input(f"{self.prompt_name}, your call> ")
            except EOFError:
                # No more input to read (e.g. piped stdin ran dry) - pass rather than
                # hang or crash, so a non-interactive run still terminates cleanly.
                print("(no input available, passing)")
                return None
            try:
                return parse_call(text, auction)
            except ValueError as exc:
                print(f"  {exc}")


def format_play_view(game: Bridge, state: BridgeState) -> str:
    """A human-readable summary of one card-play decision.

    `state.to_play` is whichever hand the card actually comes from - the human's own
    when they're playing their own cards, or the dummy's when the human is declarer
    choosing for it, exactly as real bridge works. Labelled explicitly either way
    rather than always saying "your hand", since the two are genuinely different.
    """
    to_play = state.to_play
    whose = "dummy's" if to_play == state.dummy else "your"
    trick = ", ".join(f"seat {seat}: {card}" for seat, card in state.trick) or "(you lead)"
    legal = _format_hand(game.legal_actions(state))
    lines = [
        f"--- Trick {len(state.completed) + 1} - {format_contract(state.contract)} ---",
        f"trick so far: {trick}",
        f"{whose} hand: {_format_hand(state.hands[to_play])}",
        f"legal cards: {legal}",
    ]
    return "\n".join(lines)


def parse_card(text: str, legal: Sequence[Card]) -> Card:
    """Turn one line of input into a Card.

    Raises ValueError with a human-readable reason on anything unparseable or not
    in `legal` - CLIPlayStrategy catches this and re-prompts rather than letting a
    typo end the trick.
    """
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
    """Prompts the terminal for one card per play decision, re-prompting on bad
    input."""

    def __init__(self, prompt_name: str = "You") -> None:
        self.prompt_name = prompt_name

    def __call__(self, game: Bridge, state: BridgeState) -> Card:
        print()
        print(format_play_view(game, state))
        legal = game.legal_actions(state)
        while True:
            try:
                text = input(f"{self.prompt_name}, your card> ")
            except EOFError:
                # No more input to read - play the first legal card rather than hang
                # or crash, so a non-interactive run still terminates cleanly.
                print(f"(no input available, playing {legal[0]})")
                return legal[0]
            try:
                return parse_card(text, legal)
            except ValueError as exc:
                print(f"  {exc}")
