"""Court Piece's play phase: the trick-taking state machine, as an engine.game.Game.

One state machine, both trump variants (ARCHITECTURE.md §5's Court Piece build plan
frames running trump as an extension of fixed trump for exactly this reason - "the
only new work is treating trump as Optional[Suit]"). Fixed trump: `TrumpCall.trump`
is a concrete `Suit` from the very first deal (games/court_piece/declare.py decides
it from a 5-card preview before the rest is even dealt) and the logic below never
fires. Running trump (Be-ranga Double Sar): the deal starts with `trump=None`, and
`CourtPiece.apply()` sets it the moment any player, unable to follow the suit led,
plays a card - that card's suit becomes trump for the rest of the deal, permanently.
There is no separate "declare" decision in this variant at all: whichever card a void
player was going to play anyway is what sets trump, not an extra choice layered on
top of it.

Deliberately unlike bridge in the one way that actually matters: **there is no
dummy.** All four hands stay private for the whole hand - `current_player()` is just
`state.to_play`, with no redirection, and `information_set()` shows a player only
their own hand. That's the real source of Court Piece's harder inference problem
despite its otherwise-lighter build: PIMC has three unseen hands to sample per
decision, not bridge's two (games/court_piece/pimc.py).
"""

import random
from collections.abc import Mapping
from dataclasses import dataclass, replace

from engine.cards import Card, Deck, Suit
from engine.trick_taking.resolution import PlayedCard, Seat, legal_plays, trick_winner
from games.court_piece.deal import CARDS_PER_SEAT, SEATS, deal_hands

TRICKS = CARDS_PER_SEAT  # every card in a hand is played to exactly one trick


def partner(seat: Seat) -> Seat:
    """The seat opposite. Seats are numbered clockwise, so partners sit two apart:
    0 with 2 against 1 with 3 - identical convention to games/bridge/rules.py."""
    return (seat + 2) % SEATS


def opening_leader(caller: Seat) -> Seat:
    """The player on the trump-caller's left, who leads to the first trick."""
    return (caller + 1) % SEATS


@dataclass(frozen=True)
class TrumpCall:
    """Trump, and a reference seat, for one deal.

    Fixed trump: `trump` is a concrete `Suit` from the start, and `caller` is exactly
    who called it - both fixed before the first card is led. Running trump: `trump`
    starts `None` (undetermined until the first void play sets it, in
    `CourtPiece.apply()` below) and `caller` is simply whichever seat's turn it is to
    lead first this hand (the same per-hand rotation fixed trump's caller uses, via
    `opening_leader()` below - there is no "calling" decision in this variant, so the
    name is a strained fit, but the seat still does the same structural job: fixing
    where the first lead comes from, and anchoring the solver's perspective in
    games/court_piece/solver.py, which needs some seat's side to keep the maximizing
    convention consistent across a whole search regardless of who's actually deciding
    at each node - any fixed seat works for that math, not specifically "the caller").

    No `target` the way bridge's Contract has one - Court Piece has no bid to make or
    fail, only tricks won, so scoring (games/court_piece/scoring.py) works from the
    completed trick sequence directly rather than against a declared number.
    """

    trump: Suit | None
    caller: Seat

    def __post_init__(self) -> None:
        if not 0 <= self.caller < SEATS:
            raise ValueError(f"caller must be a seat in 0-{SEATS - 1}, got {self.caller}")


@dataclass(frozen=True)
class CompletedTrick:
    """A trick played out in full, kept card by card so the history stays exact -
    scoring needs the actual order tricks were won in, not just final counts, to tell
    a kot (all 13, or the first 7 straight) apart from an ordinary piece."""

    cards: tuple[PlayedCard, ...]
    winner: Seat

    @property
    def led_suit(self) -> Suit:
        _seat, card = self.cards[0]
        return card.suit


@dataclass(frozen=True)
class CourtPieceState:
    """Everything true about a deal in progress.

    Immutable: `apply()` returns a new state rather than mutating this one, the same
    reasoning as BridgeState - it costs a copy of four shrinking hands per card, which
    is nothing here, and lets the double-dummy search walk the tree without undoing
    moves.
    """

    call: TrumpCall
    hands: tuple[tuple[Card, ...], ...]  # remaining cards per seat, in seat order
    trick: tuple[PlayedCard, ...] = ()  # the trick in progress, in the order played
    completed: tuple[CompletedTrick, ...] = ()

    def __post_init__(self) -> None:
        if len(self.hands) != SEATS:
            raise ValueError(f"a deal has {SEATS} hands, got {len(self.hands)}")

    @property
    def leader(self) -> Seat:
        """Seat on lead to the current trick: whoever won the last one, or the
        player to the caller's left before any trick has been played."""
        if self.trick:
            leader, _card = self.trick[0]
            return leader
        if self.completed:
            return self.completed[-1].winner
        return opening_leader(self.call.caller)

    @property
    def to_play(self) -> Seat:
        """The hand the next card comes from - play runs clockwise from the leader."""
        return (self.leader + len(self.trick)) % SEATS

    @property
    def led_suit(self) -> Suit | None:
        """Suit led to the current trick, or None when nobody has played to it yet."""
        if not self.trick:
            return None
        _seat, card = self.trick[0]
        return card.suit


@dataclass(frozen=True)
class CourtPieceInformationSet:
    """What one player can actually see of a deal in progress: their own hand, plus
    everything public - the call, the trick in progress, and every completed trick.

    Unlike bridge's BridgeInformationSet, exactly one hand is ever visible here -
    there is no dummy to add a second one."""

    player: Seat
    hands: Mapping[Seat, tuple[Card, ...]]
    call: TrumpCall
    trick: tuple[PlayedCard, ...]
    completed: tuple[CompletedTrick, ...]


class CourtPiece:
    """Court Piece's play phase, implementing engine.game.Game.

    Stateless: every method takes the state it works on, so one instance can drive
    any number of deals - or a branch of a search - the same as games/bridge/rules.py's
    Bridge.
    """

    def current_player(self, state: CourtPieceState) -> Seat:
        """Who decides the next card - always whoever's hand it's coming from. No
        dummy redirection: every seat plays its own cards."""
        return state.to_play

    def legal_actions(self, state: CourtPieceState) -> list[Card]:
        """Cards playable from the hand on turn, following suit wherever it can."""
        return legal_plays(state.hands[state.to_play], state.led_suit)

    def apply(self, state: CourtPieceState, action: Card) -> CourtPieceState:
        """Play `action` from the hand on turn, resolving the trick if it completes
        it. Legality is checked here rather than trusted from the caller, same
        discipline as Bridge.apply().

        Running trump's whole mechanic lives in one place: if trump is still
        undetermined and `seat` couldn't follow the suit led (checked against their
        hand *before* this card leaves it), `action`'s own suit becomes trump, right
        now - not a separate decision layered on the play, just what that play
        happens to mean. The newly-set trump applies to resolving *this* trick too,
        not only future ones - the card that reveals the void can itself win the
        trick outright by being trump, exactly as in real play.
        """
        seat = state.to_play
        if action not in self.legal_actions(state):
            raise ValueError(f"{action} is not a legal play for seat {seat}")

        call = state.call
        if call.trump is None and state.led_suit is not None:
            was_void = not any(card.suit is state.led_suit for card in state.hands[seat])
            if was_void:
                call = replace(call, trump=action.suit)

        hands = list(state.hands)
        remaining = list(hands[seat])
        remaining.remove(action)
        hands[seat] = tuple(remaining)

        trick = (*state.trick, (seat, action))
        completed = state.completed
        if len(trick) == SEATS:
            _leader, led = trick[0]
            winner = trick_winner(trick, led.suit, call.trump)
            completed = (*completed, CompletedTrick(cards=trick, winner=winner))
            trick = ()  # the winner leads the next one, via CourtPieceState.leader

        return replace(state, call=call, hands=tuple(hands), trick=trick, completed=completed)

    def is_terminal(self, state: CourtPieceState) -> bool:
        return len(state.completed) == TRICKS

    def payoff(self, state: CourtPieceState) -> dict[Seat, float]:
        """Tricks won by each partnership, both partners given the same number -
        raw trick counts, the same way Bridge.payoff() defers scoring; real Court
        Piece scoring (kot vs. piece, caller multiplier) lives in scoring.py and
        needs the completed-trick sequence itself, not just this total."""
        counts = dict.fromkeys(range(SEATS), 0.0)
        for trick in state.completed:
            counts[trick.winner] += 1.0
            counts[partner(trick.winner)] += 1.0
        return counts

    def information_set(self, state: CourtPieceState, player: Seat) -> CourtPieceInformationSet:
        """What `player` can see: only their own hand, plus the public record. No
        second hand is ever visible - the real difference from Bridge.information_set()."""
        return CourtPieceInformationSet(
            player=player,
            hands={player: state.hands[player]},
            call=state.call,
            trick=state.trick,
            completed=state.completed,
        )


def new_deal(
    call: TrumpCall, *, deck: Deck | None = None, rng: random.Random | None = None
) -> CourtPieceState:
    """A fresh deal under `call`: cards dealt, nothing played yet."""
    return CourtPieceState(call=call, hands=deal_hands(deck=deck, rng=rng))
