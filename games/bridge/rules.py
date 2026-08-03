"""Bridge's play phase: the trick-taking state machine, as an engine.game.Game.

Phase 1 scope (ARCHITECTURE.md §5). The contract - trump suit, declarer's seat, and how
many tricks declarer's side needs - is a fixed input rather than the outcome of an
auction, and never changes once play starts. Bidding is a separate, later track.

Three rules shape the interface here, all deliberate:

**Declarer plays the dummy's cards.** `current_player()` returns declarer's seat both
when declarer's own card is being chosen and when the dummy's is, because the dummy
never makes a decision - exactly as in real bridge. Which hand the card actually comes
from is `BridgeState.to_play`, and that is what legality is checked against; declarer
therefore chooses from both hands as a single decision-maker.

**The dummy's hand is public.** Once exposed it lies face up on the table, so
`information_set()` shows it to all four players, not only to declarer. Nobody ever
sees the other opponent's hand.

**Scoring is deferred.** `payoff()` returns raw trick counts per partnership. Whether
the contract was made, and what that is worth in points, is a light layer to add once
trick counting itself is proven correct - not part of this state machine.
"""

import random
from collections.abc import Mapping
from dataclasses import dataclass, replace

from engine.cards import Card, Deck, Suit
from engine.trick_taking.resolution import PlayedCard, Seat, legal_plays, trick_winner
from games.bridge.deal import CARDS_PER_SEAT, SEATS, deal_hands

TRICKS = CARDS_PER_SEAT  # every card in a hand is played to exactly one trick


def partner(seat: Seat) -> Seat:
    """The seat opposite. Seats are numbered clockwise, so partners sit two apart:
    0 with 2 against 1 with 3."""
    return (seat + 2) % SEATS


def opening_leader(declarer: Seat) -> Seat:
    """The player on declarer's left, who leads to the first trick."""
    return (declarer + 1) % SEATS


@dataclass(frozen=True)
class Contract:
    """What declarer's side is playing, fixed before the first card is led.

    `trump` of None is a notrump contract - trick resolution already handles a
    trumpless deal, so it needs nothing special here. `target` is how many of the
    thirteen tricks declarer's side has to win; Phase 1 carries it without scoring
    against it.
    """

    trump: Suit | None
    declarer: Seat
    target: int

    def __post_init__(self) -> None:
        if not 0 <= self.declarer < SEATS:
            raise ValueError(f"declarer must be a seat in 0-{SEATS - 1}, got {self.declarer}")
        if not 1 <= self.target <= TRICKS:
            raise ValueError(f"target must be between 1 and {TRICKS}, got {self.target}")


@dataclass(frozen=True)
class CompletedTrick:
    """A trick played out in full, kept card by card so the history stays exact."""

    cards: tuple[PlayedCard, ...]
    winner: Seat

    @property
    def led_suit(self) -> Suit:
        _seat, card = self.cards[0]
        return card.suit


@dataclass(frozen=True)
class BridgeState:
    """Everything true about a deal in progress.

    Immutable: `apply()` returns a new state rather than mutating this one. That costs
    a copy of four shrinking hands per card, which is nothing here, and means Phase 2's
    double-dummy search can walk the game tree without having to undo moves.
    """

    contract: Contract
    hands: tuple[tuple[Card, ...], ...]  # remaining cards per seat, in seat order
    trick: tuple[PlayedCard, ...] = ()  # the trick in progress, in the order played
    completed: tuple[CompletedTrick, ...] = ()

    def __post_init__(self) -> None:
        if len(self.hands) != SEATS:
            raise ValueError(f"a deal has {SEATS} hands, got {len(self.hands)}")

    @property
    def dummy(self) -> Seat:
        return partner(self.contract.declarer)

    @property
    def leader(self) -> Seat:
        """Seat on lead to the current trick: whoever won the last one, or the player
        to declarer's left before any trick has been played."""
        if self.trick:
            leader, _card = self.trick[0]
            return leader
        if self.completed:
            return self.completed[-1].winner
        return opening_leader(self.contract.declarer)

    @property
    def to_play(self) -> Seat:
        """The hand the next card comes from - play runs clockwise from the leader.

        Not the same thing as `Bridge.current_player()`: when this is the dummy, it is
        declarer who chooses the card.
        """
        return (self.leader + len(self.trick)) % SEATS

    @property
    def led_suit(self) -> Suit | None:
        """Suit led to the current trick, or None when nobody has played to it yet."""
        if not self.trick:
            return None
        _seat, card = self.trick[0]
        return card.suit


@dataclass(frozen=True)
class BridgeInformationSet:
    """What one player can actually see of a deal in progress.

    Their own hand and the dummy's - the same hand, for the dummy itself - plus
    everything public: the contract, the trick in progress, and every completed trick.
    A hand this player cannot see is simply absent from `hands`.
    """

    player: Seat
    hands: Mapping[Seat, tuple[Card, ...]]
    contract: Contract
    trick: tuple[PlayedCard, ...]
    completed: tuple[CompletedTrick, ...]


class Bridge:
    """Bridge's play phase, implementing engine.game.Game.

    Stateless: every method takes the state it works on, so one instance can drive any
    number of deals - or, later, any number of branches of a search.
    """

    def current_player(self, state: BridgeState) -> Seat:
        """Who decides the next card.

        Declarer's seat whenever the card comes from declarer's own hand *or* from the
        dummy's, since the dummy never makes a decision and declarer chooses from both.
        An opponent's turn is simply their own seat.
        """
        seat = state.to_play
        return state.contract.declarer if seat == state.dummy else seat

    def legal_actions(self, state: BridgeState) -> list[Card]:
        """Cards playable from the hand on turn, following suit wherever it can."""
        return legal_plays(state.hands[state.to_play], state.led_suit)

    def apply(self, state: BridgeState, action: Card) -> BridgeState:
        """Play `action` from the hand on turn, resolving the trick if it completes it.

        Legality is checked here rather than trusted from the caller, the same way
        poker's betting round does it: a revoke is a rules violation, and keeping the
        check in one place means no strategy has to be trusted to police itself.
        """
        seat = state.to_play
        if action not in self.legal_actions(state):
            raise ValueError(f"{action} is not a legal play for seat {seat}")

        hands = list(state.hands)
        remaining = list(hands[seat])
        remaining.remove(action)
        hands[seat] = tuple(remaining)

        trick = (*state.trick, (seat, action))
        completed = state.completed
        if len(trick) == SEATS:
            _leader, led = trick[0]
            winner = trick_winner(trick, led.suit, state.contract.trump)
            completed = (*completed, CompletedTrick(cards=trick, winner=winner))
            trick = ()  # the winner leads the next one, via BridgeState.leader

        return replace(state, hands=tuple(hands), trick=trick, completed=completed)

    def is_terminal(self, state: BridgeState) -> bool:
        return len(state.completed) == TRICKS

    def payoff(self, state: BridgeState) -> dict[Seat, float]:
        """Tricks won by each partnership, both partners given the same number.

        Counted from the completed tricks rather than tallied as play goes, so the
        count cannot drift out of step with the history it comes from.

        Phase 1 stops here on purpose: this is the raw trick count, not a contract
        score. Whether `contract.target` was made, and what that is worth, is a
        separate layer to add once trick counting is proven (ARCHITECTURE.md §5).
        """
        counts = dict.fromkeys(range(SEATS), 0.0)
        for trick in state.completed:
            counts[trick.winner] += 1.0
            counts[partner(trick.winner)] += 1.0
        return counts

    def information_set(self, state: BridgeState, player: Seat) -> BridgeInformationSet:
        """What `player` can see: their own hand, the dummy's, and the public record.

        The dummy's hand is face up for everyone once exposed, so an opponent sees it
        just as declarer does - but neither opponent ever sees the other's hand, and
        declarer never sees either of them.
        """
        visible = {player, state.dummy}
        return BridgeInformationSet(
            player=player,
            hands={seat: state.hands[seat] for seat in sorted(visible)},
            contract=state.contract,
            trick=state.trick,
            completed=state.completed,
        )


def new_deal(
    contract: Contract, *, deck: Deck | None = None, rng: random.Random | None = None
) -> BridgeState:
    """A fresh deal under `contract`: cards dealt, nothing played yet."""
    return BridgeState(contract=contract, hands=deal_hands(deck=deck, rng=rng))
