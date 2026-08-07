"""Drives one complete bridge hand end to end, and a multi-hand session on top of it
(ARCHITECTURE.md §6, bridge Phase 5) - the glue games/poker/hand.py and session.py
provide for poker, which nothing in bridge has had until now. `bidding.py` runs the
auction and `rules.py` runs the play; this module is what actually calls them in
order and turns the result into a score.

Genuinely simpler than poker's session: bridge is always exactly four seats, with no
variable table size, no bankroll, no bust concept and no elimination. The only
session-level state that persists across hands is the dealer (rotating clockwise,
the same principle as poker's button) and a cumulative points total per side.

Both `bid_strategies` and `play_strategies` are indexed by seat (0-3), one callable
per seat, and every seat needs both - even the dummy's, since the dummy itself never
decides anything (`Bridge.current_player()` always resolves to declarer). A seat's
`play_strategies` entry is only ever invoked for that seat's own decisions, which for
declarer includes the dummy's cards too, exactly as real bridge works.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from engine.cards import Card, Deck
from engine.trick_taking.resolution import Seat
from games.bridge.bidding import Auction, Call, new_auction
from games.bridge.deal import SEATS, deal_hands
from games.bridge.personas import BiddingStrategy, BridgePersona, CardPlayStrategy
from games.bridge.rules import Bridge, BridgeState, Contract, partner
from games.bridge.scoring import ScoreResult, score_contract


class BidStrategy(Protocol):
    """Chooses a call for one seat's bidding decision."""

    def __call__(self, hand: Sequence[Card], auction: Auction) -> Call: ...


class PlayStrategy(Protocol):
    """Chooses a card for one seat's card-play decision (including, for declarer,
    the dummy's)."""

    def __call__(self, game: Bridge, state: BridgeState) -> Card: ...


@dataclass(frozen=True)
class HandResult:
    """What happened in one complete hand, from a fresh deal through to a score.

    `contract`, `tricks_won` and `score` are all None together for a passed-out
    hand - a real, explicit outcome with no play phase at all, not an error and not
    a zero-trick contract standing in for "nothing happened."
    """

    dealer: Seat
    hands: tuple[tuple[Card, ...], ...]
    auction: Auction
    contract: Contract | None
    tricks_won: int | None
    score: ScoreResult | None


def _run_auction(
    bid_strategies: Sequence[BidStrategy], hands: Sequence[Sequence[Card]], dealer: Seat
) -> Auction:
    auction = new_auction(dealer)
    while not auction.is_over():
        seat = auction.to_call
        call = bid_strategies[seat](hands[seat], auction)
        auction = auction.apply(call)
    return auction


def _run_play(
    play_strategies: Sequence[PlayStrategy], contract: Contract, hands: Sequence[Sequence[Card]]
) -> BridgeState:
    game = Bridge()
    state = BridgeState(contract=contract, hands=tuple(tuple(hand) for hand in hands))
    while not game.is_terminal(state):
        player = game.current_player(state)
        action = play_strategies[player](game, state)
        state = game.apply(state, action)
    return state


def play_hand(
    bid_strategies: Sequence[BidStrategy],
    play_strategies: Sequence[PlayStrategy],
    *,
    dealer: Seat,
    deck: Deck | None = None,
    rng: random.Random | None = None,
) -> HandResult:
    """Play one hand: deal, run the auction, and - unless it's passed out - play it
    out and score it.

    Pass `deck` to control exactly which cards come out (tests do this); otherwise a
    fresh deck is shuffled, seeded by `rng` when reproducibility matters.
    """
    hands = deal_hands(deck=deck, rng=rng)
    auction = _run_auction(bid_strategies, hands, dealer)

    if auction.is_passed_out():
        return HandResult(
            dealer=dealer, hands=hands, auction=auction, contract=None, tricks_won=None, score=None
        )

    contract = auction.contract()
    assert contract is not None  # is_passed_out() already excluded the only other None case
    final_state = _run_play(play_strategies, contract, hands)
    tricks_won = int(Bridge().payoff(final_state)[contract.declarer])
    score = score_contract(contract, tricks_won)

    return HandResult(
        dealer=dealer,
        hands=hands,
        auction=auction,
        contract=contract,
        tricks_won=tricks_won,
        score=score,
    )


@dataclass(frozen=True)
class SeatConfig:
    """Static identity of one seat, fixed for the whole session."""

    name: str
    is_human: bool
    persona: BridgePersona | None = None  # required for bots, unused for the human

    def __post_init__(self) -> None:
        if self.is_human and self.persona is not None:
            raise ValueError("the human seat does not have a persona")
        if not self.is_human and self.persona is None:
            raise ValueError("a bot seat needs a persona")


@dataclass(frozen=True)
class SessionConfig:
    """Exactly four seats, one of them human - bridge has no variable table size,
    unlike poker."""

    seats: tuple[SeatConfig, SeatConfig, SeatConfig, SeatConfig]

    def __post_init__(self) -> None:
        humans = [seat for seat in self.seats if seat.is_human]
        if len(humans) != 1:
            raise ValueError("a session needs exactly one human seat")


class Session:
    """One bridge session: fixed seats, dealer rotation, cumulative score per side.

    Partnerships are fixed for the whole session the moment seats are assigned -
    seat 0 and seat 2 are always partners, as are seat 1 and seat 3
    (`games.bridge.rules.partner`), and nothing in a session ever changes that.
    `scores` gives both partners in a side the same running total, the same
    convention `Bridge.payoff()` already uses for raw trick counts - reading a
    side's score is just `scores[either_partners_seat]`.
    """

    def __init__(
        self,
        config: SessionConfig,
        human_bid_strategy: BidStrategy,
        human_play_strategy: PlayStrategy,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self._rng = rng if rng is not None else random.Random()

        self.roster: tuple[SeatConfig, ...] = config.seats
        self.human_seat: Seat = next(i for i, seat in enumerate(self.roster) if seat.is_human)
        self.dealer: Seat = 0
        self.hands_played: int = 0
        self.scores: dict[Seat, int] = dict.fromkeys(range(SEATS), 0)

        self._bid_strategies: list[BidStrategy] = []
        self._play_strategies: list[PlayStrategy] = []
        for _seat, config_seat in enumerate(self.roster):
            if config_seat.is_human:
                self._bid_strategies.append(human_bid_strategy)
                self._play_strategies.append(human_play_strategy)
            else:
                assert config_seat.persona is not None  # enforced by SeatConfig.__post_init__
                self._bid_strategies.append(BiddingStrategy(config_seat.persona, self._rng))
                self._play_strategies.append(CardPlayStrategy(config_seat.persona, self._rng))

    def play_next_hand(self) -> HandResult:
        result = play_hand(
            self._bid_strategies, self._play_strategies, dealer=self.dealer, rng=self._rng
        )
        if result.score is not None:
            assert result.contract is not None  # score is only ever set alongside a contract
            declarer_side = {result.contract.declarer, partner(result.contract.declarer)}
            for seat in range(SEATS):
                gain = result.score.points if seat in declarer_side else -result.score.points
                self.scores[seat] += gain

        self.hands_played += 1
        self.dealer = (self.dealer + 1) % SEATS
        return result
