"""Drives one complete Court Piece hand end to end, and a multi-hand session on top
of it - the same role games/bridge/session.py plays for bridge, genuinely simpler
here: one single-shot declare decision instead of a multi-round auction, and no
dummy/contract-target bookkeeping at all.

Match length is a real session-level choice, not fixed the way bridge's is
open-ended: `target_courts` ends the session the instant either side's cumulative
court total reaches it; `None` means free play, exactly like bridge's own
open-ended-until-you-quit session.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from engine.cards import Card, Deck, Suit
from engine.trick_taking.resolution import Seat
from games.court_piece.deal import DECLARE_PREVIEW, SEATS, deal_hands
from games.court_piece.personas import CardPlayStrategy, CourtPiecePersona
from games.court_piece.personas import DeclareStrategy as PersonaDeclareStrategy
from games.court_piece.rules import CourtPiece, CourtPieceState, TrumpCall, partner
from games.court_piece.scoring import HandScore, score_hand, score_running_hand


class DeclareStrategy(Protocol):
    """Chooses the trump suit for one seat's declare-phase decision."""

    def __call__(self, preview: Sequence[Card]) -> Suit: ...


class PlayStrategy(Protocol):
    """Chooses a card for one seat's card-play decision."""

    def __call__(self, game: CourtPiece, state: CourtPieceState) -> Card: ...


@dataclass(frozen=True)
class HandResult:
    """What happened in one complete hand, from a fresh deal through to a score."""

    caller: Seat
    hands: tuple[tuple[Card, ...], ...]
    call: TrumpCall
    final_state: CourtPieceState
    score: HandScore


def _run_play(
    play_strategies: Sequence[PlayStrategy], call: TrumpCall, hands: Sequence[Sequence[Card]]
) -> CourtPieceState:
    game = CourtPiece()
    state = CourtPieceState(call=call, hands=tuple(tuple(hand) for hand in hands))
    while not game.is_terminal(state):
        player = game.current_player(state)
        action = play_strategies[player](game, state)
        state = game.apply(state, action)
    return state


def play_hand(
    declare_strategies: Sequence[DeclareStrategy],
    play_strategies: Sequence[PlayStrategy],
    *,
    caller: Seat,
    deck: Deck | None = None,
    rng: random.Random | None = None,
) -> HandResult:
    """Play one hand: deal, run the single declare-phase decision, play it out, score it.

    All 13 cards are dealt upfront (the same block deal every game in this project
    uses) - `caller` only ever sees the first `DECLARE_PREVIEW` of their own 13 before
    calling, matching the real game's "call from a 5-card preview" rule, even though
    the rest of their hand already physically exists in `hands` by the time they call.
    """
    hands = deal_hands(deck=deck, rng=rng)
    preview = hands[caller][:DECLARE_PREVIEW]
    trump = declare_strategies[caller](preview)
    call = TrumpCall(trump=trump, caller=caller)

    final_state = _run_play(play_strategies, call, hands)
    score = score_hand(call, final_state.completed)

    return HandResult(caller=caller, hands=hands, call=call, final_state=final_state, score=score)


def play_running_hand(
    play_strategies: Sequence[PlayStrategy],
    *,
    first_leader: Seat,
    deck: Deck | None = None,
    rng: random.Random | None = None,
) -> HandResult:
    """Play one running-trump hand: deal, play it out from an undetermined trump,
    score it. No declare phase at all - `CourtPiece.apply()` sets trump itself, the
    moment any player first can't follow suit (games/court_piece/rules.py).

    `first_leader` is the seat leading the very first trick, rotated each hand the
    same way fixed trump rotates its caller (`TrumpCall`'s own docstring explains why
    the same field still does real work here despite there being no "call" at all).
    """
    hands = deal_hands(deck=deck, rng=rng)
    call = TrumpCall(trump=None, caller=first_leader)

    final_state = _run_play(play_strategies, call, hands)
    score = score_running_hand(final_state.completed)

    # final_state.call, not the original `call` - trump starts undetermined and
    # CourtPiece.apply() resolves it into final_state.call as play goes; the initial
    # call is always trump=None and would misreport what this hand actually settled on.
    return HandResult(
        caller=first_leader,
        hands=hands,
        call=final_state.call,
        final_state=final_state,
        score=score,
    )


@dataclass(frozen=True)
class SeatConfig:
    """Static identity of one seat, fixed for the whole session."""

    name: str
    is_human: bool
    persona: CourtPiecePersona | None = None  # required for bots, unused for the human

    def __post_init__(self) -> None:
        if self.is_human and self.persona is not None:
            raise ValueError("the human seat does not have a persona")
        if not self.is_human and self.persona is None:
            raise ValueError("a bot seat needs a persona")


@dataclass(frozen=True)
class SessionConfig:
    """Exactly four seats, one of them human - no variable table size, the same as
    bridge's SessionConfig."""

    seats: tuple[SeatConfig, SeatConfig, SeatConfig, SeatConfig]
    target_courts: float | None = None  # None = free play, no match-ending threshold

    def __post_init__(self) -> None:
        humans = [seat for seat in self.seats if seat.is_human]
        if len(humans) != 1:
            raise ValueError("a session needs exactly one human seat")
        if self.target_courts is not None and self.target_courts <= 0:
            raise ValueError("target_courts must be positive")


class Session:
    """One Court Piece session: fixed seats, caller rotation, cumulative courts per
    side. Partnerships are fixed for the whole session the moment seats are
    assigned - seat 0 and seat 2 are always partners, as are seat 1 and seat 3.
    """

    def __init__(
        self,
        config: SessionConfig,
        human_declare_strategy: DeclareStrategy,
        human_play_strategy: PlayStrategy,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self._rng = rng if rng is not None else random.Random()

        self.roster: tuple[SeatConfig, ...] = config.seats
        self.human_seat: Seat = next(i for i, seat in enumerate(self.roster) if seat.is_human)
        self.caller: Seat = 0
        self.hands_played: int = 0
        self.courts: dict[Seat, float] = dict.fromkeys(range(SEATS), 0.0)

        self._declare_strategies: list[DeclareStrategy] = []
        self._play_strategies: list[PlayStrategy] = []
        for config_seat in self.roster:
            if config_seat.is_human:
                self._declare_strategies.append(human_declare_strategy)
                self._play_strategies.append(human_play_strategy)
            else:
                assert config_seat.persona is not None  # enforced by SeatConfig.__post_init__
                persona = config_seat.persona
                self._declare_strategies.append(PersonaDeclareStrategy(persona, self._rng))
                self._play_strategies.append(CardPlayStrategy(persona, self._rng))

    def is_over(self) -> bool:
        if self.config.target_courts is None:
            return False
        return any(total >= self.config.target_courts for total in self.courts.values())

    def play_next_hand(self) -> HandResult:
        if self.is_over():
            raise RuntimeError("the session already reached its target; no more hands to play")

        result = play_hand(
            self._declare_strategies, self._play_strategies, caller=self.caller, rng=self._rng
        )
        if result.score.winner is not None:
            winning_side = {result.score.winner, partner(result.score.winner)}
            for seat in winning_side:
                self.courts[seat] += result.score.courts

        self.hands_played += 1
        self.caller = (self.caller + 1) % SEATS
        return result


class RunningSession:
    """One running-trump Court Piece session: fixed seats, first-leader rotation,
    cumulative courts per side. Deliberately a separate small class rather than a
    branch inside `Session` - the two variants differ in enough places at once
    (no declare phase or DeclareStrategy at all, flat scoring with no caller
    multiplier) that a shared class would need conditionals threaded through nearly
    every method for little real reuse; duplicating this much bookkeeping reads more
    plainly than that. `SeatConfig`/`SessionConfig` and each bot's `CardPlayStrategy`
    are still the exact same types `Session` uses - personas are already
    trump-mode-agnostic (games/court_piece/personas.py), and a bot seat's unused
    `declare_bias` simply never gets called here.
    """

    def __init__(
        self,
        config: SessionConfig,
        human_play_strategy: PlayStrategy,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self._rng = rng if rng is not None else random.Random()

        self.roster: tuple[SeatConfig, ...] = config.seats
        self.human_seat: Seat = next(i for i, seat in enumerate(self.roster) if seat.is_human)
        self.first_leader: Seat = 0
        self.hands_played: int = 0
        self.courts: dict[Seat, float] = dict.fromkeys(range(SEATS), 0.0)

        self._play_strategies: list[PlayStrategy] = []
        for config_seat in self.roster:
            if config_seat.is_human:
                self._play_strategies.append(human_play_strategy)
            else:
                assert config_seat.persona is not None  # enforced by SeatConfig.__post_init__
                self._play_strategies.append(CardPlayStrategy(config_seat.persona, self._rng))

    def is_over(self) -> bool:
        if self.config.target_courts is None:
            return False
        return any(total >= self.config.target_courts for total in self.courts.values())

    def play_next_hand(self) -> HandResult:
        if self.is_over():
            raise RuntimeError("the session already reached its target; no more hands to play")

        result = play_running_hand(
            self._play_strategies, first_leader=self.first_leader, rng=self._rng
        )
        if result.score.winner is not None:
            winning_side = {result.score.winner, partner(result.score.winner)}
            for seat in winning_side:
                self.courts[seat] += result.score.courts

        self.hands_played += 1
        self.first_leader = (self.first_leader + 1) % SEATS
        return result
