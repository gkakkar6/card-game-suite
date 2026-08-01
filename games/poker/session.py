"""Runs a multiplayer poker session: table setup, persistent bankrolls, button
rotation, and bot elimination on top of games/poker/hand.py's single-hand engine.

No terminal I/O in this module - every call returns structured state and lets the
caller (scripts/play_cli.py today, a web UI later) decide how to display it or prompt
for input. That is the point of putting this logic here rather than in the CLI script:
the same Session drives both.
"""

import random
from dataclasses import dataclass, field
from typing import Literal

from engine.cards import Card
from engine.personas.quantal import Persona
from games.poker.betting import ActionType
from games.poker.hand import DEFAULT_STARTING_STACK, Blinds, HandResult, Strategy, play_hand
from games.poker.personas import PersonaStrategy, strategy_for

MIN_BOTS = 1
MAX_BOTS = 5

# A common small-tournament pace, not an arbitrary number: enough hands per blind
# level for skill to matter, but not so many that blinds stay trivial relative to
# stacks for most of the session - which is the point of "increasing" mode, forcing
# the session to actually end rather than continuing indefinitely at flat stakes.
DEFAULT_DOUBLE_EVERY = 10


@dataclass(frozen=True)
class SeatConfig:
    """Static identity of one seat, fixed for the whole session."""

    name: str
    is_human: bool
    persona: Persona[ActionType] | None = None  # required for bots, unused for the human

    def __post_init__(self) -> None:
        if self.is_human and self.persona is not None:
            raise ValueError("the human seat does not have a persona")
        if not self.is_human and self.persona is None:
            raise ValueError("a bot seat needs a persona")


@dataclass(frozen=True)
class BlindSchedule:
    """How blinds change as a session progresses.

    "fixed": the same Blinds every hand - casual practice, nothing forcing an end.
    "increasing": doubles every `double_every` hands - a real tournament structure,
    since eventually stacks can't cover the blinds and the session is forced to a
    conclusion rather than continuing forever.
    """

    mode: Literal["fixed", "increasing"] = "fixed"
    base: Blinds = field(default_factory=Blinds)
    double_every: int = DEFAULT_DOUBLE_EVERY

    def for_hand(self, hand_number: int) -> Blinds:
        """Blinds for the hand about to be played (`hand_number` is 0-indexed)."""
        if self.mode == "fixed":
            return self.base
        doublings = hand_number // self.double_every
        factor = 2**doublings
        return Blinds(small=self.base.small * factor, big=self.base.big * factor)


@dataclass(frozen=True)
class SessionConfig:
    """What a session is set up with, fixed for its whole lifetime.

    `strict` and the blind schedule are independent toggles, not coupled: a
    tournament is "increasing" blinds plus `strict=True` together, but either can be
    set alone (e.g. increasing blinds with bots auto-replenishing, for a longer casual
    session that still ramps up pressure).
    """

    seats: tuple[SeatConfig, ...]
    starting_stack: int = DEFAULT_STARTING_STACK
    blind_schedule: BlindSchedule = field(default_factory=BlindSchedule)
    strict: bool = False

    def __post_init__(self) -> None:
        humans = [seat for seat in self.seats if seat.is_human]
        if len(humans) != 1:
            raise ValueError("a session needs exactly one human seat")
        num_bots = len(self.seats) - 1
        if not (MIN_BOTS <= num_bots <= MAX_BOTS):
            raise ValueError(f"need {MIN_BOTS}-{MAX_BOTS} bots, got {num_bots}")


@dataclass(frozen=True)
class HandSummary:
    """What happened in one hand - everything a caller needs to render it."""

    hand_number: int
    order: tuple[int, ...]  # seat ids, aligned with `result`'s player indices
    result: HandResult
    blinds: Blinds
    eliminated: tuple[int, ...]  # bot seat ids removed this hand
    replenished: int | None  # bot seat id kept alive as the last bot, if any
    human_busted: bool
    strict_end: bool  # True if strict mode ended the session on this hand
    button_before: int
    button_after: int | None  # None if the session ended

    @property
    def winner_seats(self) -> tuple[int, ...]:
        """`result.winners` translated from hand-local indices to seat ids."""
        return tuple(self.order[index] for index in self.result.winners)

    def hole_cards_for(self, seat: int) -> tuple[Card, ...]:
        return self.result.hole_cards[self.order.index(seat)]


def _seats_from(num_seats: int, active: frozenset[int], start: int) -> list[int]:
    """Active seat ids in table order, beginning at `start` (included if active)."""
    return [seat for offset in range(num_seats) if (seat := (start + offset) % num_seats) in active]


def _next_active_seat(num_seats: int, active: frozenset[int], after: int) -> int:
    """The next active seat strictly clockwise from `after`."""
    ordered = _seats_from(num_seats, active, (after + 1) % num_seats)
    if not ordered:
        raise ValueError("no active seats remain")
    return ordered[0]


def deal_order(num_seats: int, active: frozenset[int], button: int) -> list[int]:
    """Seat ids in the order hand.py expects: index 0 is the small blind.

    hand.py fixes the small blind at its own seat 0 and the button at its last seat,
    except heads-up where the button and small blind are the same seat. Rather than
    adding a button parameter to hand.py itself, a shrinking, rotating table is just a
    different permutation of who currently occupies which hand.py seat - this mirrors
    that existing convention instead of duplicating it.
    """
    if button not in active:
        raise ValueError("the button must be on an active seat")
    from_button = _seats_from(num_seats, active, button)  # button is first in this list
    if len(from_button) == 2:
        return from_button  # heads-up: the button doubles as the small blind
    return from_button[1:] + from_button[:1]  # small blind first, button last


class Session:
    """One multiplayer session: table, bankrolls, button, blinds, eliminations.

    Bot elimination: a bot whose stack hits zero is removed from the table, unless it
    is the only bot left. In that case, by default (`SessionConfig.strict=False`) it
    is auto-replenished to the starting stack and the session continues, so the human
    always has an opponent to play against; in strict mode it is not replenished and
    the session ends there instead.

    The human busting is a different case, signalled rather than handled: the returned
    HandSummary's `human_busted` is set, and no further hand can be played until the
    caller calls `rebuy_human()` - deciding whether to rebuy or quit is the caller's
    call, not this class's.
    """

    def __init__(
        self,
        config: SessionConfig,
        human_strategy: Strategy,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self.human_strategy = human_strategy
        self._rng = rng if rng is not None else random.Random()

        self.roster: tuple[SeatConfig, ...] = config.seats
        self.human_seat: int = next(i for i, seat in enumerate(self.roster) if seat.is_human)
        self.stacks: dict[int, int] = dict.fromkeys(range(len(self.roster)), config.starting_stack)
        self.active: set[int] = set(range(len(self.roster)))
        self.button: int = 0
        self.hands_played: int = 0
        self.human_busted: bool = False
        self.human_total_buyins: int = config.starting_stack
        self.ended: bool = False
        self.end_reason: str | None = None

        self._bots: dict[int, PersonaStrategy] = {}
        for i, seat in enumerate(self.roster):
            if seat.is_human:
                continue
            assert seat.persona is not None  # enforced by SeatConfig.__post_init__
            self._bots[i] = strategy_for(seat.persona, self._rng)

    @property
    def num_seats(self) -> int:
        return len(self.roster)

    @property
    def human_net(self) -> int:
        """What the human is up or down across the whole session, including rebuys."""
        return self.stacks[self.human_seat] - self.human_total_buyins

    def rebuy_human(self, amount: int | None = None) -> None:
        """Reset the human's stack and clear the busted flag so play can resume."""
        if not self.human_busted:
            raise RuntimeError("the human is not busted")
        amount = amount if amount is not None else self.config.starting_stack
        self.stacks[self.human_seat] = amount
        self.human_total_buyins += amount
        self.human_busted = False

    def play_next_hand(self) -> HandSummary:
        if self.ended:
            raise RuntimeError("this session has already ended")
        if self.human_busted:
            raise RuntimeError("the human is busted; call rebuy_human() before playing again")

        order = deal_order(self.num_seats, frozenset(self.active), self.button)
        blinds = self.config.blind_schedule.for_hand(self.hands_played)

        for seat in order:
            bot = self._bots.get(seat)
            if bot is not None:
                bot.start_hand()  # equity estimates do not carry between hands

        strategies = [
            self.human_strategy if seat == self.human_seat else self._bots[seat] for seat in order
        ]
        starting_stacks = [self.stacks[seat] for seat in order]

        result = play_hand(
            strategies, blinds=blinds, starting_stacks=starting_stacks, rng=self._rng
        )
        for index, seat in enumerate(order):
            self.stacks[seat] = result.final_stacks[index]

        button_before = self.button
        if self.stacks[self.human_seat] == 0:
            self.human_busted = True

        eliminated, replenished, strict_end = self._process_bot_eliminations(order)

        self.hands_played += 1
        button_after = None
        if not self.ended:
            button_after = _next_active_seat(self.num_seats, frozenset(self.active), button_before)
            self.button = button_after

        return HandSummary(
            hand_number=self.hands_played - 1,
            order=tuple(order),
            result=result,
            blinds=blinds,
            eliminated=tuple(eliminated),
            replenished=replenished,
            human_busted=self.human_busted,
            strict_end=strict_end,
            button_before=button_before,
            button_after=button_after,
        )

    def _process_bot_eliminations(
        self, order: list[int]
    ) -> tuple[list[int], int | None, bool]:
        """Remove busted bots, replenishing (or ending in strict mode) the last one.

        Processes busted bots one at a time so ties - more than one bot busting in the
        same hand - resolve deterministically: whichever is checked last, once no
        other bot remains active, is the one kept alive (or that ends the session).
        Which specific bot that is among simultaneous busts is an arbitrary but stable
        tie-break, not a meaningful choice.
        """
        eliminated: list[int] = []
        replenished: int | None = None
        strict_end = False

        busted_bots = [seat for seat in order if seat != self.human_seat and self.stacks[seat] == 0]
        for seat in busted_bots:
            other_bots_active = {s for s in self.active if s != self.human_seat and s != seat}
            if other_bots_active:
                self.active.discard(seat)
                eliminated.append(seat)
                continue

            # `seat` is the last bot standing, at zero chips.
            if self.config.strict:
                self.ended = True
                self.end_reason = f"{self.roster[seat].name} busted as the last bot (strict mode)"
                strict_end = True
                break
            self.stacks[seat] = self.config.starting_stack
            replenished = seat

        return eliminated, replenished, strict_end
