"""One betting round (a single street) among N seats: fold/check/call/bet/raise until settled.

Side-pot allocation for multiway all-ins is out of scope here (ARCHITECTURE.md §5 poker
stretch goals) - stacks/bets are tracked per player, but pot splitting on uneven all-ins
is left to a later evaluation-time step.
"""

from dataclasses import dataclass, field
from enum import Enum, auto


class ActionType(Enum):
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    BET = auto()
    RAISE = auto()


@dataclass(frozen=True)
class BettingAction:
    type: ActionType
    amount: int = 0  # total this player is putting in this action; BET/RAISE only, "raise to"


@dataclass
class BettingRound:
    """Seat `first_to_act` acts first. Settled once every player still in the hand has
    folded, gone all-in, or matched the current bet since the last bet/raise."""

    stacks: list[int]
    first_to_act: int = 0
    min_bet: int = 1
    initial_bets: list[int] | None = None  # e.g. posted blinds, already deducted from stacks

    pot: int = field(init=False, default=0)
    bets: list[int] = field(init=False)
    folded: set[int] = field(init=False)
    all_in: set[int] = field(init=False)
    current_player: int = field(init=False)
    _acted_since_last_raise: set[int] = field(init=False)
    # players who can no longer raise this round because the last raise was a short
    # (incomplete) all-in - real poker rule: an incomplete raise doesn't reopen the
    # betting for players who already matched the previous bet, it only owes them a
    # call of the extra amount (or a fold)
    _capped_from_short_raise: set[int] = field(init=False)

    def __post_init__(self) -> None:
        self.bets = (
            list(self.initial_bets) if self.initial_bets is not None else [0] * len(self.stacks)
        )
        self.pot = sum(self.bets)
        self.folded = set()
        self.all_in = {p for p, stack in enumerate(self.stacks) if stack == 0}
        self._acted_since_last_raise = set()
        self._capped_from_short_raise = set()
        self.current_player = self.first_to_act

    def _active_players(self) -> list[int]:
        return [p for p in range(len(self.stacks)) if p not in self.folded]

    def _other_active_non_all_in_players(self) -> list[int]:
        """Players besides the current one who could still respond to a raise."""
        return [
            p
            for p in range(len(self.stacks))
            if p != self.current_player and p not in self.folded and p not in self.all_in
        ]

    @property
    def current_bet(self) -> int:
        return max(self.bets, default=0)

    def _can_raise(self, to_call: int) -> bool:
        return (
            self.stacks[self.current_player] > to_call
            and self.current_player not in self._capped_from_short_raise
            and bool(self._other_active_non_all_in_players())
        )

    def legal_actions(self) -> list[ActionType]:
        if self.is_settled():
            return []
        to_call: int = self.current_bet - self.bets[self.current_player]
        actions: list[ActionType] = [ActionType.FOLD]
        if to_call == 0:
            actions.append(ActionType.CHECK)
            if self.stacks[self.current_player] > 0:
                actions.append(ActionType.BET)
        else:
            if self.stacks[self.current_player] > 0:
                actions.append(ActionType.CALL)
                if self._can_raise(to_call):
                    actions.append(ActionType.RAISE)
        return actions

    def apply(self, action: BettingAction) -> None:
        if action.type not in self.legal_actions():
            raise ValueError(f"{action.type} is not legal for player {self.current_player}")

        player = self.current_player
        if action.type is ActionType.FOLD:
            self.folded.add(player)
        elif action.type is ActionType.CHECK:
            self._acted_since_last_raise.add(player)
        elif action.type is ActionType.CALL:
            self._put_in(player, self.current_bet - self.bets[player])
            self._acted_since_last_raise.add(player)
        else:  # BET or RAISE
            self._apply_bet_or_raise(player, action)

        if self.stacks[player] == 0 and player not in self.folded:
            self.all_in.add(player)

        self._advance_to_next_player()

    def _apply_bet_or_raise(self, player: int, action: BettingAction) -> None:
        additional: int = action.amount - self.bets[player]  # chips this player adds now
        going_all_in: bool = additional == self.stacks[player]
        is_full_raise: bool = action.amount >= self.current_bet + self.min_bet
        if action.amount <= self.current_bet:
            raise ValueError("bet/raise must exceed the current bet")
        if additional > self.stacks[player]:
            raise ValueError("cannot bet more than remaining stack")
        # a short all-in raise (below the usual minimum) is still legal - a player can only
        # raise what they have
        if not is_full_raise and not going_all_in:
            min_raise = self.current_bet + self.min_bet
            raise ValueError(f"raise must be to at least {min_raise} (or all-in)")

        if is_full_raise:
            # a full raise reopens the action for everyone still in the hand
            self._capped_from_short_raise = set()
        else:
            # a short all-in doesn't reopen the action for players who already matched
            # the previous bet - they can only call the extra amount or fold, not re-raise
            self._capped_from_short_raise |= self._acted_since_last_raise

        self._put_in(player, additional)
        self._acted_since_last_raise = {player}

    def _put_in(self, player: int, amount: int) -> None:
        self.stacks[player] -= amount
        self.bets[player] += amount
        self.pot += amount

    def _advance_to_next_player(self) -> None:
        n = len(self.stacks)
        for offset in range(1, n + 1):
            candidate = (self.current_player + offset) % n
            if candidate not in self.folded and candidate not in self.all_in:
                self.current_player = candidate
                return

    def is_settled(self) -> bool:
        active: list[int] = self._active_players()
        if len(active) <= 1:
            return True
        # all-in players have nothing left to decide, so only non-folded, non-all-in
        # players need to have matched the bet for the round to be over
        contenders: list[int] = [p for p in active if p not in self.all_in]
        if not contenders:
            return True
        return all(
            p in self._acted_since_last_raise and self.bets[p] == self.current_bet
            for p in contenders
        )
