"""The shared, game-agnostic `Game` protocol every game implements. See ARCHITECTURE.md §2."""

from typing import Protocol

PlayerId = int


class InformationSet(Protocol):
    """What a player observes about `state` — the basis for determinization/PIMC sampling
    in imperfect-information games. Shape is intentionally left to each game."""


# State: game-specific state representation. Action: game-specific — poker's Action != bridge's
# Action, only the shape below is shared.
class Game[State, Action](Protocol):
    def current_player(self, state: State) -> PlayerId: ...

    def legal_actions(self, state: State) -> list[Action]: ...

    def apply(self, state: State, action: Action) -> State: ...

    def is_terminal(self, state: State) -> bool: ...

    def payoff(self, state: State) -> dict[PlayerId, float]: ...

    def information_set(self, state: State, player: PlayerId) -> InformationSet: ...
