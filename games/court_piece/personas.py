"""Court Piece personas: named playing styles, covering both decisions a seat ever
makes - the declare-phase trump call, and card play - one combined profile per seat,
the same shape as games/bridge/personas.py's BridgePersona.

**Baseline only, deliberately, for this first version.** Bridge's six styles were
each measured against real bot-vs-bot play before being trusted (ARCHITECTURE.md §3,
§5) - inventing Court Piece equivalents (an "aggressive" declare bias, a bridge-style
"baiter") without that same measurement, or without knowing yet how they actually
feel at this table, would be guessing rather than building on evidence. Baseline
gives a genuinely playable near-optimal bot end to end; more named styles are a real,
easy extension on top of this same structure once there's something to test them
against (see games/bridge/personas.py's _tiered_bidding_bias /
_cards_above_the_mean for the pattern this would follow).
"""

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from engine.cards import Card, Suit
from engine.personas.quantal import choose
from games.court_piece.action_values import analyse
from games.court_piece.declare import DeclareBias, _no_declare_bias, declare_values
from games.court_piece.rules import CourtPiece, CourtPieceState

PlayBias = Callable[[CourtPiece, CourtPieceState, dict[Card, float]], dict[Card, float]]


def _no_play_bias(
    game: CourtPiece, state: CourtPieceState, values: dict[Card, float]
) -> dict[Card, float]:
    return {}


@dataclass(frozen=True)
class CourtPiecePersona:
    """One named style, covering both domains a seat ever decides in."""

    name: str
    declare_temperature: float
    declare_bias: DeclareBias
    play_temperature: float
    play_bias: PlayBias


# Low but nonzero, matching bridge's own baseline temperatures (games/bridge/personas.py) -
# near-optimal without being perfectly deterministic, which would make every baseline-vs-
# baseline hand identical given the same deal.
BASELINE_DECLARE_TEMPERATURE = 0.3
BASELINE_PLAY_TEMPERATURE = 0.05

BASELINE = CourtPiecePersona(
    name="baseline",
    declare_temperature=BASELINE_DECLARE_TEMPERATURE,
    declare_bias=_no_declare_bias,
    play_temperature=BASELINE_PLAY_TEMPERATURE,
    play_bias=_no_play_bias,
)

PERSONAS: dict[str, CourtPiecePersona] = {BASELINE.name: BASELINE}


class DeclareStrategy:
    """Chooses the trump suit for one seat's declare-phase decision, from its persona."""

    def __init__(self, persona: CourtPiecePersona, rng: random.Random | None = None) -> None:
        self.persona = persona
        self.rng = rng if rng is not None else random.Random()

    def __call__(self, preview: Sequence[Card]) -> Suit:
        values = declare_values(preview)
        bias = self.persona.declare_bias(values)
        return choose(values, self.persona.declare_temperature, bias, self.rng)


class CardPlayStrategy:
    """Chooses a card by scoring the legal ones via action_values.analyse() and its
    reported scale, then letting the persona pick - same normalisation discipline as
    games/bridge/personas.py's CardPlayStrategy."""

    def __init__(self, persona: CourtPiecePersona, rng: random.Random | None = None) -> None:
        self.persona = persona
        self.rng = rng if rng is not None else random.Random()

    def __call__(self, game: CourtPiece, state: CourtPieceState) -> Card:
        result = analyse(game, state, rng=self.rng)
        values = {card: value / result.scale for card, value in result.values.items()}
        bias = self.persona.play_bias(game, state, values)
        return choose(values, self.persona.play_temperature, bias, self.rng)
