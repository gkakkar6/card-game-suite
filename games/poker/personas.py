"""Poker personas: named playing styles built on the shared quantal engine.

Each persona is just a temperature and a per-action bias fed to engine/personas/quantal.py
(ARCHITECTURE.md §3) - no poker-specific decision mechanism, and no new machinery to
make a bot play badly on purpose. A near-optimal baseline plus four ways of being wrong:
too aggressive, too passive, unwilling to fold, and simply noisy. The baseline should
beat all four (§5), which tests the architecture against deviation in several
directions rather than a single axis.

Values are normalised by what is at stake before temperature and bias touch them, which
is the game-specific half of §3's rule. Poker's action values are in chips and chips
scale with the pot, so without it a bias tuned to look right in one pot size is either
invisible or overwhelming in another. quantal.py stays a generic value-to-probability
mapper and knows nothing about pots; the normalisation belongs here.
"""

import random
from collections.abc import Mapping, Sequence

from engine.cards import Card
from engine.evaluation import (
    DEFAULT_HANDS,
    DEFAULT_RESAMPLES,
    HandOutcome,
    MatchupResult,
    evaluate_matchup,
)
from engine.personas.quantal import Persona, choose_for
from games.poker.action_values import (
    DECISION_ITERATIONS,
    DECISION_MAX_EXACT_ENUMERATION,
    Sizing,
    action_for,
    analyse,
)
from games.poker.betting import ActionType, BettingAction
from games.poker.equity import hand_equity
from games.poker.hand import DEFAULT_STARTING_STACK, DecisionView, HandResult, play_hand

# Every number below is on the normalised scale - a bias of 0.1 means "a tenth of what
# is at stake", not a tenth of a chip - and every one was measured rather than guessed,
# over a sample of 1,723 real decisions from 400 baseline-vs-baseline hands. The two
# figures quoted are how often a persona takes the top-valued action (how mechanical it
# is) and how often it takes its lean-direction action (how strong the lean is).

# Near-optimal: takes the best action 82% of the time. Lower would be near-deterministic
# (92% at 0.02) and stop being a probabilistic policy in any meaningful sense.
BASELINE_TEMPERATURE = 0.05

# Noisy rather than leaning: takes the best action 38% of the time, against the
# baseline's 82%, with three or four actions available so pure chance is around 33%.
# Still slightly better than random, which is the point - it is erratic, not broken.
ERRATIC_TEMPERATURE = 2.0

# Bet or raise 81% of the time, against the baseline's 51%, still choosing the
# best-valued action 66% of the time. Pushing to 0.2 tips it over 87% aggressive, which
# starts being a bot that just bets rather than one that leans towards betting.
BLUFFER_BIAS = 0.15

# The mirror image by measured effect rather than by matching magnitude: bets or raises
# 20% of the time, so it plays passively on 80% of decisions - the same lean strength as
# the bluffer's 81%, reached at a different number because the values either side of the
# decision are not symmetric.
CONSERVATIVE_BIAS = -0.20

# Applied to folding, a different axis from the two above. Facing a bet, this folds 1.1%
# of the time against the baseline's 7.5%.
#
# Worth being straight about: this is the weakest of the four deviations by construction.
# Against a uniform opponent range, calling is nearly always better than folding - fold
# is the top-valued action on only 6% of decisions that face a bet - so the baseline
# already almost never folds, and a persona that folds even less has little room to be
# different. That is a direct consequence of the no-opponent-model limitation documented
# in action_values.py, not a badly chosen number.
CALLING_STATION_BIAS = -0.20

AGGRESSIVE_ACTIONS = (ActionType.BET, ActionType.RAISE)

# Keeps the deal seeds of different runs from overlapping when `seed` is varied.
DEAL_SEED_STRIDE = 1_000_003

# A duplicate deal is two hands: the same cards played from each seat in turn.
DUPLICATE_PAIR = 2

# Near-optimal reference point: low noise, no lean, plays the values as scored.
BASELINE = Persona[ActionType](name="baseline", temperature=BASELINE_TEMPERATURE, bias={})

# Deviation one: bets and raises far more than the values justify.
BLUFFER = Persona[ActionType](
    name="bluffer",
    temperature=BASELINE_TEMPERATURE,
    bias=dict.fromkeys(AGGRESSIVE_ACTIONS, BLUFFER_BIAS),
)

# Deviation two: the mirror image, suppressing aggression unless it is clearly strong.
CONSERVATIVE = Persona[ActionType](
    name="conservative",
    temperature=BASELINE_TEMPERATURE,
    bias=dict.fromkeys(AGGRESSIVE_ACTIONS, CONSERVATIVE_BIAS),
)

# Deviation three, on a different axis: loose-passive. It leans against folding rather
# than for or against aggression, so it pays off bets it should let go.
CALLING_STATION = Persona[ActionType](
    name="calling station",
    temperature=BASELINE_TEMPERATURE,
    bias={ActionType.FOLD: CALLING_STATION_BIAS},
)

# Deviation four, on no axis at all: noise without direction. This is the one that
# closes §4's original claim, that a low-temperature bot beats a high-temperature one.
ERRATIC = Persona[ActionType](name="erratic", temperature=ERRATIC_TEMPERATURE, bias={})

DEVIATIONS = (BLUFFER, CONSERVATIVE, CALLING_STATION, ERRATIC)
PERSONAS = {persona.name: persona for persona in (BASELINE, *DEVIATIONS)}


def decision_scale(view: DecisionView) -> int:
    """What is at stake at this decision - the natural unit to measure values against.

    Poker's action values are in chips, and chips scale with the pot, so the same
    action can be worth 8 chips in one spot and 140 in another without the decision
    being any different in character. Dividing by what is at stake turns a bias into
    "a fraction of the pot", which means the same thing in both (ARCHITECTURE.md §3).
    """
    return max(view.pot + view.to_call, 1)


def normalised(
    values: Mapping[ActionType, float], view: DecisionView
) -> dict[ActionType, float]:
    """Action values as fractions of what is at stake, for temperature and bias."""
    scale = decision_scale(view)
    return {action: value / scale for action, value in values.items()}


def strategy_for(
    persona: Persona[ActionType],
    rng: random.Random | None = None,
    sizing: Sizing | None = None,
) -> "PersonaStrategy":
    return PersonaStrategy(persona, rng=rng, sizing=sizing)


class PersonaStrategy:
    """Plays a hand by scoring the legal actions, then letting the persona pick one.

    Equity is memoised on (hole cards, board) for the duration of a single hand: it
    does not change when the pot or the bet facing the player does, so acting more than
    once on a street costs nothing the second time.

    The cache is cleared between hands rather than kept for a whole run, and that is a
    correctness point, not tidiness. A decision-time equity is a noisy estimate, so
    holding one across a long run means any (hole cards, board) that recurs by chance
    is answered with the same stale draw instead of a fresh one. Those errors then stop
    behaving like independent noise that averages out over the run, and each persona
    carries a different set of them - a small systematic asymmetry between exactly the
    two players an evaluation is trying to compare.
    """

    def __init__(
        self,
        persona: Persona[ActionType],
        rng: random.Random | None = None,
        sizing: Sizing | None = None,
    ) -> None:
        self.persona = persona
        self.rng = rng if rng is not None else random.Random()
        self.sizing = sizing
        self._equity: dict[tuple[tuple[Card, ...], tuple[Card, ...]], float] = {}

    def start_hand(self) -> None:
        """Forget the previous hand's equity estimates. See the class docstring."""
        self._equity.clear()

    def _equity_for(self, view: DecisionView) -> float:
        key = (view.hole_cards, view.board)
        cached = self._equity.get(key)
        if cached is None:
            cached = hand_equity(
                view.hole_cards,
                view.board,
                iterations=DECISION_ITERATIONS,
                max_exact_enumeration=DECISION_MAX_EXACT_ENUMERATION,
                rng=self.rng,
            ).equity
            self._equity[key] = cached
        return cached

    def __call__(self, view: DecisionView) -> BettingAction:
        analysis = analyse(
            view, sizing=self.sizing, rng=self.rng, equity=self._equity_for(view)
        )
        chosen = choose_for(self.persona, normalised(analysis.values, view), self.rng)
        return action_for(chosen, analysis)


def _seat_of(hand_index: int) -> int:
    """Which seat the evaluated persona takes. Alternates so neither side keeps the
    small blind - position is worth real chips, and it would otherwise be baked into
    the result as a fake edge."""
    return hand_index % 2


def play_matchup(
    persona: Persona[ActionType],
    opponent: Persona[ActionType],
    *,
    hands: int = DEFAULT_HANDS,
    rng: random.Random | None = None,
    starting_stack: int = DEFAULT_STARTING_STACK,
    resamples: int = DEFAULT_RESAMPLES,
    duplicate: bool = True,
    seed: int = 0,
) -> MatchupResult:
    """Play `persona` against `opponent` heads-up and report `persona`'s edge.

    With `duplicate` set, each deal is played twice - once from each seat - by reusing
    the same shuffle for consecutive pairs of hands. Whoever was dealt the better cards,
    both personas face them, so most of the luck cancels instead of drowning the skill
    difference in noise. Without it, the swings from card luck are so much larger than
    the gap between two personas that an honest interval stays far too wide to conclude
    anything from a feasible number of hands.
    """
    shared = rng if rng is not None else random.Random()
    evaluated = strategy_for(persona, shared)
    against = strategy_for(opponent, shared)

    def play_one(hand_index: int) -> HandOutcome:
        seat = _seat_of(hand_index)
        seated: Sequence[PersonaStrategy] = (
            [evaluated, against] if seat == 0 else [against, evaluated]
        )
        for strategy in seated:
            strategy.start_hand()  # equity estimates do not carry between hands
        # Pairs of hands share a deal seed, so the second replays the first's cards.
        deal = random.Random(seed * DEAL_SEED_STRIDE + hand_index // 2) if duplicate else shared
        result: HandResult = play_hand(seated, rng=deal, starting_stack=starting_stack)
        return HandOutcome(profit=result.net(seat), won=seat in result.winners)

    return evaluate_matchup(
        play_one,
        hands=hands,
        rng=shared,
        resamples=resamples,
        block=DUPLICATE_PAIR if duplicate else 1,
    )
