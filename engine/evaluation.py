"""Bot-vs-bot evaluation: does one player actually beat another, or does it just look
like it (ARCHITECTURE.md §4)?

Any claimed edge is reported with a confidence interval, so "this bot is better" is a
number with error bars rather than a vibe. An interval that straddles zero means the
result is consistent with no difference at all, however good the raw average looked.

Game-agnostic per §1: this never imports a game. Callers hand it a callable that plays
one hand and reports the outcome, so the same harness scores poker, bridge or Court
Piece without changing.

Confidence intervals use the **bootstrap**, not Wilson score. Wilson is specific to a
binomial proportion, which would mean scoring a hand as simply won or lost - and in
poker that is the wrong question: a bot can win most hands while losing money, if the
pots it loses are bigger than the ones it wins. The statistic that matters is average
profit per hand, whose distribution is skewed and heavy-tailed (most hands are small,
a few are huge). The bootstrap makes no assumption about that shape, so it stays
honest where a normal approximation would not.
"""

import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

DEFAULT_HANDS = 5_000  # fixed count per matchup; adaptive stopping is a later refinement
DEFAULT_RESAMPLES = 2_000
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class HandOutcome:
    """One hand's result from the evaluated player's point of view."""

    profit: float  # chips won or lost
    won: bool  # whether they took (or shared) the pot


@dataclass(frozen=True)
class MatchupResult:
    hands: int
    win_rate: float  # share of hands won, reported for context
    mean_profit: float  # chips per hand - the statistic the interval is built on
    ci_low: float
    ci_high: float
    confidence: float

    @property
    def beats_opponent(self) -> bool:
        """True only if the whole interval sits above break-even."""
        return self.ci_low > 0.0

    @property
    def is_significant(self) -> bool:
        """True if the interval excludes zero in either direction."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def summary(self) -> str:
        return (
            f"{self.mean_profit:+.3f} chips/hand "
            f"[{self.ci_low:+.3f}, {self.ci_high:+.3f}] at {self.confidence:.0%}, "
            f"win rate {self.win_rate:.1%} over {self.hands:,} hands"
        )


def bootstrap_ci(
    samples: Sequence[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    block: int = 1,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of `samples`.

    Resamples the observed hands with replacement many times and takes the spread of
    the resulting means, which needs no assumption that the underlying distribution is
    normal - and chip results are decidedly not.

    `block` groups consecutive samples and resamples those groups instead of individual
    hands. Samples that were deliberately paired - the same deal played from both seats
    - are not independent, and resampling them separately would discard exactly the
    cancellation the pairing was set up to get, reporting an interval as wide as if the
    pairing had never happened.
    """
    if not samples:
        raise ValueError("need at least one sample")
    if block < 1:
        raise ValueError("block size must be at least one")
    if len(samples) == 1:
        only = float(samples[0])
        return (only, only)

    blocks = [samples[start : start + block] for start in range(0, len(samples), block)]
    draw = (rng or random.Random()).choices
    means = sorted(
        statistics.fmean([value for chunk in draw(blocks, k=len(blocks)) for value in chunk])
        for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    low = means[int(tail * resamples)]
    high = means[min(int((1.0 - tail) * resamples), resamples - 1)]
    return (low, high)


def evaluate_matchup(
    play_hand: Callable[[int], HandOutcome],
    *,
    hands: int = DEFAULT_HANDS,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    block: int = 1,
    rng: random.Random | None = None,
) -> MatchupResult:
    """Play `hands` hands and report the evaluated player's edge with an interval.

    `play_hand` is called with the hand number, so the caller can alternate seats or
    otherwise vary the setup, and returns that hand's outcome for the evaluated player.
    Set `block` to the size of any deliberately paired group of hands, so the interval
    accounts for the pairing rather than treating those hands as independent.
    """
    if hands < 1:
        raise ValueError("need at least one hand to evaluate")

    outcomes = [play_hand(index) for index in range(hands)]
    profits = [outcome.profit for outcome in outcomes]
    low, high = bootstrap_ci(
        profits, confidence=confidence, resamples=resamples, block=block, rng=rng
    )
    return MatchupResult(
        hands=hands,
        win_rate=sum(outcome.won for outcome in outcomes) / hands,
        mean_profit=statistics.fmean(profits),
        ci_low=low,
        ci_high=high,
        confidence=confidence,
    )
