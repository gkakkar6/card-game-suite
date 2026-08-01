"""The validating experiment for the whole architecture (ARCHITECTURE.md §4 and §5).

Marked `slow` and skipped by default (see pyproject.toml's addopts) - every test here
plays a real bot-vs-bot matchup with bootstrap resampling, so the module takes minutes,
not the sub-second cost of the rest of the suite. Run it explicitly with:

    uv run pytest -m slow

Worth running after touching action_values.py, quantal.py, equity.py, or any persona's
bias/temperature constants in personas.py - anything that can move a persona's actual
behaviour rather than just its code.

The roster covers four ways of being wrong: too aggressive, too passive, unwilling to
fold, and simply noisy. §5 expects the baseline to beat all four.

Measured over 1,000 duplicate-dealt hands per matchup by scripts/validate_personas.py,
which is the authority on the claim - three of the four hold:

    vs bluffer          +8.69 chips/hand, 95% CI [+5.09, +12.44]   beats
    vs calling station  +7.12 chips/hand, 95% CI [+3.79, +10.57]   beats
    vs erratic         +10.71 chips/hand, 95% CI [+7.83, +13.63]   beats
    vs conservative     -5.23 chips/hand, 95% CI [-7.84,  -2.60]   LOSES

The erratic result closes §4's original claim - that a low-temperature bot beats a
high-temperature one - end to end rather than only at the mechanism level in
tests/test_quantal.py.

The conservative result is a real failure, not noise: the whole interval sits below
zero. The cause is the no-opponent-model limitation named in action_values.py. Equity
is computed against a uniform random range, so when a tight opponent puts money in the
baseline has no way to notice their betting range is far stronger than random. Measured
over 120 hands, the conservative persona bets and raises holding mean equity 0.74
against the baseline's 0.65, and the baseline pays off the difference. Fixing it needs
opponent modelling - the weighted-range work already listed as a stretch goal in §5 -
not a different bias number, so it is recorded here rather than tuned away.

What this file asserts is direction at a hand count fast enough to sit in a suite. At
120 hands none of these intervals clear zero (the bluffer edge alone needs roughly a
thousand), so significance is the script's job; these are regression guards that catch
a change flipping a matchup the wrong way.
"""

import random

import pytest

from engine.evaluation import MatchupResult
from engine.personas.quantal import Persona
from games.poker.betting import ActionType
from games.poker.personas import (
    BASELINE,
    BLUFFER,
    CALLING_STATION,
    CONSERVATIVE,
    ERRATIC,
    play_matchup,
)

pytestmark = pytest.mark.slow

# Every hand runs real equity calculations, so this is the dominant cost in the suite.
VALIDATION_HANDS = 120


def _matchup(opponent: Persona[ActionType]) -> MatchupResult:
    return play_matchup(
        BASELINE, opponent, hands=VALIDATION_HANDS, rng=random.Random(1), resamples=300, seed=1
    )


@pytest.mark.parametrize("opponent", [BLUFFER, CALLING_STATION, ERRATIC])
def test_baseline_out_earns_the_deviations_it_beats(opponent: Persona[ActionType]) -> None:
    result = _matchup(opponent)
    assert result.mean_profit > 0.0, result.summary()


def test_beating_the_erratic_bot_isolates_temperature() -> None:
    # The erratic persona carries no bias at all, so this is noise being punished on its
    # own - §4's original claim, with nothing else varying alongside it.
    assert ERRATIC.bias == {}
    assert ERRATIC.temperature > BASELINE.temperature
    assert _matchup(ERRATIC).mean_profit > 0.0


@pytest.mark.xfail(
    reason=(
        "Known and measured: the baseline loses to a tight opponent because equity is "
        "computed against a uniform range, so it cannot see that a conservative bot's "
        "betting range is strong. -5.23 chips/hand, 95% CI [-7.84, -2.60] over 1,000 "
        "hands. Needs opponent modelling (a stretch goal in ARCHITECTURE.md §5), not a "
        "different bias number."
    ),
    strict=True,
)
def test_baseline_should_also_beat_the_conservative_bot() -> None:
    # Recorded as an expected failure rather than deleted, so the §5 claim stays visible
    # and turns into a passing result the day opponent modelling lands.
    assert _matchup(CONSERVATIVE).mean_profit > 0.0


def test_winning_more_hands_is_not_the_same_as_winning() -> None:
    # The bluffer takes about as many pots as the baseline and still loses clearly,
    # which is why the harness reports chips per hand rather than a win rate.
    result = _matchup(BLUFFER)
    assert result.win_rate <= 0.5
    assert result.mean_profit > 0.0
