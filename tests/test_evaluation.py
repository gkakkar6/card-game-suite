import random

import pytest

from engine.evaluation import HandOutcome, bootstrap_ci, evaluate_matchup


def constant_outcome(profit: float, won: bool = True) -> HandOutcome:
    return HandOutcome(profit=profit, won=won)


def test_bootstrap_interval_brackets_the_mean() -> None:
    rng = random.Random(1)
    samples = [rng.gauss(5.0, 2.0) for _ in range(500)]
    low, high = bootstrap_ci(samples, resamples=500, rng=random.Random(2))
    assert low < 5.0 < high
    assert low > 0.0  # a clear positive effect should not straddle zero


def test_bootstrap_interval_of_noise_includes_zero() -> None:
    rng = random.Random(3)
    samples = [rng.gauss(0.0, 5.0) for _ in range(500)]
    low, high = bootstrap_ci(samples, resamples=500, rng=random.Random(4))
    assert low < 0.0 < high


def test_bootstrap_of_a_constant_has_no_width() -> None:
    low, high = bootstrap_ci([7.0] * 50, resamples=100, rng=random.Random(5))
    assert low == high == 7.0


def test_wider_confidence_gives_a_wider_interval() -> None:
    samples = [random.Random(6).gauss(1.0, 3.0) for _ in range(200)]
    narrow = bootstrap_ci(samples, confidence=0.80, resamples=400, rng=random.Random(7))
    wide = bootstrap_ci(samples, confidence=0.99, resamples=400, rng=random.Random(7))
    assert wide[0] <= narrow[0] and wide[1] >= narrow[1]


def test_block_resampling_respects_paired_samples() -> None:
    # Pairs of (+100, -100) plus a small steady edge: every pair nets +2, so the true
    # per-hand edge is +1 with almost no uncertainty. Resampling hands individually
    # breaks the pairs apart and reports a hugely wide interval instead.
    paired: list[float] = []
    for _ in range(200):
        paired.extend([101.0, -99.0])

    unpaired_low, unpaired_high = bootstrap_ci(paired, resamples=500, rng=random.Random(8))
    blocked_low, blocked_high = bootstrap_ci(paired, block=2, resamples=500, rng=random.Random(8))

    assert unpaired_low < 0.0 < unpaired_high  # looks like noise
    assert blocked_low == blocked_high == 1.0  # every block is identical, so no spread
    assert (blocked_high - blocked_low) < (unpaired_high - unpaired_low)


def test_empty_samples_and_bad_block_sizes_are_rejected() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], block=0)


def test_matchup_reports_win_rate_and_mean_profit() -> None:
    def play(index: int) -> HandOutcome:
        return constant_outcome(profit=2.0 if index % 2 == 0 else -1.0, won=index % 2 == 0)

    result = evaluate_matchup(play, hands=100, resamples=200, rng=random.Random(9))
    assert result.hands == 100
    assert result.win_rate == 0.5
    assert result.mean_profit == pytest.approx(0.5)


def test_a_clear_winner_is_reported_as_significant() -> None:
    rng = random.Random(10)

    def play(index: int) -> HandOutcome:
        profit = rng.gauss(3.0, 1.0)
        return constant_outcome(profit, won=profit > 0)

    result = evaluate_matchup(play, hands=400, resamples=400, rng=random.Random(11))
    assert result.beats_opponent
    assert result.is_significant
    assert result.ci_low > 0.0


def test_a_coin_flip_matchup_is_not_significant() -> None:
    rng = random.Random(12)

    def play(index: int) -> HandOutcome:
        profit = rng.gauss(0.0, 10.0)
        return constant_outcome(profit, won=profit > 0)

    result = evaluate_matchup(play, hands=400, resamples=400, rng=random.Random(13))
    assert not result.beats_opponent
    assert not result.is_significant


def test_a_clear_loser_is_significant_but_does_not_beat_the_opponent() -> None:
    # Losing clearly is still a real result: the interval excludes zero, but on the
    # wrong side. Reporting those as the same thing would hide a losing bot.
    def play(index: int) -> HandOutcome:
        return constant_outcome(profit=-5.0, won=False)

    result = evaluate_matchup(play, hands=50, resamples=100, rng=random.Random(14))
    assert result.is_significant
    assert not result.beats_opponent


def test_summary_mentions_the_interval_and_hand_count() -> None:
    result = evaluate_matchup(
        lambda index: constant_outcome(1.0), hands=10, resamples=50, rng=random.Random(15)
    )
    summary = result.summary()
    assert "chips/hand" in summary
    assert "10" in summary


def test_at_least_one_hand_is_required() -> None:
    with pytest.raises(ValueError):
        evaluate_matchup(lambda index: constant_outcome(1.0), hands=0)
