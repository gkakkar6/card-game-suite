"""Run the validating experiment from ARCHITECTURE.md §4: does the baseline persona
beat a deviation in both directions by a statistically significant margin?

Kept as a script rather than a test because a trustworthy interval on the narrower
matchup needs about a thousand hands, which takes minutes - far too slow for the test
suite. games/poker/tests/test_validation.py covers the same claim at a hand count that
fits in a normal run.

    uv run python scripts/validate_personas.py
"""

import random
import time

from games.poker.personas import BASELINE, DEVIATIONS, play_matchup

HANDS = 1_000
SEED = 1


def main() -> None:
    print(f"baseline against each deviation, {HANDS:,} duplicate-dealt hands each\n")
    failures = []
    for opponent in DEVIATIONS:
        started = time.perf_counter()
        result = play_matchup(
            BASELINE, opponent, hands=HANDS, rng=random.Random(SEED), seed=SEED
        )
        verdict = "beats it" if result.beats_opponent else "DOES NOT beat it"
        # Each matchup takes minutes, so flush as they land rather than leaving the
        # whole run looking hung behind a buffer when output is redirected to a file.
        print(f"vs {opponent.name:<16} {result.summary()}", flush=True)
        print(f"{'':<19} {verdict}  ({time.perf_counter() - started:.0f}s)\n", flush=True)
        if not result.beats_opponent:
            failures.append(opponent.name)

    if failures:
        print(f"NOT validated against: {', '.join(failures)}")
    else:
        print("baseline beats all four deviations by a significant margin")


if __name__ == "__main__":
    main()
