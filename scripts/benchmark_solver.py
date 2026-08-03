"""How long the double-dummy solver actually takes, and what each optimization buys.

Run it rather than trusting an estimate - these numbers are what ARCHITECTURE.md §5's
Phase 2b work was aimed at and what README's honest-scope note quotes. Rerun after any
change to the search, the same way scripts/check_persona_keyword_fragility.py is rerun
after any change to the keyword table.

    uv run python scripts/benchmark_solver.py
"""

import random
import time

from engine.cards import Deck, Suit
from games.bridge.rules import Bridge, BridgeState, Contract
from games.bridge.solver import Solution, solve

SEED = 2026
MAX_TRICKS = 9  # past this a single deal takes minutes; measure those one at a time

# Each column adds one switch to the one before it, so the difference between two
# neighbouring columns is what that switch is actually worth.
SETTINGS: dict[str, dict[str, bool]] = {
    "minimax": {"prune": False, "transpositions": False, "equivalence": False, "narrow": False},
    "alpha-beta": {"prune": True, "transpositions": False, "equivalence": False, "narrow": False},
    "+ table": {"prune": True, "transpositions": True, "equivalence": False, "narrow": False},
    "+ equivalence": {"prune": True, "transpositions": True, "equivalence": True, "narrow": False},
    "+ narrow": {"prune": True, "transpositions": True, "equivalence": True, "narrow": True},
}

# The unoptimized columns blow up quickly - past these sizes they are not worth waiting
# for, which is itself part of what the table is meant to show. Checked directly rather
# than guessed: plain minimax at 5 tricks ranges roughly 1-30s across random seeds, still
# worth measuring properly, but 6 tricks is genuinely minutes even for a single deal.
MAX_TRICKS_FOR_MINIMAX = 5
MAX_TRICKS_WITHOUT_TABLE = 5


def deals_for(tricks: int) -> int:
    """Average over several deals while that is cheap - one deal is very noisy, since
    how hard a deal is to solve varies far more than its size alone suggests."""
    return 5 if tricks <= 7 else 2


def endgame(rng: random.Random, tricks: int) -> BridgeState:
    """A position with `tricks` cards in each hand, off a shuffled deck."""
    deck = Deck()
    deck.shuffle(rng)
    hands = tuple(tuple(deck.deal(tricks)) for _ in range(4))
    return BridgeState(
        contract=Contract(trump=Suit.SPADES, declarer=0, target=7), hands=hands
    )


def time_one(game: Bridge, state: BridgeState, settings: dict[str, bool]) -> tuple[float, int]:
    start = time.perf_counter()
    solution: Solution = solve(game, state, **settings)
    return time.perf_counter() - start, solution.nodes


def main() -> None:
    game = Bridge()
    print("mean seconds and mean nodes visited, over random deals of each size\n")
    header = f"{'tricks':>6} {'deals':>5}  " + "  ".join(f"{name:>20}" for name in SETTINGS)
    print(header)
    print("-" * len(header))

    for tricks in range(1, MAX_TRICKS + 1):
        deals = deals_for(tricks)
        states = [endgame(random.Random(SEED + index), tricks) for index in range(deals)]
        cells: list[str] = []
        for name, settings in SETTINGS.items():
            too_big_for_minimax = name == "minimax" and tricks > MAX_TRICKS_FOR_MINIMAX
            no_table = not settings["transpositions"] and tricks > MAX_TRICKS_WITHOUT_TABLE
            if too_big_for_minimax or no_table:
                cells.append(f"{'-':>20}")
                continue
            results = [time_one(game, state, settings) for state in states]
            seconds = sum(result[0] for result in results) / len(results)
            nodes = sum(result[1] for result in results) / len(results)
            cells.append(f"{seconds:>8.3f}s {nodes:>10,.0f}")
        # flushed per row so a long run can be watched, and so stopping it early still
        # leaves the sizes it did get through
        print(f"{tricks:>6} {deals:>5}  " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    main()
