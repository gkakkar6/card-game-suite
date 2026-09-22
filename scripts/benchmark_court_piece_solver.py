"""How long the Court Piece double-dummy solver actually takes, and what each
optimization buys - the same measurement games/bridge's own
scripts/benchmark_solver.py does, rerun here because Court Piece's solver is a
separate, adapted implementation (games/court_piece/solver.py), not a shared one -
bridge's numbers don't transfer.

The real reason this needs its own run, not just bridge's numbers reused: this
benchmark measures raw solver speed given a full, already-known deal - which is
identical work to bridge's (both solve an arbitrary 4-hands-known trick-taking
position). What genuinely differs between the two games is PIMC's *sampling* cost
(3 unseen hands here vs bridge's 2, games/court_piece/pimc.py), which this script
does not measure - see scripts/benchmark_court_piece_pimc.py for that, once written.
This script exists to check that assumption directly rather than leave it assumed:
if per-solve timings come out close to bridge's own numbers at the same trick count,
that confirms the solver itself carries over cleanly and the entire depth-gate gap
between bridge's DEPTH_GATE=6 and Court Piece's DEPTH_GATE=4
(games/court_piece/action_values.py) is really about PIMC's sampling cost, not this.

    uv run python scripts/benchmark_court_piece_solver.py
"""

import random
import time

from engine.cards import Deck, Suit
from games.court_piece.rules import CourtPiece, CourtPieceState, TrumpCall
from games.court_piece.solver import Solution, solve

SEED = 2026
MAX_TRICKS = 9  # past this a single deal takes minutes; measure those one at a time

# Each column adds one switch to the one before it, so the difference between two
# neighbouring columns is what that switch is actually worth - identical structure to
# bridge's own benchmark table.
SETTINGS: dict[str, dict[str, bool]] = {
    "minimax": {"prune": False, "transpositions": False, "equivalence": False, "narrow": False},
    "alpha-beta": {"prune": True, "transpositions": False, "equivalence": False, "narrow": False},
    "+ table": {"prune": True, "transpositions": True, "equivalence": False, "narrow": False},
    "+ equivalence": {"prune": True, "transpositions": True, "equivalence": True, "narrow": False},
    "+ narrow": {"prune": True, "transpositions": True, "equivalence": True, "narrow": True},
}

MAX_TRICKS_FOR_MINIMAX = 5
MAX_TRICKS_WITHOUT_TABLE = 5


def deals_for(tricks: int) -> int:
    """Average over several deals while that is cheap - one deal is very noisy, since
    how hard a deal is to solve varies far more than its size alone suggests."""
    return 5 if tricks <= 7 else 2


def endgame(rng: random.Random, tricks: int) -> CourtPieceState:
    """A position with `tricks` cards in each hand, off a shuffled deck."""
    deck = Deck()
    deck.shuffle(rng)
    hands = tuple(tuple(deck.deal(tricks)) for _ in range(4))
    call = TrumpCall(trump=Suit.SPADES, caller=0)
    return CourtPieceState(call=call, hands=hands)


def time_one(
    game: CourtPiece, state: CourtPieceState, settings: dict[str, bool]
) -> tuple[float, int]:
    start = time.perf_counter()
    solution: Solution = solve(game, state, **settings)
    return time.perf_counter() - start, solution.nodes


def main() -> None:
    game = CourtPiece()
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
        print(f"{tricks:>6} {deals:>5}  " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    main()
