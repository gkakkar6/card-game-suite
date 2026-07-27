"""Best 5-card poker hand from 5-7 cards, and a comparable rank for it."""

from collections import Counter
from enum import IntEnum
from itertools import combinations

from engine.cards import Card

HandRank = tuple[int, ...]  # (HandCategory, *tiebreakers) - compares correctly via tuple ordering


class HandCategory(IntEnum):
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9


def _straight_high(distinct_ranks_desc: list[int]) -> int | None:
    """Highest card of a straight among 5 distinct ranks, treating A-2-3-4-5 as a wheel (high=5)."""
    if distinct_ranks_desc == [14, 5, 4, 3, 2]:
        return 5
    # 5 distinct ranks spanning exactly 4 apart (e.g. 9,8,7,6,5) must be consecutive
    if distinct_ranks_desc[0] - distinct_ranks_desc[4] == 4:
        return distinct_ranks_desc[0]
    return None


def evaluate_five(cards: list[Card]) -> HandRank:
    """Rank a single 5-card hand. Higher tuples beat lower ones."""
    if len(cards) != 5:
        raise ValueError("evaluate_five requires exactly 5 cards")

    ranks: list[int] = sorted((c.rank.value for c in cards), reverse=True)
    is_flush: bool = len({c.suit for c in cards}) == 1
    distinct_ranks: list[int] = sorted(set(ranks), reverse=True)
    straight_high = _straight_high(distinct_ranks) if len(distinct_ranks) == 5 else None

    if straight_high and is_flush:
        return (HandCategory.STRAIGHT_FLUSH, straight_high)

    # how many of each rank we hold, e.g. {14: 2, 9: 1, 7: 1, 2: 1} for a pair of aces
    counts: Counter[int] = Counter(ranks)
    # sorted (rank, count) groups, biggest group first and highest rank breaking ties within
    # a group size - this order is what makes the returned tuple compare correctly card-by-card
    by_count: list[tuple[int, int]] = sorted(
        counts.items(), key=lambda item: (item[1], item[0]), reverse=True
    )
    group_ranks: list[int] = [rank for rank, _count in by_count]
    group_sizes: list[int] = [count for _rank, count in by_count]

    if group_sizes[0] == 4:
        return (HandCategory.FOUR_OF_A_KIND, *group_ranks)
    if group_sizes[0] == 3 and group_sizes[1] == 2:
        return (HandCategory.FULL_HOUSE, *group_ranks)
    if is_flush:
        return (HandCategory.FLUSH, *ranks)
    if straight_high:
        return (HandCategory.STRAIGHT, straight_high)
    if group_sizes[0] == 3:
        return (HandCategory.THREE_OF_A_KIND, *group_ranks)
    if group_sizes[0] == 2 and group_sizes[1] == 2:
        return (HandCategory.TWO_PAIR, *group_ranks)
    if group_sizes[0] == 2:
        return (HandCategory.PAIR, *group_ranks)
    return (HandCategory.HIGH_CARD, *ranks)


def best_hand(cards: list[Card]) -> HandRank:
    """Best 5-card HandRank achievable from 5-7 cards (e.g. 2 hole + 5 board)."""
    if len(cards) < 5:
        raise ValueError("need at least 5 cards to form a hand")
    return max(evaluate_five(list(combo)) for combo in combinations(cards, 5))
