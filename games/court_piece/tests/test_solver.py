import random
from itertools import product

from engine.cards import Card, Deck, Suit
from games.court_piece.rules import CourtPiece, CourtPieceState, TrumpCall
from games.court_piece.solver import INFINITY, solve
from games.court_piece.tests.test_rules import card, hand

# Endgames, not fresh deals - a full 13-trick solve is real work (measure with
# scripts/benchmark_solver.py before ever assuming it's fast), and none of these
# answers would be hand-checkable at that size anyway.
#
# The caller sits in seat 0 throughout, so seat 2 is their partner and the two of
# them are the caller's side; seat 1 leads whenever a position starts at a trick
# boundary. Seats in order below: N(0), E(1), S(2), W(3).
CALLER = 0
CALL = TrumpCall(trump=Suit.SPADES, caller=CALLER)
NOTRUMP_LIKE_CALL = TrumpCall(trump=Suit.CLUBS, caller=CALLER)  # trump is mandatory here


def endgame(
    call: TrumpCall,
    north: tuple[Card, ...],
    east: tuple[Card, ...],
    south: tuple[Card, ...],
    west: tuple[Card, ...],
    trick: tuple[tuple[int, Card], ...] = (),
) -> CourtPieceState:
    return CourtPieceState(call=call, hands=(north, east, south, west), trick=trick)


def test_a_single_trick_callers_side_wins_is_worth_one() -> None:
    game = CourtPiece()
    state = endgame(CALL, hand("AH"), hand("3H"), hand("4H"), hand("2H"))
    solution = solve(game, state)
    assert solution.tricks == 1.0
    assert solution.values == {card("3H"): 1.0}  # east's only legal card


def test_a_single_trick_the_opponents_win_is_worth_nothing() -> None:
    game = CourtPiece()
    state = endgame(CALL, hand("KH"), hand("AH"), hand("4H"), hand("2H"))
    assert solve(game, state).tricks == 0.0


def test_a_trick_the_callers_partner_wins_counts_for_the_callers_side() -> None:
    # No dummy here, unlike bridge - both cases are simply "which seat won," but
    # worth checking explicitly since it's the same kind of gap mutation testing
    # would find: crediting only the caller's own seat, not their partner's.
    game = CourtPiece()
    state = endgame(CALL, hand("2H"), hand("3H"), hand("AH"), hand("4H"))
    assert solve(game, state).tricks == 1.0

    defended = endgame(CALL, hand("2H"), hand("3H"), hand("4H"), hand("AH"))
    assert solve(game, defended).tricks == 0.0


def test_a_single_trick_a_ruff_wins_is_worth_one() -> None:
    game = CourtPiece()
    state = endgame(CALL, hand("2S"), hand("AH"), hand("4H"), hand("3H"))
    assert solve(game, state).tricks == 1.0


def test_caller_takes_both_tricks_holding_ace_queen_over_the_king() -> None:
    # Trump is clubs, but none of these cards are clubs, so this is effectively the
    # same notrump-like ace-queen-over-king tenace as bridge's own equivalent test.
    # Caller (north) holds A-Q of hearts, east holds the king, and the other four
    # cards are all lower than the king - nothing but those three can win a trick.
    #
    # Worked out by hand, both of east's leads: leading the king lets north win with
    # the ace and cash the queen; leading the three, north plays the queen straight
    # away (nothing above it is out except north's own ace) and keeps the ace for the
    # second trick. Two tricks either way - the same double-dummy-sees-everything
    # reasoning bridge's equivalent test documents, not a card-reading skill the
    # solver actually has.
    game = CourtPiece()
    state = endgame(
        NOTRUMP_LIKE_CALL, hand("AH", "QH"), hand("KH", "3H"), hand("5H", "4H"), hand("7H", "6H")
    )
    solution = solve(game, state)

    assert solution.tricks == 2.0
    assert solution.values == {card("KH"): 2.0, card("3H"): 2.0}


def test_playing_the_wrong_card_of_a_tenace_throws_a_trick_away() -> None:
    # Same holding, picked up mid-trick with the caller to play: east led the three,
    # caller's partner (south) followed with the four, west with the six.
    game = CourtPiece()
    state = endgame(
        NOTRUMP_LIKE_CALL,
        hand("AH", "QH"),
        hand("KH"),
        hand("5H"),
        hand("7H"),
        trick=((1, card("3H")), (2, card("4H")), (3, card("6H"))),
    )
    solution = solve(game, state)

    assert state.to_play == CALLER  # picked up part-way through a trick, not at a boundary
    assert solution.values == {card("QH"): 2.0, card("AH"): 1.0}
    assert solution.best == card("QH")
    assert solution.tricks == 2.0


def test_the_callers_partner_also_maximizes_not_just_the_caller() -> None:
    # The same ace-queen-over-king tenace as the two tests above, but rooted with the
    # tenace held by the caller's *partner* (seat 2) instead of the caller (seat 0).
    # A bug that only recognizes the caller's own seat as "on the caller's side" (and
    # treats the partner's own decisions as minimizing, as if they were an opponent)
    # would misplay this: east leads the king, and if seat 2's own turn is wrongly
    # minimized instead of maximized, the search would duck under it with the ace
    # rather than winning with the queen and cashing the ace after - throwing a trick
    # away. Found by deliberately introducing exactly this bug and confirming this
    # test (and no other existing one) catches it - see DECISIONS.md.
    game = CourtPiece()
    state = endgame(
        NOTRUMP_LIKE_CALL, hand("5H", "4H"), hand("KH", "3H"), hand("AH", "QH"), hand("7H", "6H")
    )
    solution = solve(game, state)
    assert solution.tricks == 2.0


def test_transposition_table_does_not_conflate_identical_hands_under_different_trump() -> None:
    # The real correctness risk running trump introduces, found and fixed while
    # building it (see DECISIONS.md): two genuinely different positions - same seat
    # to move, same remaining cards in every hand - can still need different
    # answers if trump differs between them, since running trump lets two different
    # move orders reach identical remaining hands while having set trump
    # differently along the way. The old key (seat, hands), with no trump component
    # at all, would have conflated these; both `solve()` calls below share one
    # `_Search`/table instance specifically to exercise that cache, not just check
    # each answer in isolation with a fresh table each time.
    #
    # South holds a lone spade against three hearts elsewhere. Worked out by hand:
    # if spades is trump, south's spade ruffs and wins outright regardless of the
    # hearts (1.0 for caller's side, since south is caller's partner). If hearts is
    # trump instead, south's spade is just an ordinary off-suit card that can't win
    # at all, and west's 4H - the highest heart - takes the trick instead (0.0 for
    # caller's side, since west is an opponent).
    from games.court_piece.solver import _Search

    game = CourtPiece()
    hands = (hand("2H"), hand("3H"), hand("9S"), hand("4H"))
    spade_trump = CourtPieceState(call=TrumpCall(trump=Suit.SPADES, caller=0), hands=hands)
    heart_trump = CourtPieceState(call=TrumpCall(trump=Suit.HEARTS, caller=0), hands=hands)
    assert spade_trump.to_play == 1  # same to_play in both - same key but for trump
    assert heart_trump.to_play == 1

    search = _Search(
        game, spade_trump, prune=True, transpositions=True, equivalence=True, narrow=True
    )
    assert search.future_tricks(spade_trump, -INFINITY, INFINITY) == 1.0
    assert search.future_tricks(heart_trump, -INFINITY, INFINITY) == 0.0  # not the cached 1.0

    # And the reverse order, in case only one direction happened to work by luck.
    reordered = _Search(
        game, heart_trump, prune=True, transpositions=True, equivalence=True, narrow=True
    )
    assert reordered.future_tricks(heart_trump, -INFINITY, INFINITY) == 0.0
    assert reordered.future_tricks(spade_trump, -INFINITY, INFINITY) == 1.0  # not the cached 0.0


def test_alpha_beta_matches_plain_minimax() -> None:
    # Every optimization is supposed to change nothing about the answer, only which
    # provably-irrelevant branches get skipped - checked on a small endgame across
    # every combination of options, the same discipline as bridge's own solver tests.
    game = CourtPiece()
    rng = random.Random(7)
    deck = Deck().cards
    rng.shuffle(deck)
    cards = deck[:16]  # 4 cards per hand - small enough to brute-force safely
    hands = tuple(tuple(cards[i * 4 : (i + 1) * 4]) for i in range(4))
    call = TrumpCall(trump=Suit.HEARTS, caller=0)
    state = CourtPieceState(call=call, hands=hands)

    unpruned = solve(
        game, state, prune=False, transpositions=False, equivalence=False, narrow=False
    )
    baseline = unpruned.tricks
    for prune, transpositions, equivalence, narrow in product([True, False], repeat=4):
        if not prune and narrow:
            continue  # narrow does nothing without pruning
        result = solve(
            game,
            state,
            prune=prune,
            transpositions=transpositions,
            equivalence=equivalence,
            narrow=narrow,
        )
        assert result.tricks == baseline
