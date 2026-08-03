import random

import pytest

from engine.cards import Card, Deck, Suit
from games.bridge.rules import Bridge, BridgeState, Contract
from games.bridge.solver import solve
from games.bridge.tests.test_rules import card

# Endgames, not fresh deals: a full 13-trick solve is Phase 2b's problem, and none of
# the answers below would be checkable by hand at that size anyway. Cards missing from
# these hands are simply ones that have already been played and gone.
#
# Declarer sits in seat 0 throughout, so seat 2 is the dummy, the two of them are
# declarer's side, and seat 1 leads whenever a position starts at a trick boundary.
DECLARER = 0
SPADES_CONTRACT = Contract(trump=Suit.SPADES, declarer=DECLARER, target=7)
NOTRUMP_CONTRACT = Contract(trump=None, declarer=DECLARER, target=7)


def hand(*texts: str) -> tuple[Card, ...]:
    return tuple(card(text) for text in texts)


def endgame(
    contract: Contract,
    north: tuple[Card, ...],
    east: tuple[Card, ...],
    south: tuple[Card, ...],
    west: tuple[Card, ...],
    trick: tuple[tuple[int, Card], ...] = (),
) -> BridgeState:
    """A position with only the given cards left. Seats in order: N, E, S, W."""
    return BridgeState(contract=contract, hands=(north, east, south, west), trick=trick)


# ---------------------------------------------------------------------------
# One trick left: the answer is whatever trick_winner() already says
# ---------------------------------------------------------------------------


def test_a_single_trick_declarers_side_wins_is_worth_one() -> None:
    # Nobody has a choice with one card each, so this is purely a check that the search
    # wraps Phase 1's machinery correctly: east leads the three, and north's ace is the
    # highest heart, so declarer's side takes it.
    game = Bridge()
    state = endgame(
        SPADES_CONTRACT, hand("AH"), hand("3H"), hand("4H"), hand("2H")
    )
    solution = solve(game, state)

    assert solution.tricks == 1.0
    assert solution.values == {card("3H"): 1.0}  # east's only legal card


def test_a_single_trick_the_defence_wins_is_worth_nothing() -> None:
    # Same shape, but now east's ace is the high heart and north can only follow suit.
    game = Bridge()
    state = endgame(
        SPADES_CONTRACT, hand("KH"), hand("AH"), hand("4H"), hand("2H")
    )
    assert solve(game, state).tricks == 0.0


def test_a_trick_the_dummy_wins_counts_for_declarers_side() -> None:
    # Declarer's side is two seats, not one. Here the ace sits in the dummy rather than
    # in declarer's own hand, and the trick still counts.
    #
    # This test exists because of a gap mutation testing found: every other endgame in
    # this file has declarer personally winning declarer's tricks, so crediting only
    # declarer's own seat and ignoring the dummy's passed the whole suite.
    game = Bridge()
    state = endgame(SPADES_CONTRACT, hand("2H"), hand("3H"), hand("AH"), hand("4H"))

    assert solve(game, state).tricks == 1.0

    # and the same trick counted from the other side: an opponent's partner winning is
    # not declarer's side, however the seats are numbered
    defended = endgame(SPADES_CONTRACT, hand("2H"), hand("3H"), hand("4H"), hand("AH"))
    assert solve(game, defended).tricks == 0.0


def test_a_single_trick_a_ruff_wins_is_worth_one() -> None:
    # North is void in hearts and holds the two of trumps, which beats east's ace -
    # the same rule resolution.py is tested on, reached through the search this time.
    game = Bridge()
    state = endgame(
        SPADES_CONTRACT, hand("2S"), hand("AH"), hand("4H"), hand("3H")
    )
    assert solve(game, state).tricks == 1.0


# ---------------------------------------------------------------------------
# Two tricks: declarer has a real choice to get right
# ---------------------------------------------------------------------------


def test_declarer_takes_both_tricks_holding_ace_queen_over_the_king() -> None:
    # Notrump. North holds A-Q, east holds the king, and the other four cards are all
    # lower than the king, so nothing but those three can win a trick.
    #
    # Worked out by hand, both of east's leads:
    #   east leads the king - north wins with the ace, then leads the queen, which is
    #     now the highest card left. Two tricks.
    #   east leads the three - north plays the *queen*, which is high enough to win
    #     (nothing above the three is out except north's own ace), and keeps the ace to
    #     take the second trick. Two tricks again.
    # So it is two either way, and the defence has nothing to choose between.
    #
    # Worth being clear about what this really shows: declarer plays the queen because
    # the search can see the king sitting in east's hand. That is the double-dummy
    # assumption doing the work, not a card-reading skill the solver has.
    game = Bridge()
    state = endgame(
        NOTRUMP_CONTRACT, hand("AH", "QH"), hand("KH", "3H"), hand("5H", "4H"), hand("7H", "6H")
    )
    solution = solve(game, state)

    assert solution.tricks == 2.0
    assert solution.values == {card("KH"): 2.0, card("3H"): 2.0}


def test_playing_the_wrong_card_of_a_tenace_throws_a_trick_away() -> None:
    # The same holding, picked up mid-trick with north to play: east led the three,
    # dummy followed with the four and west with the six.
    #
    #   north plays the queen - it wins (only the ace is higher and north holds it),
    #     and the ace takes the last trick. Two.
    #   north plays the ace - it wins, but then north has to lead the queen into east's
    #     king, which takes the last trick. One.
    #
    # This is the per-action value the search is meant to expose: both cards are legal,
    # both win the trick in front of them, and one of them costs a trick later.
    game = Bridge()
    state = endgame(
        NOTRUMP_CONTRACT,
        hand("AH", "QH"),
        hand("KH"),
        hand("5H"),
        hand("7H"),
        trick=((1, card("3H")), (2, card("4H")), (3, card("6H"))),
    )
    solution = solve(game, state)

    assert state.to_play == DECLARER  # picked up part-way through a trick, not at a boundary
    assert solution.values == {card("QH"): 2.0, card("AH"): 1.0}
    assert solution.best == card("QH")
    assert solution.tricks == 2.0


# ---------------------------------------------------------------------------
# Three tricks: now the defence has a real choice to get right
# ---------------------------------------------------------------------------


# Notrump, three cards each, east on lead. North holds A-Q of hearts over east's king,
# and the defence can see it, so the whole question is whether east has to lead hearts.
#
# East leads the king: north takes it with the ace, cashes the queen (both other hearts
#   are gone), and the last trick goes to west's eight of diamonds. Two for declarer.
# East leads a diamond: west's eight wins it. West then exits with the nine of clubs -
#   north is void and has to throw one of the two hearts, keeping the ace. West leads
#   the last heart, north's ace wins. One for declarer.
#   (If west led a heart instead, north's ace would win and the queen would take the
#   last trick as well - so west's club exit is the whole point of the defence.)
#
# East is minimizing, so the position is worth one trick, and leading the king costs a
# whole trick over either diamond.
TENACE_ENDGAME = (
    hand("AH", "QH", "2D"),
    hand("KH", "3D", "4D"),
    hand("5D", "6D", "7C"),
    hand("3H", "8D", "9C"),
)


def test_the_defence_must_avoid_leading_into_declarers_tenace() -> None:
    game = Bridge()
    state = endgame(NOTRUMP_CONTRACT, *TENACE_ENDGAME)
    solution = solve(game, state)

    assert solution.tricks == 1.0
    assert solution.values == {card("KH"): 2.0, card("3D"): 1.0, card("4D"): 1.0}
    assert solution.best in (card("3D"), card("4D"))  # east minimizes, so not the king


def test_the_two_low_diamonds_are_reported_as_worth_the_same() -> None:
    # East's three and four of diamonds are interchangeable here - the two of diamonds
    # is in north's hand, but nothing sits *between* three and four, so no trick can
    # tell them apart. The search only tries one of them; both still get a value.
    game = Bridge()
    solution = solve(game, endgame(NOTRUMP_CONTRACT, *TENACE_ENDGAME))
    assert solution.values[card("3D")] == solution.values[card("4D")]


# ---------------------------------------------------------------------------
# Tricks already won, and playing on from a solved position
# ---------------------------------------------------------------------------


def test_the_value_counts_tricks_already_in_the_bag() -> None:
    # Play the first trick of the tenace endgame out, then solve what is left. East has
    # already given up a trick by leading the king, so the position is now worth two -
    # one won, one still to come - even though only two tricks remain to be played.
    game = Bridge()
    state = endgame(NOTRUMP_CONTRACT, *TENACE_ENDGAME)
    for text in ("KH", "5D", "3H", "AH"):
        state = game.apply(state, card(text))

    assert len(state.completed) == 1
    assert game.payoff(state)[DECLARER] == 1.0  # the trick north just took
    assert solve(game, state).tricks == 2.0


def test_the_value_holds_steady_while_both_sides_play_the_solvers_own_cards() -> None:
    # If the number really is what both sides can force, then following the solver's own
    # choice at every turn must not move it - a wrong maximize/minimize test or a
    # mis-scaled window would show up as the value drifting as the deal is played out.
    game = Bridge()
    state = endgame(NOTRUMP_CONTRACT, *TENACE_ENDGAME)
    expected = solve(game, state).tricks

    while any(state.hands):
        state = game.apply(state, solve(game, state).best)
        if any(state.hands):
            assert solve(game, state).tricks == expected
    assert game.payoff(state)[DECLARER] == expected


# ---------------------------------------------------------------------------
# The optimizations must not change any answer
# ---------------------------------------------------------------------------

# Plain minimax: every optimization off. The reference every other setting is checked
# against, since it is the one version with nothing clever in it to get wrong.
PLAIN = {"prune": False, "transpositions": False, "equivalence": False, "narrow": False}

# Every combination of the four switches, so no single one and no pair of them can
# quietly change a result. All-on is the default the rest of the tests use.
SETTINGS = [
    {"prune": prune, "transpositions": table, "equivalence": groups, "narrow": narrow}
    for prune in (False, True)
    for table in (False, True)
    for groups in (False, True)
    for narrow in (False, True)
]


def random_endgame(rng: random.Random, tricks: int) -> BridgeState:
    """A position with `tricks` cards in each hand, dealt off a shuffled deck."""
    deck = Deck()
    deck.shuffle(rng)
    hands = tuple(tuple(deck.deal(tricks)) for _ in range(4))
    trump = rng.choice([None, *list(Suit)])
    contract = Contract(trump=trump, declarer=rng.randrange(4), target=7)
    return BridgeState(contract=contract, hands=hands)


@pytest.mark.parametrize("settings", SETTINGS)
def test_every_combination_of_optimizations_agrees_on_the_worked_endgames(
    settings: dict[str, bool],
) -> None:
    game = Bridge()
    positions = [
        endgame(SPADES_CONTRACT, hand("AH"), hand("3H"), hand("4H"), hand("2H")),
        endgame(
            NOTRUMP_CONTRACT, hand("AH", "QH"), hand("KH", "3H"), hand("5H", "4H"), hand("7H", "6H")
        ),
        endgame(NOTRUMP_CONTRACT, *TENACE_ENDGAME),
        endgame(
            NOTRUMP_CONTRACT,
            hand("AH", "QH"),
            hand("KH"),
            hand("5H"),
            hand("7H"),
            trick=((1, card("3H")), (2, card("4H")), (3, card("6H"))),
        ),
    ]
    for state in positions:
        plain = solve(game, state, **PLAIN)
        tuned = solve(game, state, **settings)
        assert tuned.tricks == plain.tricks
        assert tuned.values == plain.values


@pytest.mark.parametrize("tricks", [2, 3])
def test_alpha_beta_matches_plain_minimax_on_random_endgames(tricks: int) -> None:
    # The check that pruning only skips branches it can prove are irrelevant. Random
    # positions rather than chosen ones, because a hand-picked position is exactly the
    # kind of thing that might happen to miss a case where a bound is wrong.
    #
    # Three tricks is the ceiling here, and that limit is the point rather than a
    # shortcut: unpruned minimax on four cards each is already minutes, which is why
    # ARCHITECTURE.md §5 calls alpha-beta a precondition rather than a speed extra.
    game = Bridge()
    rng = random.Random(11)
    for _ in range(12):
        state = random_endgame(rng, tricks)
        plain = solve(game, state, **PLAIN)
        pruned = solve(
            game, state, prune=True, transpositions=False, equivalence=False, narrow=False
        )

        assert pruned.tricks == plain.tricks
        assert pruned.values == plain.values
        assert pruned.nodes <= plain.nodes  # pruning alone can only remove work


@pytest.mark.parametrize("tricks", [2, 3])
def test_all_the_optimizations_together_match_plain_minimax_on_random_endgames(
    tricks: int,
) -> None:
    game = Bridge()
    rng = random.Random(23)
    for _ in range(12):
        state = random_endgame(rng, tricks)
        plain = solve(game, state, **PLAIN)
        tuned = solve(game, state)

        assert tuned.tricks == plain.tricks
        assert tuned.values == plain.values


@pytest.mark.parametrize("tricks", [4, 5])
def test_all_the_optimizations_together_match_bare_alpha_beta_on_bigger_endgames(
    tricks: int,
) -> None:
    # Past three tricks plain minimax is out of reach, so bare alpha-beta becomes the
    # reference instead - itself tied to plain minimax by the test above. This is what
    # covers the table, the grouping and the narrow windows at sizes where positions
    # actually repeat often enough for any of them to do something.
    game = Bridge()
    rng = random.Random(29)
    for _ in range(6):
        state = random_endgame(rng, tricks)
        reference = solve(
            game, state, prune=True, transpositions=False, equivalence=False, narrow=False
        )
        tuned = solve(game, state)

        assert tuned.tricks == reference.tricks
        assert tuned.values == reference.values
        assert tuned.nodes <= reference.nodes


# ---------------------------------------------------------------------------
# The transposition key
# ---------------------------------------------------------------------------


def test_positions_are_hashable_and_compare_by_value() -> None:
    # The transposition table keys on parts of a BridgeState directly, so this has to
    # hold: two positions built independently from the same cards must be the same key.
    left = endgame(NOTRUMP_CONTRACT, *TENACE_ENDGAME)
    right = endgame(NOTRUMP_CONTRACT, *TENACE_ENDGAME)

    assert (left.to_play, left.trick, left.hands) == (right.to_play, right.trick, right.hands)
    assert hash((left.to_play, left.trick, left.hands)) == hash(
        (right.to_play, right.trick, right.hands)
    )
    assert len({left, right}) == 1  # the whole state is hashable too


def test_the_table_is_reached_through_different_move_orders() -> None:
    # The table only earns its place if positions actually repeat. Solving with it on
    # has to visit strictly fewer positions than solving with it off, on a position big
    # enough for move orders to converge.
    game = Bridge()
    state = random_endgame(random.Random(5), 4)
    without = solve(game, state, transpositions=False, equivalence=False, narrow=False)
    with_table = solve(game, state, transpositions=True, equivalence=False, narrow=False)

    assert with_table.tricks == without.tricks
    assert with_table.nodes < without.nodes


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def test_solving_a_position_with_no_cards_left_raises() -> None:
    game = Bridge()
    with pytest.raises(ValueError):
        solve(game, endgame(SPADES_CONTRACT, (), (), (), ()))


def test_every_legal_card_gets_a_value() -> None:
    game = Bridge()
    state = endgame(NOTRUMP_CONTRACT, *TENACE_ENDGAME)
    solution = solve(game, state)
    assert set(solution.values) == set(game.legal_actions(state))


def test_a_value_never_exceeds_the_tricks_that_exist() -> None:
    game = Bridge()
    rng = random.Random(31)
    for _ in range(10):
        state = random_endgame(rng, 3)
        solution = solve(game, state)
        assert 0.0 <= solution.tricks <= 3.0
