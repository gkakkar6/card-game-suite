import random

import pytest

from games.poker.betting import ActionType, BettingAction
from games.poker.hand import Blinds, DecisionView
from games.poker.personas import BASELINE, BLUFFER, CONSERVATIVE
from games.poker.session import (
    BlindSchedule,
    SeatConfig,
    Session,
    SessionConfig,
    _next_active_seat,
    deal_order,
)


def check_or_call(view: DecisionView) -> BettingAction:
    if ActionType.CHECK in view.legal_actions:
        return BettingAction(ActionType.CHECK)
    return BettingAction(ActionType.CALL)


def human_bot_seats(num_bots: int = 2) -> tuple[SeatConfig, ...]:
    personas = [BASELINE, BLUFFER, CONSERVATIVE]
    seats = [SeatConfig(name="Human", is_human=True)]
    for i in range(num_bots):
        persona = personas[i % len(personas)]
        seats.append(SeatConfig(name=f"Bot{i}", is_human=False, persona=persona))
    return tuple(seats)


# ---------------------------------------------------------------------------
# SeatConfig / SessionConfig validation
# ---------------------------------------------------------------------------


def test_seat_config_rejects_a_persona_on_the_human_seat() -> None:
    with pytest.raises(ValueError):
        SeatConfig(name="Human", is_human=True, persona=BASELINE)


def test_seat_config_requires_a_persona_for_a_bot() -> None:
    with pytest.raises(ValueError):
        SeatConfig(name="Bot", is_human=False)


def test_session_config_requires_exactly_one_human() -> None:
    with pytest.raises(ValueError):
        SessionConfig(seats=(SeatConfig(name="Bot", is_human=False, persona=BASELINE),) * 3)
    with pytest.raises(ValueError):
        SessionConfig(seats=(SeatConfig(name="Human", is_human=True),) * 2)


@pytest.mark.parametrize("num_bots", [0, 6])
def test_session_config_rejects_bot_counts_outside_one_to_five(num_bots: int) -> None:
    with pytest.raises(ValueError):
        SessionConfig(seats=human_bot_seats(num_bots))


@pytest.mark.parametrize("num_bots", [1, 2, 3, 4, 5])
def test_session_config_accepts_one_to_five_bots(num_bots: int) -> None:
    SessionConfig(seats=human_bot_seats(num_bots))  # does not raise


# ---------------------------------------------------------------------------
# deal_order / _next_active_seat: the pure seat-rotation math
# ---------------------------------------------------------------------------


def test_deal_order_matches_hand_py_convention_full_table() -> None:
    # hand.py: seat 0 is always the small blind, and the button sits at the last
    # seat - except heads-up, where the button and small blind are the same seat.
    assert deal_order(2, frozenset({0, 1}), button=0) == [0, 1]
    assert deal_order(2, frozenset({0, 1}), button=1) == [1, 0]
    assert deal_order(3, frozenset({0, 1, 2}), button=0) == [1, 2, 0]
    assert deal_order(4, frozenset({0, 1, 2, 3}), button=2) == [3, 0, 1, 2]
    assert deal_order(5, frozenset({0, 1, 2, 3, 4}), button=4) == [0, 1, 2, 3, 4]
    assert deal_order(6, frozenset(range(6)), button=1) == [2, 3, 4, 5, 0, 1]


def test_deal_order_skips_inactive_seats() -> None:
    # A 5-seat table with seats 1 and 3 eliminated: only 0, 2, 4 remain, and the
    # rotation must skip straight over the gaps rather than treating them as seats.
    active = frozenset({0, 2, 4})
    assert deal_order(5, active, button=2) == [4, 0, 2]


def test_deal_order_requires_the_button_to_be_active() -> None:
    with pytest.raises(ValueError):
        deal_order(3, frozenset({0, 2}), button=1)


def test_next_active_seat_wraps_around_the_table() -> None:
    active = frozenset({0, 1, 2})
    assert _next_active_seat(3, active, after=0) == 1
    assert _next_active_seat(3, active, after=2) == 0  # wraps


def test_next_active_seat_skips_gaps_and_works_from_an_inactive_seat() -> None:
    # The seat the button just left can itself have been eliminated this hand - the
    # search still has to work starting from a seat that is no longer active.
    active = frozenset({0, 2, 4})
    assert _next_active_seat(5, active, after=1) == 2
    assert _next_active_seat(5, active, after=4) == 0  # wraps, skipping 1 and 3


def test_next_active_seat_raises_with_no_active_seats() -> None:
    with pytest.raises(ValueError):
        _next_active_seat(3, frozenset(), after=0)


# ---------------------------------------------------------------------------
# BlindSchedule
# ---------------------------------------------------------------------------


def test_fixed_schedule_never_changes() -> None:
    schedule = BlindSchedule(mode="fixed", base=Blinds(small=1, big=2))
    assert schedule.for_hand(0) == Blinds(1, 2)
    assert schedule.for_hand(999) == Blinds(1, 2)


def test_increasing_schedule_doubles_on_the_expected_hands() -> None:
    schedule = BlindSchedule(mode="increasing", base=Blinds(small=1, big=2), double_every=3)
    assert schedule.for_hand(0) == Blinds(1, 2)
    assert schedule.for_hand(2) == Blinds(1, 2)  # still the first level
    assert schedule.for_hand(3) == Blinds(2, 4)  # one doubling
    assert schedule.for_hand(5) == Blinds(2, 4)
    assert schedule.for_hand(6) == Blinds(4, 8)  # two doublings
    assert schedule.for_hand(9) == Blinds(8, 16)  # three doublings


# ---------------------------------------------------------------------------
# Bot elimination (_process_bot_eliminations), tested directly against
# deliberately-set stacks so ties and edge cases are exact rather than luck of the deal
# ---------------------------------------------------------------------------


def test_elimination_removes_a_bot_when_others_remain() -> None:
    session = Session(SessionConfig(seats=human_bot_seats(3)), check_or_call, rng=random.Random(1))
    session.stacks[2] = 0  # one of three bots busts

    eliminated, replenished, strict_end = session._process_bot_eliminations([0, 1, 2, 3])

    assert eliminated == [2]
    assert replenished is None
    assert not strict_end
    assert 2 not in session.active
    assert not session.ended


def test_last_bot_is_replenished_instead_of_removed_by_default() -> None:
    session = Session(SessionConfig(seats=human_bot_seats(2)), check_or_call, rng=random.Random(1))
    session.stacks[2] = 0  # eliminate one bot first, so seat 1 becomes the last
    session.active.discard(2)
    session.stacks[1] = 0  # now the only remaining bot busts too

    eliminated, replenished, strict_end = session._process_bot_eliminations([0, 1])

    assert eliminated == []
    assert replenished == 1
    assert not strict_end
    assert not session.ended
    assert session.stacks[1] == session.config.starting_stack
    assert 1 in session.active


def test_last_bot_ends_the_session_in_strict_mode() -> None:
    config = SessionConfig(seats=human_bot_seats(2), strict=True)
    session = Session(config, check_or_call, rng=random.Random(1))
    session.stacks[2] = 0
    session.active.discard(2)
    session.stacks[1] = 0

    eliminated, replenished, strict_end = session._process_bot_eliminations([0, 1])

    assert eliminated == []
    assert replenished is None
    assert strict_end
    assert session.ended
    assert session.end_reason is not None
    assert session.stacks[1] == 0  # not replenished in strict mode


def test_the_only_bot_is_immediately_the_last_bot_at_the_minimum_table_size() -> None:
    # 1 bot: the new lower boundary. There is no "some bots remain" branch to take at
    # all here - the only bot busting is always the last-bot case, from the very first
    # bust, with no prior elimination needed to get there.
    session = Session(SessionConfig(seats=human_bot_seats(1)), check_or_call, rng=random.Random(1))
    session.stacks[1] = 0

    eliminated, replenished, strict_end = session._process_bot_eliminations([0, 1])

    assert eliminated == []
    assert replenished == 1
    assert not strict_end
    assert session.stacks[1] == session.config.starting_stack
    assert 1 in session.active


def test_the_only_bot_ends_the_session_in_strict_mode_at_the_minimum_table_size() -> None:
    config = SessionConfig(seats=human_bot_seats(1), strict=True)
    session = Session(config, check_or_call, rng=random.Random(1))
    session.stacks[1] = 0

    eliminated, replenished, strict_end = session._process_bot_eliminations([0, 1])

    assert eliminated == []
    assert strict_end
    assert session.ended
    assert session.stacks[1] == 0


def test_elimination_removes_one_bot_among_five_at_the_maximum_table_size() -> None:
    # 5 bots: the new upper boundary. One busting among five others is the ordinary
    # "others remain" case, same as the 3-bot version above, just at the new ceiling.
    session = Session(SessionConfig(seats=human_bot_seats(5)), check_or_call, rng=random.Random(1))
    session.stacks[3] = 0  # one of five bots busts

    eliminated, replenished, strict_end = session._process_bot_eliminations([0, 1, 2, 3, 4, 5])

    assert eliminated == [3]
    assert replenished is None
    assert not strict_end
    assert 3 not in session.active
    assert session.active == {0, 1, 2, 4, 5}
    assert not session.ended


def test_simultaneous_bust_of_all_bots_replenishes_exactly_one() -> None:
    # Both bots hit zero in the same hand. Removing either individually would leave
    # a bot-less table, so exactly one has to survive - which one is an arbitrary,
    # stable tie-break (see the docstring on _process_bot_eliminations), not a
    # meaningful choice, so this only asserts the aggregate outcome.
    session = Session(SessionConfig(seats=human_bot_seats(2)), check_or_call, rng=random.Random(1))
    session.stacks[1] = 0
    session.stacks[2] = 0

    eliminated, replenished, strict_end = session._process_bot_eliminations([0, 1, 2])

    assert len(eliminated) == 1
    assert replenished is not None
    assert replenished not in eliminated
    assert {eliminated[0], replenished} == {1, 2}
    assert session.stacks[replenished] == session.config.starting_stack
    assert session.active == {0, replenished}


def test_elimination_ignores_the_human_seat() -> None:
    session = Session(SessionConfig(seats=human_bot_seats(2)), check_or_call, rng=random.Random(1))
    session.stacks[session.human_seat] = 0

    eliminated, replenished, strict_end = session._process_bot_eliminations(
        [0, 1, 2]
    )

    assert eliminated == []
    assert replenished is None
    assert not strict_end
    assert session.human_seat in session.active  # untouched - not this method's job


# ---------------------------------------------------------------------------
# rebuy_human
# ---------------------------------------------------------------------------


def test_rebuy_human_requires_being_busted() -> None:
    session = Session(SessionConfig(seats=human_bot_seats(2)), check_or_call, rng=random.Random(1))
    with pytest.raises(RuntimeError):
        session.rebuy_human()


def test_rebuy_human_resets_stack_and_tracks_net_result() -> None:
    session = Session(SessionConfig(seats=human_bot_seats(2)), check_or_call, rng=random.Random(1))
    session.stacks[session.human_seat] = 0
    session.human_busted = True

    session.rebuy_human(amount=150)

    assert session.stacks[session.human_seat] == 150
    assert not session.human_busted
    total_buyins = session.config.starting_stack + 150
    assert session.human_total_buyins == total_buyins
    assert session.human_net == 150 - total_buyins  # current stack minus everything bought in


# ---------------------------------------------------------------------------
# Integration: real play_next_hand() calls
# ---------------------------------------------------------------------------


def test_persistent_bankroll_carries_over_between_hands() -> None:
    session = Session(
        SessionConfig(seats=human_bot_seats(2), starting_stack=500),
        check_or_call,
        rng=random.Random(7),
    )
    hand1 = session.play_next_hand()
    stacks_after_hand1 = dict(session.stacks)

    hand2 = session.play_next_hand()

    for index, seat in enumerate(hand2.order):
        assert hand2.result.starting_stacks[index] == stacks_after_hand1[seat]
    # sanity: this genuinely changed hand to hand, it isn't trivially true
    assert stacks_after_hand1 != {seat: session.config.starting_stack for seat in session.active}
    assert hand1.hand_number == 0
    assert hand2.hand_number == 1


def _stable_table_seats(num_bots: int = 2) -> tuple[SeatConfig, ...]:
    """Baseline-only bots: calmer than a bluffer/conservative mix, so a table this
    well-capitalised is very unlikely to see a bust within the few hands these tests
    play - they're testing button/blind bookkeeping, not survival."""
    seats = [SeatConfig(name="Human", is_human=True)]
    for i in range(num_bots):
        seats.append(SeatConfig(name=f"Bot{i}", is_human=False, persona=BASELINE))
    return tuple(seats)


def fold_to_any_bet(view: DecisionView) -> BettingAction:
    """A human strategy that never risks its stack - removes the human as a source of
    bust risk in tests that only care about button/blind bookkeeping."""
    if view.to_call > 0:
        return BettingAction(ActionType.FOLD)
    return BettingAction(ActionType.CHECK)


def test_button_rotates_through_active_seats_each_hand() -> None:
    # Measured: this seed keeps all three seats active for at least 8 hands, so the
    # button cycles cleanly with no elimination interference (that case is covered
    # separately, via deal_order's own tests and the simultaneous-bust tests below).
    session = Session(
        SessionConfig(seats=_stable_table_seats(), starting_stack=20_000),
        fold_to_any_bet,
        rng=random.Random(7),
    )
    buttons = [session.button]
    for _ in range(6):
        summary = session.play_next_hand()
        assert summary.button_after is not None
        buttons.append(summary.button_after)

    assert buttons == [0, 1, 2, 0, 1, 2, 0]  # cycles through all 3 seats in order
    assert sum(session.stacks.values()) == 3 * 20_000  # chips only moved, none created


def test_button_rotates_correctly_at_the_minimum_table_size() -> None:
    # 1 bot: the new lower boundary (2 total players, heads-up). Measured: this seed
    # is clean of eliminations, so the button strictly alternates - the heads-up case
    # in deal_order, where the button and small blind are the same seat.
    session = Session(
        SessionConfig(seats=_stable_table_seats(1), starting_stack=20_000),
        fold_to_any_bet,
        rng=random.Random(0),
    )
    buttons = [session.button]
    for _ in range(6):
        summary = session.play_next_hand()
        assert summary.button_after is not None
        buttons.append(summary.button_after)

    assert buttons == [0, 1, 0, 1, 0, 1, 0]
    assert sum(session.stacks.values()) == 2 * 20_000


def test_button_rotates_correctly_at_the_maximum_table_size() -> None:
    # 5 bots: the new upper boundary (6 total players). Five real bots betting
    # against each other is volatile enough that no seed stays elimination-free for
    # several hands even at a very high stack - unlike the smaller tables above, this
    # is a genuine property of a fuller table, not a fixable test setup. So instead of
    # asserting an exact button sequence, this asserts what must hold regardless of
    # how many eliminations happen along the way: the button always lands on a seat
    # that is actually still active, and always moves somewhere while more than one
    # seat remains - exercising deal_order, elimination and the button-advance
    # together at the largest table session.py supports.
    session = Session(
        SessionConfig(seats=_stable_table_seats(5), starting_stack=50_000),
        fold_to_any_bet,
        rng=random.Random(1),
    )
    for _ in range(6):
        if session.ended:
            break
        before = session.button
        summary = session.play_next_hand()
        if summary.button_after is None:
            break
        assert summary.button_after in session.active
        if len(session.active) > 1:
            assert summary.button_after != before
    assert sum(session.stacks.values()) == 6 * 50_000  # chips only moved, none created


def test_fixed_blinds_never_change_across_hands() -> None:
    session = Session(
        SessionConfig(seats=_stable_table_seats(), starting_stack=20_000),
        fold_to_any_bet,
        rng=random.Random(7),
    )
    blinds_seen = [session.play_next_hand().blinds for _ in range(4)]
    assert all(b == Blinds() for b in blinds_seen)


def test_increasing_blinds_change_across_hands() -> None:
    schedule = BlindSchedule(mode="increasing", base=Blinds(small=1, big=2), double_every=2)
    session = Session(
        SessionConfig(
            seats=_stable_table_seats(), starting_stack=20_000, blind_schedule=schedule
        ),
        fold_to_any_bet,
        rng=random.Random(7),
    )
    blinds_seen = [session.play_next_hand().blinds for _ in range(4)]
    assert blinds_seen == [Blinds(1, 2), Blinds(1, 2), Blinds(2, 4), Blinds(2, 4)]


def test_human_bust_is_signalled_and_blocks_further_play() -> None:
    # Starting stack equals the blind, so every seat is forced all-in before any
    # voluntary action - this hand's winner is decided purely by the deal, but *some*
    # player finishing at zero is certain. Seed 1 is a measured case where the human
    # is the one who loses.
    seats = human_bot_seats(2)
    blinds = BlindSchedule(base=Blinds(small=1, big=1))
    session = Session(
        SessionConfig(seats=seats, starting_stack=1, blind_schedule=blinds),
        check_or_call,
        rng=random.Random(1),
    )
    summary = session.play_next_hand()

    assert summary.human_busted
    assert session.human_busted
    assert session.stacks[session.human_seat] == 0

    with pytest.raises(RuntimeError):
        session.play_next_hand()  # blocked until rebuy_human() is called

    session.rebuy_human()
    session.play_next_hand()  # now succeeds


def test_simultaneous_double_bust_through_real_play_replenishes_the_survivor() -> None:
    # Measured case (seed 0): the human wins outright and both bots bust in the same
    # hand, exercising the real elimination path end to end rather than through
    # directly-poked stacks.
    seats = human_bot_seats(2)
    blinds = BlindSchedule(base=Blinds(small=1, big=1))
    session = Session(
        SessionConfig(seats=seats, starting_stack=1, blind_schedule=blinds),
        check_or_call,
        rng=random.Random(0),
    )
    summary = session.play_next_hand()

    assert not summary.human_busted
    assert len(summary.eliminated) == 1
    assert summary.replenished is not None
    assert session.active == {session.human_seat, summary.replenished}
    assert session.stacks[summary.replenished] == session.config.starting_stack


def test_simultaneous_double_bust_ends_the_session_in_strict_mode() -> None:
    seats = human_bot_seats(2)
    blinds = BlindSchedule(base=Blinds(small=1, big=1))
    session = Session(
        SessionConfig(seats=seats, starting_stack=1, blind_schedule=blinds, strict=True),
        check_or_call,
        rng=random.Random(0),
    )
    summary = session.play_next_hand()

    assert summary.strict_end
    assert session.ended
    assert summary.button_after is None
    with pytest.raises(RuntimeError):
        session.play_next_hand()


def test_hand_summary_seat_lookups_translate_hand_local_indices() -> None:
    session = Session(SessionConfig(seats=human_bot_seats(2)), check_or_call, rng=random.Random(5))
    summary = session.play_next_hand()

    for seat in summary.winner_seats:
        assert seat in summary.order
    for seat in summary.order:
        assert len(summary.hole_cards_for(seat)) == 2
