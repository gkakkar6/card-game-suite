import random
from collections.abc import Sequence

import pytest

from engine.cards import Card, Suit
from games.court_piece import personas as personas_module
from games.court_piece.action_values import ActionAnalysis
from games.court_piece.deal import DECLARE_PREVIEW
from games.court_piece.personas import BASELINE
from games.court_piece.rules import CourtPiece, CourtPieceState
from games.court_piece.session import (
    HandResult,
    RunningSession,
    SeatConfig,
    Session,
    SessionConfig,
    play_hand,
    play_running_hand,
)


def _lowest_legal(game: CourtPiece, state: CourtPieceState) -> Card:
    """A trivial, deterministic card-play strategy: whatever sorts first. Used
    wherever a test calls play_hand() directly and doesn't care which specific card
    is played, just that play proceeds - the same precedent
    games/bridge/tests/test_session.py sets, so a whole hand completes without paying
    for real PIMC solving at every decision."""
    return min(game.legal_actions(state), key=lambda c: (c.suit.value, c.rank.value))


def _always_hearts(preview: Sequence[Card]) -> Suit:
    """A trivial, deterministic declare strategy - the actual suit called doesn't
    matter to any test in this file, only that a hand can proceed past the
    declare phase."""
    return Suit.HEARTS


def test_play_hand_produces_a_complete_thirteen_trick_hand() -> None:
    rng = random.Random(11)
    declare_strategies = [_always_hearts for _ in range(4)]
    play_strategies = [_lowest_legal for _ in range(4)]
    result: HandResult = play_hand(declare_strategies, play_strategies, caller=0, rng=rng)

    assert len(result.final_state.completed) == 13
    assert result.call.caller == 0
    assert result.call.trump == Suit.HEARTS
    assert result.score.tricks_won >= 7  # someone always reaches a majority in 13 tricks


def test_play_hand_only_shows_the_caller_a_five_card_preview() -> None:
    # The real fairness rule you specified: the caller decides trump from only their
    # first 5 cards, not the full 13 - even though the rest of their hand already
    # physically exists in `hands` by the time play_hand() calls the declare
    # strategy. A bug here would silently leak information no real caller would have.
    seen: list[Sequence[Card]] = []

    def recording_declare_strategy(preview: Sequence[Card]) -> Suit:
        seen.append(preview)
        return Suit.HEARTS

    declare_strategies = [recording_declare_strategy] + [_always_hearts for _ in range(3)]
    play_strategies = [_lowest_legal for _ in range(4)]
    result = play_hand(declare_strategies, play_strategies, caller=0, rng=random.Random(4))

    assert len(seen) == 1
    assert len(seen[0]) == DECLARE_PREVIEW
    assert tuple(seen[0]) == result.hands[0][:DECLARE_PREVIEW]


def test_play_running_hand_produces_a_complete_thirteen_trick_hand() -> None:
    play_strategies = [_lowest_legal for _ in range(4)]
    result: HandResult = play_running_hand(
        play_strategies, first_leader=0, rng=random.Random(13)
    )

    assert len(result.final_state.completed) == 13
    assert result.caller == 0  # first_leader, reported the same way play_hand()'s caller is
    assert result.score.tricks_won >= 7  # someone always reaches a majority in 13 tricks
    # HandResult.call must reflect what trump actually resolved to by the end, not
    # the initial always-None call - see the bug this exact check was added to catch.
    assert result.call == result.final_state.call


def test_play_running_hand_has_no_declare_phase_at_all() -> None:
    # There is nothing to record here the way test_play_hand_only_shows_the_caller_
    # a_five_card_preview() checks for fixed trump - play_running_hand() takes no
    # DeclareStrategy argument at all, which this signature check makes structural
    # rather than just "nobody happened to call one."
    import inspect

    parameters = inspect.signature(play_running_hand).parameters
    assert "declare_strategies" not in parameters


# ---------------------------------------------------------------------------
# Session()-level tests. Unlike play_hand() above, Session always wires each bot
# seat's play through its own assigned persona internally (SeatConfig.persona) - a
# test can't hand a Session a trivial per-seat function directly the way play_hand()
# allows. Court Piece has no bridge-style "unaware" fast-path persona yet to skip
# real PIMC, so these tests monkeypatch analyse() instead: it exercises the real
# Session/CardPlayStrategy wiring end to end, just without paying for real solving,
# since these tests care about caller rotation and court bookkeeping, not play
# quality (which test_action_values.py, test_pimc.py and test_personas.py already
# cover directly against the real machinery).
# ---------------------------------------------------------------------------

_SEATS = (
    SeatConfig(name="You", is_human=True),
    SeatConfig(name="Opponent 1", is_human=False, persona=BASELINE),
    SeatConfig(name="Partner", is_human=False, persona=BASELINE),
    SeatConfig(name="Opponent 2", is_human=False, persona=BASELINE),
)


def _fast_session(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> Session:
    def fast_analyse(
        game: CourtPiece, state: CourtPieceState, **analyse_kwargs: object
    ) -> ActionAnalysis:
        return ActionAnalysis(values=dict.fromkeys(game.legal_actions(state), 0.0), scale=1.0)

    monkeypatch.setattr(personas_module, "analyse", fast_analyse)
    config = SessionConfig(seats=_SEATS, **kwargs)  # type: ignore[arg-type]
    return Session(config, _always_hearts, _lowest_legal, rng=random.Random(9))


def test_session_rotates_the_caller_and_accumulates_courts(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _fast_session(monkeypatch)

    assert session.caller == 0
    session.play_next_hand()
    assert session.caller == 1
    session.play_next_hand()
    assert session.caller == 2

    assert session.hands_played == 2
    assert sum(session.courts.values()) > 0  # every hand hands out a majority winner


def test_caller_rotation_completes_a_full_cycle_back_to_the_original_seat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _fast_session(monkeypatch)

    seen_callers = []
    for _ in range(4):
        seen_callers.append(session.caller)
        session.play_next_hand()

    assert seen_callers == [0, 1, 2, 3]
    assert session.caller == 0  # back to the start after a full 4-hand cycle


def test_session_ends_once_a_side_reaches_the_target(monkeypatch: pytest.MonkeyPatch) -> None:
    # A tiny target guarantees the session ends almost immediately, since every hand
    # scores at least PIECE_VALUE (0.1) to somebody.
    session = _fast_session(monkeypatch, target_courts=0.05)

    assert not session.is_over()
    session.play_next_hand()
    assert session.is_over()


def test_is_over_is_false_right_up_to_the_exact_target_and_true_from_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _fast_session(monkeypatch, target_courts=0.2)

    # Manually control the court total rather than depending on real hands, so the
    # boundary itself - not incidental scoring - is what's under test.
    session.courts[0] = 0.1
    assert not session.is_over()
    session.courts[0] = 0.2
    assert session.is_over()  # exactly at the target counts as over, not just past it


def test_play_next_hand_raises_once_the_session_is_already_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _fast_session(monkeypatch, target_courts=0.05)
    session.play_next_hand()
    assert session.is_over()

    with pytest.raises(RuntimeError):
        session.play_next_hand()


# ---------------------------------------------------------------------------
# RunningSession: same rotation/bookkeeping shape as Session, no declare phase.
# ---------------------------------------------------------------------------


def _fast_running_session(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> RunningSession:
    def fast_analyse(
        game: CourtPiece, state: CourtPieceState, **analyse_kwargs: object
    ) -> ActionAnalysis:
        return ActionAnalysis(values=dict.fromkeys(game.legal_actions(state), 0.0), scale=1.0)

    monkeypatch.setattr(personas_module, "analyse", fast_analyse)
    config = SessionConfig(seats=_SEATS, **kwargs)  # type: ignore[arg-type]
    return RunningSession(config, _lowest_legal, rng=random.Random(9))


def test_running_session_rotates_the_first_leader_and_accumulates_courts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _fast_running_session(monkeypatch)

    assert session.first_leader == 0
    session.play_next_hand()
    assert session.first_leader == 1
    session.play_next_hand()
    assert session.first_leader == 2

    assert session.hands_played == 2
    assert sum(session.courts.values()) > 0


def test_running_session_first_leader_completes_a_full_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _fast_running_session(monkeypatch)

    seen_leaders = []
    for _ in range(4):
        seen_leaders.append(session.first_leader)
        session.play_next_hand()

    assert seen_leaders == [0, 1, 2, 3]
    assert session.first_leader == 0


def test_running_session_ends_once_a_side_reaches_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _fast_running_session(monkeypatch, target_courts=0.05)

    assert not session.is_over()
    session.play_next_hand()
    assert session.is_over()


def test_running_session_play_next_hand_raises_once_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _fast_running_session(monkeypatch, target_courts=0.05)
    session.play_next_hand()
    assert session.is_over()

    with pytest.raises(RuntimeError):
        session.play_next_hand()
