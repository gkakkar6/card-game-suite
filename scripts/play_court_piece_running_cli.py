"""Thin terminal entry point for a running-trump Court Piece (Be-ranga Double Sar)
session against bots.

All the actual session logic (first-leader rotation, running court total) lives in
games/court_piece/session.py's RunningSession, with no I/O of its own - this file's
only job is prompting for setup, running the loop, and printing results.

The real difference from scripts/play_court_piece_cli.py (fixed trump): there is no
declare phase at all, so no partial-hand preview and no trump-call prompt before play
starts. Trump begins undetermined and is set mid-play, automatically, the moment any
player can't follow the suit led (games/court_piece/rules.py) - the play view itself
announces it ("trump: not yet set" until it fires), nothing more for a human to do.

Seats are You(0), Opponent 1(1), Partner(2), Opponent 2(3) - the same table-order
convention as the fixed-trump script.

Only the baseline persona exists today - see that script's own docstring for why.

    uv run python scripts/play_court_piece_running_cli.py
"""

from games.court_piece.cli_strategy import CLIPlayStrategy, format_trump
from games.court_piece.deal import SEATS
from games.court_piece.personas import BASELINE
from games.court_piece.session import HandResult, RunningSession, SeatConfig, SessionConfig


def _prompt_target_courts() -> float | None:
    raw = input(
        "Play to a target number of courts (e.g. 3), or press Enter for free play: "
    ).strip()
    if not raw:
        return None
    try:
        target = float(raw)
    except ValueError:
        print("  didn't understand that - starting free play instead")
        return None
    if target <= 0:
        print("  target must be positive - starting free play instead")
        return None
    return target


def build_config() -> SessionConfig:
    print("=== Table setup ===")
    print("Court Piece (running trump) is partnerships: you and your partner against")
    print(
        "the other two. Nobody calls trump - it starts undetermined, and whoever "
        "first can't follow the suit led sets it, automatically, to whatever they play.\n"
    )

    target_courts = _prompt_target_courts()
    seats = (
        SeatConfig(name="You", is_human=True),
        SeatConfig(name="Opponent 1 (baseline)", is_human=False, persona=BASELINE),
        SeatConfig(name="Partner (baseline)", is_human=False, persona=BASELINE),
        SeatConfig(name="Opponent 2 (baseline)", is_human=False, persona=BASELINE),
    )
    return SessionConfig(seats=seats, target_courts=target_courts)


def _print_hand_result(session: RunningSession, result: HandResult) -> None:
    leader_name = session.roster[result.caller].name
    print(f"\n=== Hand {session.hands_played} (first to lead: {leader_name}) ===")
    print(f"Trump ended up: {format_trump(result.call.trump)}")

    score = result.score
    if score.winner is None:
        print("Nobody reached 7 tricks - no score this hand.")
    else:
        winner_name = session.roster[score.winner].name
        kind = "kot" if score.is_kot else "piece"
        print(
            f"Winner: {winner_name}'s side, {score.tricks_won} tricks - "
            f"a {kind}, +{score.courts:.1f} courts"
        )

    your_side = session.courts[session.human_seat]
    other_seat = (session.human_seat + 1) % SEATS
    other_side = session.courts[other_seat]
    print(f"\nRunning courts — your side: {your_side:.1f}   other side: {other_side:.1f}")


def _print_session_summary(session: RunningSession) -> None:
    print("\n=== Session over ===")
    print(f"hands played: {session.hands_played}")
    your_side = session.courts[session.human_seat]
    other_seat = (session.human_seat + 1) % SEATS
    other_side = session.courts[other_seat]
    print(f"final courts — your side: {your_side:.1f}   other side: {other_side:.1f}")
    if your_side > other_side:
        print("You came out ahead.")
    elif your_side < other_side:
        print("The other side came out ahead.")
    else:
        print("Dead even.")


def main() -> None:
    config = build_config()
    session = RunningSession(config, CLIPlayStrategy())

    print("\nStarting the session. Good luck.")
    while not session.is_over():
        result = session.play_next_hand()
        _print_hand_result(session, result)

        if session.is_over():
            break
        choice = input("\nPress Enter for the next hand, or type 'quit' to stop: ").strip().lower()
        if choice == "quit":
            break

    _print_session_summary(session)


if __name__ == "__main__":
    main()
