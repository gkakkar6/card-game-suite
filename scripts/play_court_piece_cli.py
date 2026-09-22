"""Thin terminal entry point for a Court Piece (fixed trump) session against bots.

All the actual session logic (caller rotation, running court total) lives in
games/court_piece/session.py, with no I/O of its own - this file's only job is
prompting for setup, running the loop, and printing results.

Seats are You(0), Opponent 1(1), Partner(2), Opponent 2(3) - the same table-order
convention as scripts/play_bridge_cli.py, so seat 0/2 and seat 1/3 land on the same
sides that games.court_piece.rules.partner() expects.

Only the baseline persona exists today (games/court_piece/personas.py) - every bot
plays it, so there's no persona prompt yet, unlike bridge's CLI. This is a real,
temporary gap, not an oversight: more named styles are a planned, easy extension once
there's real play to measure them against.

    uv run python scripts/play_court_piece_cli.py
"""

from games.court_piece.cli_strategy import CLIDeclareStrategy, CLIPlayStrategy, format_call
from games.court_piece.deal import SEATS
from games.court_piece.personas import BASELINE
from games.court_piece.session import HandResult, SeatConfig, Session, SessionConfig


def _prompt_yes_no(question: str, default: bool) -> bool:
    default_text = "yes" if default else "no"
    while True:
        raw = input(f"{question} [{default_text}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  answer yes or no")


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
    print("Court Piece is partnerships: you and your partner against the other two.\n")

    target_courts = _prompt_target_courts()
    seats = (
        SeatConfig(name="You", is_human=True),
        SeatConfig(name="Opponent 1 (baseline)", is_human=False, persona=BASELINE),
        SeatConfig(name="Partner (baseline)", is_human=False, persona=BASELINE),
        SeatConfig(name="Opponent 2 (baseline)", is_human=False, persona=BASELINE),
    )
    return SessionConfig(seats=seats, target_courts=target_courts)


def _print_hand_result(session: Session, result: HandResult) -> None:
    caller_name = session.roster[result.caller].name
    print(f"\n=== Hand {session.hands_played} (caller: {caller_name}) ===")
    print(f"Trump called: {format_call(result.call)}")

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


def _print_session_summary(session: Session) -> None:
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
    session = Session(config, CLIDeclareStrategy(), CLIPlayStrategy())

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
