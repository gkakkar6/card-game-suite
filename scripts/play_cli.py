"""Thin terminal entry point for a multiplayer poker session against bots.

All the actual session logic (bankrolls, button rotation, elimination, blinds) lives
in games/poker/session.py, with no I/O of its own - this file's only job is prompting
for setup, running the loop, and printing results.

Accepting every default produces tournament mode (increasing blinds, strict
elimination): a session that runs down to a single winner rather than one that could
continue forever. Fixed blinds and/or bot auto-replenish (non-strict) stay available
as explicit overrides for casual practice - the two toggles remain independent, as in
session.py; tournament is just what answering "yes" to both defaults gives you.

    uv run python scripts/play_cli.py
"""

from engine.personas.quantal import Persona
from games.poker.betting import ActionType
from games.poker.cli_strategy import CLIStrategy
from games.poker.hand import DEFAULT_STARTING_STACK
from games.poker.personas import BLUFFER, CALLING_STATION, CONSERVATIVE, ERRATIC, PERSONAS
from games.poker.session import BlindSchedule, HandSummary, SeatConfig, Session, SessionConfig

# Cycled through for the default bot personas, skipping baseline: this is a CLI to
# play against, and showcasing the range of styles out of the box is more interesting
# than an all-baseline table. Baseline is still one prompt away for anyone who wants it.
DEFAULT_BOT_PERSONAS = (BLUFFER, CONSERVATIVE, CALLING_STATION, ERRATIC)


def _prompt(question: str, default: str) -> str:
    raw = input(f"{question} [{default}]: ").strip()
    return raw or default


def _prompt_int(question: str, default: int, low: int, high: int) -> int:
    while True:
        raw = _prompt(question, str(default))
        try:
            value = int(raw)
        except ValueError:
            print(f"  enter a whole number between {low} and {high}")
            continue
        if not (low <= value <= high):
            print(f"  must be between {low} and {high}")
            continue
        return value


def _prompt_persona(question: str, default: Persona[ActionType]) -> Persona[ActionType]:
    names = ", ".join(PERSONAS)
    while True:
        raw = _prompt(f"{question} ({names})", default.name)
        persona = PERSONAS.get(raw)
        if persona is not None:
            return persona
        print(f"  unknown persona '{raw}' - choose one of: {names}")


def _prompt_yes_no(question: str, default: bool) -> bool:
    default_text = "yes" if default else "no"
    while True:
        raw = _prompt(question, default_text).lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  answer yes or no")


def build_config() -> SessionConfig:
    print("=== Table setup ===")
    print("Defaults give you tournament mode: increasing blinds, strict elimination -")
    print("a session that actually ends. Answer 'n' to either for casual play instead.\n")

    num_bots = _prompt_int("How many bots?", default=3, low=2, high=4)
    seats = [SeatConfig(name="You", is_human=True)]
    for i in range(num_bots):
        default_persona = DEFAULT_BOT_PERSONAS[i % len(DEFAULT_BOT_PERSONAS)]
        persona = _prompt_persona(f"Persona for bot {i + 1}?", default_persona)
        seats.append(
            SeatConfig(name=f"Bot {i + 1} ({persona.name})", is_human=False, persona=persona)
        )

    starting_stack = _prompt_int(
        "Starting stack?", default=DEFAULT_STARTING_STACK, low=10, high=1_000_000
    )
    increasing = _prompt_yes_no("Increasing blinds (tournament-style)?", default=True)
    strict = _prompt_yes_no("Strict mode (session ends when the last bot busts)?", default=True)

    return SessionConfig(
        seats=tuple(seats),
        starting_stack=starting_stack,
        blind_schedule=BlindSchedule(mode="increasing" if increasing else "fixed"),
        strict=strict,
    )


def _print_hand_result(session: Session, summary: HandSummary) -> None:
    blinds = f"{summary.blinds.small}/{summary.blinds.big}"
    print(f"\n=== Hand {summary.hand_number + 1} (blinds {blinds}) ===")
    result = summary.result

    if result.went_to_showdown:
        board = " ".join(str(card) for card in result.board)
        print(f"board: {board}")
        for seat in result.contenders:
            cards = " ".join(str(card) for card in summary.hole_cards_for(seat))
            print(f"  {session.roster[seat].name}: {cards}")

    winners = ", ".join(session.roster[seat].name for seat in summary.winner_seats)
    print(f"pot {result.pot} won by: {winners}")

    for seat in range(session.num_seats):
        stack = session.stacks[seat]
        tag = " (out)" if seat not in session.active and seat != session.human_seat else ""
        print(f"  {session.roster[seat].name}: {stack}{tag}")

    for seat in summary.eliminated:
        print(f"*** {session.roster[seat].name} is eliminated ***")
    if summary.replenished is not None:
        name = session.roster[summary.replenished].name
        print(f"*** {name} busted as the last bot standing and is re-staked ***")
    if summary.strict_end:
        print(f"*** session over: {session.end_reason} ***")
    elif summary.button_after is not None:
        print(f"button: {session.roster[summary.button_after].name}")


def _handle_human_bust(session: Session) -> bool:
    """Returns True if play should continue (the human rebought), False to stop."""
    print(f"\nYou're out of chips. Net for the session so far: {session.human_net:+d}")
    if _prompt_yes_no("Rebuy and keep playing?", default=False):
        session.rebuy_human()
        return True
    return False


def _print_session_summary(session: Session) -> None:
    print("\n=== Session over ===")
    print(f"hands played: {session.hands_played}")
    print(f"your net result: {session.human_net:+d}")


def main() -> None:
    config = build_config()
    session = Session(config, CLIStrategy())

    print("\nStarting the game. Good luck.")
    while True:
        summary = session.play_next_hand()
        _print_hand_result(session, summary)

        if summary.strict_end:
            break

        if summary.human_busted:
            if not _handle_human_bust(session):
                break
            continue

        choice = input("\nPress Enter for the next hand, or type 'quit' to stop: ").strip().lower()
        if choice == "quit":
            break

    _print_session_summary(session)


if __name__ == "__main__":
    main()
