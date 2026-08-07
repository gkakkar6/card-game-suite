"""Thin terminal entry point for a bridge session against bots.

All the actual session logic (dealer rotation, running score) lives in
games/bridge/session.py, with no I/O of its own - this file's only job is prompting
for setup, running the loop, and printing results.

Seats are fixed at You(0), Opponent 1(1), Partner(2), Opponent 2(3) - real bridge
table order (clockwise), so seat 0 and seat 2 land on the same side the way
games.bridge.rules.partner() expects, the same as seat 1 and seat 3.

    uv run python scripts/play_bridge_cli.py
"""

from games.bridge.cli_strategy import (
    CLIBidStrategy,
    CLIPlayStrategy,
    format_auction,
    format_contract,
)
from games.bridge.deal import SEATS
from games.bridge.persona_selection import persona_for_text
from games.bridge.personas import PERSONAS, BridgePersona
from games.bridge.session import HandResult, SeatConfig, Session, SessionConfig

PERSONA_DESCRIPTION_HELP = (
    "Describe your partner or opponent, or type a persona name directly. I can "
    "currently recognise: best/optimal/expert play, aggressive/bold, "
    "conservative/tight/careful, selfish/glory-seeking, sneaky/tricky/baiting, or "
    "beginner/inexperienced - anything else falls back to baseline, so ask for one "
    "of those styles if you want a specific lean."
)


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


def _prompt_persona(question: str) -> BridgePersona:
    names = ", ".join(PERSONAS)
    while True:
        raw = input(f"{question}\n  ({names}, or describe a style): ").strip()
        exact = PERSONAS.get(raw.lower())
        if exact is not None:
            return exact

        persona, _name, matched = persona_for_text(raw)
        if not matched:
            print(
                f"  didn't recognise '{raw}' as a persona name or a described style - "
                f"type one of {names}, or describe a style (e.g. 'aggressive', 'sneaky')"
            )
            continue

        print(f"  Interpreting that as: {persona.name} (matched: {', '.join(matched)})")
        if _prompt_yes_no("  Use this persona?", default=True):
            return persona
        # Declining just loops back to the same prompt above, which already accepts
        # either an exact name or another description - covering both "re-describe"
        # and "name it exactly" without a separate branch for each.
        print("  OK - describe again, or type an exact name.")


def build_config() -> SessionConfig:
    print("=== Table setup ===")
    print("Bridge is partnerships: you and your partner against the other two.\n")

    print(
        "Your partner is the highest-stakes pick of the three - the dummy rule means "
        "that on any hand where you end up as dummy, you make zero decisions at all; "
        "your partner's skill alone determines how that hand goes."
    )
    print(f"\n{PERSONA_DESCRIPTION_HELP}\n")
    partner_persona = _prompt_persona("Persona for your partner?")
    opp1_persona = _prompt_persona("\nPersona for your first opponent?")
    opp2_persona = _prompt_persona("\nPersona for your second opponent?")

    seats = (
        SeatConfig(name="You", is_human=True),
        SeatConfig(
            name=f"Opponent 1 ({opp1_persona.name})", is_human=False, persona=opp1_persona
        ),
        SeatConfig(
            name=f"Partner ({partner_persona.name})", is_human=False, persona=partner_persona
        ),
        SeatConfig(
            name=f"Opponent 2 ({opp2_persona.name})", is_human=False, persona=opp2_persona
        ),
    )
    return SessionConfig(seats=seats)


def _print_hand_result(session: Session, result: HandResult) -> None:
    dealer_name = session.roster[result.dealer].name
    print(f"\n=== Hand {session.hands_played} (dealer: {dealer_name}) ===")
    print("Auction:")
    print(format_auction(result.auction))

    if result.contract is None:
        print("\nPassed out - no contract, no score this hand.")
    else:
        assert result.tricks_won is not None and result.score is not None
        declarer_name = session.roster[result.contract.declarer].name
        print(f"\nContract: {format_contract(result.contract)} ({declarer_name})")
        print(f"Tricks won by declarer's side: {result.tricks_won}")
        if result.score.made:
            over = result.score.margin
            extra = f", {over} over" if over > 0 else ""
            print(f"Made{extra} — {result.score.points:+d} points")
        else:
            print(f"Down {-result.score.margin} — {result.score.points:+d} points")

    your_side = session.scores[session.human_seat]
    other_seat = (session.human_seat + 1) % SEATS
    other_side = session.scores[other_seat]
    print(f"\nRunning score — your side: {your_side:+d}   other side: {other_side:+d}")


def _print_session_summary(session: Session) -> None:
    print("\n=== Session over ===")
    print(f"hands played: {session.hands_played}")
    your_side = session.scores[session.human_seat]
    other_seat = (session.human_seat + 1) % SEATS
    other_side = session.scores[other_seat]
    print(f"final score — your side: {your_side:+d}   other side: {other_side:+d}")
    if your_side > other_side:
        print("You came out ahead.")
    elif your_side < other_side:
        print("The other side came out ahead.")
    else:
        print("Dead even.")


def main() -> None:
    config = build_config()
    session = Session(config, CLIBidStrategy(), CLIPlayStrategy())

    print("\nStarting the session. Good luck.")
    while True:
        result = session.play_next_hand()
        _print_hand_result(session, result)

        choice = input("\nPress Enter for the next hand, or type 'quit' to stop: ").strip().lower()
        if choice == "quit":
            break

    _print_session_summary(session)


if __name__ == "__main__":
    main()
