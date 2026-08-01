"""A Strategy (games/poker/hand.py's protocol) that reads a human's actions from the
terminal, instead of computing one.

Deliberately does not import games.poker.action_values or games.poker.equity - a human
player sees only what a real player would see (their own cards, the board, the pot,
what they owe), never a computed equity or action value. That is a hard boundary, not
just a display choice: this file has no way to show one even if someone tried to add
it carelessly later, since the modules that compute it are never imported here.
"""

from games.poker.betting import ActionType, BettingAction
from games.poker.hand import DecisionView

_ACTION_WORDS: dict[str, ActionType] = {
    "fold": ActionType.FOLD,
    "check": ActionType.CHECK,
    "call": ActionType.CALL,
    "bet": ActionType.BET,
    "raise": ActionType.RAISE,
}

_AMOUNT_ACTIONS = (ActionType.BET, ActionType.RAISE)


def _command_for(action: ActionType) -> str:
    return f"{action.name.lower()} <amount>" if action in _AMOUNT_ACTIONS else action.name.lower()


def format_view(view: DecisionView) -> str:
    """A human-readable summary of one decision point."""
    hole = " ".join(str(card) for card in view.hole_cards)
    board = " ".join(str(card) for card in view.board) or "-"
    legal = ", ".join(_command_for(action) for action in view.legal_actions)
    lines = [
        f"--- {view.street} ---",
        f"your cards: {hole}    board: {board}",
        f"pot: {view.pot}    to call: {view.to_call}    your stack: {view.stack}",
        f"actions: {legal}",
    ]
    return "\n".join(lines)


def parse_action(text: str, view: DecisionView) -> BettingAction:
    """Turn one line of input into a BettingAction.

    Raises ValueError with a human-readable reason on anything unparseable, illegal
    for `view`, or missing/invalid an amount - CLIStrategy catches this and re-prompts
    rather than letting a typo end the hand.
    """
    parts = text.strip().lower().split()
    if not parts:
        raise ValueError("type an action, e.g. fold / check / call / bet 20 / raise 50")

    word, *rest = parts
    action = _ACTION_WORDS.get(word)
    if action is None:
        raise ValueError(f"'{word}' isn't a recognised action")
    if action not in view.legal_actions:
        legal = ", ".join(_command_for(a) for a in view.legal_actions)
        raise ValueError(f"{word} isn't legal right now - try one of: {legal}")

    if action not in _AMOUNT_ACTIONS:
        return BettingAction(action)

    if not rest:
        raise ValueError(f"{word} needs an amount, e.g. '{word} {view.current_bet + view.min_bet}'")
    try:
        amount = int(rest[0])
    except ValueError as exc:
        raise ValueError(f"'{rest[0]}' isn't a whole number") from exc
    if amount <= 0:
        raise ValueError("amount must be positive")
    return BettingAction(action, amount=amount)


class CLIStrategy:
    """Prompts the terminal for one action per decision, re-prompting on bad input."""

    def __init__(self, prompt_name: str = "You") -> None:
        self.prompt_name = prompt_name

    def __call__(self, view: DecisionView) -> BettingAction:
        print()
        print(format_view(view))
        while True:
            try:
                text = input(f"{self.prompt_name}> ")
            except EOFError:
                # No more input to read (e.g. piped stdin ran dry) - fold rather than
                # hang or crash, so a non-interactive run still terminates cleanly.
                print("(no input available, folding)")
                return BettingAction(ActionType.FOLD)
            try:
                return parse_action(text, view)
            except ValueError as exc:
                print(f"  {exc}")
