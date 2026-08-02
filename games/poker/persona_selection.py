"""Poker's keyword table for free-text opponent selection - engine/personas/nlp.py's
generic matching mechanism, given poker's specific families and personas.

Deliberately maps onto the five personas already measured in games/poker/personas.py -
never sets temperature or bias directly from parsed text (ARCHITECTURE.md §3). Those
personas behave sanely because their numbers were empirically measured against real
decisions, not guessed. Letting free text set a dial value, even through a lookup
table, would reintroduce that exact unmeasured-parameter risk behind a friendlier
interface. If more nuance is wanted later, the right way to get it is more presets,
each separately measured the same way - not a continuous dial nobody's validated.
"""

from engine.personas.nlp import matched_keywords, parse_intent
from engine.personas.quantal import Persona
from games.poker.betting import ActionType
from games.poker.personas import BASELINE, BLUFFER, CALLING_STATION, CONSERVATIVE, ERRATIC

# Family name -> (trigger keywords, the persona it maps to). Dict order is meaningful:
# it is the tie-break when free text matches more than one family at once (e.g.
# "aggressive but careful") - the first family listed here wins. A simple, documented
# rule rather than trying to weigh conflicting signals against each other, which would
# need a model of what "more aggressive-sounding than tight-sounding" even means.
FAMILIES: dict[str, tuple[tuple[str, ...], Persona[ActionType]]] = {
    "aggressive": (
        ("bluff", "wild", "maniac", "loose", "reckless", "aggressive"),
        BLUFFER,
    ),
    "tight": (
        ("tight", "careful", "conservative", "nit", "patient", "cautious"),
        CONSERVATIVE,
    ),
    "calling": (
        ("calling station", "sticky", "never folds", "loose-passive", "calls everything"),
        CALLING_STATION,
    ),
    "random": (
        ("random", "unpredictable", "chaotic", "wildcard", "erratic"),
        ERRATIC,
    ),
    "optimal": (
        ("best", "optimal", "expert", "tough", "hard", "pro", "good"),
        BASELINE,
    ),
}

_KEYWORDS_BY_FAMILY = {family: keywords for family, (keywords, _persona) in FAMILIES.items()}


def persona_for_text(text: str) -> tuple[Persona[ActionType], str, list[str]]:
    """Interpret free text as an opponent persona.

    Returns the chosen persona, its name, and which keyword(s) in `text` actually
    triggered the match - so a caller can show an honest "interpreting that as: X
    (matched: ...)" message rather than inferring silently. An empty keyword list
    means nothing matched at all, distinct from a genuine "optimal" match that also
    happens to resolve to the baseline persona.

    Text that matches nothing falls back to baseline, the safe default, rather than
    raising - free text is expected to sometimes not match anything.
    """
    intent = parse_intent(text, _KEYWORDS_BY_FAMILY)
    if not intent.tags:
        return BASELINE, BASELINE.name, []

    family = next(name for name in FAMILIES if name in intent.tags)
    keywords, persona = FAMILIES[family]
    return persona, persona.name, matched_keywords(text, keywords)
