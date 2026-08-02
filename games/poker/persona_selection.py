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

# Single-word keywords that are too close (by edit distance) to an unrelated real
# English word to safely fuzz - matched exactly only, regardless of length, never via
# nlp.py's typo tolerance. Found systematically, not by inspection: every single-word
# keyword in FAMILIES was checked against a real system dictionary
# (scripts/check_persona_keyword_fragility.py, ~200,000 words) and flagged if its
# closest match fell within what its length's tolerance would otherwise allow.
#
# Exempted, with what each collides with and why it's a real practical risk:
#
#   "loose" - edit-distance 1 from "goose", "louse", "moose", "noose", "lose". A
#       common animal-name rhyme family plus "lose", itself central poker vocabulary
#       ("afraid to lose") that means something quite different from a loose
#       playing style.
#   "tight" - edit-distance 1 from "eight", "fight", "light", "might", "night",
#       "right", "sight", "tights" - an unusually dense collision, the whole -ight
#       rhyme family lands at distance 1.
#   "tough" - edit-distance 1 from "touch", "though", "dough", "cough", "rough",
#       "bough", "trough". "though" is one of the most common function words in
#       English (would appear in almost any longer description); "dough" is itself
#       poker slang for money. Either makes this the highest-traffic exemption here.
#   "unpredictable" - edit-distance 2 from "predictable" - THE serious case, and not
#       like the others. This isn't an unrelated word that would produce an
#       off-topic guess; it's the literal semantic opposite. Describing an opponent
#       as "predictable" is a very ordinary thing to type, and if it fuzzy-matched
#       "unpredictable" the tool would confidently recommend the exact wrong
#       persona (erratic instead of, if anything, the reverse). An unrelated-word
#       collision produces a bad guess; a same-axis, opposite-meaning collision
#       produces a plausible, confidently wrong one - worth exempting even though
#       "unpredictable" (13 letters) would otherwise be one of the safer, longer
#       keywords.
#   "bluff" - edit-distance 1 from "buff". "buff" alone reads as enthusiasm
#       ("a real poker buff"), not aggression - fuzzy-matching it to bluffer would
#       misdirect toward a different concept entirely, not just tolerate a typo.
#   "expert" - edit-distance 1 from "expect", an extremely common, general-purpose
#       verb ("I expect he folds a lot") likely to appear in any reasonably long
#       free-text answer regardless of whether the writer meant to describe skill.
#   "sticky" - edit-distance 1 from "stocky", a common physical descriptor
#       plausible in a "describe your opponent" answer even when the person means
#       appearance, not playing style.
#
# Considered and deliberately left un-exempted, with why:
#
#   "aggressive" collides with its own grammatical family ("aggression",
#       "aggressively" - matching these is the point) and otherwise only obscure
#       technical words ("degressive", "ingressive", "egressive", "regressive",
#       "unaggressive") nobody types describing a poker opponent.
#   "careful" only collides with obscure/archaic words ("cageful", "carful",
#       "cartful", "caseful", "dareful", "scareful") - no real risk.
#   "cautious" collides with its own grammatical family ("caution", "cautiously" -
#       desirable) and with "curious", a real word but not a natural play-style
#       descriptor in this context. It also collides with "incautious"/"uncautious"
#       - the same semantic-inversion pattern as unpredictable/predictable above,
#       genuinely considered for exemption on that basis - but judged lower risk in
#       practice: those are far rarer, more formal words that a casual description
#       is unlikely to reach for (someone would write "reckless" or "careless"
#       instead). A closer call than the others left un-exempted, worth naming
#       explicitly rather than silently deciding.
#   "conservative" collides with its own grammatical family ("conservation",
#       "conservatism", "conservatist", "conservatively" - desirable) and with
#       "unconservative", the same negation-prefix pattern as "incautious" above,
#       left un-exempted for the same reason: real but rarely used in practice.
#   "erratic" only collides with one obscure word ("serratic").
#   "maniac" collides with its own grammatical family ("mania", "manic" -
#       desirable) and "manioc" (cassava), too obscure to matter.
#   "optimal" collides with "optical" - real, but not a plausible way to describe a
#       player's style (a visual/technical term, contextually out of place here).
#   "patient" collides with "patent" - real, but likewise not a natural
#       personality or style descriptor for this context.
#   "random" collides with "fandom" and "ransom" - real words, neither a natural
#       style descriptor here.
#   "reckless" collides with its own grammatical family ("recklessly" - desirable)
#       and, among a large family of mostly nonsense "-less" words, two real ones
#       worth naming: "restless" and "luckless" (the latter genuinely poker-
#       adjacent). Judged lower practical risk than "predictable": the prompt asks
#       to describe an opponent's *style*, and commentary on luck or restlessness
#       is a less likely answer to that specific question than an ordinary style
#       adjective would be.
#   "wildcard" collides with "wildcat" (a similar reckless/unpredictable
#       connotation anyway, so not actually misleading if matched), "windward"
#       (real but implausible in this context), and "willyard" (too obscure to
#       matter).
EXEMPT_FROM_FUZZY_MATCHING: frozenset[str] = frozenset(
    {"loose", "tight", "tough", "unpredictable", "bluff", "expert", "sticky"}
)


def persona_for_text(text: str) -> tuple[Persona[ActionType], str, list[str]]:
    """Interpret free text as an opponent persona.

    Returns the chosen persona, its name, and which keyword(s) in `text` actually
    triggered the match - so a caller can show an honest "interpreting that as: X
    (matched: ...)" message rather than inferring silently. An empty keyword list
    means nothing matched at all, distinct from a genuine "optimal" match that also
    happens to resolve to the baseline persona.

    Text that matches nothing falls back to baseline, the safe default, rather than
    raising - free text is expected to sometimes not match anything. Single-word
    keywords tolerate small typos (nlp.py's edit-distance matching), except the
    EXEMPT_FROM_FUZZY_MATCHING keywords, which stay exact-match only.
    """
    intent = parse_intent(text, _KEYWORDS_BY_FAMILY, EXEMPT_FROM_FUZZY_MATCHING)
    if not intent.tags:
        return BASELINE, BASELINE.name, []

    family = next(name for name in FAMILIES if name in intent.tags)
    keywords, persona = FAMILIES[family]
    return persona, persona.name, matched_keywords(text, keywords, EXEMPT_FROM_FUZZY_MATCHING)
