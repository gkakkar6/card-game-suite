"""Bridge's keyword table for free-text opponent/partner selection -
engine/personas/nlp.py's generic matching mechanism, given bridge's six personas.

Deliberately maps onto the six personas already measured in games/bridge/personas.py -
never sets bias or temperature directly from parsed text (ARCHITECTURE.md §3), same
discipline as games/poker/persona_selection.py. Those personas behave sanely because
their numbers were empirically measured against real decisions, not guessed.

The matching mechanism itself is untouched, game-agnostic code shared with poker -
this file only supplies bridge's own keywords and persona targets.
"""

from engine.personas.nlp import matched_keywords, parse_intent
from games.bridge.personas import (
    AGGRESSIVE,
    BAITER,
    BASELINE,
    CONSERVATIVE,
    SELFISH,
    UNAWARE,
    BridgePersona,
)

# Family name -> (trigger keywords, the persona it maps to). Dict order is meaningful:
# it is the tie-break when free text matches more than one family at once, the same
# rule poker's own FAMILIES table uses - the first family listed here wins.
FAMILIES: dict[str, tuple[tuple[str, ...], BridgePersona]] = {
    "baseline": (
        ("best", "optimal", "expert", "tough", "hard", "pro", "good", "solid"),
        BASELINE,
    ),
    "aggressive": (
        ("aggressive", "bold", "pushy", "wild", "loose"),
        AGGRESSIVE,
    ),
    "conservative": (
        ("conservative", "tight", "careful", "cautious", "safe", "timid", "passive"),
        CONSERVATIVE,
    ),
    "selfish": (
        ("selfish", "glory", "show-off", "self-centered", "greedy", "individual", "hog"),
        SELFISH,
    ),
    "baiter": (
        ("sneaky", "tricky", "deceptive", "sly", "cunning", "misleading", "wily"),
        BAITER,
    ),
    "unaware": (
        ("beginner", "inexperienced", "clueless", "novice", "naive", "basic", "simple"),
        UNAWARE,
    ),
}

_KEYWORDS_BY_FAMILY = {family: keywords for family, (keywords, _persona) in FAMILIES.items()}

# Single-word keywords too close (by edit distance) to an unrelated real English word to
# safely fuzz - matched exactly only, regardless of length, never via nlp.py's typo
# tolerance. Found systematically against a real system dictionary
# (scripts/check_bridge_persona_keyword_fragility.py, ~210,000 words), the same
# discipline as poker's own list - a genuinely fresh scan, not an assumption that
# poker's exemptions transfer.
#
# Carried forward unchanged from poker's own findings, because they're the literal same
# word with the literal same collision:
#
#   "loose" - edit-distance 1 from "goose"/"louse"/"moose"/"noose"/"lose". Identical to
#       poker's own reasoning; not rediscovered here, just reused.
#   "tight" - edit-distance 1 from "eight"/"fight"/"light"/"might"/"night"/"right"/
#       "sight"/"tights". Same as above.
#   "expert" - edit-distance 1 from "expect", an extremely common general-purpose verb.
#       Poker already exempted this word for exactly this reason.
#   "tough" - edit-distance 1 from "touch"/"though"/"dough"/"cough"/"rough"/"bough"/
#       "trough". "though" alone is decisive (one of the most common function words in
#       English). Same word, same finding as poker's.
#
# New to bridge's own table, found by this scan:
#
#   "inexperienced" - edit-distance 2 from "experienced" - THE serious case here, the
#       same severity as poker's own unpredictable/predictable exemption. Not an
#       unrelated word but its literal semantic opposite: "he's pretty experienced"
#       fuzzy-matching "inexperienced" would confidently recommend the beginner-level
#       unaware persona for someone described as exactly the opposite of a beginner.
#   "tricky" - edit-distance 1 from "trick"/"tricks" - bridge's own most fundamental
#       piece of domain vocabulary (every hand is about who wins how many tricks).
#       Ordinary bridge language ("he won every trick", "a tricky end position") would
#       almost certainly trip this if it weren't exempted, unrelated to describing
#       anyone's personality as tricky/sneaky.
#   "cunning" - edit-distance 1 from "running", a common word and itself real bridge
#       vocabulary ("running the diamond suit") - a genuinely plausible thing to type
#       that has nothing to do with cunning.
#   "deceptive" - edit-distance 1-2 from "receptive" and "perceptive", both real,
#       common, and plausible bridge-partner descriptors in their own right - the
#       opposite problem from an unrelated-word collision, since either would
#       misdirect toward "sneaky/tricky" for a partner described in positive,
#       unrelated terms.
#   "glory" - edit-distance 1 from "gory", a common, unrelated word.
#   "novice" - edit-distance 1 from "notice", a common verb ("I notice he plays
#       cautiously") plausible in ordinary descriptive sentences having nothing to do
#       with skill level.
#   "passive" - edit-distance 1 from "massive", a common physical descriptor plausible
#       in a "describe your partner" answer about appearance, not style - the same
#       class of collision as poker's own "sticky"/"stocky" exemption.
#   "simple" - edit-distance 1 from "sample", itself real bridge vocabulary ("a sample
#       hand") and likely to appear in bridge-related text for reasons having nothing
#       to do with describing a player as simple.
#   "timid" - edit-distance 1 from "timed", a common word very plausible in game
#       context ("a well-timed bid").
#
# Considered and deliberately left un-exempted, with why:
#
#   "aggressive" collides with its own grammatical family (desirable) and otherwise
#       only obscure technical words (degressive, ingressive, egressive, regressive,
#       unaggressive) - identical finding to poker's own "aggressive".
#   "careful" only collides with obscure/archaic words (cageful, carful, cartful,
#       caseful, dareful, scareful) - identical finding to poker's own "careful".
#   "cautious" collides with its own grammatical family (desirable), "curious" (real
#       but not a natural style descriptor here), and "incautious"/"uncautious" (the
#       same negation-prefix pattern as "inexperienced" above, but judged the same way
#       poker judged it for this identical word: rarer, more formal words a casual
#       description is unlikely to reach for).
#   "conservative" collides with its own grammatical family (desirable) and
#       "unconservative" - identical finding and identical judgement to poker's own
#       "conservative".
#   "optimal" collides with "optical" - real, but not a plausible style descriptor,
#       identical finding to poker's own "optimal".
#   "basic" collides with "basin"/"basis" - real words, not plausible style
#       descriptors in this context (same class as poker's "patient"/"optimal" calls).
#   "clueless" collides with "careless" (real, edit-distance 2, borderline) among a
#       long tail of obscure "-less" words. Not a semantic opposite the way
#       "inexperienced"/"experienced" is - "careless" and "clueless" point in a
#       similar rather than contradictory direction for a bridge partner - judged
#       lower risk, the same way poker judged "reckless"/"restless" lower risk than
#       "unpredictable"/"predictable".
#   "greedy" collides with "reedy" and "greeny" - real but implausible descriptors
#       here; "greed" itself is the desirable own-family match.
#   "naive" collides with "native" - real and common, but not a natural style
#       descriptor for a bridge partner's play (an origin/nativeness word, out of
#       place here, the same class of call as poker's "optimal"/"optical").
#   "pushy" collides with a handful of real but implausible words ("cushy", "mushy")
#       and otherwise obscure ones - no real risk.
#   "solid" collides with "stolid" (real, style-relevant, but a rare, formal word -
#       judged the same way poker judged "incautious": unlikely to be reached for
#       casually) among mostly obscure/archaic words.
#   "selfish", "beginner", "individual", "misleading", "sneaky" only collide with
#       obscure or nonsense words - no real risk.
EXEMPT_FROM_FUZZY_MATCHING: frozenset[str] = frozenset(
    {
        "loose",
        "tight",
        "expert",
        "tough",
        "inexperienced",
        "tricky",
        "cunning",
        "deceptive",
        "glory",
        "novice",
        "passive",
        "simple",
        "timid",
    }
)


def persona_for_text(text: str) -> tuple[BridgePersona, str, list[str]]:
    """Interpret free text as a bridge persona (for a partner or an opponent).

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
