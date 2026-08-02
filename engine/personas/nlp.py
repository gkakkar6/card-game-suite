"""Free-text opponent selection: turn a loose description into a small set of tags a
game can map onto its own personas.

Rule-based, not an LLM call (ARCHITECTURE.md §3) - fully deterministic, testable with
plain unit tests, and needs no external dependency or API key to clone and run the
repo, which matters more given this is a public repo, not just a private tool.

Game-agnostic on purpose: this module only knows about keyword families (a family
name mapped to the keywords that trigger it) and plain text matching - nothing about
personas, temperature, bias, or any specific game's playing styles. Each game supplies
its own keyword-family mapping (e.g. games/poker/persona_selection.py is poker's) and
decides for itself what a matched tag, or several matched tags at once, should mean.

Matching tolerates typos on single words, via edit distance scaled by word length -
see TOLERANCE_BY_MIN_LENGTH. Multi-word keyword phrases ("calling station") always
require an exact match; fuzzing a whole phrase is out of scope. A caller-supplied
exemptions set forces specific single words to stay exact-match only regardless of
length - for keywords too close to an unrelated common word to safely fuzz (see
games/poker/persona_selection.py's EXEMPT_FROM_FUZZY_MATCHING for a worked example:
"tight" is edit-distance 1 from "eight").
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

KeywordFamilies = Mapping[str, Iterable[str]]

_WORD_RE = re.compile(r"[a-z0-9']+")

# Length -> allowed edit-distance tolerance for fuzzy single-word matching. Below the
# first threshold: no fuzzing at all - a short word has too few characters for a typo
# to be distinguishable from a genuinely different, unrelated word. Calibrated in
# games/poker/persona_selection.py against curated typo and near-miss word lists, the
# same way persona bias/temperature values are measured rather than guessed.
TOLERANCE_BY_MIN_LENGTH: tuple[tuple[int, int], ...] = (
    (8, 2),  # 8+ letters: tolerance 2
    (5, 1),  # 5-7 letters: tolerance 1
    (0, 0),  # under 5 letters: exact match only
)


@dataclass(frozen=True)
class OpponentIntent:
    """Tags matched in free text - one per keyword family whose keyword appeared.

    Deliberately just tags, not scores or confidence levels: rule-based matching is
    binary (a keyword either appears in the text or it doesn't), so anything richer
    would be manufacturing precision the mechanism doesn't actually have.
    """

    tags: frozenset[str] = field(default_factory=frozenset)


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance: the minimum number of single-character insertions,
    deletions, or substitutions needed to turn `a` into `b`."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current[j] = min(
                previous[j] + 1,  # delete a character from `a`
                current[j - 1] + 1,  # insert a character into `a`
                previous[j - 1] + cost,  # substitute (or match, if cost is 0)
            )
        previous = current
    return previous[len(b)]


def _tolerance_for_length(length: int) -> int:
    for min_length, tolerance in TOLERANCE_BY_MIN_LENGTH:
        if length >= min_length:
            return tolerance
    return 0  # unreachable: the table's last entry covers length 0 and up


def _normalize(text: str) -> str:
    """Lowercase and collapse to single-spaced word tokens, so matching is
    insensitive to punctuation, hyphens, and extra whitespace while still respecting
    word boundaries - "tight" should not match inside "tighten"."""
    return " ".join(_WORD_RE.findall(text.lower()))


def _exact_contains(normalized_text: str, normalized_keyword: str) -> bool:
    if not normalized_keyword:
        return False
    # Padding both sides makes this land on token boundaries rather than a raw
    # substring search, so multi-word keywords ("calling station") match correctly
    # and single-word ones don't fire on a word that merely contains them.
    return f" {normalized_keyword} " in f" {normalized_text} "


def _keyword_matches(normalized_text: str, keyword: str, exemptions: frozenset[str]) -> bool:
    """Does `keyword` match somewhere in `normalized_text`?

    A multi-word keyword requires an exact phrase match, always. A single word
    matches exactly, or - unless it's in `exemptions` - fuzzily, within the edit-
    distance tolerance its length allows.
    """
    normalized_keyword = _normalize(keyword)
    if not normalized_keyword:
        return False
    if " " in normalized_keyword:
        return _exact_contains(normalized_text, normalized_keyword)
    if _exact_contains(normalized_text, normalized_keyword):
        return True
    if normalized_keyword in exemptions:
        return False
    tolerance = _tolerance_for_length(len(normalized_keyword))
    if tolerance <= 0:
        return False
    return any(
        _edit_distance(normalized_keyword, token) <= tolerance
        for token in normalized_text.split()
    )


def matched_keywords(
    text: str, keywords: Iterable[str], exemptions: frozenset[str] = frozenset()
) -> list[str]:
    """Which of `keywords` actually appear in `text`, in the order given.

    The same matching rule `parse_intent` uses, exposed directly so a caller can
    report which specific words triggered a match - not just that a family did.
    `exemptions` are keywords (already lowercased, single words) to match exactly
    only, never fuzzily, regardless of length.
    """
    normalized = _normalize(text)
    return [keyword for keyword in keywords if _keyword_matches(normalized, keyword, exemptions)]


def parse_intent(
    text: str, keyword_families: KeywordFamilies, exemptions: frozenset[str] = frozenset()
) -> OpponentIntent:
    """Match `text` against `keyword_families`, tagging every family that fired.

    A family fires if any one of its keywords appears in `text`, exactly or (for
    single words not in `exemptions`) within a small edit-distance tolerance. Which
    family should "win" when more than one fires, and what to do about text that
    matches nothing, are the calling game's business, not this function's - it just
    reports everything that matched.
    """
    matched = {
        family
        for family, keywords in keyword_families.items()
        if matched_keywords(text, keywords, exemptions)
    }
    return OpponentIntent(tags=frozenset(matched))
