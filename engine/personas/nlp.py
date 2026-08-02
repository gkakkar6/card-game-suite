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
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

KeywordFamilies = Mapping[str, Iterable[str]]

_WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class OpponentIntent:
    """Tags matched in free text - one per keyword family whose keyword appeared.

    Deliberately just tags, not scores or confidence levels: rule-based matching is
    binary (a keyword either appears in the text or it doesn't), so anything richer
    would be manufacturing precision the mechanism doesn't actually have.
    """

    tags: frozenset[str] = field(default_factory=frozenset)


def _normalize(text: str) -> str:
    """Lowercase and collapse to single-spaced word tokens, so matching is
    insensitive to punctuation, hyphens, and extra whitespace while still respecting
    word boundaries - "tight" should not match inside "tighten"."""
    return " ".join(_WORD_RE.findall(text.lower()))


def _contains(normalized_text: str, keyword: str) -> bool:
    normalized_keyword = _normalize(keyword)
    if not normalized_keyword:
        return False
    # Padding both sides makes this land on token boundaries rather than a raw
    # substring search, so multi-word keywords ("calling station") match correctly
    # and single-word ones don't fire on a word that merely contains them.
    return f" {normalized_keyword} " in f" {normalized_text} "


def matched_keywords(text: str, keywords: Iterable[str]) -> list[str]:
    """Which of `keywords` actually appear in `text`, in the order given.

    The same matching rule `parse_intent` uses, exposed directly so a caller can
    report which specific words triggered a match - not just that a family did.
    """
    normalized = _normalize(text)
    return [keyword for keyword in keywords if _contains(normalized, keyword)]


def parse_intent(text: str, keyword_families: KeywordFamilies) -> OpponentIntent:
    """Match `text` against `keyword_families`, tagging every family that fired.

    A family fires if any one of its keywords appears in `text`; keywords may be
    single words ("tight") or short phrases ("calling station", "never folds"). Which
    family should "win" when more than one fires, and what to do about text that
    matches nothing, are the calling game's business, not this function's - it just
    reports everything that matched.
    """
    matched = {
        family for family, keywords in keyword_families.items() if matched_keywords(text, keywords)
    }
    return OpponentIntent(tags=frozenset(matched))
