"""One-off diagnostic, not part of the test suite: is any single-word keyword in
poker's persona-selection FAMILIES table close enough (by edit distance, within its
length's fuzzy-matching tolerance) to an unrelated real English word to risk a
spurious match? And does the calibration - the tolerance table plus the resulting
exemption list - actually hold against curated typo and near-miss word lists?

Uses a real system dictionary rather than a small embedded word list, so the scan
covers actual English vocabulary rather than whatever a few hundred hand-picked words
happen to include. /usr/share/dict/american-english is the standard path on Debian/
Ubuntu; this machine (macOS) doesn't have that file, so this falls back to
/usr/share/dict/words - a symlink to the classic 1934 Webster's Second word list
(~236,000 entries, public domain) that ships with the OS. Either way: a real system
dictionary already on a normal dev machine, no new dependency to install.

Re-run this after editing FAMILIES, TOLERANCE_BY_MIN_LENGTH, or
EXEMPT_FROM_FUZZY_MATCHING, to confirm the calibration still holds.

    uv run python scripts/check_persona_keyword_fragility.py
"""

import os

from engine.personas.nlp import (
    TOLERANCE_BY_MIN_LENGTH,
    _edit_distance,
    _tolerance_for_length,
    matched_keywords,
)
from games.poker.persona_selection import EXEMPT_FROM_FUZZY_MATCHING, FAMILIES

# Checked in order; the first that exists is used. Both are the same underlying word
# list on systems that ship it (aspell/hunspell-derived on Linux, Webster's Second on
# BSD/macOS) - see the module docstring.
DICTIONARY_PATHS = ("/usr/share/dict/american-english", "/usr/share/dict/words")

# A word's length rarely differs from the dictionary word it's compared against by
# more than this and still be within tolerance, since edit distance is always at
# least abs(len(a) - len(b)). Used to skip the O(len(a) * len(b)) distance
# computation entirely for words nowhere near a keyword's length - the difference
# between a few seconds and several minutes over a ~200,000-word dictionary.
MAX_POSSIBLE_TOLERANCE = max(tolerance for _length, tolerance in TOLERANCE_BY_MIN_LENGTH)

MIN_WORD_LENGTH = 3
MAX_WORD_LENGTH = 20

# (typo text, the keyword it should still match) - realistic typos, at least a couple
# per family where the family still has a non-exempted single-word keyword to test.
# "bluff", "expert", "sticky", "tough" and "unpredictable" are deliberately absent
# here: they're exempted below, so their typo'd forms correctly stop matching -
# that's the intended tradeoff, not a regression to test against.
SHOULD_MATCH = [
    ("agressive", "aggressive"),
    ("manic", "maniac"),
    ("reckles", "reckless"),
    ("conservitive", "conservative"),
    ("carefull", "careful"),
    ("cautous", "cautious"),
    ("patiant", "patient"),
    ("eratic", "erratic"),
    ("chaoic", "chaotic"),
    ("wildcart", "wildcard"),
    ("optimel", "optimal"),
]

# (text, keyword it must NOT match) - the exemption-list collisions plus a general
# sanity check that keywords under 5 letters never fuzzy-match anything at all.
SHOULD_NOT_MATCH = [
    ("goose", "loose"), ("louse", "loose"), ("moose", "loose"), ("noose", "loose"),
    ("lose", "loose"),
    ("eight", "tight"), ("fight", "tight"), ("light", "tight"), ("might", "tight"),
    ("night", "tight"), ("right", "tight"), ("sight", "tight"), ("tights", "tight"),
    ("touch", "tough"), ("though", "tough"), ("dough", "tough"), ("cough", "tough"),
    ("rough", "tough"), ("bough", "tough"), ("trough", "tough"),
    ("predictable", "unpredictable"),
    ("buff", "bluff"),
    ("expect", "expert"),
    ("stocky", "sticky"),
    ("mild", "wild"), ("knit", "nit"), ("prod", "pro"),
    ("rest", "best"), ("gold", "good"), ("hare", "hard"),
]


def load_dictionary() -> tuple[str, list[str]]:
    for path in DICTIONARY_PATHS:
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="ignore") as handle:
                raw_words = [line.strip() for line in handle]
            filtered = {
                word
                for word in raw_words
                if MIN_WORD_LENGTH <= len(word) <= MAX_WORD_LENGTH
                and word.isalpha()
                and word.islower()
            }
            return path, sorted(filtered)
    raise FileNotFoundError(f"no system dictionary found (checked {DICTIONARY_PATHS})")


def single_word_keywords() -> list[str]:
    seen: set[str] = set()
    for keywords, _persona in FAMILIES.values():
        for keyword in keywords:
            if " " in keyword or "-" in keyword:
                continue  # multi-word phrase: never fuzzy-matched, not at risk
            seen.add(keyword)
    return sorted(seen)


def scan_for_fragile_keywords() -> dict[str, list[tuple[str, int]]]:
    path, words = load_dictionary()
    keywords = single_word_keywords()
    print(f"dictionary: {path} ({len(words):,} words after filtering)")
    print(f"scanning {len(keywords)} single-word keywords: {keywords}\n")

    by_length: dict[int, list[str]] = {}
    for word in words:
        by_length.setdefault(len(word), []).append(word)

    flagged: dict[str, list[tuple[str, int]]] = {}
    for keyword in keywords:
        tolerance = _tolerance_for_length(len(keyword))
        candidate_lengths = range(
            len(keyword) - MAX_POSSIBLE_TOLERANCE, len(keyword) + MAX_POSSIBLE_TOLERANCE + 1
        )
        candidates = [word for length in candidate_lengths for word in by_length.get(length, [])]
        collisions = [
            (word, _edit_distance(keyword, word))
            for word in candidates
            if word != keyword and _edit_distance(keyword, word) <= tolerance
        ]
        if collisions:
            flagged[keyword] = sorted(collisions, key=lambda item: (item[1], item[0]))

    for keyword, collisions in flagged.items():
        tolerance = _tolerance_for_length(len(keyword))
        collision_text = ", ".join(f"{w} (d={d})" for w, d in collisions)
        exempt = " [EXEMPTED]" if keyword in EXEMPT_FROM_FUZZY_MATCHING else ""
        print(f"  {keyword!r} (len={len(keyword)}, tol={tolerance}){exempt}: {collision_text}")

    print(f"\n{len(flagged)} of {len(keywords)} single-word keywords flagged")
    return flagged


def check_calibration() -> bool:
    ok = True
    print("\n=== positive: realistic typos that should match ===")
    for typo, keyword in SHOULD_MATCH:
        matched = matched_keywords(typo, [keyword], EXEMPT_FROM_FUZZY_MATCHING)
        if not matched:
            ok = False
        print(f"  {'ok  ' if matched else 'FAIL'} {typo!r} -> {keyword!r}: matched={matched}")

    print("\n=== negative: near-miss real words that must NOT match ===")
    for text, keyword in SHOULD_NOT_MATCH:
        matched = matched_keywords(text, [keyword], EXEMPT_FROM_FUZZY_MATCHING)
        if matched:
            ok = False
        print(f"  {'ok  ' if not matched else 'FAIL'} {text!r} vs {keyword!r}: matched={matched}")

    print("\n=== sanity: a multi-word phrase never fuzzes, even with one word off ===")
    phrase_typo = matched_keywords(
        "calling staton", ["calling station"], EXEMPT_FROM_FUZZY_MATCHING
    )
    if phrase_typo:
        ok = False
    print(f"  {'ok  ' if not phrase_typo else 'FAIL'} 'calling staton': matched={phrase_typo}")

    return ok


def main() -> None:
    scan_for_fragile_keywords()
    calibration_ok = check_calibration()
    print(f"\n{'CALIBRATION OK' if calibration_ok else 'CALIBRATION FAILED'}")


if __name__ == "__main__":
    main()
