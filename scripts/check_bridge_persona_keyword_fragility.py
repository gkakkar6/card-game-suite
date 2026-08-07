"""One-off diagnostic, not part of the test suite: is any single-word keyword in
bridge's persona-selection FAMILIES table close enough (by edit distance, within its
length's fuzzy-matching tolerance) to an unrelated real English word to risk a
spurious match? And does the calibration - the tolerance table plus the resulting
exemption list - actually hold against curated typo and near-miss word lists?

Same discipline, same dictionary, same structure as
scripts/check_persona_keyword_fragility.py (poker's own scan) - see that file's
docstring for why a real system dictionary rather than a small embedded list. This is
a genuinely fresh scan against bridge's own keyword table, not an assumption that
poker's exemption list transfers - only "loose" and "tight" (shared, unchanged words)
carry an exemption forward from poker's own findings.

Re-run this after editing FAMILIES, TOLERANCE_BY_MIN_LENGTH, or
EXEMPT_FROM_FUZZY_MATCHING, to confirm the calibration still holds.

    uv run python scripts/check_bridge_persona_keyword_fragility.py
"""

import os

from engine.personas.nlp import (
    TOLERANCE_BY_MIN_LENGTH,
    _edit_distance,
    _tolerance_for_length,
    matched_keywords,
)
from games.bridge.persona_selection import EXEMPT_FROM_FUZZY_MATCHING, FAMILIES

# Checked in order; the first that exists is used. Both are the same underlying word
# list on systems that ship it (aspell/hunspell-derived on Linux, Webster's Second on
# BSD/macOS) - see poker's scan script for detail.
DICTIONARY_PATHS = ("/usr/share/dict/american-english", "/usr/share/dict/words")

MAX_POSSIBLE_TOLERANCE = max(tolerance for _length, tolerance in TOLERANCE_BY_MIN_LENGTH)

MIN_WORD_LENGTH = 3
MAX_WORD_LENGTH = 20

# (typo text, the keyword it should still match) - realistic typos, at least one per
# family where the family still has a non-exempted single-word keyword to test.
# "expert", "tough", "inexperienced", "tricky", "cunning", "deceptive", "glory",
# "novice", "passive", "simple", "timid", "loose" and "tight" are deliberately absent
# here: they're exempted below, so their typo'd forms correctly stop matching - that's
# the intended tradeoff, not a regression to test against.
SHOULD_MATCH = [
    ("optimel", "optimal"),
    ("solidd", "solid"),
    ("agressive", "aggressive"),
    ("pushi", "pushy"),
    ("conservitive", "conservative"),
    ("carefull", "careful"),
    ("cautous", "cautious"),
    ("selfsh", "selfish"),
    ("greedu", "greedy"),
    ("indivdual", "individual"),
    ("sneaki", "sneaky"),
    ("misleadin", "misleading"),
    ("biginner", "beginner"),
    ("clueles", "clueless"),
    ("naiv", "naive"),
    ("basik", "basic"),
]

# (text, keyword it must NOT match) - the exemption-list collisions.
SHOULD_NOT_MATCH = [
    ("goose", "loose"), ("louse", "loose"), ("moose", "loose"), ("noose", "loose"),
    ("lose", "loose"),
    ("eight", "tight"), ("fight", "tight"), ("light", "tight"), ("might", "tight"),
    ("night", "tight"), ("right", "tight"), ("sight", "tight"), ("tights", "tight"),
    ("exert", "expert"), ("expect", "expert"), ("export", "expert"), ("exsert", "expert"),
    ("touch", "tough"), ("though", "tough"), ("dough", "tough"), ("cough", "tough"),
    ("rough", "tough"), ("bough", "tough"), ("trough", "tough"),
    ("experienced", "inexperienced"),
    ("trick", "tricky"), ("tricks", "tricky"),
    ("canning", "cunning"), ("conning", "cunning"), ("gunning", "cunning"),
    ("running", "cunning"),
    ("receptive", "deceptive"), ("perceptive", "deceptive"), ("defective", "deceptive"),
    ("gory", "glory"),
    ("notice", "novice"),
    ("massive", "passive"),
    ("sample", "simple"),
    ("timed", "timid"),
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

    return ok


def main() -> None:
    scan_for_fragile_keywords()
    calibration_ok = check_calibration()
    print(f"\n{'CALIBRATION OK' if calibration_ok else 'CALIBRATION FAILED'}")


if __name__ == "__main__":
    main()
