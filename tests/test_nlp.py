from engine.personas.nlp import (
    OpponentIntent,
    _edit_distance,
    _tolerance_for_length,
    matched_keywords,
    parse_intent,
)

# A small made-up keyword-family map, deliberately not poker's real one (see
# games/poker/tests/test_persona_selection.py for that) - this file exercises the
# generic matching mechanism only, so it should not need to know poker exists.
FAMILIES = {
    "sunny": ("sunny", "bright", "clear skies"),
    "rainy": ("rain", "wet", "storm clouds"),
}


def test_matches_a_single_family() -> None:
    intent = parse_intent("it's a bright sunny day", FAMILIES)
    assert intent.tags == frozenset({"sunny"})


def test_matches_multiple_families_at_once() -> None:
    intent = parse_intent("bright morning, storm clouds by afternoon", FAMILIES)
    assert intent.tags == frozenset({"sunny", "rainy"})


def test_no_match_returns_empty_tags() -> None:
    intent = parse_intent("a perfectly ordinary Tuesday", FAMILIES)
    assert intent.tags == frozenset()
    assert intent == OpponentIntent()


def test_matching_is_case_insensitive() -> None:
    assert parse_intent("SUNNY", FAMILIES).tags == frozenset({"sunny"})
    assert parse_intent("Heavy RAIN expected", FAMILIES).tags == frozenset({"rainy"})


def test_respects_word_boundaries_not_substrings() -> None:
    # "rain" should not match inside "training" or "brainstorm".
    intent = parse_intent("a training session, some brainstorming", FAMILIES)
    assert intent.tags == frozenset()


def test_multi_word_keywords_match_as_phrases() -> None:
    intent = parse_intent("forecast says clear skies all week", FAMILIES)
    assert intent.tags == frozenset({"sunny"})


def test_multi_word_keywords_are_insensitive_to_punctuation_and_hyphens() -> None:
    intent = parse_intent("clear-skies, apparently", FAMILIES)
    assert intent.tags == frozenset({"sunny"})


def test_matched_keywords_lists_which_specific_words_fired() -> None:
    assert matched_keywords("sunny and bright", FAMILIES["sunny"]) == ["sunny", "bright"]
    assert matched_keywords("a perfectly ordinary Tuesday", FAMILIES["sunny"]) == []


def test_matched_keywords_preserves_input_order_not_match_order() -> None:
    # "bright" appears in the text before "sunny", but the keyword list is checked in
    # its own given order, not the order the words appear in the text.
    assert matched_keywords("bright and sunny", ("sunny", "bright")) == ["sunny", "bright"]


def test_empty_text_matches_nothing() -> None:
    assert parse_intent("", FAMILIES).tags == frozenset()


def test_empty_family_map_matches_nothing() -> None:
    assert parse_intent("sunny and bright", {}).tags == frozenset()


def test_opponent_intent_defaults_to_empty_tags() -> None:
    assert OpponentIntent().tags == frozenset()


# ---------------------------------------------------------------------------
# _edit_distance: standard reference cases before trusting it for anything else
# ---------------------------------------------------------------------------


def test_edit_distance_reference_cases() -> None:
    assert _edit_distance("kitten", "sitting") == 3  # the textbook example
    assert _edit_distance("flaw", "lawn") == 2
    assert _edit_distance("abc", "abc") == 0  # identical strings
    assert _edit_distance("", "") == 0
    assert _edit_distance("", "abc") == 3  # empty-string edges: pure insertion/deletion
    assert _edit_distance("abc", "") == 3


def test_edit_distance_is_symmetric() -> None:
    assert _edit_distance("kitten", "sitting") == _edit_distance("sitting", "kitten")


def test_edit_distance_single_edits() -> None:
    assert _edit_distance("cat", "cats") == 1  # insertion
    assert _edit_distance("cats", "cat") == 1  # deletion
    assert _edit_distance("cat", "bat") == 1  # substitution


def test_tolerance_scales_with_keyword_length() -> None:
    assert _tolerance_for_length(4) == 0  # under 5: exact only
    assert _tolerance_for_length(5) == 1
    assert _tolerance_for_length(7) == 1
    assert _tolerance_for_length(8) == 2
    assert _tolerance_for_length(20) == 2


# ---------------------------------------------------------------------------
# Fuzzy matching: typo tolerance, exemptions, and multi-word phrases staying exact
# ---------------------------------------------------------------------------


def test_a_single_word_keyword_matches_a_close_typo() -> None:
    # "bright" is 6 letters (tolerance 1); "brght" is missing one letter, distance 1.
    assert matched_keywords("a brght morning", ["bright"]) == ["bright"]


def test_a_typo_beyond_tolerance_does_not_match() -> None:
    # "brt" is distance 3 from "bright" - too far even for a 6-letter word.
    assert matched_keywords("a brt morning", ["bright"]) == []


def test_short_keyword_never_fuzzy_matches_even_at_distance_one() -> None:
    # "wet" is 3 letters - tolerance 0, exact only, however close a typo is.
    assert matched_keywords("a wot day", ["wet"]) == []
    assert matched_keywords("a wet day", ["wet"]) == ["wet"]  # exact still works


def test_exempted_keyword_ignores_tolerance_and_stays_exact() -> None:
    # "sunny" would normally tolerate distance 1 (5 letters), so "sunnu" would match -
    # unless "sunny" is exempted, in which case only an exact match counts.
    assert matched_keywords("a sunnu day", ["sunny"], exemptions=frozenset()) == ["sunny"]
    assert matched_keywords("a sunnu day", ["sunny"], exemptions=frozenset({"sunny"})) == []
    assert matched_keywords("a sunny day", ["sunny"], exemptions=frozenset({"sunny"})) == [
        "sunny"
    ]  # exact match still works even when exempted


def test_multi_word_phrase_never_fuzzes_even_one_word_off() -> None:
    # "storm clouds" typo'd to "storm cloud" (missing the s) should not match - multi-
    # word phrases require an exact match, full stop, regardless of tolerance.
    assert matched_keywords("storm cloud approaching", ["storm clouds"]) == []
    assert matched_keywords("storm clouds approaching", ["storm clouds"]) == ["storm clouds"]


def test_parse_intent_accepts_exemptions_and_applies_them_per_family() -> None:
    intent = parse_intent("a sunnu day", FAMILIES, exemptions=frozenset({"sunny"}))
    assert intent.tags == frozenset()  # "sunny" exempted, "sunnu" doesn't match exactly

    intent_unexempted = parse_intent("a sunnu day", FAMILIES)
    assert intent_unexempted.tags == frozenset({"sunny"})  # not exempted, fuzzes normally
