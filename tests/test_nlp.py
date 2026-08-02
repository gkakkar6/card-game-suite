from engine.personas.nlp import OpponentIntent, matched_keywords, parse_intent

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
