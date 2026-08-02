import pytest

from games.poker.persona_selection import FAMILIES, persona_for_text
from games.poker.personas import BASELINE, BLUFFER, CALLING_STATION, CONSERVATIVE, ERRATIC


def test_all_five_families_have_the_keyword_table_documented() -> None:
    # A real, editable data structure, not buried in logic - this pins its shape down.
    assert set(FAMILIES) == {"aggressive", "tight", "calling", "random", "optimal"}
    for keywords, persona in FAMILIES.values():
        assert len(keywords) >= 1
        assert persona.name  # every family maps to a real, named persona


@pytest.mark.parametrize(
    ("text", "expected_persona", "expected_matches"),
    [
        ("he plays wild and loves to bluff", BLUFFER, {"wild", "bluff"}),
        ("a tight, patient, careful nit", CONSERVATIVE, {"tight", "patient", "careful", "nit"}),
        ("classic calling station, never folds anything", CALLING_STATION,
         {"calling station", "never folds"}),
        ("totally random, chaotic, unpredictable", ERRATIC, {"random", "chaotic", "unpredictable"}),
        ("plays like a real pro, tough and optimal", BASELINE, {"pro", "tough", "optimal"}),
    ],
)
def test_one_clearly_matching_phrase_per_family(
    text: str, expected_persona: object, expected_matches: set[str]
) -> None:
    persona, name, matched = persona_for_text(text)
    assert persona is expected_persona
    assert name == expected_persona.name
    assert set(matched) == expected_matches


def test_text_matching_nothing_falls_back_to_baseline() -> None:
    persona, name, matched = persona_for_text("someone who only plays when they have the nuts")
    assert persona is BASELINE
    assert name == "baseline"
    assert matched == []  # empty, not just "resolved to baseline" - a real non-match


def test_multi_family_match_ties_break_by_family_order() -> None:
    # "aggressive but careful" matches both the aggressive family ("aggressive") and
    # the tight family ("careful"). FAMILIES lists "aggressive" first, so it wins -
    # matched keywords come only from that family, not "careful" too.
    persona, name, matched = persona_for_text("aggressive but careful")
    assert persona is BLUFFER
    assert name == "bluffer"
    assert matched == ["aggressive"]
    assert "careful" not in matched


def test_tie_break_follows_families_dict_order_generally() -> None:
    # Confirms the rule itself (first family in FAMILIES order wins), not just the one
    # worked example above - swap which two families collide and the winner should
    # still be whichever is listed first.
    family_names = list(FAMILIES)
    for earlier, later in zip(family_names, family_names[1:], strict=False):
        earlier_keyword = FAMILIES[earlier][0][0]
        later_keyword = FAMILIES[later][0][0]
        _persona, _name, matched = persona_for_text(f"{later_keyword} {earlier_keyword}")
        assert matched[0] in FAMILIES[earlier][0], (
            f"expected a keyword from {earlier!r} to win over {later!r}"
        )


def test_matched_keywords_are_accurate_not_just_the_right_persona() -> None:
    # Two different phrases landing on the same persona should report different
    # matched keywords - the list has to reflect the actual input, not a canned one.
    _, _, matched_one = persona_for_text("what a maniac, total loose cannon")
    _, _, matched_two = persona_for_text("reckless and aggressive")
    assert set(matched_one) == {"maniac", "loose"}
    assert set(matched_two) == {"reckless", "aggressive"}
    assert matched_one != matched_two


def test_calling_family_multi_word_keywords_match_correctly() -> None:
    persona, _, matched = persona_for_text("a real sticky type who calls everything")
    assert persona is CALLING_STATION
    assert set(matched) == {"sticky", "calls everything"}


def test_known_ambiguity_loose_passive_collides_with_the_bare_loose_keyword() -> None:
    # A real gap in the keyword table, not a mechanism bug: "loose-passive" (calling
    # family) shares the word "loose" with the standalone "loose" keyword (aggressive
    # family, meant for "loose-aggressive"). "loose" alone is genuinely ambiguous in
    # real poker terms - loose-aggressive and loose-passive are opposite archetypes -
    # so this text resolves to bluffer rather than the calling station it actually
    # describes, because aggressive is listed before calling in FAMILIES. Recorded
    # here deliberately so a future change to the keyword table can see this was a
    # known, understood behaviour rather than accidentally fixing (or re-breaking) it.
    persona, _, matched = persona_for_text("a real loose-passive type")
    assert persona is BLUFFER
    assert matched == ["loose"]
