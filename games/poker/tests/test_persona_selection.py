import pytest

from games.poker.persona_selection import EXEMPT_FROM_FUZZY_MATCHING, FAMILIES, persona_for_text
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


# ---------------------------------------------------------------------------
# Typo tolerance: realistic typos should match, known collisions should not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected_persona", "expected_keyword"),
    [
        ("he's a real agressive player", BLUFFER, "aggressive"),
        ("very conservitive at the table", CONSERVATIVE, "conservative"),
        ("plays eratic, hard to read", ERRATIC, "erratic"),
        ("quite the manic at the table", BLUFFER, "maniac"),
        ("carefull and cautous, a real nit", CONSERVATIVE, "careful"),
    ],
)
def test_realistic_typos_still_match_the_intended_family(
    text: str, expected_persona: object, expected_keyword: str
) -> None:
    # "bluff", "expert", "sticky", "tough" and "unpredictable" are deliberately not
    # exercised here - they're exempted (see EXEMPT_FROM_FUZZY_MATCHING below), so a
    # typo of any of them correctly stops matching. That's the intended tradeoff, and
    # is exactly what test_exempted_keywords_do_not_fuzzy_match_their_known_collision
    # checks for.
    persona, _, matched = persona_for_text(text)
    assert persona is expected_persona
    assert expected_keyword in matched


@pytest.mark.parametrize(
    ("collision_word", "exempted_keyword"),
    [
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
    ],
)
def test_exempted_keywords_do_not_fuzzy_match_their_known_collision(
    collision_word: str, exempted_keyword: str
) -> None:
    assert exempted_keyword in EXEMPT_FROM_FUZZY_MATCHING
    persona, _, matched = persona_for_text(f"a real {collision_word} player")
    assert exempted_keyword not in matched
    # this specific text shouldn't accidentally trip a different family either
    assert persona is BASELINE
    assert matched == []


def test_unpredictable_does_not_fuzzy_match_predictable() -> None:
    # The MUST-fix case, called out on its own rather than only as one row in the
    # parametrized collision test above: "predictable" is edit-distance 2 from
    # "unpredictable" - not just an unrelated word, but its literal semantic
    # opposite. Describing an opponent as predictable is an ordinary thing to type;
    # if it fuzzy-matched "unpredictable" this would confidently recommend erratic
    # for someone described as the exact opposite of erratic.
    persona, _, matched = persona_for_text("he's honestly pretty predictable")
    assert "unpredictable" not in matched
    assert persona is not ERRATIC
    assert persona is BASELINE


def test_short_keyword_never_fuzzy_matches_even_at_distance_one() -> None:
    # "nit" is 3 letters - tolerance 0 by length alone, before exemptions even apply.
    persona, _, matched = persona_for_text("a real knit at the table")  # "knit", not "nit"
    assert matched == []
    assert persona is BASELINE


def test_multi_word_phrase_keeps_requiring_an_exact_match_despite_typo_tolerance() -> None:
    # "calling station" typo'd to "calling staton" must not match, even though other
    # single-word keywords elsewhere in the table do tolerate typos.
    persona, _, matched = persona_for_text("a calling staton type of player")
    assert "calling station" not in matched
    assert persona is BASELINE


def test_exemption_list_matches_the_documented_decisions() -> None:
    # games/poker/persona_selection.py's comment on EXEMPT_FROM_FUZZY_MATCHING is the
    # source of truth for why each of these is here (and why others considered were
    # deliberately left out); this pins the actual contents down so a silent edit
    # doesn't go unnoticed.
    assert EXEMPT_FROM_FUZZY_MATCHING == frozenset(
        {"loose", "tight", "tough", "unpredictable", "bluff", "expert", "sticky"}
    )
