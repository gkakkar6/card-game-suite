import pytest

from games.bridge.persona_selection import EXEMPT_FROM_FUZZY_MATCHING, FAMILIES, persona_for_text
from games.bridge.personas import AGGRESSIVE, BAITER, BASELINE, CONSERVATIVE, SELFISH, UNAWARE


def test_all_six_families_have_the_keyword_table_documented() -> None:
    # A real, editable data structure, not buried in logic - this pins its shape down.
    assert set(FAMILIES) == {
        "baseline", "aggressive", "conservative", "selfish", "baiter", "unaware",
    }
    for keywords, persona in FAMILIES.values():
        assert len(keywords) >= 1
        assert persona.name  # every family maps to a real, named persona


@pytest.mark.parametrize(
    ("text", "expected_persona", "expected_matches"),
    [
        ("plays like a real pro, tough and optimal", BASELINE, {"pro", "tough", "optimal"}),
        ("he's bold and aggressive, plays really wild", AGGRESSIVE, {"bold", "aggressive", "wild"}),
        ("a tight, careful, cautious partner", CONSERVATIVE, {"tight", "careful", "cautious"}),
        ("total glory hog, so selfish and greedy", SELFISH, {"glory", "hog", "selfish", "greedy"}),
        ("sneaky and tricky, a real cunning player", BAITER, {"sneaky", "tricky", "cunning"}),
        ("a total beginner, pretty clueless and naive", UNAWARE, {"beginner", "clueless", "naive"}),
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
    persona, name, matched = persona_for_text("someone who always leads their fourth highest")
    assert persona is BASELINE
    assert name == "baseline"
    assert matched == []  # empty, not just "resolved to baseline" - a real non-match


def test_multi_family_match_ties_break_by_family_order() -> None:
    # "aggressive but careful" matches both the aggressive family ("aggressive") and
    # the conservative family ("careful"). FAMILIES lists "aggressive" before
    # "conservative", so it wins - matched keywords come only from that family.
    persona, name, matched = persona_for_text("aggressive but careful")
    assert persona is AGGRESSIVE
    assert name == "aggressive"
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
    _, _, matched_one = persona_for_text("what a show-off, total glory hog")
    _, _, matched_two = persona_for_text("greedy and self-centered")
    assert set(matched_one) == {"show-off", "glory", "hog"}
    assert set(matched_two) == {"greedy", "self-centered"}
    assert matched_one != matched_two


def test_selfish_family_multi_word_keywords_match_correctly() -> None:
    persona, _, matched = persona_for_text("a real show-off, totally self-centered")
    assert persona is SELFISH
    assert set(matched) == {"show-off", "self-centered"}


# ---------------------------------------------------------------------------
# Typo tolerance: realistic typos should match, known collisions should not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected_persona", "expected_keyword"),
    [
        ("plays it pretty solidd, dependable", BASELINE, "solid"),
        ("he's agressive at the table", AGGRESSIVE, "aggressive"),
        ("very conservitive with the bidding", CONSERVATIVE, "conservative"),
        ("carefull and cautous, a real nit", CONSERVATIVE, "careful"),
        ("total greedu glory-seeker", SELFISH, "greedy"),
        ("a real sneaki player, watch out", BAITER, "sneaky"),
        ("still a biginner at this", UNAWARE, "beginner"),
    ],
)
def test_realistic_typos_still_match_the_intended_family(
    text: str, expected_persona: object, expected_keyword: str
) -> None:
    # "expert", "tough", "loose", "tight", "inexperienced", "tricky", "cunning",
    # "deceptive", "glory", "novice", "passive", "timid" and "simple" are deliberately
    # not exercised here - they're exempted (see EXEMPT_FROM_FUZZY_MATCHING), so a
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
        ("exert", "expert"), ("expect", "expert"), ("export", "expert"), ("exsert", "expert"),
        ("touch", "tough"), ("though", "tough"), ("dough", "tough"), ("cough", "tough"),
        ("rough", "tough"), ("bough", "tough"), ("trough", "tough"),
        ("canning", "cunning"), ("conning", "cunning"), ("gunning", "cunning"),
        ("running", "cunning"),
        ("receptive", "deceptive"), ("perceptive", "deceptive"), ("defective", "deceptive"),
        ("gory", "glory"),
        ("notice", "novice"),
        ("massive", "passive"),
        ("sample", "simple"),
        ("timed", "timid"),
    ],
)
def test_exempted_keywords_do_not_fuzzy_match_their_known_collision(
    collision_word: str, exempted_keyword: str
) -> None:
    assert exempted_keyword in EXEMPT_FROM_FUZZY_MATCHING
    persona, _, matched = persona_for_text(f"a real {collision_word} player")
    assert exempted_keyword not in matched


def test_experienced_does_not_fuzzy_match_inexperienced() -> None:
    # The MUST-fix case, called out on its own rather than only as one row in a
    # parametrized list: "experienced" is edit-distance 2 from "inexperienced" - not
    # just an unrelated word, but its literal semantic opposite. Describing a partner
    # as experienced is an ordinary thing to type; if it fuzzy-matched
    # "inexperienced" this would confidently recommend the beginner-level unaware
    # persona for someone described as the exact opposite of a beginner.
    persona, _, matched = persona_for_text("a really experienced player")
    assert "inexperienced" not in matched
    assert persona is not UNAWARE
    assert persona is BASELINE


def test_trick_and_tricks_do_not_fuzzy_match_tricky() -> None:
    # Bridge's own most fundamental piece of domain vocabulary - "he won every
    # trick" or "a tricky end position" (in the literal card-play sense) must not
    # spuriously fire the baiter persona.
    persona, _, matched = persona_for_text("he won every trick in that hand")
    assert "tricky" not in matched
    assert persona is BASELINE

    persona, _, matched = persona_for_text("counted all the tricks carefully")
    assert "tricky" not in matched
    assert persona is BASELINE


def test_short_keyword_never_fuzzy_matches_even_at_distance_one() -> None:
    # "hog" is 3 letters - tolerance 0 by length alone, before exemptions even apply.
    persona, _, matched = persona_for_text("a real hug at the table")  # "hug", not "hog"
    assert matched == []
    assert persona is BASELINE


def test_multi_word_phrase_keeps_requiring_an_exact_match_despite_typo_tolerance() -> None:
    # "show-off" typo'd (missing the "w") must not match, even though other
    # single-word keywords elsewhere in the table do tolerate typos. Note the
    # unhyphenated "show off" alone WOULD match - hyphens normalize to spaces the
    # same as the keyword itself, so that isn't a typo at all, just different
    # punctuation for the same phrase.
    persona, _, matched = persona_for_text("a real sho-off type of player")
    assert "show-off" not in matched
    assert persona is BASELINE


def test_exemption_list_matches_the_documented_decisions() -> None:
    # games/bridge/persona_selection.py's comment on EXEMPT_FROM_FUZZY_MATCHING is the
    # source of truth for why each of these is here (and why others considered were
    # deliberately left out); this pins the actual contents down so a silent edit
    # doesn't go unnoticed.
    assert EXEMPT_FROM_FUZZY_MATCHING == frozenset(
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
