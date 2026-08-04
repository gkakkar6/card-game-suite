from games.bridge.hand_evaluation import hcp, is_balanced, longest_suits, suit_lengths
from games.bridge.tests.test_rules import card


def test_hcp_counts_the_four_honors_and_nothing_else() -> None:
    hand = [card("AS"), card("KH"), card("QD"), card("JC"), card("TS"), card("2H")]
    assert hcp(hand) == 4 + 3 + 2 + 1  # ten and two count for nothing


def test_hcp_of_an_empty_hand_is_zero() -> None:
    assert hcp([]) == 0


def test_suit_lengths_counts_every_suit_including_ones_held_zero_times() -> None:
    hand = [card("AS"), card("2S"), card("3S"), card("KH")]
    lengths = suit_lengths(hand)
    assert lengths[card("AS").suit] == 3
    assert lengths[card("KH").suit] == 1
    assert lengths[card("2D").suit] == 0
    assert lengths[card("2C").suit] == 0


def test_longest_suits_returns_every_suit_tied_for_longest() -> None:
    hand = [
        card("AS"), card("2S"), card("3S"), card("4S"), card("5S"),  # 5 spades
        card("AH"), card("2H"), card("3H"), card("4H"), card("5H"),  # 5 hearts, tied
        card("2D"), card("3D"), card("2C"),
    ]
    result = longest_suits(hand)
    assert set(result) == {card("AS").suit, card("AH").suit}


def test_longest_suits_returns_a_single_entry_when_there_is_no_tie() -> None:
    hand = [
        card("AS"), card("2S"), card("3S"), card("4S"), card("5S"), card("6S"),
        card("AH"), card("2H"), card("2D"), card("3D"), card("2C"), card("3C"), card("4C"),
    ]
    assert longest_suits(hand) == [card("AS").suit]


def test_is_balanced_accepts_the_three_standard_shapes() -> None:
    four_three_three_three = [
        card("AS"), card("2S"), card("3S"), card("4S"),
        card("AH"), card("2H"), card("3H"),
        card("AD"), card("2D"), card("3D"),
        card("AC"), card("2C"), card("3C"),
    ]
    four_four_three_two = [
        card("AS"), card("2S"), card("3S"), card("4S"),
        card("AH"), card("2H"), card("3H"), card("4H"),
        card("AD"), card("2D"), card("3D"),
        card("AC"), card("2C"),
    ]
    five_three_three_two = [
        card("AS"), card("2S"), card("3S"), card("4S"), card("5S"),
        card("AH"), card("2H"), card("3H"),
        card("AD"), card("2D"), card("3D"),
        card("AC"), card("2C"),
    ]
    assert is_balanced(four_three_three_three)
    assert is_balanced(four_four_three_two)
    assert is_balanced(five_three_three_two)


def test_is_balanced_rejects_a_void_or_a_singleton() -> None:
    with_a_void = [
        card("AS"), card("2S"), card("3S"), card("4S"), card("5S"), card("6S"),
        card("AH"), card("2H"), card("3H"), card("4H"),
        card("AD"), card("2D"), card("3D"),
    ]
    with_a_singleton = [
        card("AS"), card("2S"), card("3S"), card("4S"), card("5S"),
        card("AH"), card("2H"), card("3H"), card("4H"),
        card("AD"), card("2D"), card("3D"),
        card("AC"),
    ]
    assert not is_balanced(with_a_void)
    assert not is_balanced(with_a_singleton)


def test_is_balanced_rejects_two_doubletons_even_with_no_short_suit() -> None:
    # 5-4-2-2 has no void or singleton, but two doubletons is one too many - a real,
    # easy-to-get-wrong case: it looks balanced at a glance but standard SAYC excludes it.
    five_four_two_two = [
        card("AS"), card("2S"), card("3S"), card("4S"), card("5S"),
        card("AH"), card("2H"), card("3H"), card("4H"),
        card("AD"), card("2D"),
        card("AC"), card("2C"),
    ]
    assert not is_balanced(five_four_two_two)
