from engine.cards import Suit
from games.bridge.bidding import Bid, bid_values, choose_bid, new_auction
from games.bridge.tests.test_rules import card

# ---------------------------------------------------------------------------
# Bid legality
# ---------------------------------------------------------------------------


def test_suit_ordering_matches_engine_cards_suit_exactly() -> None:
    # The whole "level dominates, same-level rank is C<D<H<S<NT" scheme only works if
    # it lines up with engine.cards.Suit's own ordering - checked directly here rather
    # than assumed, since a silent mismatch would make every same-level comparison
    # below subtly wrong without any single test pinpointing why.
    assert list(Suit) == [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]
    ranks = [
        Bid(1, Suit.CLUBS).rank,
        Bid(1, Suit.DIAMONDS).rank,
        Bid(1, Suit.HEARTS).rank,
        Bid(1, Suit.SPADES).rank,
        Bid(1, None).rank,
    ]
    assert ranks == sorted(ranks)


def test_level_dominates_suit_rank_entirely() -> None:
    # Any higher-level bid outranks every lower-level bid, however low its own strain
    # ranks - 2C beats 1NT, even though clubs is the lowest strain of all.
    assert Bid(2, Suit.CLUBS).rank > Bid(1, None).rank
    assert Bid(2, Suit.CLUBS).rank > Bid(1, Suit.SPADES).rank


def test_legal_calls_include_pass_and_only_bids_that_outrank_the_current_high() -> None:
    auction = new_auction(dealer=0).apply(Bid(1, Suit.HEARTS))
    legal = auction.legal_calls()

    assert None in legal  # Pass is always legal
    assert Bid(1, Suit.SPADES) in legal  # same level, higher strain
    assert Bid(1, None) in legal  # same level, notrump
    assert Bid(2, Suit.CLUBS) in legal  # higher level, lower strain
    assert Bid(1, Suit.HEARTS) not in legal  # the bid just made
    assert Bid(1, Suit.DIAMONDS) not in legal  # same level, lower strain


def test_apply_rejects_an_illegal_call() -> None:
    auction = new_auction(dealer=0).apply(Bid(1, Suit.SPADES))
    try:
        auction.apply(Bid(1, Suit.HEARTS))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_bid_level_must_be_one_through_seven() -> None:
    try:
        Bid(0, Suit.CLUBS)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        Bid(8, Suit.CLUBS)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Auction termination
# ---------------------------------------------------------------------------


def test_three_passes_after_a_bid_ends_the_auction() -> None:
    auction = new_auction(dealer=0)
    for call in (Bid(1, Suit.SPADES), None, None):
        auction = auction.apply(call)
        assert not auction.is_over()
    auction = auction.apply(None)
    assert auction.is_over()
    assert not auction.is_passed_out()


def test_four_passes_with_no_bid_is_a_distinct_passed_out_outcome() -> None:
    auction = new_auction(dealer=1)
    for call in (None, None, None):
        auction = auction.apply(call)
        assert not auction.is_over()
    auction = auction.apply(None)
    assert auction.is_over()
    assert auction.is_passed_out()
    assert auction.contract() is None


def test_legal_calls_is_empty_once_the_auction_is_over() -> None:
    auction = new_auction(dealer=0)
    for _ in range(4):
        auction = auction.apply(None)
    assert auction.legal_calls() == []


def test_contract_raises_if_the_auction_is_not_over_yet() -> None:
    auction = new_auction(dealer=0).apply(Bid(1, Suit.SPADES))
    try:
        auction.contract()
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Declarer determination
# ---------------------------------------------------------------------------


def test_the_final_contract_is_correct_for_a_simple_auction() -> None:
    auction = new_auction(dealer=0)
    for call in (Bid(1, Suit.SPADES), None, Bid(2, Suit.SPADES), None, None, None):
        auction = auction.apply(call)
    contract = auction.contract()
    assert contract is not None
    assert contract.trump == Suit.SPADES
    assert contract.declarer == 0
    assert contract.target == 8  # level + 6


def test_declarer_is_whoever_first_named_the_strain_not_the_final_bidder() -> None:
    # Seat 0 opens 1C, seat 2 (partner) responds 1S - the FIRST spade bid. Seat 0 then
    # raises to 2S, supporting but not re-introducing the suit. The auction's last bid
    # is 2S by seat 0, but declarer must be seat 2, who named spades first.
    auction = new_auction(dealer=0)
    for call in (
        Bid(1, Suit.CLUBS), None, Bid(1, Suit.SPADES), None,
        Bid(2, Suit.SPADES), None, None, None,
    ):
        auction = auction.apply(call)
    assert auction.last_bidder == 0  # confirms this genuinely isn't the trivial case
    contract = auction.contract()
    assert contract is not None
    assert contract.declarer == 2


# ---------------------------------------------------------------------------
# bid_values(): shape
# ---------------------------------------------------------------------------


def test_bid_values_scores_every_legal_call_not_just_the_winner() -> None:
    hand = [
        card("AS"), card("KS"), card("QS"), card("2S"), card("3S"), card("4S"),
        card("KH"), card("2H"),
        card("2D"), card("3D"),
        card("2C"), card("3C"), card("4C"),
    ]
    auction = new_auction(dealer=0)
    values = bid_values(hand, auction)

    assert set(values) == set(auction.legal_calls())
    assert len(values) > 2  # a real dict of many scored calls, not just one chosen bid
    assert values[None] > 0  # Pass always gets a real score, even when it loses
    assert values[Bid(1, Suit.SPADES)] > values[None]  # the textbook opening still wins


# ---------------------------------------------------------------------------
# Opening bids - the sourced tie-break rules
#
# Sourced from the ACBL SAYC System Booklet (Revised January 2006): "Open the higher
# of long suits of equal length: 5-5 or 6-6"; "Normally open 1D with 4-4 in the
# minors"; "Normally open 1C with 3-3 in the minors" - the same discipline as the NLP
# keyword matching, checked against a real reference rather than assumed.
# ---------------------------------------------------------------------------


def test_opening_suit_with_two_five_card_suits_tied_opens_the_higher() -> None:
    hand = [
        card("AS"), card("2S"), card("3S"), card("4S"), card("5S"),
        card("AH"), card("2H"), card("3H"), card("4H"), card("5H"),
        card("KD"), card("QD"), card("2C"),
    ]
    assert choose_bid(hand, new_auction(dealer=0)) == Bid(1, Suit.SPADES)


def test_opening_suit_with_four_four_in_the_minors_opens_diamonds() -> None:
    # Kept safely below the 1NT range (12, not 15-17) even though the shape is
    # balanced - otherwise this would collide with the "1NT preferred" rule and test
    # the wrong thing entirely.
    hand = [
        card("2S"), card("3S"), card("QS"),
        card("2H"), card("3H"),
        card("AD"), card("KD"), card("QD"), card("2D"),
        card("2C"), card("3C"), card("4C"), card("JC"),
    ]
    assert choose_bid(hand, new_auction(dealer=0)) == Bid(1, Suit.DIAMONDS)


def test_opening_suit_with_three_three_in_the_minors_as_longest_opens_clubs() -> None:
    # "Longest" here means longest among suits actually eligible to open - this hand
    # also holds a 4-card spade suit, but it doesn't qualify (majors need 5+), so the
    # 3-3 minor tie is what decides it. A literal "3-3-3-3 with nothing else" shape is
    # impossible in a 13-card hand (some suit always reaches 4), so this is the real,
    # reachable version of the rule, not a simplified stand-in for it.
    hand = [
        card("2S"), card("3S"), card("4S"), card("5S"),
        card("2H"), card("3H"), card("4H"),
        card("AD"), card("KD"), card("QD"),
        card("AC"), card("KC"), card("QC"),
    ]
    assert choose_bid(hand, new_auction(dealer=0)) == Bid(1, Suit.CLUBS)


def test_a_four_card_major_alone_is_not_eligible_to_open() -> None:
    # Five-card majors: a bare 4-card major with nothing else that long does not
    # qualify to be opened - the hand must open its longest eligible suit instead.
    # Diamonds is the unique longest eligible suit here (4, against clubs' 3) - not
    # a minor tie, which is already covered by its own test above.
    hand = [
        card("AS"), card("KS"), card("2S"), card("3S"),  # 4 spades, not enough
        card("2H"), card("3H"),
        card("AD"), card("KD"), card("2D"), card("3D"),
        card("2C"), card("3C"), card("4C"),
    ]
    assert choose_bid(hand, new_auction(dealer=0)) == Bid(1, Suit.DIAMONDS)


# ---------------------------------------------------------------------------
# Opening bids - textbook examples, worked out by hand
# ---------------------------------------------------------------------------


def test_opens_one_of_a_suit_with_a_minimum_opening_hand() -> None:
    # 14 HCP, 6-card spade suit - safely unbalanced, no notrump competes for it.
    hand = [
        card("AS"), card("KS"), card("QS"), card("2S"), card("3S"), card("4S"),
        card("KH"), card("2H"),
        card("QD"), card("2D"), card("3D"),
        card("2C"), card("3C"),
    ]
    assert choose_bid(hand, new_auction(dealer=0)) == Bid(1, Suit.SPADES)


def test_opens_1nt_with_15_to_17_balanced() -> None:
    hand = [
        card("AS"), card("KS"), card("2S"), card("5S"),
        card("AH"), card("2H"), card("3H"),
        card("KD"), card("2D"), card("7D"),
        card("QC"), card("2C"), card("3C"),
    ]
    assert choose_bid(hand, new_auction(dealer=0)) == Bid(1, None)


def test_opens_2nt_with_20_to_21_balanced() -> None:
    hand = [
        card("AS"), card("KS"), card("QS"), card("JS"),
        card("AH"), card("KH"), card("QH"),
        card("JD"), card("2D"), card("3D"),
        card("2C"), card("3C"), card("4C"),
    ]
    assert choose_bid(hand, new_auction(dealer=0)) == Bid(2, None)


def test_1nt_is_preferred_over_a_same_ranged_suit_opening() -> None:
    # 5-3-3-2 shape with a 5-card major, 15-17 HCP: real SAYC practice opens 1NT here,
    # not the suit - notrump openings "may be made with a five-card major" per the
    # source, meaning notrump itself is the preferred bid, not merely an option.
    hand = [
        card("AS"), card("KS"), card("QS"), card("2S"), card("3S"),
        card("AH"), card("2H"), card("3H"),
        card("KD"), card("2D"), card("3D"),
        card("2C"), card("3C"),
    ]
    assert choose_bid(hand, new_auction(dealer=0)) == Bid(1, None)


def test_opens_the_strong_two_club_with_22_or_more_points() -> None:
    hand = [
        card("AS"), card("KS"), card("QS"), card("JS"),
        card("AH"), card("KH"), card("QH"),
        card("AD"), card("KD"), card("2D"),
        card("AC"), card("2C"), card("3C"),
    ]
    assert choose_bid(hand, new_auction(dealer=0)) == Bid(2, Suit.CLUBS)


def test_passes_below_opening_strength() -> None:
    hand = [
        card("2S"), card("3S"), card("4S"), card("5S"),
        card("2H"), card("3H"), card("4H"),
        card("2D"), card("3D"), card("4D"),
        card("2C"), card("3C"), card("4C"),
    ]
    assert choose_bid(hand, new_auction(dealer=0)) is None


# ---------------------------------------------------------------------------
# Responses - textbook examples
# ---------------------------------------------------------------------------


def test_responder_raises_opener_with_support_and_minimum_values() -> None:
    responder = [
        card("JS"), card("2S"), card("3S"), card("4S"),
        card("AH"), card("2H"), card("3H"),
        card("KD"), card("2D"), card("3D"),
        card("2C"), card("3C"), card("4C"),
    ]
    auction = new_auction(dealer=0).apply(Bid(1, Suit.SPADES)).apply(None)
    assert choose_bid(responder, auction) == Bid(2, Suit.SPADES)


def test_responder_shows_a_new_suit_before_raising_or_bidding_1nt() -> None:
    # Real SAYC priority: "1S = at least four spades, 6 or more points. Tends to deny
    # a heart fit" outranks "1NT = 6-9 points, denies four spades" for a hand that
    # actually holds four spades - showing shape comes first.
    responder = [
        card("JS"), card("QS"), card("2S"), card("3S"),
        card("2H"), card("3H"),
        card("KD"), card("2D"), card("3D"),
        card("KC"), card("2C"), card("3C"), card("4C"),
    ]
    auction = new_auction(dealer=0).apply(Bid(1, Suit.HEARTS)).apply(None)
    assert choose_bid(responder, auction) == Bid(1, Suit.SPADES)

    # Checked by score, not just by outcome: 1S must genuinely outrank 1NT here, not
    # merely happen to win a tie because of which order legal_calls() lists them in -
    # a real gap found during mutation testing (see DECISIONS.md).
    values = bid_values(responder, auction)
    assert values[Bid(1, Suit.SPADES)] > values[Bid(1, None)]


def test_responder_bids_1nt_when_nothing_else_fits() -> None:
    responder = [
        card("2S"), card("3S"),
        card("JH"), card("QH"), card("2H"), card("3H"),
        card("KD"), card("2D"), card("3D"),
        card("KC"), card("2C"), card("3C"),
    ]
    auction = new_auction(dealer=0).apply(Bid(1, Suit.SPADES)).apply(None)
    assert choose_bid(responder, auction) == Bid(1, None)


def test_responder_passes_below_response_strength() -> None:
    responder = [
        card("2S"), card("3S"), card("4S"), card("5S"),
        card("2H"), card("3H"), card("4H"),
        card("2D"), card("3D"), card("4D"),
        card("2C"), card("3C"), card("4C"),
    ]
    auction = new_auction(dealer=0).apply(Bid(1, Suit.SPADES)).apply(None)
    assert choose_bid(responder, auction) is None


def test_natural_1nt_response_signs_off_in_a_weak_five_card_suit() -> None:
    responder = [
        card("2S"), card("3S"),
        card("AH"), card("2H"), card("3H"), card("4H"), card("5H"),
        card("2D"), card("3D"), card("4D"),
        card("2C"), card("3C"), card("4C"),
    ]
    auction = new_auction(dealer=0).apply(Bid(1, None)).apply(None)
    assert choose_bid(responder, auction) == Bid(2, Suit.HEARTS)


def test_natural_1nt_response_invites_with_8_to_9() -> None:
    responder = [
        card("AS"), card("2S"), card("3S"), card("4S"),
        card("QH"), card("2H"), card("3H"),
        card("QD"), card("2D"), card("3D"),
        card("2C"), card("3C"), card("4C"),
    ]
    auction = new_auction(dealer=0).apply(Bid(1, None)).apply(None)
    assert choose_bid(responder, auction) == Bid(2, None)


def test_natural_1nt_response_bids_game_with_10_or_more() -> None:
    responder = [
        card("AS"), card("KS"), card("QS"),
        card("AH"), card("2H"), card("3H"),
        card("JD"), card("2D"), card("3D"),
        card("2C"), card("3C"), card("4C"),
    ]
    auction = new_auction(dealer=0).apply(Bid(1, None)).apply(None)
    assert choose_bid(responder, auction) == Bid(3, None)


# ---------------------------------------------------------------------------
# Opener's rebid - textbook examples
# ---------------------------------------------------------------------------


def test_opener_passes_a_raise_with_a_minimum_hand() -> None:
    opener = [
        card("AS"), card("KS"), card("QS"), card("2S"), card("3S"), card("4S"),
        card("KH"), card("2H"),
        card("QD"), card("2D"), card("3D"),
        card("2C"), card("3C"),
    ]
    auction = new_auction(dealer=0)
    for call in (Bid(1, Suit.SPADES), None, Bid(2, Suit.SPADES), None):
        auction = auction.apply(call)
    assert choose_bid(opener, auction) is None


def test_opener_bids_on_over_a_raise_with_extra_strength() -> None:
    opener = [
        card("AS"), card("KS"), card("QS"), card("2S"), card("3S"), card("4S"),
        card("AH"), card("KH"),
        card("QD"), card("2D"), card("3D"),
        card("2C"), card("3C"),
    ]
    auction = new_auction(dealer=0)
    for call in (Bid(1, Suit.SPADES), None, Bid(2, Suit.SPADES), None):
        auction = auction.apply(call)
    assert choose_bid(opener, auction) == Bid(3, Suit.SPADES)


def test_opener_rebids_a_genuinely_long_own_suit_over_a_1nt_response() -> None:
    opener = [
        card("AS"), card("KS"), card("QS"), card("2S"), card("3S"), card("4S"),
        card("KH"), card("2H"),
        card("2D"), card("3D"),
        card("2C"), card("3C"), card("4C"),
    ]
    auction = new_auction(dealer=0)
    for call in (Bid(1, Suit.SPADES), None, Bid(1, None), None):
        auction = auction.apply(call)
    assert choose_bid(opener, auction) == Bid(2, Suit.SPADES)


def test_opener_raises_partners_new_suit_with_support() -> None:
    opener = [
        card("AS"), card("KS"), card("2S"), card("3S"),
        card("2H"), card("3H"),
        card("AD"), card("KD"), card("QD"), card("2D"), card("3D"),
        card("2C"), card("3C"),
    ]
    auction = new_auction(dealer=0)
    for call in (Bid(1, Suit.DIAMONDS), None, Bid(1, Suit.SPADES), None):
        auction = auction.apply(call)
    assert choose_bid(opener, auction) == Bid(2, Suit.SPADES)


def test_opener_rebids_notrump_with_a_balanced_hand_and_no_fit() -> None:
    opener = [
        card("2S"), card("3S"),
        card("KH"), card("2H"), card("3H"),
        card("AD"), card("KD"), card("2D"), card("3D"),
        card("KC"), card("2C"), card("3C"), card("4C"),
    ]
    auction = new_auction(dealer=0)
    for call in (Bid(1, Suit.DIAMONDS), None, Bid(1, Suit.SPADES), None):
        auction = auction.apply(call)
    assert choose_bid(opener, auction) == Bid(1, None)


# ---------------------------------------------------------------------------
# Out of scope, handled honestly rather than guessed at
# ---------------------------------------------------------------------------


def test_rebidding_after_a_notrump_opening_falls_back_rather_than_guesses() -> None:
    # No Stayman or Jacoby transfers - a notrump opener's second bid, after partner's
    # response to it, is beyond what V1 has real judgement about. Every real call
    # scores UNSCORED so Pass always wins, rather than a made-up preference.
    opener = [
        card("AS"), card("KS"), card("2S"), card("5S"),
        card("AH"), card("2H"), card("3H"),
        card("KD"), card("2D"), card("7D"),
        card("QC"), card("2C"), card("3C"),
    ]
    auction = new_auction(dealer=0)
    for call in (Bid(1, None), None, Bid(2, Suit.CLUBS), None):
        auction = auction.apply(call)
    assert choose_bid(opener, auction) is None
