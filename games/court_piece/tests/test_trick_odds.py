from engine.cards import Suit
from games.court_piece.rules import CompletedTrick, CourtPiece, CourtPieceState, TrumpCall
from games.court_piece.tests.test_rules import card
from games.court_piece.trick_odds import trick_win_probabilities

TRUMP = Suit.SPADES
CALL = TrumpCall(trump=TRUMP, caller=0)


def _won_by(winner: int) -> CompletedTrick:
    cards = ((0, card("2C")), (1, card("3C")), (2, card("4C")), (3, card("5C")))
    return CompletedTrick(cards=cards, winner=winner)


def test_the_last_card_of_a_trick_has_a_computable_certain_answer() -> None:
    # Seat 0 is on lead to the fourth and final card, holding the ace - nothing left
    # can beat it, so this has to come out as a certain win, no probability about it.
    state = CourtPieceState(
        call=CALL,
        hands=((card("AH"),), (), (), ()),
        trick=((1, card("2H")), (2, card("3H")), (3, card("4H"))),
        completed=(_won_by(1),),
    )
    game = CourtPiece()
    assert state.to_play == 0
    assert trick_win_probabilities(game, state) == {card("AH"): 1.0}


def test_a_card_already_beaten_by_what_is_down_wins_with_probability_zero() -> None:
    # The nine has already lost to the earlier nine of hearts on the table - no later
    # card of anyone else's can rescue it, so this is a sure loss, not just a bad bet.
    state = CourtPieceState(
        call=CALL,
        hands=((card("2H"),), (), (), ()),
        trick=((1, card("9H")), (2, card("3H")), (3, card("4H"))),
        completed=(_won_by(1),),
    )
    game = CourtPiece()
    assert trick_win_probabilities(game, state) == {card("2H"): 0.0}


def test_a_beater_that_happens_to_sit_with_partner_is_still_genuinely_uncertain() -> None:
    # The real, honest difference from bridge: there is no dummy here, so partner's
    # hand is exactly as unseen to the deciding player as either opponent's. Seat 1
    # leads the nine; the ace happens to actually sit with seat 3 (seat 1's partner)
    # in this ground-truth deal - but the heuristic has no way to know that (a real
    # player never could either, with no dummy exposed), so it correctly can't credit
    # partner's card as safe just because it happens not to be a real threat this time.
    # The ace and king are pooled uniformly across all three unseen seats, and only
    # seats 0 and 2 (the two genuine opponents) count as "draw" - so this comes out as
    # a real, fractional probability, not the certain win bridge's known-dummy
    # equivalent test gets.
    #
    # Caller is seat 1 here specifically so seat 1 leads (opening_leader() puts the
    # caller themselves on lead) - the seat/hand assignments are otherwise identical.
    state = CourtPieceState(
        call=TrumpCall(trump=TRUMP, caller=1),
        hands=((card("2H"),), (card("9H"),), (card("KH"),), (card("AH"),)),
    )
    game = CourtPiece()
    assert state.to_play == 1
    # pool = {2H, KH, AH} (3 cards, seats 0/2/3); beaters = {KH, AH} (2); draw = seats
    # 0 and 2's cards (2 real opponent cards) -> survival = C(1,2)/C(3,2) = 0/3 = 0.0.
    assert trick_win_probabilities(game, state) == {card("9H"): 0.0}


def test_no_threat_anywhere_is_a_certain_win() -> None:
    # The only unseen card left in the whole deal is lower than the candidate too -
    # nothing can beat it.
    state = CourtPieceState(
        call=CALL,
        hands=((card("9H"),), (card("5H"),), (card("2H"),), ()),
        trick=((3, card("3H")),),
        completed=(_won_by(3),),
    )
    game = CourtPiece()
    assert trick_win_probabilities(game, state) == {card("9H"): 1.0}


def test_a_lone_beater_split_across_three_unseen_hands_is_only_a_fractional_threat() -> None:
    # Leading, so all three other seats are still to play. The one beater (KH) could
    # honestly be in any of the three unseen hands from seat 0's perspective - genuine
    # opponents (seats 1, 3) or partner (seat 2) - and only landing with a genuine
    # opponent actually costs the trick. This is the real, honest difference from a
    # hypothetical bridge-style version with a known dummy: with no dummy, the beater's
    # location is never narrowed down to "definitely a real opponent," so the answer
    # is a real fraction, not a certainty either way.
    state = CourtPieceState(
        call=CALL,
        hands=((card("9H"),), (card("KH"),), (card("2H"),), (card("4H"),)),
        trick=(),
        completed=(_won_by(0),),
    )
    game = CourtPiece()
    assert state.to_play == 0
    # pool = {KH, 2H, 4H} (3 cards, seats 1/2/3); beaters = {KH} (1); draw = seats 1
    # and 3's cards (the two genuine opponents, partner/seat 2 excluded) = 2 cards.
    # survival = C(2,2)/C(3,2) = 1/3.
    assert trick_win_probabilities(game, state) == {card("9H"): 1 / 3}


def test_a_genuinely_fractional_probability_from_a_shared_pool() -> None:
    # All of seats 1 and 2's cards are unseen from seat 0's perspective (no dummy to
    # know exactly) and pooled together: {KH, 3H}, one of them (KH) a beater. Seat 2
    # is seat 0's own partner and never a real threat, so the only genuine adversary
    # still to play is seat 1, holding exactly 1 of those 2 pooled cards.
    # P(seat 1's one card is the safe 3H, not the beating KH) = C(1,1)/C(2,1) = 0.5.
    state = CourtPieceState(
        call=CALL,
        hands=(
            (card("9H"),),
            (card("KH"),),
            (card("3H"),),
            (),
        ),
        trick=((3, card("2H")),),
        completed=(_won_by(3),),
    )
    game = CourtPiece()
    assert state.to_play == 0
    assert trick_win_probabilities(game, state) == {card("9H"): 0.5}


def test_a_beater_that_happens_to_sit_with_partner_is_not_treated_as_certainly_safe() -> None:
    # Seat 0 leads a middling card; the king actually sits with seat 2 (seat 0's own
    # partner) in this ground-truth deal - but exactly like the fractional test above,
    # the heuristic has no dummy to confirm that, so it can't credit the king as safe
    # just because it happens not to be a real opponent's. This is the same honest
    # extra uncertainty Court Piece has that bridge's dummy-known equivalent doesn't.
    state = CourtPieceState(
        call=CALL,
        hands=((card("9H"),), (card("2H"),), (card("KH"),), (card("3H"),)),
        trick=(),
        completed=(_won_by(0),),
    )
    game = CourtPiece()
    assert state.to_play == 0
    # pool = {2H, KH, 3H} (3 cards, seats 1/2/3); beaters = {KH} (1); draw = seats 1
    # and 3's cards (2 cards). survival = C(2,2)/C(3,2) = 1/3.
    assert trick_win_probabilities(game, state) == {card("9H"): 1 / 3}


def test_partner_holding_the_last_beater_with_no_real_opponents_left_is_a_certain_win() -> None:
    # The real, cleaner Court Piece analogue of bridge's "dummy never threatens" test:
    # both genuine opponents (seats 1, 3) are already out of cards entirely, so the
    # only unseen card left anywhere is partner's - and since partner is never a real
    # threat (a trick either side of the partnership wins is a win for the whole
    # side), there is truly nobody left who could beat this trick, known or not.
    state = CourtPieceState(
        call=CALL,
        hands=((card("9H"),), (), (card("KH"),), ()),
        trick=(),
        completed=(_won_by(0),),
    )
    game = CourtPiece()
    assert state.to_play == 0
    assert trick_win_probabilities(game, state) == {card("9H"): 1.0}


# ---------------------------------------------------------------------------
# Running trump: trump is still undetermined for most of the early hand - exactly
# where this heuristic actually runs (beyond the depth gate). A void candidate would
# set trump to its own suit if played, and that must be scored in, not ignored.
# ---------------------------------------------------------------------------

RUNNING_CALL = TrumpCall(trump=None, caller=0)


def test_a_void_candidate_is_scored_as_setting_and_winning_with_its_own_trump() -> None:
    # Seat 0 is void in the led suit (hearts) and holds one spade - playing it would
    # set trump to spades right now and, since nobody else has played a trump yet
    # (trump was undetermined until this exact play), it wins outright. Scoring this
    # candidate against the OLD undetermined trump (as an ordinary off-suit discard
    # that can never win) would wrongly give it 0.0.
    state = CourtPieceState(
        call=RUNNING_CALL,
        hands=((card("9S"),), (), (), ()),
        trick=((1, card("2H")), (2, card("3H")), (3, card("4H"))),
        completed=(_won_by(1),),
    )
    game = CourtPiece()
    assert state.to_play == 0
    assert trick_win_probabilities(game, state) == {card("9S"): 1.0}


def test_leading_with_trump_still_undetermined_behaves_like_notrump() -> None:
    # Nothing is led yet, so nobody can be void - leading never sets trump, so this
    # must score exactly like a genuine notrump position (trump=None throughout),
    # not as if the leader's own card could somehow set trump for itself.
    state = CourtPieceState(
        call=RUNNING_CALL,
        hands=((card("9H"), card("2C"), card("5S")), (), (), ()),
        trick=(),
        completed=(_won_by(0),),
    )
    game = CourtPiece()
    result = trick_win_probabilities(game, state)
    assert set(result) == {card("9H"), card("2C"), card("5S")}


def test_following_suit_with_trump_still_undetermined_never_sets_it() -> None:
    # Seat 0 holds the suit led (hearts) and must follow - not void, so playing it
    # can never set trump, and this must score as an ordinary notrump comparison.
    state = CourtPieceState(
        call=RUNNING_CALL,
        hands=((card("AH"),), (), (), ()),
        trick=((1, card("2H")), (2, card("3H")), (3, card("4H"))),
        completed=(_won_by(1),),
    )
    game = CourtPiece()
    assert state.to_play == 0
    assert trick_win_probabilities(game, state) == {card("AH"): 1.0}  # highest heart, no trump


def test_leading_offers_every_card_in_the_hand_on_turn() -> None:
    state = CourtPieceState(
        call=CALL,
        hands=((card("9H"), card("2C"), card("5S")), (), (), ()),
        trick=(),
        completed=(_won_by(0),),
    )
    game = CourtPiece()
    result = trick_win_probabilities(game, state)
    assert set(result) == {card("9H"), card("2C"), card("5S")}
