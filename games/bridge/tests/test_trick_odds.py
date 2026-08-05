from engine.cards import Suit
from games.bridge.rules import Bridge, BridgeState, CompletedTrick, Contract
from games.bridge.tests.test_rules import card
from games.bridge.trick_odds import trick_win_probabilities

TRUMP = Suit.SPADES
CONTRACT = Contract(trump=TRUMP, declarer=0, target=7)

# A throwaway completed trick, won by whichever seat needs to lead the next one - lets
# a test control who's on lead without relying on opening_leader()'s declarer-based rule.
def _won_by(winner: int) -> CompletedTrick:
    cards = ((0, card("2C")), (1, card("3C")), (2, card("4C")), (3, card("5C")))
    return CompletedTrick(cards=cards, winner=winner)


def test_the_last_card_of_a_trick_has_a_computable_certain_answer() -> None:
    # Declarer is on lead to the fourth and final card, holding the ace - nothing left
    # can beat it, so this has to come out as a certain win, no probability about it.
    state = BridgeState(
        contract=CONTRACT,
        hands=((card("AH"),), (), (), ()),
        trick=((1, card("2H")), (2, card("3H")), (3, card("4H"))),
        completed=(_won_by(1),),
    )
    game = Bridge()
    assert state.to_play == 0
    assert trick_win_probabilities(game, state) == {card("AH"): 1.0}


def test_a_card_already_beaten_by_what_is_down_wins_with_probability_zero() -> None:
    # The nine has already lost to the earlier nine of hearts on the table - no later
    # card of anyone else's can rescue it, so this is a sure loss, not just a bad bet.
    state = BridgeState(
        contract=CONTRACT,
        hands=((card("2H"),), (), (), ()),
        trick=((1, card("9H")), (2, card("3H")), (3, card("4H"))),
        completed=(_won_by(1),),
    )
    game = Bridge()
    assert trick_win_probabilities(game, state) == {card("2H"): 0.0}


def test_a_known_threat_still_to_play_is_a_certain_loss() -> None:
    # Dummy is a genuine adversary here - an opponent (seat 1) is deciding, so dummy
    # sits on the *other* side. Dummy hasn't played to this trick yet and holds the ace
    # of hearts - fully known, not a probability, so this opponent's nine is a sure
    # loss regardless of what their own, genuinely unseen partner might hold.
    state = BridgeState(
        contract=CONTRACT,
        hands=((card("2H"),), (card("9H"),), (card("AH"),), (card("3H"),)),
    )
    game = Bridge()
    assert state.to_play == 1
    assert game.current_player(state) == 1
    assert trick_win_probabilities(game, state) == {card("9H"): 0.0}


def test_no_threat_anywhere_is_a_certain_win() -> None:
    # Dummy's remaining card is low, and the only unseen card left in the whole deal
    # is lower than the candidate too - nothing can beat it.
    state = BridgeState(
        contract=CONTRACT,
        hands=((card("9H"),), (card("5H"),), (card("2H"),), ()),
        trick=((3, card("3H")),),
        completed=(_won_by(3),),
    )
    game = Bridge()
    assert trick_win_probabilities(game, state) == {card("9H"): 1.0}


def test_two_unknown_hands_sharing_the_whole_pool_is_a_certain_loss_if_a_beater_exists() -> None:
    # Leading, so both opponents (the two genuinely unseen seats) are still to play and
    # between them hold every remaining unseen card - if a beater exists anywhere in
    # that pool, one of them necessarily has it, so this is certain, not probabilistic.
    state = BridgeState(
        contract=CONTRACT,
        hands=((card("9H"),), (card("KH"),), (card("2H"),), (card("4H"),)),
        trick=(),
        completed=(_won_by(0),),
    )
    game = Bridge()
    assert state.to_play == 0
    assert trick_win_probabilities(game, state) == {card("9H"): 0.0}


def test_a_genuinely_fractional_probability_from_a_shared_pool() -> None:
    # Worked out by hand: seat 3 already played to this trick (2H, revealed and safe),
    # so only seat 1's cards are still "in play" here - but seat 1's actual holding is
    # still unknown, drawn uniformly from the combined pool of what both seat 1 and
    # seat 3 have left (KH, 3H, 5H, 6H - 4 cards, 1 of them a beater: KH).
    # P(seat 1's 2-card hand avoids the single beater) = C(3,2)/C(4,2) = 3/6 = 0.5.
    state = BridgeState(
        contract=CONTRACT,
        hands=(
            (card("9H"),),
            (card("KH"), card("3H")),
            (card("2H"),),
            (card("5H"), card("6H")),
        ),
        trick=((3, card("2H")),),
        completed=(_won_by(3),),
    )
    game = Bridge()
    assert state.to_play == 0
    assert trick_win_probabilities(game, state) == {card("9H"): 0.5}


def test_an_opponent_deciding_sees_a_different_known_set_than_declarer_does() -> None:
    # Opening lead: seat 1 (declarer's left, per opening_leader()) is on lead and
    # deciding - dummy(2) is known to them too, same as it would be for declarer, but
    # both seat 0 (declarer, the real adversary) and seat 3 (opp1's own partner, not a
    # real adversary) are genuinely unseen to seat 1, unlike from declarer's own view.
    #
    # The king could be in either unseen hand with equal likelihood - only half of that
    # is a genuine threat (declarer's), the other half (opp1's own partner) is not, so
    # this is honestly uncertain rather than a certain loss: P(the single king lands in
    # the one-card adversary hand, not the partner's) = C(1,1)/C(2,1) = 0.5.
    state = BridgeState(
        contract=CONTRACT,
        hands=((card("KH"),), (card("9H"),), (card("2H"),), (card("3H"),)),
    )
    game = Bridge()
    assert state.leader == 1  # opening_leader(declarer=0)
    assert state.to_play == 1
    assert game.current_player(state) == 1
    assert trick_win_probabilities(game, state) == {card("9H"): 0.5}


def test_a_beater_in_declarers_own_partner_dummy_never_costs_declarer_the_trick() -> None:
    # Declarer leads a middling card; dummy - declarer's own partner - holds a genuine
    # beater and hasn't played yet. A trick either partner wins is still a win for
    # declarer's side, so dummy can never be the reason a candidate loses, however high
    # dummy's card is. No other threat exists (both opponents hold only low cards).
    state = BridgeState(
        contract=CONTRACT,
        hands=((card("9H"),), (card("2H"),), (card("KH"),), (card("3H"),)),
        trick=(),
        completed=(_won_by(0),),
    )
    game = Bridge()
    assert state.to_play == 0
    assert trick_win_probabilities(game, state) == {card("9H"): 1.0}


def test_a_beater_in_declarer_never_costs_the_dummys_own_card_the_trick() -> None:
    # The other half of the same rule, from the other direction: dummy itself is on
    # turn (declarer decides for it), and declarer - dummy's own partner - holds a
    # genuine beater and hasn't played yet. Still not a threat, for the same reason.
    state = BridgeState(
        contract=CONTRACT,
        hands=((card("KH"),), (card("2H"),), (card("9H"),), (card("3H"),)),
        trick=(),
        completed=(_won_by(2),),
    )
    game = Bridge()
    assert state.to_play == 2
    assert game.current_player(state) == 0
    assert trick_win_probabilities(game, state) == {card("9H"): 1.0}


def test_a_beater_in_an_opponents_own_partner_never_costs_that_opponent_the_trick() -> None:
    # Declarer (the true adversary to a deciding opponent) has already played this
    # trick and shown safe. The only seat left unaccounted for is opp1's own partner
    # (opp3) - genuinely unseen to opp1, but not adversarial - who happens to hold the
    # only remaining beater. That's not a real threat, so this must be a certain win,
    # not the fractional or zero answer a pooled-with-declarer treatment would give.
    state = BridgeState(
        contract=CONTRACT,
        hands=((), (card("9H"),), (card("3H"),), (card("KH"),)),
        trick=((0, card("2H")),),
    )
    game = Bridge()
    assert state.to_play == 1
    assert game.current_player(state) == 1
    assert trick_win_probabilities(game, state) == {card("9H"): 1.0}


def test_not_overtaking_a_partners_own_earlier_card_is_still_a_win() -> None:
    # Declarer(0) has already played the king to this trick, then an opponent(1)
    # followed low. Dummy(2), deciding now, holds a card that does NOT beat
    # declarer's king - but declarer's king is still winning regardless, and a trick
    # either partner wins is a win for the side (same principle as the dummy-as-
    # threat tests above, just for a card already played rather than still to come).
    # No other threat exists, so this must be a certain win, not zero.
    state = BridgeState(
        contract=CONTRACT,
        hands=((), (), (card("8H"),), (card("3H"),)),
        trick=((0, card("KH")), (1, card("2H"))),
    )
    game = Bridge()
    assert state.to_play == 2
    assert game.current_player(state) == 0
    assert trick_win_probabilities(game, state) == {card("8H"): 1.0}


def test_future_threats_are_checked_against_the_partners_card_actually_winning() -> None:
    # Same shape as above, but now a genuine threat exists - the 9 of hearts,
    # sitting with the one remaining unseen opponent. 9H beats dummy's own 8H, but
    # NOT declarer's king, which is what's actually protecting the trick. A version
    # of this check that compared threats against the candidate (8H) instead of
    # whichever card is actually ahead (the king) would wrongly score this 0 - it
    # has to come out as a certain win.
    state = BridgeState(
        contract=CONTRACT,
        hands=((), (), (card("8H"),), (card("9H"),)),
        trick=((0, card("KH")), (1, card("2H"))),
    )
    game = Bridge()
    assert state.to_play == 2
    assert trick_win_probabilities(game, state) == {card("8H"): 1.0}


def test_leading_offers_every_card_in_the_hand_on_turn() -> None:
    state = BridgeState(
        contract=CONTRACT,
        hands=((card("9H"), card("2C"), card("5S")), (), (), ()),
        trick=(),
        completed=(_won_by(0),),
    )
    game = Bridge()
    result = trick_win_probabilities(game, state)
    assert set(result) == {card("9H"), card("2C"), card("5S")}
