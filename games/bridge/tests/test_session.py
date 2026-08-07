import random
from collections.abc import Sequence

import pytest

from engine.cards import Card, Deck, Rank, Suit
from games.bridge.bidding import Auction, Bid, Call
from games.bridge.personas import UNAWARE
from games.bridge.rules import Bridge, BridgeState, Contract
from games.bridge.scoring import ScoreResult
from games.bridge.session import (
    BidStrategy,
    HandResult,
    PlayStrategy,
    SeatConfig,
    Session,
    SessionConfig,
    play_hand,
)


def _lowest_legal(game: Bridge, state: BridgeState) -> Card:
    """A trivial, deterministic card-play strategy: whatever sorts first. Only used
    where a test doesn't care which specific card is played, just that play proceeds."""
    return min(game.legal_actions(state), key=lambda c: (c.suit.value, c.rank.value))


class _ScriptedBidder:
    """Returns each queued call in order, then passes forever once the queue is
    empty - lets a test control exactly one auction without hand-writing every
    seat's bids for the rest of the deal."""

    def __init__(self, calls: Sequence[Call]) -> None:
        self._calls = list(calls)

    def __call__(self, hand: Sequence[Card], auction: Auction) -> Call:
        if self._calls:
            return self._calls.pop(0)
        return None


def _always_pass(hand: Sequence[Card], auction: Auction) -> Call:
    return None


def _one_suit_per_seat_deck() -> Deck:
    """Each seat holds one suit outright: seat 0 clubs, seat 1 diamonds, seat 2
    hearts, seat 3 spades. Nobody can ever follow a trick's led suit except whoever
    led it, so - in a notrump contract - the leader's card is trivially the only
    card of the led suit on the table and always wins. Since the winner leads next,
    whoever leads trick 1 wins every one of the 13 tricks: fully deterministic and
    checkable by hand, without needing to script every individual card played."""
    clubs = [Card(rank, Suit.CLUBS) for rank in Rank]
    diamonds = [Card(rank, Suit.DIAMONDS) for rank in Rank]
    hearts = [Card(rank, Suit.HEARTS) for rank in Rank]
    spades = [Card(rank, Suit.SPADES) for rank in Rank]
    return Deck(cards=[*clubs, *diamonds, *hearts, *spades])


# ---------------------------------------------------------------------------
# play_hand(): a fully scripted deal, deal-to-score end to end
# ---------------------------------------------------------------------------


def test_a_fully_scripted_hand_from_deal_to_score() -> None:
    # Seat 0 opens 1NT and everyone else passes it out - declarer is seat 0, target 7,
    # no trump suit to complicate who beats whom. Seat 0's opening lead comes from
    # opening_leader(declarer=0) = seat 1, who holds diamonds outright and so - per
    # _one_suit_per_seat_deck()'s own reasoning - wins every one of the 13 tricks (no
    # trump means nobody can ever ruff in). Declarer's side (0, 2) therefore wins zero
    # tricks: down 7, worked out entirely by hand, no PIMC or fallback heuristics
    # involved.
    bid_strategies: list[BidStrategy] = [
        _ScriptedBidder([Bid(1, None)]),
        _always_pass,
        _always_pass,
        _always_pass,
    ]
    play_strategies: list[PlayStrategy] = [_lowest_legal] * 4

    result = play_hand(bid_strategies, play_strategies, dealer=0, deck=_one_suit_per_seat_deck())

    assert result.contract == Contract(trump=None, declarer=0, target=7)
    assert result.tricks_won == 0
    assert result.score is not None
    assert not result.score.made
    assert result.score.margin == -7
    assert result.score.points == -50 * 7


def test_a_passed_out_hand_has_no_contract_tricks_or_score() -> None:
    bid_strategies: list[BidStrategy] = [_always_pass] * 4
    play_strategies: list[PlayStrategy] = [_lowest_legal] * 4

    result = play_hand(bid_strategies, play_strategies, dealer=0, rng=random.Random(1))

    assert result.auction.is_passed_out()
    assert result.contract is None
    assert result.tricks_won is None
    assert result.score is None


# ---------------------------------------------------------------------------
# Session: dealer rotation, score accumulation, passed-out hands
# ---------------------------------------------------------------------------


def _fast_session_config() -> SessionConfig:
    # UNAWARE never invokes PIMC for card play (always the cheap myopic fallback),
    # and bidding is cheap regardless of persona - so a whole session built from it
    # runs many real hands quickly, without needing @pytest.mark.slow treatment the
    # way test_personas.py's calibration sweeps do.
    return SessionConfig(
        seats=(
            SeatConfig(name="You", is_human=True),
            SeatConfig(name="Partner", is_human=False, persona=UNAWARE),
            SeatConfig(name="Opp1", is_human=False, persona=UNAWARE),
            SeatConfig(name="Opp2", is_human=False, persona=UNAWARE),
        )
    )


def test_dealer_rotates_clockwise_across_several_hands() -> None:
    session = Session(_fast_session_config(), _always_pass, _lowest_legal, rng=random.Random(2))
    dealers = []
    for _ in range(5):
        dealers.append(session.dealer)
        session.play_next_hand()
    assert dealers == [0, 1, 2, 3, 0]
    assert session.dealer == 1


def test_a_passed_out_hand_still_advances_the_dealer_but_does_not_touch_the_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session(_fast_session_config(), _always_pass, _lowest_legal, rng=random.Random(3))

    passed_out = HandResult(
        dealer=0,
        hands=((), (), (), ()),
        auction=Auction(dealer=0, calls=(None, None, None, None)),
        contract=None,
        tricks_won=None,
        score=None,
    )
    monkeypatch.setattr("games.bridge.session.play_hand", lambda *a, **k: passed_out)

    before = dict(session.scores)
    result = session.play_next_hand()

    assert result.score is None
    assert session.scores == before
    assert session.hands_played == 1
    assert session.dealer == 1


def test_cumulative_score_accumulates_across_a_mix_of_made_and_failed_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session(_fast_session_config(), _always_pass, _lowest_legal, rng=random.Random(4))

    # Hand 1: the human's side (seats 0, 2) declares and makes - a real gain for them.
    made_by_human_side = HandResult(
        dealer=0,
        hands=((), (), (), ()),
        auction=Auction(dealer=0, calls=(Bid(1, None), None, None, None)),
        contract=Contract(trump=None, declarer=0, target=7),
        tricks_won=7,
        score=ScoreResult(made=True, margin=0, points=40),
    )
    # Hand 2: the OTHER side (seats 1, 3) declares and fails - a gain for the defenders,
    # who this time are the human's side.
    failed_by_other_side = HandResult(
        dealer=1,
        hands=((), (), (), ()),
        auction=Auction(dealer=1, calls=(Bid(1, None), None, None, None)),
        contract=Contract(trump=None, declarer=1, target=7),
        tricks_won=5,
        score=ScoreResult(made=False, margin=-2, points=-100),
    )
    queue = iter([made_by_human_side, failed_by_other_side])
    monkeypatch.setattr("games.bridge.session.play_hand", lambda *a, **k: next(queue))

    session.play_next_hand()
    session.play_next_hand()

    # Human's side (0, 2): +40 from hand 1 (declared and made), +100 from hand 2
    # (the other side went down 2, which is a gain for the defenders).
    assert session.scores[0] == session.scores[2] == 140
    # The other side (1, 3): the mirror image, -40 then -100.
    assert session.scores[1] == session.scores[3] == -140
