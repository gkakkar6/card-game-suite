"""Myopic single-trick win-probability heuristic (ARCHITECTURE.md §5, bridge Phase 3).

This is the fallback games/bridge/action_values.py reaches for when a real PIMC solve
either isn't attempted at all (too many tricks remain) or didn't finish in time - and in
practice it's the *primary* mechanism for roughly the first 3-4 decisions of every hand,
not a rare edge case, so it gets the same real testing as anything else load-bearing here.

Same conceptual shape as poker's Week 2 equity engine (games/poker/equity.py): a win
probability computed from a uniform-distribution assumption over unseen cards, no search,
no lookahead past the trick in front of the player. "Winning" here means winning the
CURRENT trick only - what happens on later tricks is not considered at all.

Computed exactly via combinatorics, not simulated. For each legal card: first check
whether the *side* playing it is currently ahead - trick_winner() decides this outright,
and if the winner is on the other side, the value is 0, no further reasoning needed. A
side being ahead is not the same as candidate itself being the highest card down: the
only way it can differ is the dummy's decision, where declarer may already have played
a higher card earlier in this same trick - candidate not beating that is not a loss,
since the side wins either way. What matters for the rest of the check is always
whichever card is actually ahead right now (candidate, or that earlier same-side card),
not candidate specifically - a card that is not currently winning is not what a future
threat has to get past.

Every seat still to play this trick is a threat to that card, *except* the one on the
same side as the hand actually playing (partner(state.to_play)) - a trick either partner
wins is a win for the whole side, so that seat can never zero out or discount a
candidate's value, known or not. The dummy's hand is known exactly, so its threat (when
it genuinely is one - an opponent deciding, not declarer or dummy itself) is checked
directly against its own actually-legal cards (legal_plays()). The other, genuinely
unseen seats are treated as one pool of cards, uniformly distributed between them, and a
hypergeometric count answers "how likely is a beating card to land in a hand that's
still to play this trick" - restricted to the seats that are actually adversarial, not
just unseen.

One honest, named simplification: whether a genuinely unseen hand holding a qualifying
card would actually be forced to follow suit instead of playing it (and so couldn't
really use it as a threat) is not modeled - that would need knowing the correlation
between "holds this card" and "is void in the led suit" within one unknown hand's own
composition, which this heuristic deliberately doesn't attempt. The error only runs one
way: this can underestimate a card's win probability, never overestimate it, since a
threat that's actually unplayable (forced to follow suit) still gets counted as live.
"""

import math

from engine.cards import Card, Suit
from engine.trick_taking.resolution import Seat, legal_plays, trick_winner
from games.bridge.rules import Bridge, BridgeState, partner


def _beats(candidate: Card, other: Card, led_suit: Suit, trump: Suit | None) -> bool:
    """Would `other` win a two-card trick against `candidate`?"""
    trick = ((0, candidate), (1, other))
    return trick_winner(trick, led_suit, trump) == 1


def _survival_probability(pool_size: int, beaters: int, draw: int) -> float:
    """Probability that none of `beaters` marked cards, out of a pool of `pool_size`,
    land among a draw of `draw` cards taken from that pool (no replacement).

    Standard hypergeometric "zero successes" count: choose `draw` cards from the
    `pool_size - beaters` safe ones, over choosing `draw` from the whole pool.
    """
    if draw == 0 or beaters == 0:
        return 1.0
    safe = pool_size - beaters
    if draw > safe:
        return 0.0  # more cards drawn than there are safe ones to fill it with
    return math.comb(safe, draw) / math.comb(pool_size, draw)


def trick_win_probabilities(game: Bridge, state: BridgeState) -> dict[Card, float]:
    """Each legal card's probability of winning the trick in progress, no lookahead
    past it. See the module docstring for exactly what is and isn't modeled.
    """
    player = game.current_player(state)
    info = game.information_set(state, player)
    seats = range(len(state.hands))
    unknown_seats = [seat for seat in seats if seat not in info.hands]
    dummy = state.dummy
    trump = state.contract.trump

    # Whoever's on the same side as the hand actually playing - never a threat to it,
    # since a trick either side of a partnership wins is a win for the whole side. Not
    # necessarily the dummy: when an opponent is deciding, their own partner is the
    # *other* opponent seat, not the dummy at all.
    same_side = partner(state.to_play)

    # Seats still to play this trick, after whichever candidate is played next.
    remaining_count = len(state.hands) - 1 - len(state.trick)
    remaining_seats: list[Seat] = [
        (state.to_play + 1 + offset) % len(state.hands) for offset in range(remaining_count)
    ]

    pool = [card for seat in unknown_seats for card in state.hands[seat]]
    pool_size = len(pool)
    threat_size = sum(
        len(state.hands[seat])
        for seat in unknown_seats
        if seat in remaining_seats and seat != same_side
    )

    values: dict[Card, float] = {}
    for candidate in game.legal_actions(state):
        led_suit = state.led_suit if state.led_suit is not None else candidate.suit
        trial_trick = (*state.trick, (state.to_play, candidate))
        winning_seat = trick_winner(trial_trick, led_suit, trump)
        if winning_seat not in (state.to_play, same_side):
            values[candidate] = 0.0
            continue
        # Whichever card is actually ahead right now - candidate itself, or (only
        # possible for the dummy) an earlier same-side card declarer already played
        # to this same trick that candidate doesn't need to beat, since either one
        # winning is equally a win for the side. Future threats have to be checked
        # against *that* card, not blindly against candidate - a candidate that
        # isn't currently winning is not what anyone still has to beat.
        current_best = next(card for seat, card in trial_trick if seat == winning_seat)

        if dummy in remaining_seats and dummy != same_side:
            dummy_legal = legal_plays(state.hands[dummy], led_suit)
            if any(_beats(current_best, other, led_suit, trump) for other in dummy_legal):
                values[candidate] = 0.0
                continue

        beaters = sum(1 for other in pool if _beats(current_best, other, led_suit, trump))
        values[candidate] = _survival_probability(pool_size, beaters, threat_size)

    return values
