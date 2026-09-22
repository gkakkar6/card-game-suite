"""Myopic single-trick win-probability heuristic - the fallback
games/court_piece/action_values.py reaches for when a real PIMC solve either isn't
attempted at all (too many tricks remain) or didn't finish in time.

Same technique as games/bridge/trick_odds.py, genuinely simpler here since there is
no dummy: every seat other than the one deciding (and its partner) is unseen and
pooled uniformly, with no special-cased known hand to check separately. "Winning"
here means winning the CURRENT trick only - no lookahead past it.

Running trump needs one real accommodation, not just a type change: this heuristic
is what actually runs for most of the early hand (beyond the depth gate), which is
exactly when running trump's trump is most likely still undetermined. If the
deciding player is void in the suit led and trump hasn't been set yet, playing
*any* candidate card sets trump to that candidate's own suit and resolves this same
trick under it (games/court_piece/rules.py's `CourtPiece.apply()` - the identical
rule, checked the identical way). Evaluating every candidate against the OLD
(still-`None`) trump instead would wrongly treat a genuine trump-setting, likely-
winning play as an ordinary powerless discard. `_effective_trump()` below applies
exactly `apply()`'s own voidness check before any card is actually scored.
"""

import math

from engine.cards import Card, Suit
from engine.trick_taking.resolution import Seat, trick_winner
from games.court_piece.rules import CourtPiece, CourtPieceState, partner


def _effective_trump(state: CourtPieceState, candidate: Card) -> Suit | None:
    """What trump actually is for resolving this trick if `candidate` is played -
    the same voidness check `CourtPiece.apply()` uses to decide whether this exact
    play would set trump, applied here before scoring rather than after playing."""
    if state.call.trump is not None or state.led_suit is None:
        return state.call.trump
    was_void = not any(card.suit is state.led_suit for card in state.hands[state.to_play])
    return candidate.suit if was_void else state.call.trump


def _beats(candidate: Card, other: Card, led_suit: Suit, trump: Suit | None) -> bool:
    """Would `other` win a two-card trick against `candidate`?"""
    trick = ((0, candidate), (1, other))
    return trick_winner(trick, led_suit, trump) == 1


def _survival_probability(pool_size: int, beaters: int, draw: int) -> float:
    """Probability that none of `beaters` marked cards, out of a pool of `pool_size`,
    land among a draw of `draw` cards taken from that pool (no replacement) - the same
    hypergeometric "zero successes" count as games/bridge/trick_odds.py's own."""
    if draw == 0 or beaters == 0:
        return 1.0
    safe = pool_size - beaters
    if draw > safe:
        return 0.0
    return math.comb(safe, draw) / math.comb(pool_size, draw)


def trick_win_probabilities(game: CourtPiece, state: CourtPieceState) -> dict[Card, float]:
    """Each legal card's probability of winning the trick in progress, no lookahead
    past it. See the module docstring for exactly what is and isn't modeled."""
    same_side = partner(state.to_play)

    remaining_count = len(state.hands) - 1 - len(state.trick)
    remaining_seats: list[Seat] = [
        (state.to_play + 1 + offset) % len(state.hands) for offset in range(remaining_count)
    ]

    # Every seat but the one deciding is unseen - no dummy to know exactly, unlike
    # bridge. Pooled together, the same as bridge pools its two unseen opponents.
    other_seats = (seat for seat in range(len(state.hands)) if seat != state.to_play)
    pool = [card for seat in other_seats for card in state.hands[seat]]
    pool_size = len(pool)
    threat_size = sum(
        len(state.hands[seat]) for seat in remaining_seats if seat != same_side
    )

    values: dict[Card, float] = {}
    for candidate in game.legal_actions(state):
        led_suit = state.led_suit if state.led_suit is not None else candidate.suit
        trump = _effective_trump(state, candidate)
        trial_trick = (*state.trick, (state.to_play, candidate))
        winning_seat = trick_winner(trial_trick, led_suit, trump)
        if winning_seat not in (state.to_play, same_side):
            values[candidate] = 0.0
            continue
        current_best = next(card for seat, card in trial_trick if seat == winning_seat)

        beaters = sum(1 for other in pool if _beats(current_best, other, led_suit, trump))
        values[candidate] = _survival_probability(pool_size, beaters, threat_size)

    return values
