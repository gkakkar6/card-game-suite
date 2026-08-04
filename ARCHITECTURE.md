# Card Game Suite — Architecture

**Repo:** `card-suite` (Python package: `card_suite`). Plain and functional on purpose — the README carries the actual story (shared architecture across poker/bridge/Court Piece, the quantal persona model, the evaluation harness), not the name.

Goal of the project, stated plainly: build a system that gets *me* better at poker and bridge by forcing me to formalise what "correct play" actually means, not just to produce a CV artifact. That goal drives several of the calls below.

---

## 1. One repo, not one-per-game

Poker and bridge share a `Game` interface, a persona engine, and an evaluation harness. Splitting them into separate repos means either duplicating that shared code or extracting it into a versioned package to solve a problem that doesn't exist in a monorepo. GitHub subfolder links (`repo/tree/main/games/poker`) give per-game visibility without per-game repos.

```
card-games/
├── engine/                      # shared, game-agnostic, zero game-specific imports
│   ├── cards/                   # Card, Suit, Rank, Deck, deal, shuffle
│   ├── game.py                  # the Game protocol (see §2)
│   ├── rollout.py               # generic Monte Carlo rollout + determinization
│   ├── evaluation.py            # bot-vs-bot runner, confidence intervals (Wilson/bootstrap)
│   ├── trick_taking/            # shared across all whist-family games (bridge, court piece)
│   │   ├── resolution.py        # resolve a trick given cards played + trump (trump may be undetermined)
│   │   ├── doubledummy.py       # exact minimax solver — generic over any Game with this shape
│   │   └── pimc.py              # sampling layer for imperfect-info determinization
│   └── personas/
│       ├── base.py              # Persona = params + Policy; registry
│       ├── quantal.py           # shared noise/bias mechanism (see §3)
│       └── nlp.py               # free-text → structured OpponentIntent (game-agnostic)
├── games/
│   ├── poker/
│   │   ├── rules.py             # state + transitions implementing engine.game.Game
│   │   ├── equity.py            # MC hand equity, action_values()
│   │   ├── personas.py          # poker-specific persona params + intent mapping
│   │   └── tests/
│   ├── bridge/
│   │   ├── rules.py             # play phase built on engine.trick_taking
│   │   ├── bidding.py           # convention-based auction, NOT a solver (see §5)
│   │   ├── personas.py
│   │   └── tests/
│   └── court_piece/
│       ├── rules.py             # trump: Optional[Suit] — set eagerly (fixed) or on first
│       │                        #   unable-to-follow trigger (running), see §5
│       ├── personas.py
│       └── tests/
├── review/                      # phase 2: hand-review tool, see §7
├── api/                         # deferred — thin FastAPI once a game is ready to serve
├── web/                         # deferred — React/TS, design-system/ + per-game screens
├── scripts/
└── tests/                       # cross-cutting engine + evaluation tests
```

---

## 2. The `Game` protocol

Shared code never imports anything game-specific. It only knows:

- `current_player(state)`
- `legal_actions(state)` — game-specific rules (e.g. bridge's follow-suit legality) live entirely inside this, not in the shared engine
- `apply(state, action) -> state`
- `is_terminal(state)`
- `payoff(state)` — per-player or per-partnership; the protocol doesn't care which
- a notion of what each player *observes* (their information set) — needed for determinization/PIMC sampling in imperfect-information games

This is already general enough for 4-player partnership trick-taking as well as 2-player betting rounds — deliberately, so poker-first doesn't risk under-specifying it for bridge.

**Concrete shape (tbc finalise the exact types, this is the contract to preserve):**

```python
from typing import Protocol, TypeVar, Generic

State = TypeVar("State")
Action = TypeVar("Action")   # game-specific — poker's Action ≠ bridge's Action, only the shape is shared
PlayerId = int

class Game(Protocol, Generic[State, Action]):
    def current_player(self, state: State) -> PlayerId: ...
    def legal_actions(self, state: State) -> list[Action]: ...
    def apply(self, state: State, action: Action) -> State: ...
    def is_terminal(self, state: State) -> bool: ...
    def payoff(self, state: State) -> dict[PlayerId, float]: ...
    def information_set(self, state: State, player: PlayerId) -> "InformationSet": ...
```

`payoff` returning a dict rather than a scalar handles both individual (poker) and partnership (bridge, Court Piece) scoring the same way — partners just get the same value. `information_set` is what PIMC samples against; for Court Piece's Hidden-Rung-adjacent mechanics it would need to cover an unknown trump suit as well as unknown opponent cards, not just cards — worth remembering if that variant ever gets built.

---

## 3. Persona model: quantal response

Given `action_values(state) -> {action: value}` (game-specific), the shared persona layer turns it into a policy via two independent knobs:

- **Temperature** — noise. Low = plays close to optimal; high = close to uniform random.
- **Bias** — systematic deviation, not noise. A shift in a specific direction (e.g. persistently overvaluing aggressive actions). This is what actually distinguishes "maniac" from "nervous beginner" — not just more noise, but a different *direction*.

`engine/personas/quantal.py` holds this mechanism. Each game only has to supply `action_values()`; the noise/bias math is identical across games.

**Bridge will need something poker's persona model has no concept of at all: partnership.** Every poker persona is a variation on pure self-interest — poker has no cooperation in it. A "selfish" bridge persona (plays for individual glory over the partnership's best result) likely still fits the existing bias mechanism, once bridge's `action_values()` can express both a true partnership-optimal value and a second, individually-flattering one to bias toward. An "unaware" persona (doesn't correctly read partner's signalling) is a different, harder problem — not a bias applied after correct reasoning, but reasoning done with worse information in the first place, which may need a genuinely different value-computation mode rather than a persona config layered on the existing one. Aggressive-vs-conservative *bidding* is the one bridge archetype with a direct, easy mapping onto the existing bias axis. None of this is buildable before bridge's `action_values()` exists (Phase 2/3), same as poker's personas didn't arrive until after the equity engine — noted here so it isn't forgotten in the meantime.

**Partner persona is a separate, higher-stakes selection than opponent persona, specifically because of the dummy rule.** Once the dummy never decides (see the `current_player` note in the Game protocol discussion), a human player who becomes dummy has made zero choices for that entire hand — their partner's competence alone determines the outcome. Table setup will need to let a human pick their partner's persona as its own explicit choice, distinct from picking the two opponents, not default it silently to baseline. A deliberately imperfect partner (mirrors playing with a real, fallible person) is arguably more true to this project's stated motivation than an always-optimal one would be.

**Trick-history recall as a persona axis, distinct from temperature/bias.** The game state always tracks the full, exact trick history — non-negotiable, the engine needs perfect knowledge to enforce rules and resolve tricks correctly. What can vary by persona is how much of that accurate history its own decision-making is allowed to consume: a "sharp" persona sees the exact trick history, a less sharp one only a degraded view (e.g. which suits have been played, not which specific cards). This is a different kind of imperfection from temperature/bias — those change what a persona does with correct information; this changes what information it's working from in the first place. Likely shares real machinery with the "unaware" idea above rather than being fully separate.

**Bias and temperature must be applied to normalised, not raw, values.** Learned the hard way building poker's decision-making phase: `action_values()` returns real chip values, and chip values scale with pot size — a fixed additive bias tuned to look right in one pot size is either invisible or completely dominant in another, since it isn't compared against a consistent reference. The fix used in poker: divide the value dict by a per-decision scale (pot + to_call) before bias/temperature are applied, so bias becomes "a fraction of what's at stake" rather than a raw number that means something different every hand. This normalisation is game-specific (poker's natural scale is the pot; bridge or Court Piece would need their own equivalent — e.g. trick value or points swing) and belongs in the calling game's code, not in `quantal.py` itself, which stays a generic value-to-probability mapper. When bridge or Court Piece personas are built, do this normalisation from the start rather than discovering the same bug independently.

**NLP opponent selection** (`engine/personas/nlp.py`) parses free text into a game-agnostic `OpponentIntent` (difficulty tier + loose descriptors). Each game maps that intent onto its own parameter space in `games/<game>/personas.py`.

**Deliberately maps onto existing, already-measured personas — never sets temperature/bias directly from parsed text.** The five poker personas behave sanely because their numbers were empirically measured against real decisions, not guessed (see the `AGGRESSION_BIAS = 8 → 40` history above). Letting free text set a dial value, even via a lookup table, would reintroduce that exact unmeasured-parameter risk behind a friendlier interface. If more nuance is wanted later, the right way to get it is more presets, each separately measured the same way — not a continuous dial nobody's validated. Also rule-based, not an LLM call: fully deterministic, testable with plain unit tests, no external dependency or API key needed to clone and run the repo — matters more given this is a public repo, not just a private tool.

---

## 4. Evaluation harness

`engine/evaluation.py` runs bot-vs-bot matches and reports win rates with confidence intervals, so any claimed "edge" is a number, not a vibe. The validating experiment for the whole architecture: a low-temperature/low-noise bot should beat a high-temperature bot by a statistically significant margin. If that doesn't show up, something's wrong before any game-specific logic is trusted.

---

## 5. Build order and rationale

**Poker → Bridge → Court Piece, fixed trump → Court Piece, running trump (extension of the third).** Four games in the portfolio story; three genuinely separate builds in effort terms — see below.

Court Piece (aka Rang / Rung / Hokm) is real, confirmed against standard rules: 4 players in fixed partnerships sitting crosswise, 13 tricks, must follow suit if able, highest card of the led suit wins unless trumped, bonus for winning the first 7 tricks in a row before the opponents win any (a "Court") vs. just winning 7 total in any order (a "Piece"). Genuinely simpler than bridge — no competitive auction, no dummy exposure, no doubling — while sharing the exact same trick-taking/double-dummy/PIMC shape.

**Scoring (locked):** point-based, per deal. Trump-calling team scores 1 point for a plain win (7+ tricks) or 3 for a court; the opposing team scores 2 for a plain win or 4 for a court. First team to 10 points wins the session. The asymmetry is deliberate and real (sourced from documented rules, not invented) — calling trump gives an information advantage, so winning as caller pays less than winning as non-caller. This is a genuine strategic input for the decision engine: whether calling trump is +EV depends partly on this payout structure, not just on hand strength.

**Running-trump variant, precise name:** what's being built is specifically **Be-ranga Double Sar** — play starts with no trump suit at all; the first player unable to follow suit plays any card, and that card's suit becomes trump for the rest of the deal. (A related but different real variant, **Hidden Rung**, has trump chosen upfront and kept secret rather than genuinely undetermined — not what's being built here, but worth knowing it exists as a distinct thing.)

Reasoning:
- Poker's solver (Monte Carlo hand equity vs. pot odds) is well-scoped and testable incrementally, and gets the *entire* architecture — engine, personas, evaluation harness — proven end to end through a predictable, low-risk sequence of steps.
- Bridge's double-dummy solver (exact minimax over a full deal, given all four hands known) is **the single hardest module in the whole project**. A correct-but-slow version is a realistic early milestone; a genuinely fast one (alpha-beta pruning, transposition tables) is its own piece of work. This sits in the *middle* of bridge's build — if it stalls, there's no working game yet. Building it second, once the surrounding scaffolding is proven on poker, is lower risk than building it first.
- Bridge bidding is **not** a solved-optimum problem — it's convention-based (point count, suit length rules), closer to an expert system than an optimiser. Different kind of work from the double-dummy solver, not a continuation of it.
- `engine/trick_taking/` (resolution, double-dummy solver, PIMC sampling) is written once against bridge and genuinely reused, not reimplemented, for Court Piece — both are whist-family games with identical trick-resolution logic. Only Court Piece's declare-phase and Court/Piece scoring are new game-specific code, which is why it's a much lighter build than bridge despite being "game 3."
- Court Piece's running-trump variant is honestly an **extension** of the fixed-trump build, not a fourth full build. The only new work is treating trump as `Optional[Suit]` — undetermined until either the declare-phase (fixed) or a forced first-can't-follow-suit trigger (running) sets it.

### Poker build plan
- **Phase 1:** cards, deck, 7-card hand evaluator (test against known rankings), betting round state machine.
- **Phase 2:** Monte Carlo equity engine — hero hand + board + opponent range → win/tie/loss via simulation. Sanity-check against known references (AA vs KK ≈ 80%). This *is* `action_values()`.
- **Phase 3:** EV-based decisions, quantal persona layer on top, bot-vs-bot evaluation with CIs.
  - **New required piece: `games/poker/hand.py`.** Orchestrates one complete hand end to end — post forced bets → deal hole cards → preflop betting → flop → betting → turn → betting → river → betting → showdown (or early stop if only one player remains). Nothing else in this phase can actually run without it; `action_values()` and the persona layer only matter once real hands are being played through all four streets, and the evaluation harness needs this to generate the hands it scores.
  - **Blinds:** small/big blind (no ante), posted via `betting.py`'s existing `initial_bets` — that mechanism already supports this, `hand.py` just needs to compute the list. Shape it as a named structure (`"blinds"`) rather than blind-specific logic scattered around, so other structures (e.g. an ante) can be added later without rework, even though only blinds are built now. Default placeholders, all configurable: SB=1, BB=2, starting stack=200.
  - **Raise cap:** `betting.py` currently has no limit on raises per round — add `max_raises: int | None` to `BettingRound`, tracked per round, `RAISE` stops appearing in `legal_actions()` once hit. `hand.py` passes max_raises=3 for preflop and river, None (uncapped) for flop and turn.
  - **Fold/call** are direct: value = equity × (pot + to_call) − to_call, straight from `hand_equity()` and the size of the bet facing the player. Fold is the 0 reference point.
  - **Bet/raise value is always computed whenever chip-legal** (per `legal_actions()`, including the new raise cap) — NOT gated by an equity threshold. A weak hand's bet/raise value comes out low or genuinely negative, honestly, rather than the action being hidden entirely. This is deliberate: it lets a persona's aggression bias choose an objectively bad-looking bet sometimes (real "bluffing" style), using the bias mechanism already designed in §3 — no new mechanism needed.
  - **Bet/raise sizing** scales continuously with equity above a reference point — not fixed sizes, not a solved equilibrium. `strength = (equity - reference) / (1 - reference)`, only meaningful when equity > reference; `reference` is a real parameter (default 0.5 = "beats a coinflip" when opening, i.e. to_call == 0; pot-odds break-even when facing a bet) — NOT hardcoded. Size = `min_size + strength × (max_size - min_size)` as a pot fraction, defaults min=0.33, max=1.5, both configurable. Always clamp to the player's actual stack and to `betting.py`'s minimum-raise legality. Note: size is computed from `max(strength, 0)` even when the reported action *value* is negative — a "bluff" still needs a sensible size, even though its value honestly reflects that it's -EV if called.
  - A true optimal size would need a model of the opponent's fold-frequency by sizing, which doesn't exist against a uniform range and would reopen the GTO-solving problem this project deliberately avoids (see `README.md` "Honest scope"). This is a principled heuristic, not an equilibrium — document it as such.
  - Explicitly **single-street EV**: values assume showdown happens now, not how the hand might develop across future streets. A real simplification, not an oversight — worth stating in the code, matching the project's honesty habit elsewhere.
  - **No fold-equity model — a real, named limitation, not a hidden bug.** Because bet/raise value assumes the opponent always calls, this bot can only ever produce a "value bets only" style, honestly — not equilibrium-balanced bluffing (the GTO sense: bluffing at the exact frequency that makes a strong opponent indifferent between calling and folding). The bias-driven "bluffer" persona above is a real, describable aggressive style — it is not that.
  - **Decision-time equity calls need different tuning from the equity engine's defaults.** `action_values()` always calls `hand_equity()` with `opponent_hole=None` (a bot never knows the opponent's cards) — meaning every decision risks landing on the expensive unknown-opponent-at-turn case (~45,540 combinations — measurably slow). Multiplied across every decision, every simulated hand, in a bot-vs-bot evaluation run needing thousands of hands for a trustworthy CI, that compounds into a real bottleneck. Fix: lower `MAX_EXACT_ENUMERATION` for decision-time calls specifically (somewhere between river's 990 and turn's 45,540, e.g. ~5,000 — keeps river exact and fast, forces turn onto Monte Carlo) and lower `iterations` (e.g. 500–1,000 vs. the equity engine's default of 10,000) — individual estimate noise is fine here since the evaluation harness's CI already aggregates over thousands of hands. This tuning is local to the `action_values()` call site; it doesn't touch `hand_equity()` itself, which keeps its precise defaults for tests, sanity checks, and the Phase 2 hand-review tool (§7).
  - **Personas for the validating experiment (upgrades §4's original single-axis check):** five named configs. A near-optimal **baseline** (low temperature, no bias); a **bluffer** (strong positive bias toward bet/raise, chosen even when the underlying value is weak or negative); a **conservative/nit** (bias suppressing bet/raise unless the value is strong); a **calling station** (bias against folding — loose-passive, a genuinely different axis from the bluffer/conservative aggression axis, not a variant of either); and an **erratic/loose cannon** (high temperature, no bias — noisy rather than leaning). Validating claims: baseline beats bluffer, conservative, and calling station, each by a statistically significant margin (proving the architecture handles deviation in three different directions, not just two); baseline also beats the erratic persona specifically on temperature alone, which finally closes §4's original claim (that a low-temperature bot beats a high-temperature one) with a real result rather than only the mechanism-level check in `test_quantal.py`.
  - **Evaluation harness stopping rule:** fixed hand count per matchup for the first version (default ~5,000 hands, a real parameter) rather than adaptive CI-width stopping — simpler to build correctly first; adaptive stopping is a reasonable later refinement, not a problem for this phase.
- **Stretch:** real ranges instead of uniform-random opponent cards, multiway pots.

### Bridge build plan
- **Phase 1:** trick-taking core — deal, trick resolution, follow-suit/trump legality, 4-player turn order, partnership scoring. Mechanically simpler than poker's betting rounds. Precise scope, locked in planning:
  - **Contract is a fixed input, not derived from bidding.** Trump suit, declarer's seat, and target trick count are given at setup and don't change during play — bidding (Phase 3) is a separate, later concern.
  - **Dummy exposure:** `current_player()` always returns declarer's seat, including on every trick where it's structurally the dummy's card being chosen — the dummy player never makes a real decision, matching real bridge exactly. Declarer's strategy needs to handle both hands as one combined decision-maker, not two.
  - **Dummy's hand is visible to all four players once exposed**, not just declarer — real bridge places it face-up on the table. `information_set()` reflects this: declarer sees their own hand plus the dummy's; the two opponents each see only their own hand plus the dummy's; nobody but declarer ever sees the other opponent's hand.
  - **Opening lead:** the player to declarer's left (seat = declarer + 1, matching the seat-numbering convention already used elsewhere) leads the first trick, before the dummy is exposed to keep it simple — no procedural distinction between "lead placed, then dummy shown" the way real physical play sequences it.
  - **State tracks full, exact trick history** — every player's remaining cards, the current trick in progress, and every completed trick — with complete accuracy always, regardless of any future persona work. What a persona's *own* decision-making consumes from this history (see §3's trick-history-recall note) is a separate, later concern layered on top, not a Phase 1 question.
  - **Shared trick-resolution logic goes in `engine/trick_taking/`**, not `games/bridge/` — Court Piece needs identical mechanics later, so this is written once as genuinely generic (given a completed trick's cards-in-order plus the led suit and trump, who won; given a hand and the led suit, what's legally playable).
  - **Scoring is explicitly deferred, not blocking.** `payoff()` for Phase 1 can return raw trick counts per partnership — real contract-success scoring (bid vs. made, points) is a light, separate layer added once trick-counting itself is proven correct, not part of this phase's core state machine.
- **Phase 2:** double-dummy solver. Flagged explicitly above as the highest-risk module in the project. Split into two, since alpha-beta is a correctness precondition here, not a later speed extra — a full 13-trick tree is large enough that even validating correctness could take impractically long without it, and it changes nothing about the answer, only which provably-irrelevant branches get skipped:
  - **Phase 2a:** plain minimax with alpha-beta, operating on any given `BridgeState` (not just a fresh deal — this is what lets Phase 3's PIMC sampling and eventual `action_values()` call it mid-hand). Returns a value **per legal action**, not a single scalar for the position — a minimax search already computes a value for every branch on the way to the best one, so exposing all of them costs almost nothing and gives something structurally identical to poker's `action_values()`. Consistent viewpoint throughout: every value is declarer's-side trick count, so the maximizing/minimizing convention is just "declarer's own turn maximizes this number, an opponent's turn minimizes it" — no separate per-player framing needed. Tested on short, hand-verifiable endgames (last 1–4 tricks or so), not a fresh 13-trick deal — full-deal speed is explicitly not this phase's job. **Verify the test suite the way Phase 1's was, not just written and trusted:** deliberately introduce specific plausible bugs one at a time (e.g. mis-handling a suit, an off-by-one in whose turn it is, a wrong base case in the search), confirm the suite actually catches each one, then revert. A suite that passes on the first try is weak evidence on its own. This is now a standing practice for anything flagged high-risk in this document, not a one-off done for bridge Phase 1.
  - **Phase 2b:** transposition tables (caching positions already solved, reached via a different move order — `BridgeState` being an immutable, hashable frozen dataclass makes this close to free to add) and suit-symmetry reduction (treating equivalent low cards within a suit as interchangeable once the higher cards are gone). Aimed specifically at making a fresh full-deal solve practical, informed by real timing measurements from 2a rather than guessed at — measure where 2a actually starts to struggle, then target that.
- **Phase 3:** PIMC sampling layer on top of the solver; start bidding as a separate, rule-based track. **Double-dummy solving is exact only because it assumes zero uncertainty** — all four hands known — so PIMC's job is real inference under uncertainty on top of it: sample many complete hands consistent with what's actually been observed, solve each exactly with Phase 2, combine into a real decision. A further, clearly-later refinement on top of plain PIMC: weight those samples by how consistent they are with the actual bidding (Bayesian reasoning about which of the unseen 26 cards an opponent likely holds, given how they've bid and played), rather than sampling uniformly — the same shape of idea as poker's deferred weighted-range stretch goal, and the same recommended sequencing: uniform PIMC first, bidding-informed weighting as a distinct, labeled refinement afterward, not bundled in from day one.
  - **Bidding scope, settled:** Standard American Yellow Card (SAYC) — the most standardized, publicly documented system, and "natural" (bids describe what's actually in the hand rather than an artificial memorized meaning), which makes it tractable as clear rules rather than a convention lookup table. V1 assumes an **uncontested auction** — the opposing side always passes, no interference, no doubles — same discipline as every earlier phase deliberately assuming something away and building the honest, complete version of what remains.
    - In scope: HCP counting and balanced-shape detection; opening bids (1-of-a-suit ~12-21 HCP longest suit with a real, correctly-sourced tie-break rule for equal-length suits — verify against an authoritative SAYC reference rather than assumed/recalled, since a wrong tie-break wouldn't be caught by anything like a chip-conservation test, it'd just quietly produce legal-but-wrong bids; 1NT 15-17 balanced; 2NT 20-21 balanced; pass below opening strength); natural constructive bidding (raises with support, new suits showing values, 1NT responses, opener's rebid); correct auction termination (three passes after a real bid ends it; four passes with none is a passed-out hand with no contract, handled explicitly).
    - Explicitly deferred, named rather than silently missing (same Honest Scope discipline as everywhere else): named conventions (Stayman, Jacoby transfers, Blackwood), competitive bidding (interference, doubles), preemptive openings, the full response structure to a strong 2♣ opening (the opening itself is in scope, its deep response tree isn't yet).
    - **Bid legality**: level dominates suit rank — any higher-level bid outranks every bid at a lower level regardless of suit; within the same level, rank is clubs < diamonds < hearts < spades < notrump, which is exactly `engine/cards`' existing `Suit` ordering (confirm this alignment explicitly when building, don't just assume it).
    - **Architected for future bias/temperature, not bolted on later:** `bid_values(hand, auction) -> dict[Bid, float]` (a fit score per legal call from the SAYC rules), not a function that returns one chosen bid directly — same shape as poker's and bridge card-play's `action_values()`. V1's actual behaviour is just picking the best-scoring legal bid, but the values exist for a future persona layer to bias between, and the underlying thresholds (minimum HCP to open, minimum support to raise) are real named constants a persona could later shift, not numbers buried inline.
    - Module split: `games/bridge/hand_evaluation.py` (HCP, shape — reusable utilities) and `games/bridge/bidding.py` (the auction's own state machine — whose turn, legal calls, termination, producing a `Contract` that feeds directly into everything already built).
    - **Testing is necessarily different in character from Phases 1-3**: there's no double-dummy-style objectively computable answer here — what a bid means is convention, not a computed fact. Test against hand-constructed hands with known, textbook-standard auctions, closer in spirit to how the NLP keyword matching was validated against curated real examples than to how the solver was validated against combinatorics.
  - **Receding-horizon, not planned once and reused.** Re-sample and re-solve completely fresh at every real decision, using only what's actually known at that exact point — never try to extend or reuse an earlier hand's computation. Each card actually played is genuine new evidence that narrows what's still possible, so refitting from the current position is the correct thing to do, not just a simplification. This also means every individual solve stays a complete, exact solve of whatever's genuinely left — never artificially depth-limited — so no heuristic evaluator is needed for this to work, unlike a fixed-depth search would require.
  - **A depth gate, not a blind universal time budget** — refined from the original plan once the actual Phase 2 numbers were in hand. Never attempt a real solve once more than 9 tricks remain; go straight to the fallback below for those. Only attempt real solving at 9 tricks or fewer, where the benchmarks already show it reliably fits comfortably inside a reasonable budget. A time limit still sits underneath *that* regime as a safety net (individual deals vary even at smaller sizes, just far less wildly than at 10+ tricks) — generous, at least ~5s per sample as a starting point to tune, not asserted — but its job is catching the occasional unlucky deal, not rescuing the routine case. This is why "0 samples completed" stays genuinely rare rather than being the expected outcome for early-hand decisions, which a blind time-budget-only design (the original version of this note) would not have avoided.
  - **Baiting/false-carding as a later persona idea, not a solver capability.** The permanent limitation above still holds — no genuine value-generating deception, ever, without real opponent-belief modeling. But `engine/trick_taking/equivalence.py` already proves which cards are genuinely tied in value from a pure double-dummy standpoint; a persona could bias *which* card from a tied group gets played (favoring one that looks less committal to an opponent who can't see the full hand) at zero real cost, since the alternatives were already proven worth exactly the same. Gives a recognizably deceptive-looking style, same honest shape as poker's bluffer persona — not balanced game theory, a real style built on a real limitation, honestly labeled as such if it's ever built.
  - **Fallback when zero full samples complete in the budget:** a myopic, single-trick win-probability heuristic — for each legal card, estimate its chance of winning the current trick assuming unseen cards are evenly split among the unseen hands, no search, no lookahead. Structurally the same computation as poker's Week 2 equity engine (uniform-distribution assumption over unseen cards → a win probability), just applied to a different game. Given the fallback is genuinely the primary mechanism early in a hand (see above), it deserves the same real testing rigor as anything else load-bearing, not throwaway-quality treatment.
  - **Explicit, permanent limitation: no deceptive play (false-carding/baiting) is or can be modeled, by either the solver or PIMC as designed.** Double-dummy assumes all hands already known, so there's no opponent belief to manipulate. PIMC doesn't fix this either — within each sampled hypothetical, the opponent's response is still computed as if *they* also see that whole sampled deal, never as genuine inference from what they've actually observed. Direct parallel to poker's already-recorded limitation: no fold-equity model means poker's bots can't produce balanced GTO bluffing either, for the same underlying reason (modeling how a specific opponent's beliefs update from observed play is real game-theoretic solving under imperfect information, deliberately out of scope for this whole project). Record this in README's Honest Scope once Phase 3 is actually built, same as poker's equivalent limitation.
  - **Depth-limited search with a heuristic evaluator** (evaluate an unfinished position directly rather than either solving it exactly or falling back to the myopic heuristic) is a real alternative, kept on record rather than built now — the receding-horizon design above makes it unnecessary for a first version, since it avoids needing an evaluator at all. Worth revisiting if the time-budget/fallback split above turns out not to perform well enough once there's real play to test it against.
- **Stretch:** convention refinements, doubling, solver speed work.

### Court Piece build plan
- **Fixed trump (game 3):** declare-phase only — one player sees their first 5–6 cards and calls trump, caller role rotates each hand; Court/Piece scoring on top of trick resolution already built for bridge. With `engine/trick_taking/` already in place, this is a much lighter build than the previous two.
- **Running trump (extension):** trump starts as `None`; trick resolution skips the trump comparison until the first player who can't follow suit sets it, then the hand switches into standard trump-mode for the rest of play. Small, well-contained addition once fixed-trump works — not a separate large build.

---

## 6. UI

Decided: **web (React/TS + FastAPI)**, shared across all games, built **after** each game's engine is proven headless (CLI/text runner — needed for testing regardless). Decision logic never leaves Python; the frontend only renders state and posts actions. Not a personally-designed showcase — functional and consistent, reasonable default choices rather than deep collaborative design work per screen.

---

## 7. Phase 2 (later): hand-review tool

Once poker's bot + persona layer work end to end. Reuses the same `action_values()` function, pointed at *my own* played hands instead of a bot's live decision — for each decision point, shows what folding/calling/raising was actually worth and flags where my choice deviated.

**Honest scope:** this is the same EV/pot-odds heuristic as the bot, evaluated against an assumed opponent range — not a true solver. It will catch real, meaningful leaks (bad calls, missed value, wrong sizing) but not the subtler stuff real solvers catch (balanced bluff frequencies, blocker effects). Genuinely useful for the stated goal (getting better), not a substitute for GTO Wizard / PioSOLVER if that level of rigour is ever wanted.

Requires real additional work beyond the bot: a way to input played hands, and pattern aggregation across many hands (not just one-off numbers).

**Related, later idea — noted, not scoped:** the same (situation, chosen action) data this tool needs is also exactly what's needed to calibrate a persona's temperature/bias to actually match how a specific real person plays (the "plays like my cousin who never folds" idea in the README). The fitting itself is a small, tractable statistics problem — maximum-likelihood fitting of a handful of parameters against observed decisions, not a big modelling effort. The real barrier is data, not computation: real hole cards are only known at showdown, so every folded hand is missing exactly the information needed, and casual home-game sample sizes are small. This becomes practical once the web UI exists and hands get logged automatically as people actually play through it — worth keeping as a genuinely later idea, not a current build target.

---

## 8. Tooling (locked)

- **Dependency management:** `uv` — fast, modern, handles venvs and lockfiles in one tool.
- **Testing:** `pytest`.
- **Lint + format:** `ruff` — one fast tool instead of separate flake8/black/isort.
- **Type checking:** `mypy` — genuinely useful here specifically, since the `Game` protocol (§2) is the load-bearing contract every game must satisfy; static type checking catches a game implementation drifting from the interface before it's caught by a test.

Override any of these if there's a reason to (e.g. existing familiarity from coursework) — none of these choices are load-bearing the way the architecture itself is.

---

## 9. Open decisions still to lock

None currently blocking — repo can be scaffolded and the poker build can start. Remaining items are implementation-level and belong in Claude Code, not here: exact numeric defaults for persona temperature/bias per difficulty tier, and the confidence-interval method to use in the evaluation harness (Wilson score vs. bootstrap — either is fine, pick one when `engine/evaluation.py` is actually written).