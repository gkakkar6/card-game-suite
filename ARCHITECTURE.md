# Card Game Suite — Architecture

**Repo:** `card-suite` (Python package: `card_suite`). Plain and functional on purpose — the README carries the actual story (shared architecture across poker/bridge/Court Piece, the quantal persona model, the evaluation harness), not the name.

Purpose of this doc: brief any Claude Code session (or future me) on what's been decided and why, without re-deriving it each time. Update it as decisions change — it should always reflect current reality, not just the plan we started with.

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

**Concrete shape (Claude Code should finalise the exact types, this is the contract to preserve):**

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

**NLP opponent selection** (`engine/personas/nlp.py`) parses free text into a game-agnostic `OpponentIntent` (difficulty tier + loose descriptors). Each game maps that intent onto its own parameter space in `games/<game>/personas.py`.

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
- Poker's solver (Monte Carlo hand equity vs. pot odds) is well-scoped, testable incrementally, and gets the *entire* architecture — engine, personas, evaluation harness — proven end to end in a predictable ~3 weeks.
- Bridge's double-dummy solver (exact minimax over a full deal, given all four hands known) is **the single hardest module in the whole project**. A correct-but-slow version is realistic in 1–2 weeks; a genuinely fast one (alpha-beta pruning, transposition tables) is its own piece of work. This sits in the *middle* of bridge's build — if it stalls, there's no working game yet. Building it second, once the surrounding scaffolding is proven on poker, is lower risk than building it first.
- Bridge bidding is **not** a solved-optimum problem — it's convention-based (point count, suit length rules), closer to an expert system than an optimiser. Different kind of work from the double-dummy solver, not a continuation of it.
- `engine/trick_taking/` (resolution, double-dummy solver, PIMC sampling) is written once against bridge and genuinely reused, not reimplemented, for Court Piece — both are whist-family games with identical trick-resolution logic. Only Court Piece's declare-phase and Court/Piece scoring are new game-specific code, which is why it's a much lighter build than bridge despite being "game 3."
- Court Piece's running-trump variant is honestly an **extension** of the fixed-trump build, not a fourth full build. The only new work is treating trump as `Optional[Suit]` — undetermined until either the declare-phase (fixed) or a forced first-can't-follow-suit trigger (running) sets it.

### Poker build plan
- **Week 1:** cards, deck, 7-card hand evaluator (test against known rankings), betting round state machine.
- **Week 2:** Monte Carlo equity engine — hero hand + board + opponent range → win/tie/loss via simulation. Sanity-check against known references (AA vs KK ≈ 80%). This *is* `action_values()`.
- **Week 3:** EV-based decisions, quantal persona layer on top, bot-vs-bot evaluation with CIs.
- **Stretch:** real ranges instead of uniform-random opponent cards, bet-sizing as a decision axis, multiway pots.

### Bridge build plan
- **Week 1:** trick-taking core — deal, trick resolution, follow-suit/trump legality, 4-player turn order, partnership scoring. Mechanically simpler than poker's betting rounds.
- **Week 2:** double-dummy solver. Flagged explicitly above as the highest-risk module in the project.
- **Week 3:** PIMC sampling layer on top of the solver; start bidding as a separate, rule-based track.
- **Stretch:** convention refinements, doubling, solver speed work.

### Court Piece build plan
- **Fixed trump (game 3):** declare-phase only — one player sees their first 5–6 cards and calls trump, caller role rotates each hand; Court/Piece scoring on top of trick resolution already built for bridge. With `engine/trick_taking/` already in place, this is closer to a 1-week build than a multi-week one.
- **Running trump (extension):** trump starts as `None`; trick resolution skips the trump comparison until the first player who can't follow suit sets it, then the hand switches into standard trump-mode for the rest of play. Small, well-contained addition once fixed-trump works — not a separate multi-week build.

---

## 6. UI

Decided: **web (React/TS + FastAPI)**, shared across all games, built **after** each game's engine is proven headless (CLI/text runner — needed for testing regardless). Decision logic never leaves Python; the frontend only renders state and posts actions. Not a personally-designed showcase — functional and consistent, reasonable default choices rather than deep collaborative design work per screen.

---

## 7. Phase 2 (later): hand-review tool

Once poker's bot + persona layer work end to end. Reuses the same `action_values()` function, pointed at *my own* played hands instead of a bot's live decision — for each decision point, shows what folding/calling/raising was actually worth and flags where my choice deviated.

**Honest scope:** this is the same EV/pot-odds heuristic as the bot, evaluated against an assumed opponent range — not a true solver. It will catch real, meaningful leaks (bad calls, missed value, wrong sizing) but not the subtler stuff real solvers catch (balanced bluff frequencies, blocker effects). Genuinely useful for the stated goal (getting better), not a substitute for GTO Wizard / PioSOLVER if that level of rigour is ever wanted.

Requires real additional work beyond the bot: a way to input played hands, and pattern aggregation across many hands (not just one-off numbers).

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
