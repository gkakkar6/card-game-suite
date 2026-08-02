# card-suite

Card-game bots that make decisions through actual probabilistic reasoning — Monte Carlo equity, exact minimax solving, statistical evaluation — rather than fixed heuristics dressed up as difficulty settings.

## What this is

Four real card games, one architecture. Every game implements the same `Game` interface. Every bot is a computed value-per-action plus a tunable amount of noise and systematic bias (a quantal-response model — not a hardcoded "easy/medium/hard"), so difficulty and playing style are genuinely parameterised rather than faked. Every claimed edge between bots is backed by a confidence interval from head-to-head evaluation, not a vibe.

Full design rationale, including why the games are built in this order and what's deliberately out of scope: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Status

Updated as things ship — this table is the actual current state, not a plan dressed up as progress.

| Component | Status |
|---|---|
| Repo scaffold / `Game` protocol | ✅ done |
| Poker (heads-up/multiway NLHE) — decision engine | ✅ done (cards, hand evaluator, betting, equity, action values, personas, evaluation harness) |
| Poker — playable (CLI) | ✅ done (1–5 bots, persistent bankroll, tournament or cash mode, free-text opponent selection) |
| Poker — free-text opponent selection (NLP) | 🚧 in progress (rule-based keyword matching works; tolerance being added now) |
| Poker — playable (web) | ⬜ not started |
| Bridge | ⬜ not started |
| Court Piece — fixed trump | ⬜ not started |
| Court Piece — running trump (Be-ranga Double Sar) | ⬜ not started |
| Web UI | ⬜ not started |
| Hand-review tool | ⬜ not started |

## Why this exists

These are the games I actually grew up playing long sessions of Court Piece and casual poker with cousins, the kind of thing that gets harder to organise as everyone grows up, everyone gets busy and we dont get the opportunity to do it as often. Part of the point here is replicating that: personas aren't meant to be generic difficulty presets, they're built so one can eventually be calibrated against how a specific person actually plays "hard mode" can mean "plays like my cousin who never folds," not just "plays optimally." The temperature/bias split in the persona model (see `ARCHITECTURE.md`) exists partly for this reason.

It's also the project that forced me to actually formalise what "correct play" means instead of just playing based off my own experience and understanding of the game and start to think of it as a mathematical problem. University modules on probabilistic modelling, inference, and Monte Carlo methods were the ones I enjoyed the most, but I kept wondering how they'd actually hold up applied to something concrete rather than a problem set. This is why this was a perfect opprunity to explore that as this is a genuinely interesting technical problem (defining and computing "optimal" in a game with hidden information, then testing it statistically rather than trusting a feeling) that also produces something I'll actually sit down and use, not a demo that gets abandoned once it's finished. A faster, more honest feedback loop than table experience alone. Not a substitute for real solvers (GTO Wizard, PioSOLVER, Equilab) where those already exist and apply — see "Honest scope" below.

## Architecture, briefly

- **`engine/`** — shared, game-agnostic. The `Game` protocol, Monte Carlo rollout, the evaluation harness (bot-vs-bot with confidence intervals), and the persona engine (temperature = noise, bias = systematic deviation from optimal).
- **`games/`** — per-game rules, and each game's `action_values()`: Monte Carlo hand equity for poker; exact double-dummy solving + Monte Carlo determinization (PIMC) for bridge and Court Piece, which share a trick-taking core.

Details, including per-game build plans and what's deliberately simplified, in `ARCHITECTURE.md`.

## Personas

Currently five: **baseline** (near-optimal), and four ways of deviating from it — **bluffer** (too aggressive), **conservative** (too passive), **calling station** (won't fold), and **erratic** (just noisy, no lean). Each is a temperature/bias setting on the same shared decision engine, not a separate implementation (`ARCHITECTURE.md` §3). More can be added the same way, and the eventual goal is calibrating one against how a specific real person actually plays (see "Why this exists" above), not just hand-picking more archetypes.

Selectable by name, or by describing a style in plain English (e.g. "someone tight and careful," "a loose aggressive maniac") — a rule-based keyword matcher, not an LLM call, deliberately: fully deterministic, testable with plain unit tests, and works with zero setup if you clone this repo. It maps free text onto these five already-measured personas rather than generating new ones from the description — letting text set temperature/bias directly, even through a lookup table, would reintroduce the exact unvalidated-parameter risk the empirical tuning exists to avoid. It always says what it inferred and asks for confirmation before committing, never silently.

## Honest scope

The poker bot plays EV/pot-odds-based decisions — a solid, testable heuristic, not solved GTO (which is computationally intractable to solve exactly outside heavy abstraction, and that's not what this project is trying to do). The bridge and Court Piece solvers compute genuinely optimal play for the perfect-information sub-problem via double-dummy solving, then sample over the real imperfect-information game via PIMC — a real, standard technique, not a toy simplification.

The poker equity engine currently treats every possible opponent hand as equally likely (a uniform range) — the same number a tool like Equilab shows as "vs. random hand." It doesn't yet model what a real opponent would plausibly be holding given how they've played. Weighted, non-uniform ranges are a planned extension, not an oversight — either hand-coded from published opening-range theory, or eventually fitted from real observed play (see "Why this exists" above).

This isn't just a theoretical gap — it's been measured. The baseline persona actually **loses** to the conservative one, −5.23 chips/hand (95% CI [−7.84, −2.60], 1,000 hands): because equity is always computed against a uniform range, baseline has no way to notice that a tight opponent's bets mean something real, so it keeps paying off a player who only bets when genuinely ahead. Recorded as an expected failure (`xfail`, `strict=True`) rather than hidden or tuned away, so it turns into a real passing result the day opponent modelling lands — see `games/poker/tests/test_validation.py`.

## Running it

```
uv sync
uv run python scripts/play_cli.py
```

Prompts for table size (1–5 bots), and a persona per bot — by name, or by describing a style in your own words (it'll say what it inferred and confirm before committing). Defaults to a real tournament — increasing blinds, strict elimination, ends with a winner — answer no to either prompt for casual, endless play instead.