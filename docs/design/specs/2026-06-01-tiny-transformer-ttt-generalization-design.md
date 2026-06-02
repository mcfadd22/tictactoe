# Tiny Transformer Generalization on Filtered Tic-Tac-Toe — Design

**Date:** 2026-06-01
**Status:** Approved (brainstorming → ready for implementation plan)

## Research Question

Do very small transformers learn the *underlying spatial symmetry* of a board game,
or do they only memorize the patterns present in their training data?

Concretely: train a tiny decoder-only transformer on tic-tac-toe move-sequences from
which **all horizontal-winning moves have been removed**, then probe whether the model:

1. **RQ1 — Offense / generalization:** spontaneously *scores* horizontal wins it was
   never trained to make (true transfer of the win concept).
2. **RQ2 — Defense / memorization:** *blocks* opponent horizontal threats (a pattern
   still present in training), even if it never scores them itself.

### Reframing via symmetry (critical context)

Tic-tac-toe has 8-fold dihedral symmetry (D₄). The 8 win-lines split into two classes:
**orthogonal** (3 horizontal + 3 vertical, interchangeable under 90° rotation) and
**diagonal** (2). We remove **only horizontals**, leaving verticals in training. A
horizontal win is therefore a 90°-rotated vertical win, so RQ1 is precisely:

> Did the tiny transformer learn the board's 90° rotational symmetry well enough to
> transfer the win-pattern from verticals to horizontals?

A decoder over move-sequences receives **no spatial prior** — cell tokens are events in
time, with no hint that {0,3,6} is a line or that {0,1,2} is its rotation. To score an
unseen horizontal the model must build an internal board geometry *and* its symmetry,
purely from vertical/diagonal wins. This is a demanding test for a tiny model; a negative
result (no transfer) is still a meaningful finding.

## Architecture / Approach

- **Model task:** next-move prediction (behavior cloning of optimal play) with a
  **decoder-only, autoregressive transformer** over move-sequences. Native transformer
  modality; tests board state-tracking + symmetry jointly.
- **Input representation:** move-sequence tokens. Vocab = cells `0–8`, plus `BOS`
  (optional `EOS`). Whose move it is is inferred from ply parity. Sequence is
  `BOS, m₁, m₂, …`; predict `m_{k+1}` from the prefix.
- **Head:** softmax over the 9 cells; illegal moves masked at evaluation time.

## Data

- **Solver:** exhaustive minimax (game is tiny), caches optimal move(s) per position.
- **Coverage:** enumerate all ~5,478 reachable positions (full coverage, no sampling
  gaps). Each position is rendered as a **move-sequence prefix**.
- **Labels:** for each prefix, the target next-token is the **solver-optimal move** for
  the side to move. Decouples coverage from label quality — every position labeled with
  good play regardless of how it was reached.
- **Order-invariance:** generate multiple move-orderings per position (capped, e.g. ≤4)
  so the model is trained and probed across orderings. Enables an order-invariance
  sub-measurement.

### The Filter (experimental manipulation)

- **Loss-masking:** any target move that **completes a horizontal three-in-a-row for the
  mover** is masked out of the training loss (branch trimmed). The model is **never
  trained to score a horizontal.**
- **Blocks preserved:** targets that *block an opponent's horizontal threat* (taking the
  third cell defensively) remain in the loss. RQ2 premise holds.
- **No symmetry augmentation** — hard constraint. Augmenting with rotations would turn
  verticals into horizontals and silently reintroduce what we filtered. If more data is
  ever needed, apply the horizontal filter *after* any augmentation.

## Model Sweep (independent variable)

- Minimal nanoGPT-style decoder: learned token + positional embeddings, causal
  self-attention, MLP, final 9-way head.
- **Grid:** `layers ∈ {1,2,4}`, `heads ∈ {1,2,4}`, `d_model ∈ {16,32,64}`.
- **Seeds:** 5–8 per config (tiny models are high-variance; claims are statistical).
- **Stack:** PyTorch, CPU-friendly (seconds–minutes per model), single small codebase.

## Evaluation

### Probe suite (primary, clean metric)

Held-out constructed positions (horizontal wins were never training targets):

- **`win-available` probes:** side-to-move has a horizontal two-in-a-row + empty third;
  playing it wins. **Metric:** horizontal-win completion rate (model's chosen move takes
  the winning cell).
- **`block-needed` probes:** opponent threatens a horizontal three; only correct move is
  to take the third cell. **Metric:** horizontal-block rate.
- **Control probes:** identical win/block probes for **vertical** and **diagonal** lines.
  The model saw these patterns, so it should ace them. Controls are the validity check —
  they distinguish "didn't generalize" from "can't play."
- Each probe presented across multiple move-orderings → **order-invariance** score.

### Game-play (secondary, naturalistic)

Model plays full games vs **random**, **perfect (minimax)**, and **self-play** opponents;
logs scanned post-hoc for spontaneous horizontal-win attempts and blocks.

### Outcome interpretation (per config, controls passing)

| Outcome | H-win rate | H-block rate | Interpretation |
|---|---|---|---|
| Full generalization | high | high | Learned rotational symmetry, transferred to offense (RQ1 ✅) |
| Defense-only | ~chance | high | Learned defensive pattern, no offensive transfer (RQ2 ✅) |
| No transfer | ~chance | ~chance | Did not build board geometry |

"~chance" = a random-legal-move baseline, defined precisely so "high" is statistically
separated across seeds. A config's horizontal result is **admissible only if its
vertical/diagonal controls pass**.

### Headline result

**Horizontal-win rate (and block rate) vs. model capacity**, mean ± spread over seeds,
with vertical/diagonal controls overlaid as the ceiling — revealing the **capacity
threshold** at which symmetry generalization emerges, or showing it does not within the
tiny range.

## Rigor / Reporting

- All metrics aggregated across seeds (mean ± spread); no single-seed claims.
- Controls must pass for a config's horizontal result to be admissible.
- Persist per config: training curves, probe scores, example games, capacity plot.

## Scope / Non-goals

- Single game (tic-tac-toe) only; no other games this iteration.
- No symmetry data augmentation (by design).
- Board-state-input control model and "remove all orthogonals" condition are noted as
  possible follow-ups but are **out of scope** for this spec.

## Implementation Refinements (added during build)

Three refinements were made while implementing the probes and reporting, to keep the
measurement clean:

1. **Pure single-line-type probes.** A win-available / block-needed probe for a given
   line type only includes positions where the winning/blocking move completes **exactly
   that one line type** (no simultaneous fork-win of another type). This removes a
   confound where a "horizontal win" would also complete a vertical the model already
   learned, which would inflate the horizontal-win rate without genuine generalization.

2. **Unique-optimal win and block probes.** A probe is included only when the
   winning/blocking move is the **unique optimal move** (`optimal_moves(board) ==
   (cell,)`). For blocks this excludes already-lost positions where the solver is
   indifferent and blocking is pointless. For wins it means the probe measures "takes the
   available win" cleanly (an alternate winning move no longer counts as a miss), and — for
   horizontal — makes the probe board *exactly* a board dropped from training (the unique
   optimal there was the filtered horizontal win), so the held-out set is airtight.
   Yields: win 457 H / 457 V / 395 D; block 264 H / 264 V / 308 D (H = V confirms D₄
   symmetry). The full sweep trains at 150 epochs.

3. **Capacity plot groups by parameter count.** Parameter count depends only on
   `(n_layer, d_model)` — `n_head` does not change it — so the headline plot groups the
   three head-count variants that share a parameter count and averages them, giving one
   point per distinct capacity. Full per-config detail is retained in `results/raw.json`.
