# Composable Symmetry-Generalization Experiments — Design

**Date:** 2026-06-02
**Status:** Brainstorming → pending user review of this spec
**Branch:** `parker/symmetry-experiments`
**Builds on:** `docs/design/specs/2026-06-01-tiny-transformer-ttt-generalization-design.md`

## Background & motivation

The original experiment trains tiny decoder-only transformers to behavior-clone
optimal tic-tac-toe play over move-*sequences*, with **all horizontal-winning moves
filtered out of the training loss** (defensive horizontal *blocks* are kept). It then
probes whether the model (RQ1) *scores* held-out horizontal wins — recognizing that a
horizontal win is a 90°-rotated vertical — or (RQ2) only *blocks* them.

The prior 60-epoch sweep (`results_prev_60ep/`) gave a clear result across the full
grid (3.8k → 202k params, layers 1→4):

| metric | small → large |
|---|---|
| **H_win** (held-out offense) | **flat ≈ 0.05 → 0.10** — essentially chance / possibly sub-chance, at every depth & width |
| **H_block** (defense; present in training) | **0.58 → 1.00** — rises strongly with capacity |
| V_win / D_win (controls; present in training) | 0.72 → 0.87 — the model *can* play wins it saw |

**Conclusion so far:** defense-only. No offensive rotational-symmetry transfer at any
tested scale. This spec designs the follow-up experiments to understand *why*, and what
would change it.

## Research questions for this iteration

- **Q1 (encoding):** Does a symmetry-friendlier *input* representation — factored
  `(row, col)` coordinates instead of a flat cell index — unlock the rotational transfer
  that flat indices did not?
- **Q3 (depth):** Does stacking layers beyond 4 (L8/L16) ever trigger the symmetry
  realization? (Prior L1→L4 trend says no; we test the edge.)
- **Q4 (positive control):** When horizontal wins are *kept* in training, does H_win rise
  to the V/D ceiling? This is the causal anchor for "the filter caused the gap."
- **Q5 (translational):** If only the **top row's** horizontal wins are kept in training
  (middle + bottom dropped), can the model generalize to the unseen rows? This tests the
  *easier* translational transfer vs. the failed rotational transfer.
- **Q-head (output shape):** Is the flat 9-way output head suboptimally shaped for this
  transfer? Two specific hypotheses: (a) **embedding-tying** propagates symmetry learned
  in embedding space to the readout; (b) a **factored `row⊕col` head** shares parameters
  across cells in a line. Plus a diagnostic: is the H_win floor *failure-to-learn* or
  *active suppression* by the softmax?

## Core design — four orthogonal, composable axes

Every experiment is a **`Condition`**: a setting of four independent knobs on one shared
runner. Individual experiments vary one knob off baseline; combinations set several. No
knob's code is aware of the others, so combining requires **no new code** — only a new
`Condition` entry.

| Axis | Values | Question |
|---|---|---|
| **Encoding** (input) | `flat` (1 token/move, current) · `rowcol` (2 tokens/move) | Q1 |
| **Output head** | `flat9` (current) · `tied` (I/O embedding tying) · `factored` (row⊕col) | Q-head |
| **Horizontal filter** | a *set of rows to drop*: `{}`=none · `{0,1,2}`=all · `{1,2}`=keep-top | Q4, Q5 |
| **Grid / depth** | layer/head/d_model list, extendable with L8 (±L16) | Q3 |

### Axis 1 — Encoding (`ttt/encoding.py`, new)

An `Encoding` value object exposing: `name`, `vocab_size`, `max_len`, `tokens_per_move`,
`bos_id`, `pad_id`, and `encode(path) -> list[int]` (BOS-prefixed). Two instances:

- **`FLAT`** — current behavior. Cell `0–8` → one token. `vocab_size=11`
  (cells 0–8, BOS=9, PAD=10), `max_len=10`, `tokens_per_move=1`.
- **`ROWCOL`** — each cell `c` → two tokens `[ROW_{c//3}, COL_{c%3}]` using **distinct
  token families**: `ROW_0..2`=ids 0–2, `COL_0..2`=ids 3–5, `BOS`=6, `PAD`=7
  (`vocab_size=8`). Sequence: `[BOS, ROW,COL, ROW,COL, …]`, `max_len=19`,
  `tokens_per_move=2`.

**Rationale.** Under flat indices a 90° rotation is an opaque permutation
(0→2, 1→5, 3→1, …). Under `(row,col)` it is the clean transform `(r,c) → (c, 2−r)` — a
row/col family swap plus reflection. This directly tests whether representational
*accessibility* of the symmetry is the bottleneck.

**Comparability caveat (must be documented in results).** Different encodings have
different embedding-table and positional-embedding sizes, so param counts differ. Across
encodings, compare at **matched `(n_layer, n_head, d_model)`**, never by raw param count.
`rowcol` also roughly doubles sequence length (more compute; different positional load) —
an acknowledged confound, not eliminated.

### Axis 2 — Output head (`ttt/model.py`)

The head is selected by name; output is always interpreted as a distribution over the
9 cells, so **H_win measures the identical quantity across head choices**.

- **`flat9`** — current `nn.Linear(d_model, 9)`. Per-cell readout `w_i`; the nine
  readouts are independent (no symmetry tying), but each is already shared across line
  types for the same cell.
- **`tied`** — flat 9-way logits computed by tying to the input cell embeddings, so
  "play cell c" reads out along cell c's embedding direction; symmetry discovered in
  embedding space propagates to the readout. (Defined for the `flat` encoding, where
  there is a 1:1 cell↔token correspondence. For `rowcol`, tying is not well-defined at
  the cell level; `tied` + `rowcol` is disallowed — see validation.)
- **`factored`** — predict a 3-way **row** distribution and a 3-way **col** distribution
  from `h`, combine to a 9-cell logit grid (`logit[r,c] = row_logit[r] + col_logit[c]`).
  Shares parameters across all cells in a row and all cells in a column — an output-side
  inductive bias aligned with lines. **Caveat:** imposes conditional independence
  (row ⫫ col | h) that the true optimal-cell distribution violates, so it may *cost*
  in-distribution accuracy; the V/D controls will reveal this.

**Diagnostic (the H_win floor).** Filtered horizontal cells are not merely
un-reinforced: in every kept training board where another cell is the optimal target,
cross-entropy pushes the horizontal-winning cell's logit *down* as a competing class. If
the H_win floor is **below** the random-legal-move baseline, the model is being trained to
*avoid* horizontal completions, not merely failing to learn them. Confirming this changes
how every condition is interpreted. **Deliverable:** compute `random_baseline_rate` on the
H-win probe set and report it alongside H_win for all conditions.

### Axis 3 — Horizontal filter as a row-set (`ttt/dataset.py`, `ttt/board.py`)

Generalize the boolean `filter_horizontal` to `drop_horizontal_rows: frozenset[int]`
(subset of `{0,1,2}`, where row 0 = `(0,1,2)`, row 1 = `(3,4,5)`, row 2 = `(6,7,8)`). A
target is dropped iff it completes a horizontal line whose row is in the set.

- `frozenset()` → drop nothing = **positive control (Q4)**
- `{0,1,2}` → drop all = **current main condition**
- `{1,2}` → keep top row, drop middle + bottom = **translational (Q5)**

New helper in `board.py`: `horizontal_row_completed_by_move(board, cell) -> int | None`
(0/1/2 or None) — the single source of truth for both the filter and per-row probes.
`move_completes_horizontal` is re-expressed in terms of it (kept for back-compat).

### Axis 4 — Grid / depth

Baseline grid unchanged: `layers ∈ {1,2,4} × heads ∈ {1,2,4} × d_model ∈ {16,32,64}`,
keeping the `d_model % n_head == 0` filter. The depth experiment appends **L8** (L16
deferred — see Decisions). L8/D64 roughly doubles L4 params/time.

## Probes, generalized per-row (`ttt/probes.py`, `ttt/evaluate.py`)

- `win_available_probes(line_type, rows=None)` and
  `block_needed_probes(line_type, rows=None)` — `rows` restricts horizontal probes to
  specific rows. The existing **pure-single-line-type** and **unique-optimal** constraints
  are preserved, so a held-out horizontal-win probe board is *exactly* a board dropped
  from training.
- The **translational** condition reports H_win split into **seen row {0}** vs
  **held-out rows {1,2}** — that split is the experiment's signal. The row-subset of the
  probe set aligns exactly with the train/dropped split.
- `probe_prefixes`, `model_move`, and `evaluate_probes` thread the active `Encoding`
  through instead of calling `encode_prefix` directly.

## Shared runner & rigor scaffolding (`ttt/sweep.py`)

A `Condition` dataclass (`name`, `encoding`, `head`, `drop_horizontal_rows`, `grid`,
`seeds`, `epochs`, `lr`, `batch_size`, `max_orderings`) and `run_condition(cond)` that
reuses the existing process-pool sweep. **Every** condition automatically gets:

- **5 seeds**, results reported as **mean ± std** (no single-seed claims).
- **V/D controls always evaluated** → admissibility gate: a condition's horizontal result
  is admissible only if its vertical/diagonal controls separate from chance.
- **Random baseline per probe set** (`random_baseline_rate`), plotted as the "chance"
  line so "near chance" / "sub-chance" is precise.
- The `none`-filter (positive control) condition serves as the H_win **ceiling** overlay.
- Deterministic, seeded training (unchanged).
- Persisted to `results/<condition_name>/{raw.json, agg.json, *.png}`.

## Experiment matrix & sequencing

**Phase 1 — individual.** Each varies one axis off the baseline
(`flat` encoding, `flat9` head, drop-all filter, standard grid):

| ID | Encoding | Head | Drop rows | Grid | Purpose / expectation |
|---|---|---|---|---|---|
| **E0** | flat | flat9 | {0,1,2} | std | **Re-baseline** — reproduce prior result on refactored pipeline (regression guard) |
| **E1** | flat | flat9 | {} | std | **Positive control (Q4)** — H_win should rise to V/D ceiling |
| **E2** | rowcol | flat9 | {0,1,2} | std | **Encoding (Q1)** — does factored input unlock rotational H_win vs E0? |
| **E3** | flat | flat9 | {1,2} | std | **Translational (Q5)** — held-out rows {1,2} vs seen row {0} |
| **E4** | flat | flat9 | {0,1,2} | +L8 | **Depth (Q3)** — does depth > 4 trigger transfer? |
| **E5a** | flat | tied | {0,1,2} | std | **Head (Q-head)** — embedding-tying effect, isolated vs E0 |
| **E5b** | flat | factored | {0,1,2} | std | **Head (Q-head)** — factored row⊕col head, isolated vs E0 |

**Phase 2 — combinations** (selected *after* reading Phase-1 results; each is just a new
`Condition`, no new code). The marquee combination is **factored head × rowcol input**
(strongest structural alignment of input and output to board lines). Other candidates:
rowcol × translational, rowcol × deeper, tied × translational. Exact set chosen based on
which individual axes move H_win.

## Module / file plan

- **New:** `ttt/encoding.py`; `tests/test_encoding.py`.
- **Modify:**
  - `ttt/board.py` — add `horizontal_row_completed_by_move`; re-express
    `move_completes_horizontal` on top of it.
  - `ttt/dataset.py` — `build_examples(encoding, drop_horizontal_rows, max_orderings)`;
    row-set filter; `collate_fn` uses `encoding.pad_id`. A thin back-compat shim keeps the
    old boolean working for E0's regression test.
  - `ttt/model.py` — `GPTConfig` takes `vocab_size`/`max_len` from the encoding and a
    `head` selector (`flat9`/`tied`/`factored`); implement the three heads.
  - `ttt/probes.py` — `rows=` parameter on both probe builders; `probe_prefixes(encoding)`.
  - `ttt/evaluate.py` — thread `encoding` through `model_move` / `evaluate_probes`.
  - `ttt/sweep.py` — `Condition` dataclass + `run_condition`; per-row metrics; per-row
    H_win split; positive-control overlay in the plot.
  - `run_experiment.py` — define conditions E0–E5; `--condition <name>` selector (default:
    run all Phase-1); write to `results/<name>/`.

## Edge cases & validation

- `tied` head requires the `flat` encoding (1:1 cell↔token). `tied` + `rowcol` is rejected
  at `Condition` construction with a clear error.
- `rowcol` sets `max_len=19`; positional embedding sized to match.
- Possibly-small per-row probe sets → handle `n=0` gracefully (rate 0.0, flagged), as the
  current code partially does.
- `factored` head: `n_out` semantics change to a 3+3 head internally but still emit 9
  cell logits for masking/argmax, so `evaluate.py` is unchanged.
- Compute note: L8 ≈ 2× L4 time; `rowcol` ≈ 2× sequence length. The runner is already
  parallel across cores; document expected runtime per condition.

## Testing strategy

- **Regression:** E0 (`flat`, drop-all) produces the **same example set** as the legacy
  `filter_horizontal=True`, proving the refactor changes nothing in the baseline.
- **Encoding:** round-trip — `encode(path)` decodes back to the originating board;
  `rowcol` emits exactly 2 tokens/move; `vocab_size`/`max_len` correct per encoding.
- **Filter:** row-set drops exactly the specified rows and keeps the others
  (`{}`, `{0,1,2}`, `{1,2}` each checked).
- **Probes:** per-row probes are pure single-line-type, unique-optimal, and on the
  requested row; seen/held-out split matches the filter.
- **Head:** `flat9`/`tied`/`factored` all output shape `(B, 9)`; `factored` row⊕col
  combination is correct; `tied`+`rowcol` raises.
- **Smoke E2E:** per-condition tiny run (small grid, 1 seed, few epochs) writes a plot,
  mirroring the existing `tests/test_end_to_end.py`.

## Prerequisites

- **No Python is currently installed** on this machine (only the Microsoft Store stub).
  A real Python 3.11+ and a venv with `requirements.txt` must be set up before any
  experiment runs. This is the first implementation task.

## Decisions (resolved, open to change during review)

1. **Output head is a full 4th axis** (`flat9`/`tied`/`factored`) with dedicated E5.
   *(Resolved with user.)*
2. **Marquee Phase-2 combination = factored head × rowcol input.** *(Resolved with user.)*
3. **Seeds = 5** per condition (compute parity with prior; the matrix is now large).
   Could bump to 8 for tighter error bars on the highest-signal conditions — flagged.
4. **L16 deferred.** E4 includes L8 only; add L16 as a follow-up *if* L8 shows any H_win
   movement. The flat L1→L4 trend predicts low payoff.
5. **Output is always a 9-cell distribution**, even for `factored`/`rowcol`, so H_win is
   comparable across all conditions.

## Diagnostic result (E0/E1 fast run)

Run on the refactored pipeline as the first deliverable: 3 configs
(L1H1D16, L2H2D32, L4H4D64) × 3 seeds × 20 epochs, `max_orderings=4`. Artifacts in
`results/E0_diag/` and `results/E1_diag/`. Mean H_win/V_win over seeds:

| config | E0 H_win (drop-all) | E1 H_win (keep-all) | E0 V_win (control) |
|---|---|---|---|
| L1H1D16 | 0.247 | 0.572 | 0.642 |
| L2H2D32 | 0.125 | 0.956 | 0.957 |
| L4H4D64 | 0.169 | 0.995 | 0.994 |

**Random-legal-move baseline on the H-win probe set: 0.40.**

**Finding — the H_win floor is *active suppression*, not failure-to-learn.** Under the
drop-all filter (E0), H_win (0.12–0.25) sits **below** the 0.40 random baseline at every
capacity — i.e. *sub-chance* — even as the V_win control climbs to ~0.99. A model that
had merely failed to learn horizontal offense would score *at* chance; scoring *below* it
means cross-entropy is actively pushing the horizontal-winning cell's logit down as a
competing class. The positive control (E1, keep-all) lifts H_win to the V/D ceiling
(0.57 → 0.99), confirming the filter *causes* the gap rather than some intrinsic
difficulty of horizontal lines. Every condition must therefore be read against the
plotted 0.40 chance line, and "near chance" for the main conditions actually means
"suppressed below chance." (Absolute rates are higher than the prior 60-epoch sweep's
≈0.05–0.10 because this is a 20-epoch, 3-config diagnostic; the *relationship* — E0 below
baseline, E1 at ceiling — is the signal.)

## Scope / non-goals

- Single game (tic-tac-toe) only.
- **No symmetry data augmentation** (hard constraint, inherited). `(row,col)` encoding and
  factored head are *representation* choices, not augmentation; the row-subset filter is
  applied on board semantics.
- Other games, board-state (non-sequence) input models, and "remove all orthogonals"
  conditions remain out of scope.
