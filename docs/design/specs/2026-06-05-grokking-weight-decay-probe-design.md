# Grokking / Double-Descent Weight-Decay Probe — Design

**Date:** 2026-06-05
**Branch:** `parker/symmetry-experiments`
**Builds on:** `docs/design/specs/2026-06-02-composable-symmetry-experiments-design.md`

## Motivation

Phase-1 (E0–E5b) answered the central question — *does a tiny transformer learn
tic-tac-toe's rotational symmetry well enough to transfer it to held-out
horizontal offense?* — with a clear **no**. Every true generalization condition
failed:

| Condition | Held-out H-win | Reading |
|-----------|----------------|---------|
| **E0** (rotational, zero horizontal exposure) | 0.158 | sub-chance |
| **E3** (translational, top row seen, rows 1–2 held out) | rows 1–2 ≈ 0.15 | sub-chance |
| E2 (rowcol), E4 (depth), E5a/b (heads) | 0.13–0.17 | sub-chance |

(E1 keep-all and E_AUG augmentation reach ~0.95–0.99, but both put horizontal
wins *in the training data* — they are positive controls, not transfer. E_AUG in
fact violates the Phase-1 "no symmetry augmentation" hard constraint.)

**The anticipated objection.** Phase-1 trained with plain `Adam`, no weight
decay, for a fixed **150 epochs**. A reviewer can reasonably push back: *the
"no transfer" claim is unproven because the model was never given the conditions
under which late generalization is known to appear* — specifically **grokking**
(Power et al. 2022: delayed generalization that emerges far past the
memorization plateau, tied to **weight decay**) and **epoch-wise double descent**
(test error descends, rises, then descends again over long training).

This probe pre-empts that objection by giving the model exactly those
conditions on the two genuine generalization conditions (E0, E3) and recording
the **full held-out trajectory** so the result — grok or no grok — is shown as a
curve, not a single endpoint.

## Goal

Add **weight decay** as a fifth composable axis on the existing `Condition`
runner, plus a **per-epoch held-out evaluation hook**, and run a focused long
training probe on E0 and E3 across a weight-decay sweep. Produce the held-out
H-win-vs-epoch curves that demonstrate whether any late transition occurs.

Non-goal: re-running the full Phase-1 capacity grid at long epochs. Model-wise
double descent is already covered by the existing E0 capacity sweep (held-out
H-win is flat sub-chance across 3.8k→200k params); this probe cites that result
rather than reproducing it.

## What changes (composable, default-preserving)

Every new parameter defaults to current behavior, so E0–E5b and the entire test
suite stay **bit-for-bit identical**.

### 1. Optimizer + weight decay (`ttt/train.py`)

`train_model` gains a `weight_decay: float = 0.0` parameter and switches
`torch.optim.Adam` → `torch.optim.AdamW(lr=lr, weight_decay=weight_decay)`. With
`weight_decay=0.0`, AdamW is equivalent to Adam for this setup, preserving the
baseline. `weight_decay` is threaded `Condition` → `run_condition` →
`_cond_worker_task` → `_train_and_eval` → `train_model`.

### 2. Per-epoch eval hook (`ttt/train.py` + `ttt/sweep.py`)

The one genuinely new capability. Grokking and epoch-wise double descent are
**temporal** — the current harness records only the final metric and cannot
observe them.

- `train_model` gains optional `eval_every: int = 0` and
  `eval_hook: Callable[[model, epoch], dict] | None = None`. When
  `eval_every > 0` and a hook is given, every `eval_every` epochs (and on the
  final epoch) `train_model` calls the hook and appends
  `{"epoch": e, "train_loss": l, "metrics": {...}}` to a **trajectory** list,
  returned as a third value: `(model, history, trajectory)`. When `eval_every == 0`
  the trajectory is empty and the existing code path is unchanged.
- `train.py` stays ignorant of probe internals: the hook is built in `sweep.py`
  as a closure over `probe_sets / paths / encoding`, calling the existing
  `evaluate_probes`. Clean boundary — `train.py` knows only "call this every N
  epochs and collect what it returns."
- `_train_and_eval` returns the trajectory alongside the final metrics row so it
  reaches `run_condition`'s raw rows.

### 3. New `Condition` fields + grok conditions (`ttt/sweep.py`, `run_experiment.py`)

- `Condition` gains `weight_decay: float = 0.0` and `eval_every: int = 0`.
- New `GROK_GRID`: two small configs — `L1H1D16` (3,801 params) and
  `L2H2D32` (26,441). Grokking is reliably observed on *small* models under
  weight decay, so the probe weights small. The large `L4H2D64` (201,993) is
  deliberately excluded: it is the least grokking-relevant config and at 20k
  epochs costs ~29 h/run (measured), ~72% of the probe's total compute on its
  own. Its overfitting behavior is already documented under E_AUG and is cited
  rather than re-run here.
- New condition entries: **{E0, E3} base × weight_decay {0.0, 0.01, 0.1, 1.0}**,
  at long epochs. Naming: `E_GROK_<base>_wd<value>` (e.g. `E_GROK_E0_wd0.1`).
  The `wd=0.0` arm is the **long-training control** — it isolates "did weight
  decay matter?" from "did simply training longer matter?"
- Parameters (all easily tunable in one place):
  - epochs = **20,000**, `eval_every` = **100** → 200-point curves
  - seeds = **(0, 1, 2)** (reduced from 5; each run is ~130× longer)
  - Runs: 2 configs × 4 WD × 2 bases × 3 seeds = **48 runs × 20k epochs**.
  - Measured cost (i9-14900HX, 1 torch thread/worker, single-process timing):
    L1H1D16 ≈ 665 ms/epoch → ~3.9 h/run; L2H2D32 ≈ 1.28 s/epoch → ~7.4 h/run.
    The per-100-epoch eval hook is ≤15 min/run (negligible). With 24 cores /
    32 threads the 48 runs fit in roughly one parallel wave, so wall-clock is
    bounded by the longest single run (~7.4 h) plus core-contention/thermal
    margin → **expect ~overnight (≈8–12 h)**. (Earlier "low single-digit hours"
    estimate was wrong by ~10×; this dataset is ~21.5k examples, ~84
    batches/epoch.)

### 4. Outputs (`ttt/sweep.py`)

- `results/<name>/trajectory.json` — full per-epoch curves per (config, seed).
- `results/<name>/grok_curve.png` — **held-out H-win vs epoch**, one line per
  (config, weight_decay) averaged over seeds, with the random baseline (0.40)
  drawn. For E3, the held-out rows {1,2} mean is plotted (the seen row {0} is
  excluded from the "transfer" line). This is the figure that answers the
  objection.
- Existing `raw.json` / `agg.json` / `baselines.json` / `capacity.png` continue
  to be written from the final-epoch metrics.

## Decision rule (stated up front, before running)

- **No grok / no double descent (expected):** held-out H-win stays at or below
  ~the random baseline (0.40) across **all** epochs, weight-decay values, seeds,
  and configs. The flat curves are the evidence.
- **Grok / late transition (would be a major result):** a **sustained** rise of
  held-out H-win above ~0.6 at any epoch, in any (config, WD, seed). "Sustained"
  = holds across consecutive eval points, not a single-point spike.

Either outcome is publishable: we either close the objection with curves, or we
discover the tiny transformer *can* grok the symmetry given weight decay and
time — which would reframe the whole project.

## What this defuses

- *"You didn't let it grok."* → AdamW + weight-decay sweep + 20k epochs on both
  genuine generalization conditions, with the per-epoch curve as direct evidence.
- *"You didn't allow double descent."* → **epoch-wise** covered by the same long
  curves; **model-wise** already covered by the existing E0 capacity sweep
  (sub-chance H-win across the full 3.8k→200k param range), cited not re-run.

## Out of scope

- Full-grid long-epoch re-runs (compute; model-wise DD already answered).
- Learning-rate / batch-size / optimizer-beta sweeps (weight decay is the lever
  the grokking literature singles out).
- Per-epoch streaming to git or any results-sync change.
- L16 / deeper grids.

## Testing

- New: `train_model` with `weight_decay=0` is numerically unchanged vs. the prior
  Adam path on a tiny fixture (guards the optimizer swap).
- New: `eval_every > 0` produces a trajectory of the expected length and shape;
  `eval_every = 0` returns an empty trajectory and leaves `(model, history)`
  behavior intact.
- New: `Condition` with `weight_decay`/`eval_every` round-trips through
  `run_condition` and the trajectory reaches the raw rows.
- Unchanged: the full existing suite stays green (defaults preserve all behavior).
