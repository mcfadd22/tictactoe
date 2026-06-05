# Grokking / Double-Descent Weight-Decay Probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add weight decay as a fifth composable axis on the `Condition` runner plus a per-epoch held-out evaluation hook, then run a focused long-training probe on E0 + E3 across a weight-decay sweep to pre-empt the "you didn't let it grok / allow double descent" objection.

**Architecture:** Thread `weight_decay` into `train_model` (swap `Adam`→`AdamW`) and `Condition`. Add an optional `eval_hook(model, epoch, train_loss)` callback that `train_model` calls every `eval_every` epochs; the caller (in `sweep.py`) builds the hook as a closure over the probe sets and appends results to a caller-owned trajectory list — so `train_model`'s return signature is unchanged and logging cannot perturb the run. New plot + persistence emit held-out-H-win-vs-epoch curves. Every new parameter defaults to current behavior, so E0–E5b and the whole test suite stay bit-for-bit identical.

**Tech Stack:** Python 3.12, PyTorch 2.12+cpu, matplotlib (Agg), pytest, `ProcessPoolExecutor`. Run python via `.\.venv\Scripts\python.exe`.

**Spec:** `docs/design/specs/2026-06-05-grokking-weight-decay-probe-design.md`

---

## File Structure

- `ttt/train.py` — **Modify.** `train_model` gains `weight_decay`, `eval_every`, `eval_hook`; `Adam`→`AdamW`.
- `ttt/sweep.py` — **Modify.** `Condition` gains `weight_decay`/`eval_every`; `_train_and_eval` builds the eval hook + returns a trajectory; `run_condition` threads the new fields; new `held_out_curve` + `plot_grok_curves` helpers; `save_condition` persists trajectories + curve. Add `GROK_GRID`.
- `run_experiment.py` — **Modify.** `WD_SWEEP`, `GROK_BASES`, `run_grok()` orchestration, combined per-base curve, `--grok` CLI flag.
- `tests/test_train.py` — **Modify.** Weight-decay + eval-hook + non-perturbation tests.
- `tests/test_sweep.py` — **Modify.** Trajectory threading + persistence + curve-plot tests.
- `tests/test_run_experiment.py` — **Modify.** `run_grok` smoke + CLI test.

---

## Task 1: Weight decay (AdamW) in `train_model`

**Files:**
- Modify: `ttt/train.py:21-50`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_train.py`:

```python
def test_weight_decay_zero_matches_default_behavior():
    # AdamW(weight_decay=0) must reproduce the prior Adam path exactly (same seed).
    examples = [([BOS_ID, 0, 1], 2), ([BOS_ID, 3, 4], 5)]
    cfg = GPTConfig(n_layer=1, n_head=1, d_model=16)
    m1, _ = train_model(cfg, examples, epochs=5, lr=1e-2, batch_size=8, seed=7)
    m2, _ = train_model(cfg, examples, epochs=5, lr=1e-2, batch_size=8, seed=7,
                        weight_decay=0.0)
    p1 = torch.cat([p.flatten() for p in m1.parameters()])
    p2 = torch.cat([p.flatten() for p in m2.parameters()])
    assert torch.allclose(p1, p2)


def test_weight_decay_changes_trained_weights():
    examples = [([BOS_ID, 0, 1], 2), ([BOS_ID, 3, 4], 5)]
    cfg = GPTConfig(n_layer=1, n_head=1, d_model=16)
    m0, _ = train_model(cfg, examples, epochs=20, lr=1e-2, batch_size=8, seed=7,
                        weight_decay=0.0)
    mw, _ = train_model(cfg, examples, epochs=20, lr=1e-2, batch_size=8, seed=7,
                        weight_decay=1.0)
    p0 = torch.cat([p.flatten() for p in m0.parameters()])
    pw = torch.cat([p.flatten() for p in mw.parameters()])
    assert not torch.allclose(p0, pw)  # decay actually moved the weights
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_train.py::test_weight_decay_zero_matches_default_behavior tests/test_train.py::test_weight_decay_changes_trained_weights -v`
Expected: FAIL — `train_model() got an unexpected keyword argument 'weight_decay'`.

- [ ] **Step 3: Implement weight decay**

In `ttt/train.py`, change the `train_model` signature and optimizer:

```python
def train_model(cfg: GPTConfig, examples, *, epochs, lr, batch_size,
                seed, encoding=FLAT, device="cpu", weight_decay=0.0):
    """Train a model by next-move cross-entropy. Returns (model, loss_history).

    weight_decay (decoupled, AdamW) defaults to 0.0, where AdamW is numerically
    identical to the previous Adam path.
    """
```

Replace line 35:

```python
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_train.py -v`
Expected: PASS (all train tests, including the two new ones and the unchanged determinism test).

- [ ] **Step 5: Commit**

```bash
git add ttt/train.py tests/test_train.py
git commit -m "feat(train): weight_decay via AdamW (default 0.0 preserves Adam path)"
```

---

## Task 2: Per-epoch eval hook in `train_model`

**Files:**
- Modify: `ttt/train.py:21-50`
- Test: `tests/test_train.py`

The hook is called every `eval_every` epochs **and** on the final epoch. After the hook runs, `train_model` restores `model.train()`. The hook performs only `@torch.no_grad()` forward passes (no dropout in this model, no RNG draw), so logging must not change the trained weights — guarded by a test.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_train.py`:

```python
def test_eval_hook_called_at_expected_epochs():
    examples = [([BOS_ID, 0, 1], 2), ([BOS_ID, 3, 4], 5)]
    cfg = GPTConfig(n_layer=1, n_head=1, d_model=16)
    calls = []
    def hook(model, epoch, train_loss):
        calls.append((epoch, train_loss))
    train_model(cfg, examples, epochs=10, lr=1e-2, batch_size=8, seed=0,
                eval_every=4, eval_hook=hook)
    # epochs are 1-indexed in the hook; every 4th plus the final epoch (10)
    assert [e for e, _ in calls] == [4, 8, 10]
    assert all(isinstance(l, float) for _, l in calls)


def test_eval_hook_does_not_perturb_training():
    examples = [([BOS_ID, 0, 1], 2), ([BOS_ID, 3, 4], 5)]
    cfg = GPTConfig(n_layer=1, n_head=1, d_model=16)
    m_plain, _ = train_model(cfg, examples, epochs=10, lr=1e-2, batch_size=8, seed=0)
    m_hooked, _ = train_model(cfg, examples, epochs=10, lr=1e-2, batch_size=8, seed=0,
                              eval_every=2, eval_hook=lambda *a: None)
    p1 = torch.cat([p.flatten() for p in m_plain.parameters()])
    p2 = torch.cat([p.flatten() for p in m_hooked.parameters()])
    assert torch.allclose(p1, p2)


def test_no_eval_hook_is_default_noop():
    examples = [([BOS_ID, 0, 1], 2)]
    cfg = GPTConfig(n_layer=1, n_head=1, d_model=16)
    model, history = train_model(cfg, examples, epochs=3, lr=1e-2, batch_size=8, seed=0)
    assert len(history) == 3  # unchanged (model, history) return, no hook required
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_train.py::test_eval_hook_called_at_expected_epochs -v`
Expected: FAIL — `unexpected keyword argument 'eval_every'`.

- [ ] **Step 3: Implement the hook**

In `ttt/train.py`, extend the signature and the epoch loop. Final signature:

```python
def train_model(cfg: GPTConfig, examples, *, epochs, lr, batch_size,
                seed, encoding=FLAT, device="cpu", weight_decay=0.0,
                eval_every=0, eval_hook=None):
```

Replace the epoch loop (lines 38-49) with a 1-indexed loop that fires the hook:

```python
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss, n = 0.0, 0
        for ids, lengths, targets in loader:
            ids, lengths, targets = ids.to(device), lengths.to(device), targets.to(device)
            opt.zero_grad()
            loss = loss_fn(model(ids, lengths), targets)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(targets)
            n += len(targets)
        mean_loss = epoch_loss / max(n, 1)
        history.append(mean_loss)
        if eval_hook is not None and eval_every > 0 and (
            epoch % eval_every == 0 or epoch == epochs
        ):
            eval_hook(model, epoch, mean_loss)
            model.train()  # hook puts the model in eval mode; restore for training
    return model, history
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_train.py -v`
Expected: PASS (all train tests).

- [ ] **Step 5: Commit**

```bash
git add ttt/train.py tests/test_train.py
git commit -m "feat(train): optional per-epoch eval_hook (non-perturbing, restores train mode)"
```

---

## Task 3: Thread weight_decay + trajectory through the Condition runner

**Files:**
- Modify: `ttt/sweep.py:60-91` (`_train_and_eval`), `ttt/sweep.py:244-314` (`Condition`, `_cond_worker_task`, `run_condition`)
- Test: `tests/test_sweep.py`

`_train_and_eval` builds the eval hook (closure over `probe_sets`, `paths`, `encoding`), passes it to `train_model`, and returns the per-epoch `trajectory` in its row. `Condition` gains the two fields; the worker task forwards them.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sweep.py`:

```python
def test_run_condition_records_trajectory_when_eval_every_set():
    from ttt.sweep import Condition, run_condition
    cond = Condition("T_traj", grid=((1, 1, 16),), seeds=(0,),
                     epochs=6, eval_every=2, weight_decay=0.1)
    raw = run_condition(cond, n_workers=1)
    assert len(raw) == 1
    traj = raw[0]["trajectory"]
    assert [pt["epoch"] for pt in traj] == [2, 4, 6]
    # each point carries train_loss and the full per-row metric set
    pt = traj[0]
    assert isinstance(pt["train_loss"], float)
    assert "horizontal_win" in pt["metrics"]
    assert "horizontal_win_row1" in pt["metrics"]


def test_run_condition_no_trajectory_by_default():
    from ttt.sweep import Condition, run_condition
    cond = Condition("T_notraj", grid=((1, 1, 16),), seeds=(0,), epochs=3)
    raw = run_condition(cond, n_workers=1)
    assert raw[0]["trajectory"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_sweep.py::test_run_condition_records_trajectory_when_eval_every_set -v`
Expected: FAIL — `Condition.__init__() got an unexpected keyword argument 'eval_every'`.

- [ ] **Step 3: Add fields to `Condition`**

In `ttt/sweep.py`, add to the `Condition` dataclass (after `max_orderings: int = 4`):

```python
    weight_decay: float = 0.0
    eval_every: int = 0
```

- [ ] **Step 4: Build the hook + trajectory in `_train_and_eval`**

Change the `_train_and_eval` signature to accept the new params and an eval cadence, and assemble the trajectory. Replace the body from the `model, _ = train_model(...)` call onward:

```python
def _train_and_eval(config, seed, examples, paths, probe_sets, *,
                    epochs, lr, batch_size, max_orderings,
                    encoding=FLAT, head="flat9", weight_decay=0.0, eval_every=0):
    if head == "tied" and encoding.name != "flat":
        raise ValueError("tied head requires the flat encoding (1:1 cell<->token)")
    n_layer, n_head, d_model = config
    cfg = GPTConfig(
        n_layer=n_layer, n_head=n_head, d_model=d_model,
        vocab_size=encoding.vocab_size, max_len=encoding.max_len, head=head,
    )

    trajectory = []

    def eval_hook(m, epoch, train_loss):
        metrics = {
            name: evaluate_probes(m, probes, paths, encoding=encoding,
                                  max_orderings=max_orderings)["rate"]
            for name, probes in probe_sets.items()
        }
        trajectory.append({"epoch": epoch, "train_loss": train_loss,
                           "metrics": metrics})

    model, _ = train_model(
        cfg, examples, epochs=epochs, lr=lr, batch_size=batch_size,
        seed=seed, encoding=encoding, weight_decay=weight_decay,
        eval_every=eval_every, eval_hook=(eval_hook if eval_every > 0 else None),
    )
    metrics = {
        name: evaluate_probes(model, probes, paths, encoding=encoding,
                              max_orderings=max_orderings)["rate"]
        for name, probes in probe_sets.items()
    }
    return {
        "config": config_name(n_layer, n_head, d_model),
        "n_params": model.num_params(),
        "seed": seed,
        "metrics": metrics,
        "trajectory": trajectory,
    }
```

- [ ] **Step 5: Forward the new fields from the worker task**

In `ttt/sweep.py`, update `_cond_worker_task` to pass `weight_decay`/`eval_every`, and the `run_condition` task tuple to carry them. Replace `_cond_worker_task`:

```python
def _cond_worker_task(args):
    config, seed, head, epochs, lr, batch_size, weight_decay, eval_every = args
    return _train_and_eval(
        config, seed,
        _COND_WORKER["examples"], _COND_WORKER["paths"], _COND_WORKER["probe_sets"],
        epochs=epochs, lr=lr, batch_size=batch_size,
        max_orderings=_COND_WORKER["max_orderings"],
        encoding=_COND_WORKER["encoding"], head=head,
        weight_decay=weight_decay, eval_every=eval_every,
    )
```

In `run_condition`, replace the `tasks = [...]` comprehension:

```python
    tasks = [
        (config, seed, cond.head, cond.epochs, cond.lr, cond.batch_size,
         cond.weight_decay, cond.eval_every)
        for config in cond.grid
        for seed in cond.seeds
    ]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_sweep.py -v`
Expected: PASS — new trajectory tests pass; existing sweep tests still pass (rows now also carry an empty `trajectory`, which existing assertions ignore).

- [ ] **Step 7: Commit**

```bash
git add ttt/sweep.py tests/test_sweep.py
git commit -m "feat(sweep): weight_decay + eval_every Condition axes; per-epoch trajectory in raw rows"
```

---

## Task 4: Held-out curve helper + grok-curve plot

**Files:**
- Modify: `ttt/sweep.py` (add `held_out_curve`, `plot_grok_curves` near `plot_capacity`)
- Test: `tests/test_sweep.py`

`held_out_curve(raw, drop_rows)` returns `{config: [(epoch, mean_held_out_H_win), ...]}`, where the held-out rows are exactly the dropped rows (E0 → {0,1,2}, E3 → {1,2}), averaged over seeds. `plot_grok_curves(curves_by_label, out_path, baseline, title)` draws one line per label with the random baseline.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sweep.py`:

```python
def test_held_out_curve_averages_dropped_rows_over_seeds():
    from ttt.sweep import held_out_curve
    raw = [
        {"config": "L1H1D16", "seed": 0, "trajectory": [
            {"epoch": 2, "metrics": {"horizontal_win_row1": 0.2,
                                     "horizontal_win_row2": 0.4}},
            {"epoch": 4, "metrics": {"horizontal_win_row1": 0.6,
                                     "horizontal_win_row2": 0.8}}]},
        {"config": "L1H1D16", "seed": 1, "trajectory": [
            {"epoch": 2, "metrics": {"horizontal_win_row1": 0.4,
                                     "horizontal_win_row2": 0.6}},
            {"epoch": 4, "metrics": {"horizontal_win_row1": 0.8,
                                     "horizontal_win_row2": 1.0}}]},
    ]
    curve = held_out_curve(raw, frozenset({1, 2}))
    # epoch 2: mean of [(0.2+0.4)/2, (0.4+0.6)/2] = mean(0.3, 0.5) = 0.4
    # epoch 4: mean of [(0.6+0.8)/2, (0.8+1.0)/2] = mean(0.7, 0.9) = 0.8
    assert curve["L1H1D16"] == [(2, 0.4), (4, 0.8)]


def test_plot_grok_curves_writes_file(tmp_path):
    from ttt.sweep import plot_grok_curves
    out = tmp_path / "grok.png"
    curves = {"L1H1D16 wd0.1": [(2, 0.1), (4, 0.12)],
              "L2H2D32 wd0.1": [(2, 0.09), (4, 0.11)]}
    plot_grok_curves(curves, str(out), baseline=0.40, title="E0 grok")
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_sweep.py::test_held_out_curve_averages_dropped_rows_over_seeds -v`
Expected: FAIL — `cannot import name 'held_out_curve'`.

- [ ] **Step 3: Implement both helpers**

Add to `ttt/sweep.py` (after `plot_capacity`):

```python
def held_out_curve(raw, drop_rows):
    """{config: [(epoch, mean held-out H-win), ...]} averaged over seeds.

    The held-out rows are exactly the dropped rows; per eval point we average
    the per-row horizontal-win rates over those rows, then over seeds.
    """
    from collections import defaultdict
    rows = sorted(drop_rows)
    by_config = defaultdict(list)
    for row in raw:
        by_config[row["config"]].append(row["trajectory"])
    curves = {}
    for config, trajs in by_config.items():
        points = []
        n_pts = min(len(t) for t in trajs)
        for i in range(n_pts):
            epoch = trajs[0][i]["epoch"]
            per_seed = []
            for t in trajs:
                m = t[i]["metrics"]
                per_seed.append(
                    statistics.fmean(m[f"horizontal_win_row{r}"] for r in rows)
                )
            points.append((epoch, statistics.fmean(per_seed)))
        curves[config] = points
    return curves


def plot_grok_curves(curves_by_label, out_path, baseline=None, title="Grokking probe"):
    """Held-out horizontal-win vs. training epoch, one line per label."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, points in sorted(curves_by_label.items()):
        xs = [e for e, _ in points]
        ys = [v for _, v in points]
        ax.plot(xs, ys, marker=".", label=label)
    if baseline is not None:
        ax.axhline(baseline, ls="--", color="gray", label="random baseline (H-win)")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("held-out horizontal-win rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_sweep.py::test_held_out_curve_averages_dropped_rows_over_seeds tests/test_sweep.py::test_plot_grok_curves_writes_file -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ttt/sweep.py tests/test_sweep.py
git commit -m "feat(sweep): held_out_curve + plot_grok_curves for grokking trajectories"
```

---

## Task 5: Persist trajectories + per-condition curve in `save_condition`

**Files:**
- Modify: `ttt/sweep.py:324-340` (`save_condition`)
- Test: `tests/test_sweep.py`

When any raw row carries a non-empty trajectory, also write `trajectory.json` and a per-condition `grok_curve.png` (held-out rows = `cond.drop_horizontal_rows`, one line per config).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sweep.py`:

```python
def test_save_condition_writes_trajectory_artifacts(tmp_path):
    from ttt.sweep import Condition, run_condition, save_condition
    cond = Condition("T_save", grid=((1, 1, 16),), seeds=(0,),
                     epochs=4, eval_every=2)
    raw = run_condition(cond, n_workers=1)
    save_condition(cond, raw, str(tmp_path))
    assert (tmp_path / "trajectory.json").exists()
    assert (tmp_path / "grok_curve.png").exists()
    import json
    traj = json.load(open(tmp_path / "trajectory.json"))
    assert traj[0]["config"] == "L1H1D16"
    assert [pt["epoch"] for pt in traj[0]["trajectory"]] == [2, 4]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_sweep.py::test_save_condition_writes_trajectory_artifacts -v`
Expected: FAIL — `trajectory.json` does not exist.

- [ ] **Step 3: Extend `save_condition`**

In `ttt/sweep.py`, add to `save_condition` just before `return agg, baselines`:

```python
    if any(row.get("trajectory") for row in raw):
        traj_out = [
            {"config": row["config"], "seed": row["seed"],
             "trajectory": row["trajectory"]}
            for row in raw
        ]
        with open(os.path.join(out_dir, "trajectory.json"), "w") as f:
            json.dump(traj_out, f, indent=2)
        curves = held_out_curve(raw, cond.drop_horizontal_rows)
        plot_grok_curves(
            curves, os.path.join(out_dir, "grok_curve.png"),
            baseline=baselines.get("horizontal_win"),
            title=f"{cond.name}: held-out H-win vs epoch (wd={cond.weight_decay})",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_sweep.py -v`
Expected: PASS (new test plus all existing sweep tests).

- [ ] **Step 5: Commit**

```bash
git add ttt/sweep.py tests/test_sweep.py
git commit -m "feat(sweep): persist trajectory.json + grok_curve.png when eval_every set"
```

---

## Task 6: `GROK_GRID` + `run_grok` orchestration + CLI

**Files:**
- Modify: `ttt/sweep.py` (add `GROK_GRID` next to `STANDARD_GRID`)
- Modify: `run_experiment.py`
- Test: `tests/test_run_experiment.py`

`run_grok(base, n_workers, epochs, eval_every, seeds)` runs the 4 weight-decay conditions for one base (E0 or E3), saving each under `results/E_GROK_<base>_wd<wd>/`, then writes a combined `results/E_GROK_<base>/grok_curves.png` with one line per (config, wd) and a combined `trajectories.json`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_run_experiment.py`:

```python
def test_run_grok_smoke_writes_per_wd_and_combined(tmp_path, monkeypatch):
    import os
    import run_experiment as R
    monkeypatch.chdir(tmp_path)
    # tiny override: 1 config, 1 seed, few epochs, 2 wd values
    monkeypatch.setattr(R, "GROK_GRID", ((1, 1, 16),))
    monkeypatch.setattr(R, "WD_SWEEP", (0.0, 1.0))
    R.run_grok("E0", n_workers=1, epochs=4, eval_every=2, seeds=(0,))
    assert os.path.exists("results/E_GROK_E0_wd0.0/grok_curve.png")
    assert os.path.exists("results/E_GROK_E0_wd1.0/trajectory.json")
    assert os.path.exists("results/E_GROK_E0/grok_curves.png")
    assert os.path.exists("results/E_GROK_E0/trajectories.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_run_experiment.py::test_run_grok_smoke_writes_per_wd_and_combined -v`
Expected: FAIL — `module 'run_experiment' has no attribute 'GROK_GRID'`.

- [ ] **Step 3: Add `GROK_GRID` to `sweep.py`**

In `ttt/sweep.py`, after the `DEEP_GRID` definition:

```python
# Grokking probe: two small configs (grokking favors small models + weight decay).
GROK_GRID = ((1, 1, 16), (2, 2, 32))
```

- [ ] **Step 4: Implement `run_grok` + constants + CLI in `run_experiment.py`**

In `run_experiment.py`, update the import line to include the grok grid and curve helpers:

```python
from ttt.sweep import (
    Condition, run_condition, save_condition, horizontal_win_ceiling,
    STANDARD_GRID, DEEP_GRID, GROK_GRID, held_out_curve, plot_grok_curves,
    compute_random_baselines,
)
```

Add module-level constants (after `PHASE1 = [...]`):

```python
import json

# --- Grokking / double-descent probe ---------------------------------------
WD_SWEEP = (0.0, 0.01, 0.1, 1.0)
GROK_BASES = {
    "E0": frozenset({0, 1, 2}),  # rotational: zero horizontal exposure
    "E3": frozenset({1, 2}),     # translational: top row seen, rows 1-2 held out
}
GROK_EPOCHS = 20000
GROK_EVAL_EVERY = 100
GROK_SEEDS = (0, 1, 2)


def run_grok(base, n_workers, *, epochs=GROK_EPOCHS, eval_every=GROK_EVAL_EVERY,
             seeds=GROK_SEEDS):
    """Run the weight-decay sweep for one base (E0/E3) with per-epoch logging.

    Writes results/E_GROK_<base>_wd<wd>/ per weight-decay value, then a combined
    results/E_GROK_<base>/ with one curve line per (config, wd).
    """
    drop = GROK_BASES[base]
    combined_curves = {}
    combined_traj = []
    for wd in WD_SWEEP:
        name = f"E_GROK_{base}_wd{wd}"
        cond = Condition(name, drop_horizontal_rows=drop, grid=GROK_GRID,
                         seeds=seeds, epochs=epochs, weight_decay=wd,
                         eval_every=eval_every)
        print(f"=== {name}: {len(GROK_GRID) * len(seeds)} runs, "
              f"wd={wd}, epochs={epochs} ===", flush=True)
        raw = run_condition(cond, n_workers=n_workers, progress=True)
        save_condition(cond, raw, os.path.join("results", name))
        for cfg, points in held_out_curve(raw, drop).items():
            combined_curves[f"{cfg} wd{wd}"] = points
        combined_traj.append({"weight_decay": wd, "rows": [
            {"config": r["config"], "seed": r["seed"],
             "trajectory": r["trajectory"]} for r in raw]})
    out_dir = os.path.join("results", f"E_GROK_{base}")
    os.makedirs(out_dir, exist_ok=True)
    baseline = compute_random_baselines().get("horizontal_win")
    plot_grok_curves(combined_curves, os.path.join(out_dir, "grok_curves.png"),
                     baseline=baseline,
                     title=f"E_GROK {base}: held-out H-win vs epoch (all wd)")
    with open(os.path.join(out_dir, "trajectories.json"), "w") as f:
        json.dump(combined_traj, f, indent=2)
    print(f"  E_GROK_{base} combined -> {out_dir}/", flush=True)
```

Add a `--grok` argument in `build_parser` (after the `--condition` argument):

```python
    parser.add_argument("--grok", choices=["E0", "E3", "all"],
                        help="run the long-training weight-decay grokking probe "
                             "for a base condition (or both)")
```

At the start of `main()`, after `args = build_parser().parse_args()`, short-circuit to the grok path:

```python
    if args.grok:
        os.makedirs("results", exist_ok=True)
        bases = ["E0", "E3"] if args.grok == "all" else [args.grok]
        for base in bases:
            run_grok(base, args.workers)
        return
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_run_experiment.py -v`
Expected: PASS (the grok smoke test plus existing run_experiment tests).

- [ ] **Step 6: Commit**

```bash
git add ttt/sweep.py run_experiment.py tests/test_run_experiment.py
git commit -m "feat(run): GROK_GRID + run_grok weight-decay probe with --grok CLI"
```

---

## Task 7: Full suite green + end-to-end smoke at small epochs

**Files:**
- Test: full suite; manual smoke command.

- [ ] **Step 1: Run the full test suite**

Run: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: PASS — all prior tests (90+) plus the new ones; zero regressions (defaults preserved E0–E5b behavior).

- [ ] **Step 2: End-to-end smoke of the real CLI at tiny scale**

Run a fast version of the actual probe to confirm the artifacts render before committing to the overnight run:

Run (PowerShell):
```powershell
$env:TTT_WORKERS=4
.\.venv\Scripts\python.exe -c "import run_experiment as R; R.run_grok('E0', 4, epochs=20, eval_every=5, seeds=(0,1))"
```
Expected: creates `results/E_GROK_E0_wd0.0/ … wd1.0/` each with `trajectory.json` + `grok_curve.png`, and `results/E_GROK_E0/grok_curves.png` + `trajectories.json`. Open `results/E_GROK_E0/grok_curves.png` and confirm it shows per-(config, wd) lines with the baseline at ~0.40.

- [ ] **Step 3: Clean up smoke artifacts**

Run (PowerShell):
```powershell
Remove-Item -Recurse -Force results/E_GROK_E0, results/E_GROK_E0_wd0.0, results/E_GROK_E0_wd0.01, results/E_GROK_E0_wd0.1, results/E_GROK_E0_wd1.0
```

(`results/` is gitignored, so nothing to commit here.)

---

## Task 8: Launch the real probe + record the result

**Files:**
- Create (results, gitignored until pushed): `results/E_GROK_E0*/`, `results/E_GROK_E3*/`
- Modify: `docs/design/specs/2026-06-05-grokking-weight-decay-probe-design.md` (append findings)

This is the experiment itself (~overnight, ~8–12 h wall). Run it, then read the curves against the decision rule from the spec.

- [ ] **Step 1: Launch the full probe in the background**

Run (PowerShell, background — uses all cores):
```powershell
.\.venv\Scripts\python.exe run_experiment.py --grok all
```
This runs E0 then E3, each across `WD_SWEEP` × `GROK_GRID` × `GROK_SEEDS` at 20k epochs with per-100-epoch logging. Expect ~overnight wall.

- [ ] **Step 2: Inspect the combined curves against the decision rule**

Open `results/E_GROK_E0/grok_curves.png` and `results/E_GROK_E3/grok_curves.png`.
- **No grok (expected):** every line stays at/below ~0.40 across all 20k epochs.
- **Grok (major result):** any line shows a sustained rise above ~0.6.

Cross-check numerically:
```powershell
.\.venv\Scripts\python.exe -c "import json; d=json.load(open('results/E_GROK_E0/trajectories.json')); print([(b['weight_decay'], max(max(pt['metrics']['horizontal_win'] for pt in r['trajectory']) for r in b['rows'])) for b in d])"
```
Expected (no grok): every per-wd max held-out `horizontal_win` ≲ ~0.45.

- [ ] **Step 3: Record findings in the spec**

Append a `## Result` section to `docs/design/specs/2026-06-05-grokking-weight-decay-probe-design.md` quoting the per-wd peak held-out H-win for E0 and E3 and stating which side of the decision rule the run landed on. Then commit:

```bash
git add docs/design/specs/2026-06-05-grokking-weight-decay-probe-design.md
git commit -m "docs: record grokking/double-descent probe result (E0+E3 weight-decay sweep)"
```

- [ ] **Step 4 (optional): Sync results to the branch**

Only if results should be shared (note `origin` is the collaborator's repo). The existing `--push-results` plumbing is per-condition; for the grok dirs, force-add and commit manually:

```bash
git add -f results/E_GROK_E0 results/E_GROK_E3 results/E_GROK_E0_wd* results/E_GROK_E3_wd*
git commit -m "results: E_GROK (E0+E3 weight-decay grokking probe, 20k epochs)"
```

---

## Self-Review Notes

- **Spec coverage:** §1 optimizer → Task 1; §2 eval hook → Task 2 (callback variant, see below); §3 Condition fields/GROK_GRID/conditions → Tasks 3 & 6; §4 outputs (trajectory.json, grok_curve.png) → Tasks 4–6; decision rule → Task 8; testing bullets → Tasks 1–3, 5.
- **Intentional deviation from spec §2:** the spec described `train_model` returning a third `trajectory` value. To avoid breaking the existing `model, history = train_model(...)` contract and its tests, the plan instead passes an `eval_hook` that appends to a caller-owned list. Same captured trajectory; cleaner boundary; adds a non-perturbation guarantee (Task 2 test). Functionally equivalent to the spec's intent.
- **Type consistency:** `eval_hook(model, epoch, train_loss)` signature is identical in Tasks 2 and 3; trajectory point shape `{"epoch", "train_loss", "metrics"}` is consistent across Tasks 3–6; held-out rows = `drop_horizontal_rows` is the single rule used in Tasks 4, 5, 6.
- **Held-out metric choice:** `held_out_curve` averages `horizontal_win_row{r}` over the dropped rows — correct for both E0 (all rows) and E3 (rows 1,2), avoiding dilution by the seen row in E3.
