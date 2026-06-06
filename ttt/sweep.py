from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from ttt.model import GPTConfig
from ttt.train import train_model
from ttt.dataset import build_examples
from ttt.enumerate import reachable_paths
from ttt.probes import win_available_probes, block_needed_probes
from ttt.evaluate import evaluate_probes, random_baseline_rate
from ttt.encoding import FLAT, Encoding

# Metrics evaluated per trained model. (kind, line_type, rows-or-None)
PROBE_SPECS = {
    "horizontal_win": ("win", "horizontal", None),
    "horizontal_block": ("block", "horizontal", None),
    "vertical_win": ("win", "vertical", None),
    "vertical_block": ("block", "vertical", None),
    "diagonal_win": ("win", "diagonal", None),
    "diagonal_block": ("block", "diagonal", None),
    # Per-row horizontal wins drive the translational (E3) seen-vs-held-out split.
    "horizontal_win_row0": ("win", "horizontal", frozenset({0})),
    "horizontal_win_row1": ("win", "horizontal", frozenset({1})),
    "horizontal_win_row2": ("win", "horizontal", frozenset({2})),
}

STANDARD_GRID = tuple(
    (L, H, D)
    for L in (1, 2, 4) for H in (1, 2, 4) for D in (16, 32, 64)
    if D % H == 0
)
# Depth experiment (E4) appends L8; same width/head sub-grid.
DEEP_GRID = STANDARD_GRID + tuple(
    (8, H, D) for H in (1, 2, 4) for D in (16, 32, 64) if D % H == 0
)
# Grokking probe: two small configs (grokking favors small models + weight decay).
GROK_GRID = ((1, 1, 16), (2, 2, 32))


def config_name(n_layer, n_head, d_model):
    return f"L{n_layer}H{n_head}D{d_model}"


def _build_probe_sets():
    sets = {}
    for name, (kind, line_type, rows) in PROBE_SPECS.items():
        if kind == "win":
            sets[name] = win_available_probes(line_type, rows=rows)
        else:
            sets[name] = block_needed_probes(line_type, rows=rows)
    return sets


def _probe_metrics(model, probe_sets, paths, encoding, max_orderings):
    """Probe success rate per metric for a trained model."""
    return {
        name: evaluate_probes(model, probes, paths, encoding=encoding,
                              max_orderings=max_orderings)["rate"]
        for name, probes in probe_sets.items()
    }


def _train_and_eval(config, seed, examples, paths, probe_sets, *,
                    epochs, lr, batch_size, max_orderings,
                    encoding=FLAT, head="flat9", weight_decay=0.0, eval_every=0):
    """Train one (config, seed) and evaluate all probe metrics. Returns a raw row.

    Shared by run_sweep, run_sweep_parallel, and run_condition so all three
    produce identically shaped rows.
    """
    if head == "tied" and encoding.name != "flat":
        # Defense-in-depth: Condition rejects this too, but guard the lower
        # entry point so a tied readout never silently indexes non-cell tokens.
        raise ValueError("tied head requires the flat encoding (1:1 cell<->token)")
    n_layer, n_head, d_model = config
    cfg = GPTConfig(
        n_layer=n_layer, n_head=n_head, d_model=d_model,
        vocab_size=encoding.vocab_size, max_len=encoding.max_len, head=head,
    )

    trajectory = []

    def eval_hook(m, epoch, train_loss):
        trajectory.append({
            "epoch": epoch, "train_loss": train_loss,
            "metrics": _probe_metrics(m, probe_sets, paths, encoding, max_orderings),
        })

    model, _ = train_model(
        cfg, examples, epochs=epochs, lr=lr, batch_size=batch_size,
        seed=seed, encoding=encoding, weight_decay=weight_decay,
        eval_every=eval_every, eval_hook=(eval_hook if eval_every > 0 else None),
    )
    metrics = _probe_metrics(model, probe_sets, paths, encoding, max_orderings)
    return {
        "config": config_name(n_layer, n_head, d_model),
        "n_params": model.num_params(),
        "seed": seed,
        "metrics": metrics,
        "trajectory": trajectory,
    }


def run_sweep(grid, seeds, *, epochs, lr, batch_size, max_orderings=4):
    """Train every (config, seed) sequentially; evaluate all metrics. Raw rows."""
    examples = build_examples(max_orderings=max_orderings, filter_horizontal=True)
    paths = reachable_paths(max_orderings=max_orderings)
    probe_sets = _build_probe_sets()
    raw = []
    for config in grid:
        for seed in seeds:
            raw.append(_train_and_eval(
                config, seed, examples, paths, probe_sets,
                epochs=epochs, lr=lr, batch_size=batch_size,
                max_orderings=max_orderings,
            ))
    return raw


# --- Parallel sweep -------------------------------------------------------
# Each (config, seed) run is independent. We use a process pool; each worker
# builds the shared dataset/probes once (in its initializer) and pins torch to
# a single thread so N processes don't oversubscribe the cores.

_WORKER = {}


def _worker_init(max_orderings, filter_horizontal):
    import torch
    torch.set_num_threads(1)
    _WORKER["examples"] = build_examples(
        max_orderings=max_orderings, filter_horizontal=filter_horizontal
    )
    _WORKER["paths"] = reachable_paths(max_orderings=max_orderings)
    _WORKER["probe_sets"] = _build_probe_sets()
    _WORKER["max_orderings"] = max_orderings


def _worker_task(args):
    config, seed, epochs, lr, batch_size = args
    return _train_and_eval(
        config, seed,
        _WORKER["examples"], _WORKER["paths"], _WORKER["probe_sets"],
        epochs=epochs, lr=lr, batch_size=batch_size,
        max_orderings=_WORKER["max_orderings"],
    )


def run_sweep_parallel(grid, seeds, *, epochs, lr, batch_size, max_orderings=4,
                       n_workers=None, filter_horizontal=True, progress=False):
    """Like run_sweep but runs (config, seed) jobs across a process pool.

    n_workers defaults to os.cpu_count(). Results are deterministic per
    (config, seed) regardless of worker count, since training seeds itself.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    tasks = [
        (config, seed, epochs, lr, batch_size)
        for config in grid
        for seed in seeds
    ]
    raw = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(max_orderings, filter_horizontal),
    ) as pool:
        futures = [pool.submit(_worker_task, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            raw.append(fut.result())
            if progress:
                print(f"  [{i}/{len(tasks)}] done", flush=True)
    return raw


def aggregate(raw):
    """Mean/std per config per metric across seeds."""
    by_config = {}
    for row in raw:
        by_config.setdefault(row["config"], []).append(row)
    agg = {}
    for config, rows in by_config.items():
        entry = {"n_params": rows[0]["n_params"]}
        metric_names = rows[0]["metrics"].keys()
        for m in metric_names:
            vals = [r["metrics"][m] for r in rows]
            entry[m] = {
                "mean": statistics.fmean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }
        agg[config] = entry
    return agg


def plot_capacity(agg, out_path, baselines=None, ceiling=None):
    """Horizontal win/block vs. capacity, with vertical/diagonal controls,
    an optional random-baseline 'chance' line, and an optional positive-control
    ceiling overlay.

    Several configs can share a parameter count (n_head does not change the
    parameter count), so configs are grouped by n_params and averaged, giving
    one point per distinct capacity. Per-config detail stays in the raw results.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for cfg, entry in agg.items():
        groups[entry["n_params"]].append(entry)
    xs = sorted(groups)
    any_entry = next(iter(agg.values()))

    fig, ax = plt.subplots(figsize=(7, 5))
    for metric, label in [
        ("horizontal_win", "horizontal win (held-out)"),
        ("horizontal_block", "horizontal block"),
        ("vertical_win", "vertical win (control)"),
        ("diagonal_win", "diagonal win (control)"),
    ]:
        if metric not in any_entry:
            continue
        means, stds = [], []
        for x in xs:
            vals = [e[metric]["mean"] for e in groups[x]]
            means.append(statistics.fmean(vals))
            stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=label)
    if baselines and "horizontal_win" in baselines:
        ax.axhline(baselines["horizontal_win"], ls="--", color="gray",
                   label="random baseline (H-win)")
    if ceiling:
        cxs = sorted(ceiling)
        ax.plot(cxs, [ceiling[x] for x in cxs], ls=":", color="green",
                marker="^", label="positive-control ceiling (H-win)")
    ax.set_xlabel("model parameters")
    ax.set_ylabel("probe success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Horizontal-win generalization vs. model capacity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


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
            # Seeds of one condition share epochs/eval_every, so all trajs[t][i]
            # land on the same epoch; reading it from the first seed is safe.
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


def save_results(raw, agg, raw_path, agg_path):
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)


# --- Conditions: one composable experiment per dataclass instance -----------

@dataclass(frozen=True)
class Condition:
    """A single experiment = a setting of the four orthogonal axes plus run
    hyperparameters. Combining axes needs no new code, only a new Condition."""
    name: str
    encoding: Encoding = FLAT
    head: str = "flat9"
    drop_horizontal_rows: frozenset = frozenset({0, 1, 2})
    grid: tuple = STANDARD_GRID
    seeds: tuple = (0, 1, 2, 3, 4)
    epochs: int = 150
    lr: float = 1e-3
    batch_size: int = 256
    max_orderings: int = 4
    weight_decay: float = 0.0
    eval_every: int = 0

    def __post_init__(self):
        if self.head not in ("flat9", "tied", "factored"):
            raise ValueError(f"unknown head: {self.head}")
        if self.head == "tied" and self.encoding.name != "flat":
            raise ValueError("tied head requires the flat encoding (1:1 cell<->token)")


_COND_WORKER = {}


def _cond_worker_init(encoding, drop_horizontal_rows, max_orderings):
    import torch
    torch.set_num_threads(1)
    _COND_WORKER["examples"] = build_examples(
        encoding, drop_horizontal_rows=drop_horizontal_rows,
        max_orderings=max_orderings,
    )
    _COND_WORKER["paths"] = reachable_paths(max_orderings=max_orderings)
    _COND_WORKER["probe_sets"] = _build_probe_sets()
    _COND_WORKER["encoding"] = encoding
    _COND_WORKER["max_orderings"] = max_orderings


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


def run_condition(cond: Condition, *, n_workers=None, progress=False):
    """Train the condition's (config, seed) grid across a process pool and
    evaluate every probe metric. Returns raw rows (same shape as run_sweep)."""
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    tasks = [
        (config, seed, cond.head, cond.epochs, cond.lr, cond.batch_size,
         cond.weight_decay, cond.eval_every)
        for config in cond.grid
        for seed in cond.seeds
    ]
    raw = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_cond_worker_init,
        initargs=(cond.encoding, cond.drop_horizontal_rows, cond.max_orderings),
    ) as pool:
        futures = [pool.submit(_cond_worker_task, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            raw.append(fut.result())
            if progress:
                print(f"  [{cond.name} {i}/{len(tasks)}] done", flush=True)
    return raw


def _fmt_duration(seconds):
    """Compact H/M formatting for progress ETAs (e.g. '2h05m' or '7m')."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m = rem // 60
    return f"{h}h{m:02d}m" if h else f"{m}m"


def run_grok_base(drop_rows, wd_values, *, grid, seeds, epochs, eval_every,
                  encoding=FLAT, head="flat9", lr=1e-3, batch_size=256,
                  max_orderings=4, n_workers=None, progress=False, label="grok"):
    """Run every (wd, config, seed) job for one base in a SINGLE process pool.

    All weight-decay values share the same training dataset (they differ only in
    the optimizer), so one worker-built dataset serves every job and a many-core
    machine stays busy instead of being capped at one wd-condition at a time.
    Returns {wd: [raw_row, ...]} grouped by weight decay. When `progress`, prints
    a rough wall-clock ETA as jobs complete.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    tasks = [
        (wd, (config, seed, head, epochs, lr, batch_size, wd, eval_every))
        for wd in wd_values
        for config in grid
        for seed in seeds
    ]
    results = {wd: [] for wd in wd_values}
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_cond_worker_init,
        initargs=(encoding, drop_rows, max_orderings),
    ) as pool:
        futures = {pool.submit(_cond_worker_task, args): wd for wd, args in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            results[futures[fut]].append(fut.result())
            if progress:
                elapsed = time.time() - t0
                remaining = elapsed / i * (len(tasks) - i)
                print(f"  [{label} {i}/{len(tasks)}] done — elapsed "
                      f"{_fmt_duration(elapsed)}, ~{_fmt_duration(remaining)} "
                      f"left (rough est)", flush=True)
    return results


def compute_random_baselines():
    """Uniform-random-legal-move hit-rate per probe set (the 'chance' line)."""
    probe_sets = _build_probe_sets()
    return {name: random_baseline_rate(probes)
            for name, probes in probe_sets.items()}


def save_condition(cond: Condition, raw, out_dir, ceiling=None):
    """Aggregate, persist, and plot one condition into out_dir/.

    `ceiling`, if given, is a {n_params: horizontal_win_mean} map (e.g. from the
    positive-control E1) drawn as an upper reference on the plot.
    """
    os.makedirs(out_dir, exist_ok=True)
    agg = aggregate(raw)
    baselines = compute_random_baselines()
    save_results(raw, agg,
                 os.path.join(out_dir, "raw.json"),
                 os.path.join(out_dir, "agg.json"))
    with open(os.path.join(out_dir, "baselines.json"), "w") as f:
        json.dump(baselines, f, indent=2)
    plot_capacity(agg, os.path.join(out_dir, "capacity.png"),
                  baselines=baselines, ceiling=ceiling)
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
    return agg, baselines


def horizontal_win_ceiling(agg):
    """{n_params: horizontal_win mean} from a (positive-control) condition's agg,
    suitable to pass as `save_condition(..., ceiling=...)`."""
    return {entry["n_params"]: entry["horizontal_win"]["mean"]
            for entry in agg.values() if "horizontal_win" in entry}
