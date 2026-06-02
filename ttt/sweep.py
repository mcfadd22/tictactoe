from __future__ import annotations

import json
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from ttt.model import GPTConfig
from ttt.train import train_model
from ttt.dataset import build_examples
from ttt.enumerate import reachable_paths
from ttt.probes import win_available_probes, block_needed_probes
from ttt.evaluate import evaluate_probes

# Metrics evaluated per trained model.
PROBE_SPECS = {
    "horizontal_win": ("win", "horizontal"),
    "horizontal_block": ("block", "horizontal"),
    "vertical_win": ("win", "vertical"),
    "vertical_block": ("block", "vertical"),
    "diagonal_win": ("win", "diagonal"),
    "diagonal_block": ("block", "diagonal"),
}


def config_name(n_layer, n_head, d_model):
    return f"L{n_layer}H{n_head}D{d_model}"


def _build_probe_sets():
    sets = {}
    for name, (kind, line_type) in PROBE_SPECS.items():
        if kind == "win":
            sets[name] = win_available_probes(line_type)
        else:
            sets[name] = block_needed_probes(line_type)
    return sets


def _train_and_eval(config, seed, examples, paths, probe_sets, *,
                    epochs, lr, batch_size, max_orderings):
    """Train one (config, seed) and evaluate all probe metrics. Returns a raw row.

    Shared by the sequential and parallel sweeps so both produce identical rows.
    """
    n_layer, n_head, d_model = config
    cfg = GPTConfig(n_layer=n_layer, n_head=n_head, d_model=d_model)
    model, _ = train_model(
        cfg, examples, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed,
    )
    metrics = {
        name: evaluate_probes(model, probes, paths,
                              max_orderings=max_orderings)["rate"]
        for name, probes in probe_sets.items()
    }
    return {
        "config": config_name(n_layer, n_head, d_model),
        "n_params": model.num_params(),
        "seed": seed,
        "metrics": metrics,
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


def plot_capacity(agg, out_path):
    """Horizontal win/block vs. capacity, with vertical/diagonal controls.

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
    ax.set_xlabel("model parameters")
    ax.set_ylabel("probe success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Horizontal-win generalization vs. model capacity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_results(raw, agg, raw_path, agg_path):
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)


def prepare_run_dir(name=None, base="results"):
    """Create and return a fresh per-run output directory under `base`.

    Defaults to a timestamped name so runs never collide. Refuses to overwrite
    an existing non-empty directory, so prior experiment results are protected.
    """
    if name is None:
        name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = os.path.join(base, name)
    if os.path.isdir(path) and os.listdir(path):
        raise FileExistsError(
            f"run dir already exists and is non-empty: {path} "
            f"(choose a different run name to avoid overwriting results)"
        )
    os.makedirs(path, exist_ok=True)
    return path
