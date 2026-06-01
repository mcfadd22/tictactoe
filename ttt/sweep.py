from __future__ import annotations

import json
import statistics

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


def run_sweep(grid, seeds, *, epochs, lr, batch_size, max_orderings=4):
    """Train every (config, seed); evaluate all probe metrics. Returns raw rows."""
    examples = build_examples(max_orderings=max_orderings, filter_horizontal=True)
    paths = reachable_paths(max_orderings=max_orderings)
    probe_sets = _build_probe_sets()
    raw = []
    for (n_layer, n_head, d_model) in grid:
        cfg = GPTConfig(n_layer=n_layer, n_head=n_head, d_model=d_model)
        for seed in seeds:
            model, _ = train_model(
                cfg, examples, epochs=epochs, lr=lr,
                batch_size=batch_size, seed=seed,
            )
            metrics = {
                name: evaluate_probes(model, probes, paths,
                                      max_orderings=max_orderings)["rate"]
                for name, probes in probe_sets.items()
            }
            raw.append({
                "config": config_name(n_layer, n_head, d_model),
                "n_params": model.num_params(),
                "seed": seed,
                "metrics": metrics,
            })
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
