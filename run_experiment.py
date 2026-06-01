"""Run the full sweep and write results + capacity plot to ./results/."""
from __future__ import annotations

import itertools
import os

from ttt.sweep import run_sweep, aggregate, plot_capacity, save_results

GRID = list(itertools.product([1, 2, 4], [1, 2, 4], [16, 32, 64]))
# Keep only configs where d_model is divisible by n_head.
GRID = [(L, H, D) for (L, H, D) in GRID if D % H == 0]
SEEDS = [0, 1, 2, 3, 4]


def main():
    os.makedirs("results", exist_ok=True)
    raw = run_sweep(
        grid=GRID, seeds=SEEDS,
        epochs=60, lr=1e-3, batch_size=256, max_orderings=4,
    )
    agg = aggregate(raw)
    save_results(raw, agg, "results/raw.json", "results/agg.json")
    plot_capacity(agg, "results/capacity.png")
    print("Wrote results/raw.json, results/agg.json, results/capacity.png")


if __name__ == "__main__":
    main()
