"""Run the full sweep and write results + capacity plot to ./results/.

Runs the (config, seed) grid across a process pool. Override worker count with
the TTT_WORKERS env var (defaults to all cores).
"""
from __future__ import annotations

import itertools
import os
import time

from ttt.sweep import run_sweep_parallel, aggregate, plot_capacity, save_results

GRID = list(itertools.product([1, 2, 4], [1, 2, 4], [16, 32, 64]))
# Keep only configs where d_model is divisible by n_head.
GRID = [(L, H, D) for (L, H, D) in GRID if D % H == 0]
SEEDS = [0, 1, 2, 3, 4]


def main():
    os.makedirs("results", exist_ok=True)
    n_workers = int(os.environ.get("TTT_WORKERS", os.cpu_count() or 1))
    n_jobs = len(GRID) * len(SEEDS)
    print(f"Running {n_jobs} jobs ({len(GRID)} configs x {len(SEEDS)} seeds) "
          f"on {n_workers} workers...")
    t0 = time.time()
    raw = run_sweep_parallel(
        grid=GRID, seeds=SEEDS,
        epochs=60, lr=1e-3, batch_size=256, max_orderings=4,
        n_workers=n_workers, progress=True,
    )
    print(f"Sweep finished in {(time.time() - t0) / 60:.1f} min")
    agg = aggregate(raw)
    save_results(raw, agg, "results/raw.json", "results/agg.json")
    plot_capacity(agg, "results/capacity.png")
    print("Wrote results/raw.json, results/agg.json, results/capacity.png")


if __name__ == "__main__":
    main()
