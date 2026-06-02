"""Run the full sweep and write results to a per-run directory under ./results/.

Each run gets its own folder so experiments never overwrite each other:

    python run_experiment.py                 # -> results/<timestamp>/
    python run_experiment.py 150ep-uniqueprobes   # -> results/150ep-uniqueprobes/
    TTT_RUN_NAME=foo python run_experiment.py      # -> results/foo/

A run writes raw.json, agg.json, capacity.png, and meta.json (the exact
parameters + git commit that produced the results). It refuses to overwrite an
existing non-empty run directory. Worker count: TTT_WORKERS env (default: all cores).
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time

from ttt.sweep import (
    run_sweep_parallel, aggregate, plot_capacity, save_results, prepare_run_dir,
)

GRID = list(itertools.product([1, 2, 4], [1, 2, 4], [16, 32, 64]))
# Keep only configs where d_model is divisible by n_head.
GRID = [(L, H, D) for (L, H, D) in GRID if D % H == 0]
SEEDS = [0, 1, 2, 3, 4]

EPOCHS = 150
LR = 1e-3
BATCH_SIZE = 256
MAX_ORDERINGS = 4


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TTT_RUN_NAME")
    out_dir = prepare_run_dir(run_name)  # timestamped if run_name is None
    n_workers = int(os.environ.get("TTT_WORKERS", os.cpu_count() or 1))
    n_jobs = len(GRID) * len(SEEDS)

    print(f"Run dir: {out_dir}")
    print(f"Running {n_jobs} jobs ({len(GRID)} configs x {len(SEEDS)} seeds) "
          f"on {n_workers} workers, {EPOCHS} epochs...")
    t0 = time.time()
    raw = run_sweep_parallel(
        grid=GRID, seeds=SEEDS,
        epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, max_orderings=MAX_ORDERINGS,
        n_workers=n_workers, progress=True,
    )
    duration_min = (time.time() - t0) / 60

    agg = aggregate(raw)
    save_results(raw, agg,
                 os.path.join(out_dir, "raw.json"),
                 os.path.join(out_dir, "agg.json"))
    plot_capacity(agg, os.path.join(out_dir, "capacity.png"))

    meta = {
        "grid": GRID,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "max_orderings": MAX_ORDERINGS,
        "n_jobs": n_jobs,
        "duration_min": round(duration_min, 1),
        "git_sha": _git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {out_dir}/ (raw.json, agg.json, capacity.png, meta.json) "
          f"in {duration_min:.1f} min")


if __name__ == "__main__":
    main()
