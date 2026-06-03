"""Run symmetry-generalization conditions and write results to ./results/<name>/.

Each experiment is a composable `Condition` (four orthogonal axes). Run one with
`--condition E2`, or all Phase-1 conditions with `--condition all` (default).
Override worker count with `--workers` or the TTT_WORKERS env var.

Pass `--push-results` (for headless VM runs) to commit and push each condition's
`results/<name>/` to the current branch as it finishes. See
docs/design/specs/2026-06-03-results-git-sync-design.md for VM setup.
"""
from __future__ import annotations

import argparse
import os
import time

from ttt.encoding import FLAT, ROWCOL
from ttt.gitsync import sync_results
from ttt.sweep import (
    Condition, run_condition, save_condition, horizontal_win_ceiling,
    STANDARD_GRID, DEEP_GRID,
)

# Phase-1 matrix: each varies ONE axis off the baseline
# (flat encoding, flat9 head, drop-all filter, standard grid).
CONDITIONS = {
    "E0": Condition("E0", encoding=FLAT, head="flat9",
                    drop_horizontal_rows=frozenset({0, 1, 2}), grid=STANDARD_GRID),
    "E1": Condition("E1", encoding=FLAT, head="flat9",
                    drop_horizontal_rows=frozenset(), grid=STANDARD_GRID),
    "E2": Condition("E2", encoding=ROWCOL, head="flat9",
                    drop_horizontal_rows=frozenset({0, 1, 2}), grid=STANDARD_GRID),
    "E3": Condition("E3", encoding=FLAT, head="flat9",
                    drop_horizontal_rows=frozenset({1, 2}), grid=STANDARD_GRID),
    "E4": Condition("E4", encoding=FLAT, head="flat9",
                    drop_horizontal_rows=frozenset({0, 1, 2}), grid=DEEP_GRID),
    "E5a": Condition("E5a", encoding=FLAT, head="tied",
                     drop_horizontal_rows=frozenset({0, 1, 2}), grid=STANDARD_GRID),
    "E5b": Condition("E5b", encoding=FLAT, head="factored",
                     drop_horizontal_rows=frozenset({0, 1, 2}), grid=STANDARD_GRID),
}
PHASE1 = ["E0", "E1", "E2", "E3", "E4", "E5a", "E5b"]


def run_one(name, n_workers, ceiling=None, push_results=False):
    cond = CONDITIONS[name]
    out_dir = os.path.join("results", name)
    n_jobs = len(cond.grid) * len(cond.seeds)
    print(f"=== {name}: {n_jobs} jobs "
          f"({len(cond.grid)} configs x {len(cond.seeds)} seeds), "
          f"encoding={cond.encoding.name}, head={cond.head}, "
          f"drop={sorted(cond.drop_horizontal_rows)} ===")
    t0 = time.time()
    raw = run_condition(cond, n_workers=n_workers, progress=True)
    agg, _ = save_condition(cond, raw, out_dir, ceiling=ceiling)
    print(f"  {name} finished in {(time.time() - t0) / 60:.1f} min -> {out_dir}/")
    if push_results:
        # Forward-slash path so the pathspec is portable across shells.
        message = (f"results: {name} ({len(cond.grid)} configs x "
                   f"{len(cond.seeds)} seeds, encoding={cond.encoding.name}, "
                   f"head={cond.head})")
        sync_results([f"results/{name}"], message)
    return agg


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="all",
                        choices=["all"] + PHASE1)
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("TTT_WORKERS",
                                                   os.cpu_count() or 1)))
    parser.add_argument("--push-results", action="store_true",
                        help="commit and push each condition's results to the "
                             "current branch as it finishes (for VM runs)")
    return parser


def main():
    args = build_parser().parse_args()

    os.makedirs("results", exist_ok=True)
    names = PHASE1 if args.condition == "all" else [args.condition]

    # When running the full set, compute the positive-control (E1) ceiling first
    # and overlay it on the other flat/standard-grid conditions' plots.
    ceiling = None
    aggs = {}
    if "E1" in names:
        aggs["E1"] = run_one("E1", args.workers, push_results=args.push_results)
        ceiling = horizontal_win_ceiling(aggs["E1"])
        names = [n for n in names if n != "E1"]

    for name in names:
        # Ceiling overlay only makes sense at matched params (flat, standard grid).
        use_ceiling = ceiling if CONDITIONS[name].encoding.name == "flat" \
            and CONDITIONS[name].grid == STANDARD_GRID else None
        aggs[name] = run_one(name, args.workers, ceiling=use_ceiling,
                             push_results=args.push_results)


if __name__ == "__main__":
    main()
