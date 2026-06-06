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
import json
import os
import time

from ttt.encoding import FLAT, ROWCOL
from ttt.gitsync import sync_results
from ttt.sweep import (
    Condition, run_condition, save_condition, horizontal_win_ceiling,
    STANDARD_GRID, DEEP_GRID, GROK_GRID, held_out_curve, plot_grok_curves,
    compute_random_baselines,
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
    if base not in GROK_BASES:
        raise ValueError(f"unknown grok base: {base!r} (expected one of "
                         f"{sorted(GROK_BASES)})")
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
    parser.add_argument("--grok", choices=["E0", "E3", "all"],
                        help="run the long-training weight-decay grokking probe "
                             "for a base condition (or both)")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("TTT_WORKERS",
                                                   os.cpu_count() or 1)))
    parser.add_argument("--push-results", action="store_true",
                        help="commit and push each condition's results to the "
                             "current branch as it finishes (for VM runs)")
    return parser


def main():
    args = build_parser().parse_args()

    if args.grok:
        os.makedirs("results", exist_ok=True)
        bases = ["E0", "E3"] if args.grok == "all" else [args.grok]
        for base in bases:
            run_grok(base, args.workers)
        return

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
