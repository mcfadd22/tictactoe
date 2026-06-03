import os
import json
import pytest
from ttt.encoding import FLAT, ROWCOL
from ttt.sweep import aggregate, plot_capacity
from ttt.sweep import (
    Condition, run_condition, save_condition, compute_random_baselines,
    STANDARD_GRID, DEEP_GRID,
)


def test_standard_grid_excludes_indivisible_configs():
    for (L, H, D) in STANDARD_GRID:
        assert D % H == 0
    assert (1, 1, 16) in STANDARD_GRID
    assert all(D % H == 0 for (L, H, D) in DEEP_GRID)
    assert any(L == 8 for (L, H, D) in DEEP_GRID)  # depth grid adds L8


def test_condition_rejects_tied_with_rowcol():
    with pytest.raises(ValueError):
        Condition("bad", encoding=ROWCOL, head="tied")


def test_condition_rejects_unknown_head():
    with pytest.raises(ValueError):
        Condition("bad", head="nonsense")


def test_compute_random_baselines_has_horizontal_win_in_range():
    base = compute_random_baselines()
    assert 0.0 < base["horizontal_win"] < 1.0
    assert "horizontal_win_row0" in base


def test_run_condition_smoke_produces_per_row_metrics():
    cond = Condition(
        "smoke", encoding=FLAT, head="flat9",
        drop_horizontal_rows=frozenset({0, 1, 2}),
        grid=((1, 1, 16),), seeds=(0,),
        epochs=2, lr=1e-2, batch_size=256, max_orderings=2,
    )
    raw = run_condition(cond, n_workers=1)
    assert len(raw) == 1
    metrics = raw[0]["metrics"]
    for key in ("horizontal_win", "horizontal_win_row0",
                "horizontal_win_row1", "horizontal_win_row2", "vertical_win"):
        assert key in metrics


def test_save_condition_writes_artifacts(tmp_path):
    cond = Condition(
        "smoke", grid=((1, 1, 16),), seeds=(0,),
        epochs=2, lr=1e-2, batch_size=256, max_orderings=2,
    )
    raw = run_condition(cond, n_workers=1)
    save_condition(cond, raw, str(tmp_path))
    for fname in ("raw.json", "agg.json", "baselines.json", "capacity.png"):
        assert (tmp_path / fname).exists()
    base = json.loads((tmp_path / "baselines.json").read_text())
    assert "horizontal_win" in base


def test_aggregate_means_and_stds_over_seeds():
    raw = [
        {"config": "L1H1D16", "n_params": 100, "seed": 0,
         "metrics": {"horizontal_win": 0.0, "horizontal_block": 1.0}},
        {"config": "L1H1D16", "n_params": 100, "seed": 1,
         "metrics": {"horizontal_win": 0.2, "horizontal_block": 0.8}},
    ]
    agg = aggregate(raw)
    row = agg["L1H1D16"]
    assert row["n_params"] == 100
    assert abs(row["horizontal_win"]["mean"] - 0.1) < 1e-9
    assert row["horizontal_win"]["std"] >= 0.0
    assert abs(row["horizontal_block"]["mean"] - 0.9) < 1e-9


def test_plot_capacity_writes_file(tmp_path):
    agg = {
        "L1H1D16": {"n_params": 100,
                     "horizontal_win": {"mean": 0.1, "std": 0.05},
                     "horizontal_block": {"mean": 0.9, "std": 0.05},
                     "vertical_win": {"mean": 1.0, "std": 0.0},
                     "diagonal_win": {"mean": 1.0, "std": 0.0}},
        "L2H2D32": {"n_params": 500,
                     "horizontal_win": {"mean": 0.6, "std": 0.1},
                     "horizontal_block": {"mean": 0.95, "std": 0.02},
                     "vertical_win": {"mean": 1.0, "std": 0.0},
                     "diagonal_win": {"mean": 1.0, "std": 0.0}},
    }
    out = tmp_path / "capacity.png"
    plot_capacity(agg, str(out))
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_plot_capacity_with_baselines_and_ceiling_overlays(tmp_path):
    # Exercise the optional overlay branches (random-baseline line + ceiling line).
    agg = {
        "L1H1D16": {"n_params": 100,
                     "horizontal_win": {"mean": 0.1, "std": 0.05},
                     "horizontal_block": {"mean": 0.9, "std": 0.05},
                     "vertical_win": {"mean": 1.0, "std": 0.0},
                     "diagonal_win": {"mean": 1.0, "std": 0.0}},
        "L2H2D32": {"n_params": 500,
                     "horizontal_win": {"mean": 0.6, "std": 0.1},
                     "horizontal_block": {"mean": 0.95, "std": 0.02},
                     "vertical_win": {"mean": 1.0, "std": 0.0},
                     "diagonal_win": {"mean": 1.0, "std": 0.0}},
    }
    out = tmp_path / "capacity.png"
    plot_capacity(agg, str(out),
                  baselines={"horizontal_win": 0.12},
                  ceiling={100: 0.9, 500: 0.95})
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_parallel_matches_sequential():
    # Both pinned to a single torch thread so the sequential reference and the
    # single-thread workers produce bit-identical results.
    import torch
    from ttt.sweep import run_sweep, run_sweep_parallel

    torch.set_num_threads(1)
    grid = [(1, 1, 16), (2, 1, 16)]
    seeds = [0, 1]
    kw = dict(epochs=3, lr=1e-2, batch_size=256, max_orderings=2)

    seq = run_sweep(grid, seeds, **kw)
    par = run_sweep_parallel(grid, seeds, n_workers=2, **kw)

    assert len(par) == len(seq) == 4

    def key(r):
        return (r["config"], r["seed"])

    seq_by = {key(r): r for r in seq}
    par_by = {key(r): r for r in par}
    assert set(seq_by) == set(par_by)
    for k in seq_by:
        assert seq_by[k]["n_params"] == par_by[k]["n_params"]
        assert seq_by[k]["metrics"] == par_by[k]["metrics"]


def test_parallel_is_deterministic():
    from ttt.sweep import run_sweep_parallel

    kw = dict(epochs=3, lr=1e-2, batch_size=256, max_orderings=2)
    a = run_sweep_parallel([(1, 1, 16)], [0], n_workers=2, **kw)
    b = run_sweep_parallel([(1, 1, 16)], [0], n_workers=2, **kw)
    assert a[0]["metrics"] == b[0]["metrics"]
