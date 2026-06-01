import os
from ttt.sweep import aggregate, plot_capacity


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
