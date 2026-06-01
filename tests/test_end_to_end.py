import os
from ttt.sweep import run_sweep, aggregate, plot_capacity


def test_tiny_end_to_end_smoke(tmp_path):
    # One tiny config, one seed, few epochs: must run end-to-end and produce a plot.
    raw = run_sweep(
        grid=[(1, 1, 16)], seeds=[0],
        epochs=2, lr=1e-2, batch_size=256, max_orderings=2,
    )
    assert len(raw) == 1
    assert "horizontal_win" in raw[0]["metrics"]
    agg = aggregate(raw)
    out = tmp_path / "capacity.png"
    plot_capacity(agg, str(out))
    assert os.path.exists(out)
