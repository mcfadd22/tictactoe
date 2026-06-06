import pytest
from run_experiment import CONDITIONS, PHASE1


def test_phase1_has_all_seven_conditions():
    assert PHASE1 == ["E0", "E1", "E2", "E3", "E4", "E5a", "E5b"]
    for name in PHASE1:
        assert name in CONDITIONS
        assert CONDITIONS[name].name == name


def test_condition_axes_match_the_design_matrix():
    assert CONDITIONS["E0"].drop_horizontal_rows == frozenset({0, 1, 2})
    assert CONDITIONS["E1"].drop_horizontal_rows == frozenset()       # positive control
    assert CONDITIONS["E2"].encoding.name == "rowcol"                  # encoding axis
    assert CONDITIONS["E3"].drop_horizontal_rows == frozenset({1, 2})  # translational
    assert any(L == 8 for (L, H, D) in CONDITIONS["E4"].grid)          # depth
    assert CONDITIONS["E5a"].head == "tied"
    assert CONDITIONS["E5b"].head == "factored"


def test_all_phase1_conditions_use_five_seeds():
    for name in PHASE1:
        assert len(CONDITIONS[name].seeds) == 5


def test_push_results_flag_defaults_off():
    from run_experiment import build_parser
    assert build_parser().parse_args(["--condition", "E0"]).push_results is False
    assert build_parser().parse_args(
        ["--condition", "E0", "--push-results"]).push_results is True


def test_run_grok_smoke_writes_per_wd_and_combined(tmp_path, monkeypatch):
    import os
    import run_experiment as R
    monkeypatch.chdir(tmp_path)
    # tiny override: 1 config, 1 seed, few epochs, 2 wd values
    monkeypatch.setattr(R, "GROK_GRID", ((1, 1, 16),))
    monkeypatch.setattr(R, "WD_SWEEP", (0.0, 1.0))
    R.run_grok("E0", n_workers=1, epochs=4, eval_every=2, seeds=(0,))
    assert os.path.exists("results/E_GROK_E0_wd0.0/grok_curve.png")
    assert os.path.exists("results/E_GROK_E0_wd1.0/trajectory.json")
    assert os.path.exists("results/E_GROK_E0/grok_curves.png")
    assert os.path.exists("results/E_GROK_E0/trajectories.json")
