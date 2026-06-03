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
