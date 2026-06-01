from ttt.board import EMPTY_BOARD, is_terminal
from ttt.enumerate import reachable_paths


def test_empty_board_has_empty_path():
    paths = reachable_paths(max_orderings=4)
    assert EMPTY_BOARD in paths
    assert () in paths[EMPTY_BOARD]


def test_total_reachable_positions_count():
    # Tic-tac-toe has 5478 reachable board states (including terminal).
    paths = reachable_paths(max_orderings=4)
    assert len(paths) == 5478


def test_orderings_capped():
    paths = reachable_paths(max_orderings=4)
    assert all(len(v) <= 4 for v in paths.values())


def test_paths_actually_reach_their_board():
    from ttt.board import apply_move
    paths = reachable_paths(max_orderings=4)
    # Pick a non-empty board and replay one of its paths.
    for board, seqs in paths.items():
        if board == EMPTY_BOARD:
            continue
        seq = seqs[0]
        replay = EMPTY_BOARD
        for m in seq:
            replay = apply_move(replay, m)
        assert replay == board
        break
