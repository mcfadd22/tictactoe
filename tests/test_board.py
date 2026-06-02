from ttt.board import (
    EMPTY, P1, P2, EMPTY_BOARD,
    HORIZONTAL_LINES, VERTICAL_LINES, DIAGONAL_LINES, ALL_LINES, LINES_BY_TYPE,
    current_player, legal_moves, apply_move, winner, is_terminal,
)
from ttt.board import (
    horizontal_row_completed_by_move, move_completes_horizontal,
)


def test_horizontal_row_completed_returns_row_index():
    # X to move (equal counts). Top row: X at 0,1 -> playing 2 completes row 0.
    board = (P1, P1, EMPTY, P2, P2, EMPTY, EMPTY, EMPTY, EMPTY)
    assert horizontal_row_completed_by_move(board, 2) == 0
    # Bottom row: needs X at 6,7 already; build a board where X completes row 2.
    # X (P1) must be the current player, so X and O counts are equal.
    board2 = (P1, P2, EMPTY, P2, P2, EMPTY, P1, P1, EMPTY)
    assert horizontal_row_completed_by_move(board2, 8) == 2


def test_horizontal_row_completed_none_for_non_horizontal():
    # X at 0,3 -> playing 6 completes a vertical (col 0), not horizontal.
    board = (P1, P2, EMPTY, P1, P2, EMPTY, EMPTY, EMPTY, EMPTY)
    assert horizontal_row_completed_by_move(board, 6) is None
    assert move_completes_horizontal(board, 6) is False


def test_move_completes_horizontal_agrees_with_row_helper():
    board = (P1, P1, EMPTY, P2, P2, EMPTY, EMPTY, EMPTY, EMPTY)
    assert move_completes_horizontal(board, 2) is True
    assert horizontal_row_completed_by_move(board, 2) == 0


def test_lines_partition_into_eight():
    assert len(HORIZONTAL_LINES) == 3
    assert len(VERTICAL_LINES) == 3
    assert len(DIAGONAL_LINES) == 2
    assert len(ALL_LINES) == 8
    assert LINES_BY_TYPE["horizontal"] == HORIZONTAL_LINES


def test_current_player_starts_with_p1():
    assert current_player(EMPTY_BOARD) == P1


def test_current_player_alternates():
    b = apply_move(EMPTY_BOARD, 0)  # P1 plays 0
    assert current_player(b) == P2


def test_legal_moves_are_empty_cells():
    b = apply_move(EMPTY_BOARD, 4)
    assert 4 not in legal_moves(b)
    assert len(legal_moves(b)) == 8


def test_apply_move_places_current_players_mark():
    b = apply_move(EMPTY_BOARD, 0)
    assert b[0] == P1
    b2 = apply_move(b, 1)
    assert b2[1] == P2


def test_winner_detects_horizontal():
    # P1 at 0,1,2
    b = (P1, P1, P1, 0, 0, 0, 0, 0, 0)
    assert winner(b) == P1


def test_winner_none_on_empty():
    assert winner(EMPTY_BOARD) == 0


def test_is_terminal_on_win_and_full():
    win = (P1, P1, P1, 0, 0, 0, 0, 0, 0)
    assert is_terminal(win)
    full_draw = (P1, P2, P1, P1, P2, P2, P2, P1, P1)
    assert is_terminal(full_draw)
    assert not is_terminal(EMPTY_BOARD)
