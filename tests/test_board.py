from ttt.board import (
    EMPTY, P1, P2, EMPTY_BOARD,
    HORIZONTAL_LINES, VERTICAL_LINES, DIAGONAL_LINES, ALL_LINES, LINES_BY_TYPE,
    current_player, legal_moves, apply_move, winner, is_terminal,
)


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
