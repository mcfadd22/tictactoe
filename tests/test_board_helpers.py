from ttt.board import (
    P1, P2, EMPTY_BOARD,
    winning_line_for_move, line_type_completed_by_move, move_completes_horizontal,
)


def test_winning_line_for_move_horizontal():
    # P1 owns 0,1 and it is P1's turn (n1==n2); playing 2 completes the top row.
    b = (P1, P1, 0, P2, P2, 0, 0, 0, 0)
    assert winning_line_for_move(b, 2) == (0, 1, 2)


def test_winning_line_for_move_none():
    # P1 to move on an empty board; playing 1 completes nothing.
    assert winning_line_for_move(EMPTY_BOARD, 1) is None


def test_line_type_completed_classifies():
    horiz = (P1, P1, 0, P2, P2, 0, 0, 0, 0)   # P1 to move, play 2 -> top row
    assert line_type_completed_by_move(horiz, 2) == "horizontal"
    vert = (P1, P2, P2, P1, 0, 0, 0, 0, 0)     # P1 owns 0,3; play 6 -> left col
    assert line_type_completed_by_move(vert, 6) == "vertical"
    diag = (P1, P2, P2, 0, P1, 0, 0, 0, 0)     # P1 owns 0,4; play 8 -> main diag
    assert line_type_completed_by_move(diag, 8) == "diagonal"


def test_move_completes_horizontal_flag():
    b = (P1, P1, 0, P2, P2, 0, 0, 0, 0)        # P1 to move; play 2 -> horizontal win
    assert move_completes_horizontal(b, 2) is True
    bv = (P1, P2, P2, P1, 0, 0, 0, 0, 0)       # P1 to move; play 6 -> vertical, not horizontal
    assert move_completes_horizontal(bv, 6) is False


def test_move_completes_horizontal_respects_current_player():
    # n1=2, n2=1 -> P2 to move. P2 placing at cell 2 fills the top row with
    # mixed marks (P1,P1,P2), which is NOT a win.
    b = (P1, P1, 0, P2, 0, 0, 0, 0, 0)
    assert move_completes_horizontal(b, 2) is False
