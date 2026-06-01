from __future__ import annotations

EMPTY = 0
P1 = 1
P2 = 2

EMPTY_BOARD = (EMPTY,) * 9

HORIZONTAL_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
VERTICAL_LINES = [(0, 3, 6), (1, 4, 7), (2, 5, 8)]
DIAGONAL_LINES = [(0, 4, 8), (2, 4, 6)]
ALL_LINES = HORIZONTAL_LINES + VERTICAL_LINES + DIAGONAL_LINES
LINES_BY_TYPE = {
    "horizontal": HORIZONTAL_LINES,
    "vertical": VERTICAL_LINES,
    "diagonal": DIAGONAL_LINES,
}


def current_player(board):
    n1 = board.count(P1)
    n2 = board.count(P2)
    return P1 if n1 == n2 else P2


def legal_moves(board):
    return tuple(i for i, v in enumerate(board) if v == EMPTY)


def apply_move(board, cell):
    player = current_player(board)
    new = list(board)
    new[cell] = player
    return tuple(new)


def winner(board):
    for a, b, c in ALL_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def is_terminal(board):
    return winner(board) != 0 or EMPTY not in board


def winning_line_for_move(board, cell):
    """The line completed by the current player playing `cell`, else None."""
    player = current_player(board)
    after = apply_move(board, cell)
    for line in ALL_LINES:
        if cell in line and all(after[i] == player for i in line):
            return line
    return None


def line_type_completed_by_move(board, cell):
    line = winning_line_for_move(board, cell)
    if line is None:
        return None
    if line in HORIZONTAL_LINES:
        return "horizontal"
    if line in VERTICAL_LINES:
        return "vertical"
    return "diagonal"


def move_completes_horizontal(board, cell):
    return line_type_completed_by_move(board, cell) == "horizontal"
