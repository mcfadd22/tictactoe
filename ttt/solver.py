from __future__ import annotations

from functools import lru_cache

from ttt.board import legal_moves, apply_move, winner, is_terminal, current_player


@lru_cache(maxsize=None)
def minimax_value(board):
    """Value to the player to move: +1 win, 0 draw, -1 loss (perfect play)."""
    w = winner(board)
    if w != 0:
        # Previous player just won, so the player to move has lost.
        return -1
    if is_terminal(board):
        return 0
    best = -2
    for m in legal_moves(board):
        # Opponent's value after our move; negate for our perspective.
        val = -minimax_value(apply_move(board, m))
        if val > best:
            best = val
    return best


def optimal_moves(board):
    """All moves achieving the minimax-optimal value for the player to move."""
    assert not is_terminal(board), "no moves from a terminal board"
    best = -2
    scored = []
    for m in legal_moves(board):
        val = -minimax_value(apply_move(board, m))
        scored.append((m, val))
        if val > best:
            best = val
    return tuple(m for m, val in scored if val == best)
