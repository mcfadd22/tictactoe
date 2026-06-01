from __future__ import annotations

from ttt.board import EMPTY_BOARD, legal_moves, apply_move, is_terminal


def reachable_paths(max_orderings=4):
    """Map every reachable board -> up to `max_orderings` move-sequences reaching it.

    Boards are deduped; the first `max_orderings` distinct paths discovered are kept.
    A board is expanded only once (children depend only on the board, not the path).
    """
    paths = {}
    expanded = set()

    def dfs(board, path):
        if board not in paths:
            paths[board] = []
        if len(paths[board]) < max_orderings:
            paths[board].append(path)
        if board in expanded:
            return
        expanded.add(board)
        if is_terminal(board):
            return
        for m in legal_moves(board):
            dfs(apply_move(board, m), path + (m,))

    dfs(EMPTY_BOARD, ())
    return paths
