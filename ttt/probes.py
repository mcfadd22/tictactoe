from __future__ import annotations

from ttt.board import (
    EMPTY, P1, P2, current_player, is_terminal, winner,
    LINES_BY_TYPE, ALL_LINES, line_type_completed_by_move,
)
from ttt.enumerate import reachable_paths
from ttt.dataset import encode_prefix


def _immediate_win_cell(board, player):
    """A cell where `player` immediately completes any line, else None."""
    for a, b, c in ALL_LINES:
        cells = (a, b, c)
        vals = [board[i] for i in cells]
        if vals.count(player) == 2 and vals.count(EMPTY) == 1:
            return cells[vals.index(EMPTY)]
    return None


def win_available_probes(line_type):
    """Boards where the player to move can win on a `line_type` line next move.

    Returns list of (board, target_cell). Excludes already-terminal boards.
    """
    probes = []
    seen = set()
    for board in reachable_paths(max_orderings=1):
        if is_terminal(board) or board in seen:
            continue
        player = current_player(board)
        for line in LINES_BY_TYPE[line_type]:
            vals = [board[i] for i in line]
            if vals.count(player) == 2 and vals.count(EMPTY) == 1:
                target = line[vals.index(EMPTY)]
                # Verify the move's winning line type matches; skip if the cell
                # simultaneously completes a line of a different (higher-priority) type.
                if line_type_completed_by_move(board, target) != line_type:
                    continue
                probes.append((board, target))
                seen.add(board)
                break
    return probes


def block_needed_probes(line_type):
    """Boards where the opponent threatens a `line_type` win and blocking it is
    the clear correct move.

    Clean conditions: the player to move has NO immediate win of their own, and
    there is exactly one opponent threat cell, so the block target is unique.
    """
    probes = []
    for board in reachable_paths(max_orderings=1):
        if is_terminal(board):
            continue
        player = current_player(board)
        opp = P2 if player == P1 else P1
        if _immediate_win_cell(board, player) is not None:
            continue  # taking the win dominates blocking; skip for cleanliness
        threats = set()
        target_for_type = None
        for a, b, c in ALL_LINES:
            cells = (a, b, c)
            vals = [board[i] for i in cells]
            if vals.count(opp) == 2 and vals.count(EMPTY) == 1:
                cell = cells[vals.index(EMPTY)]
                threats.add(cell)
                if (a, b, c) in LINES_BY_TYPE[line_type]:
                    target_for_type = cell
        if len(threats) == 1 and target_for_type is not None:
            probes.append((board, target_for_type))
    return probes


def probe_prefixes(board, paths, max_orderings=4):
    """Encoded prefixes ([BOS, ...]) for the orderings that reach `board`."""
    return [encode_prefix(p) for p in paths.get(board, [])[:max_orderings]]
