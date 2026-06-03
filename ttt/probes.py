from __future__ import annotations

from ttt.board import (
    EMPTY, P1, P2, current_player, is_terminal, winner,
    LINES_BY_TYPE, ALL_LINES,
)
from ttt.enumerate import reachable_paths
from ttt.encoding import FLAT
from ttt.solver import optimal_moves


def _immediate_win_cell(board, player):
    """A cell where `player` immediately completes any line, else None."""
    for a, b, c in ALL_LINES:
        cells = (a, b, c)
        vals = [board[i] for i in cells]
        if vals.count(player) == 2 and vals.count(EMPTY) == 1:
            return cells[vals.index(EMPTY)]
    return None


def _completed_line_types_for(board, cell, player):
    """Line types that become fully `player`-owned after `player` plays `cell`."""
    after = list(board)
    after[cell] = player
    types = set()
    for ltype, lines in LINES_BY_TYPE.items():
        for line in lines:
            if cell in line and all(after[i] == player for i in line):
                types.add(ltype)
    return types


def win_available_probes(line_type, rows=None):
    """Boards where the player to move can win on a `line_type` line next move,
    that move completes ONLY a `line_type` line, AND it is the unique optimal
    move. When `rows` is given, restrict to lines at those indices within the
    type (used to split horizontal wins by row for the translational condition).
    Excludes terminal boards.
    """
    probes = []
    seen = set()
    for board in reachable_paths(max_orderings=1):
        if is_terminal(board) or board in seen:
            continue
        player = current_player(board)
        for idx, line in enumerate(LINES_BY_TYPE[line_type]):
            if rows is not None and idx not in rows:
                continue
            vals = [board[i] for i in line]
            if vals.count(player) == 2 and vals.count(EMPTY) == 1:
                target = line[vals.index(EMPTY)]
                if (_completed_line_types_for(board, target, player) == {line_type}
                        and optimal_moves(board) == (target,)):
                    probes.append((board, target))
                    seen.add(board)
                    break
    return probes


def block_needed_probes(line_type, rows=None):
    """Boards where the opponent threatens exactly one winning cell, that block
    completes ONLY a `line_type` line for the opponent, and the player to move
    has no immediate win of their own. When `rows` is given, restrict to the
    line at those indices within the type.
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
        for a, b, c in ALL_LINES:
            cells = (a, b, c)
            vals = [board[i] for i in cells]
            if vals.count(opp) == 2 and vals.count(EMPTY) == 1:
                threats.add(cells[vals.index(EMPTY)])
        if len(threats) != 1:
            continue
        target = next(iter(threats))
        if not (_completed_line_types_for(board, target, opp) == {line_type}
                and optimal_moves(board) == (target,)):
            continue
        if rows is not None:
            after = list(board)
            after[target] = opp
            line_idx = next(
                (i for i, ln in enumerate(LINES_BY_TYPE[line_type])
                 if target in ln and all(after[j] == opp for j in ln)),
                None,
            )
            if line_idx not in rows:
                continue
        probes.append((board, target))
    return probes


def probe_prefixes(board, paths, encoding=FLAT, max_orderings=4):
    """Encoded prefixes ([BOS, ...]) for the orderings that reach `board`."""
    return [encoding.encode(p) for p in paths.get(board, [])[:max_orderings]]
