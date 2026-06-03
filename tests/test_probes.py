from ttt.board import P1, P2, EMPTY, current_player, move_completes_horizontal
from ttt.probes import win_available_probes, block_needed_probes, probe_prefixes
from ttt.enumerate import reachable_paths
from ttt.encoding import ROWCOL
from ttt.board import horizontal_row_completed_by_move, current_player


def test_win_rows_filter_restricts_to_requested_rows():
    all_h = win_available_probes("horizontal")
    row0 = win_available_probes("horizontal", rows=frozenset({0}))
    rows12 = win_available_probes("horizontal", rows=frozenset({1, 2}))
    assert len(row0) > 0
    # every row-0 probe completes row 0; none of rows12 complete row 0
    for board, target in row0:
        assert horizontal_row_completed_by_move(board, target) == 0
    for board, target in rows12:
        assert horizontal_row_completed_by_move(board, target) in (1, 2)
    # the partition covers the whole set (rows 0,1,2 are exhaustive)
    assert len(row0) + len(rows12) == len(all_h)


def test_block_rows_filter_restricts_to_requested_rows():
    from ttt.board import P1, P2
    blocked = block_needed_probes("horizontal", rows=frozenset({0}))
    assert len(blocked) > 0  # guard against a vacuously-passing empty loop
    for board, target in blocked:
        player = current_player(board)
        opp = P2 if player == P1 else P1
        after = list(board)
        after[target] = opp
        # the defended horizontal line is row 0
        from ttt.board import HORIZONTAL_LINES
        assert any(target in ln and all(after[j] == opp for j in ln)
                   and HORIZONTAL_LINES.index(ln) == 0
                   for ln in HORIZONTAL_LINES)


def test_probe_prefixes_use_the_given_encoding():
    from ttt.board import EMPTY_BOARD, apply_move
    paths = reachable_paths(max_orderings=4)
    board, _ = win_available_probes("vertical")[0]
    prefixes = probe_prefixes(board, paths, encoding=ROWCOL, max_orderings=4)
    assert len(prefixes) >= 1
    for pref in prefixes:
        assert pref[0] == ROWCOL.bos_id
        replay = EMPTY_BOARD
        for cell in ROWCOL.decode_path(pref):
            replay = apply_move(replay, cell)
        assert replay == board


def test_win_available_targets_complete_their_line_type():
    probes = win_available_probes("horizontal")
    assert len(probes) > 0
    for board, target in probes:
        assert board[target] == EMPTY
        assert move_completes_horizontal(board, target)


def test_win_available_vertical_are_not_horizontal():
    probes = win_available_probes("vertical")
    assert len(probes) > 0
    for board, target in probes:
        assert not move_completes_horizontal(board, target)


def test_block_needed_target_is_opponent_threat_cell():
    probes = block_needed_probes("horizontal")
    assert len(probes) > 0
    for board, target in probes:
        player = current_player(board)
        opp = P2 if player == P1 else P1
        # The target cell, if filled by the opponent, would complete a line.
        filled = list(board)
        filled[target] = opp
        from ttt.board import winner
        assert winner(tuple(filled)) == opp


def test_probe_prefixes_reach_board():
    from ttt.board import apply_move, EMPTY_BOARD
    paths = reachable_paths(max_orderings=4)
    probes = win_available_probes("vertical")
    board, _ = probes[0]
    prefixes = probe_prefixes(board, paths, max_orderings=4)
    assert len(prefixes) >= 1
    for pref in prefixes:
        replay = EMPTY_BOARD
        for tok in pref[1:]:  # skip BOS
            replay = apply_move(replay, tok)
        assert replay == board


def test_win_available_horizontal_targets_are_pure():
    from ttt.board import VERTICAL_LINES, DIAGONAL_LINES, current_player, apply_move
    probes = win_available_probes("horizontal")
    assert len(probes) > 0
    for board, target in probes:
        player = current_player(board)
        after = apply_move(board, target)  # apply_move uses the current player
        for line in VERTICAL_LINES + DIAGONAL_LINES:
            assert not all(after[i] == player for i in line)


def test_block_needed_horizontal_threat_is_pure():
    from ttt.board import VERTICAL_LINES, DIAGONAL_LINES, current_player, P1, P2
    probes = block_needed_probes("horizontal")
    assert len(probes) > 0
    for board, target in probes:
        player = current_player(board)
        opp = P2 if player == P1 else P1
        filled = list(board)
        filled[target] = opp
        for line in VERTICAL_LINES + DIAGONAL_LINES:
            assert not all(filled[i] == opp for i in line)


def test_block_needed_probes_are_forced_unique_optimal():
    from ttt.solver import optimal_moves
    for line_type in ("horizontal", "vertical", "diagonal"):
        probes = block_needed_probes(line_type)
        assert len(probes) > 0
        for board, target in probes:
            assert optimal_moves(board) == (target,)


def test_win_available_probes_are_unique_optimal():
    from ttt.solver import optimal_moves
    for line_type in ("horizontal", "vertical", "diagonal"):
        probes = win_available_probes(line_type)
        assert len(probes) > 0
        for board, target in probes:
            assert optimal_moves(board) == (target,)
