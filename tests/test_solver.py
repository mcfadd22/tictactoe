from ttt.board import EMPTY_BOARD, P1, P2, apply_move
from ttt.solver import minimax_value, optimal_moves


def test_empty_board_is_a_draw_under_optimal_play():
    # Tic-tac-toe is a draw with perfect play.
    assert minimax_value(EMPTY_BOARD) == 0


def test_takes_immediate_win():
    # P1 has 0,1; the only winning move is 2 and it must be among optimal moves.
    b = (P1, P1, 0, 0, P2, P2, 0, 0, 0)  # n1=2,n2=2 -> P1 to move
    assert 2 in optimal_moves(b)
    assert minimax_value(b) == 1


def test_must_block_to_avoid_loss():
    # P2 to move; P1 threatens 0,1 -> 2. Optimal must block at 2.
    b = (P1, P1, 0, 0, P2, 0, 0, 0, 0)  # n1=2,n2=1 -> P2 to move
    assert optimal_moves(b) == (2,)


def test_optimal_moves_nonempty_for_nonterminal():
    b = apply_move(EMPTY_BOARD, 4)
    assert len(optimal_moves(b)) >= 1
