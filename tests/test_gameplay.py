import random
from ttt.board import EMPTY_BOARD, winner, is_terminal
from ttt.gameplay import RandomPlayer, PerfectPlayer, play_game


def test_perfect_vs_perfect_is_always_draw():
    for seed in range(5):
        rng = random.Random(seed)
        moves, w = play_game(PerfectPlayer(rng), PerfectPlayer(rng))
        assert w == 0  # perfect play -> draw


def test_play_game_returns_terminal_board():
    rng = random.Random(0)
    moves, w = play_game(RandomPlayer(rng), RandomPlayer(rng))
    board = EMPTY_BOARD
    from ttt.board import apply_move
    for m in moves:
        board = apply_move(board, m)
    assert is_terminal(board)
    assert w == winner(board)
