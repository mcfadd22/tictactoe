from __future__ import annotations

import random

from ttt.board import EMPTY_BOARD, legal_moves, apply_move, winner, is_terminal
from ttt.solver import optimal_moves
from ttt.dataset import encode_prefix
from ttt.evaluate import model_move


class RandomPlayer:
    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def choose(self, board, history):
        return self.rng.choice(legal_moves(board))


class PerfectPlayer:
    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def choose(self, board, history):
        return self.rng.choice(optimal_moves(board))


class ModelPlayer:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def choose(self, board, history):
        return model_move(self.model, encode_prefix(tuple(history)), board,
                          device=self.device)


def play_game(player1, player2):
    """Play one game. Returns (move_history, winner)."""
    board = EMPTY_BOARD
    history = []
    players = [player1, player2]
    turn = 0
    while not is_terminal(board):
        move = players[turn % 2].choose(board, list(history))
        history.append(move)
        board = apply_move(board, move)
        turn += 1
    return history, winner(board)
