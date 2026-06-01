from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ttt.board import is_terminal, move_completes_horizontal
from ttt.solver import optimal_moves
from ttt.enumerate import reachable_paths

N_CELLS = 9
BOS_ID = 9
PAD_ID = 10
VOCAB_SIZE = 11  # cells 0-8, BOS=9, PAD=10


def encode_prefix(path):
    return [BOS_ID] + list(path)


def build_examples(max_orderings=4, filter_horizontal=True):
    """List of (input_ids, target_cell).

    For every non-terminal reachable board, for each kept ordering, emit one
    example per optimal target move. When `filter_horizontal`, drop any target
    that completes a horizontal three (the experimental manipulation).
    """
    paths = reachable_paths(max_orderings=max_orderings)
    examples = []
    for board, orderings in paths.items():
        if is_terminal(board):
            continue
        targets = optimal_moves(board)
        if filter_horizontal:
            targets = tuple(
                m for m in targets if not move_completes_horizontal(board, m)
            )
        if not targets:
            continue
        for path in orderings:
            input_ids = encode_prefix(path)
            for target in targets:
                examples.append((input_ids, target))
    return examples


class TTTDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids, target = self.examples[idx]
        return input_ids, target


def collate_fn(batch):
    """Right-pad inputs to the batch max; return (ids, lengths, targets) tensors.

    Right-padding keeps every real last-token at index length-1, so the model
    reads it without attending to pad positions (which are causally 'future').
    """
    max_len = max(len(ids) for ids, _ in batch)
    ids_t = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    targets = torch.zeros(len(batch), dtype=torch.long)
    for i, (ids, tgt) in enumerate(batch):
        ids_t[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        lengths[i] = len(ids)
        targets[i] = tgt
    return ids_t, lengths, targets
