from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ttt.board import is_terminal, horizontal_row_completed_by_move
from ttt.solver import optimal_moves
from ttt.enumerate import reachable_paths
from ttt.encoding import FLAT

N_CELLS = 9
BOS_ID = 9
PAD_ID = 10
VOCAB_SIZE = 11  # cells 0-8, BOS=9, PAD=10 (legacy flat constants)
ALL_ROWS = frozenset({0, 1, 2})


def encode_prefix(path):
    """Legacy flat-encoding helper, kept for back-compat. Equals FLAT.encode."""
    return FLAT.encode(list(path))


def build_examples(encoding=FLAT, *, drop_horizontal_rows=ALL_ROWS,
                   max_orderings=4, filter_horizontal=None):
    """List of (input_ids, target_cell).

    For every non-terminal reachable board, for each kept ordering, emit one
    example per optimal target move. A target is dropped iff it completes a
    horizontal line whose row index is in `drop_horizontal_rows` (a subset of
    {0,1,2}). The legacy boolean `filter_horizontal` is still accepted:
    True -> drop all rows, False -> drop none.
    """
    if filter_horizontal is not None:
        drop_horizontal_rows = ALL_ROWS if filter_horizontal else frozenset()
    drop_horizontal_rows = frozenset(drop_horizontal_rows)
    paths = reachable_paths(max_orderings=max_orderings)
    examples = []
    for board, orderings in paths.items():
        if is_terminal(board):
            continue
        targets = optimal_moves(board)
        if drop_horizontal_rows:
            targets = tuple(
                m for m in targets
                if horizontal_row_completed_by_move(board, m)
                not in drop_horizontal_rows
            )
        if not targets:
            continue
        for path in orderings:
            input_ids = encoding.encode(path)
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


def collate_fn(batch, pad_id=PAD_ID):
    """Right-pad inputs to the batch max; return (ids, lengths, targets) tensors.

    Right-padding keeps every real last-token at index length-1, so the model
    reads it without attending to pad positions (which are causally 'future').
    """
    max_len = max(len(ids) for ids, _ in batch)
    ids_t = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    targets = torch.zeros(len(batch), dtype=torch.long)
    for i, (ids, tgt) in enumerate(batch):
        ids_t[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        lengths[i] = len(ids)
        targets[i] = tgt
    return ids_t, lengths, targets
