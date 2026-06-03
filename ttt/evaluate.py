from __future__ import annotations

import torch

from ttt.board import EMPTY, legal_moves
from ttt.probes import probe_prefixes
from ttt.encoding import FLAT


@torch.no_grad()
def model_move(model, input_ids, board, device="cpu"):
    """Greedy legal move: mask non-empty cells, argmax over 9 logits."""
    model.eval()
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(input_ids)], dtype=torch.long, device=device)
    logits = model(ids, lengths).squeeze(0)
    illegal = torch.tensor([board[i] != EMPTY for i in range(9)], device=device)
    logits = logits.masked_fill(illegal, float("-inf"))
    return int(logits.argmax().item())


def evaluate_probes(model, probes, paths, encoding=FLAT, max_orderings=4, device="cpu"):
    """Fraction of (probe, ordering) pairs where the model plays the target cell.

    Also returns order_invariance: fraction of probes whose chosen move is the
    same across all their orderings.
    """
    hits, total = 0, 0
    invariant_probes, probes_with_orderings = 0, 0
    for board, target in probes:
        prefixes = probe_prefixes(board, paths, encoding=encoding,
                                  max_orderings=max_orderings)
        if not prefixes:
            continue
        moves = []
        for pref in prefixes:
            move = model_move(model, pref, board, device=device)
            moves.append(move)
            total += 1
            if move == target:
                hits += 1
        probes_with_orderings += 1
        if len(set(moves)) == 1:
            invariant_probes += 1
    return {
        "rate": hits / total if total else 0.0,
        "order_invariance": (
            invariant_probes / probes_with_orderings if probes_with_orderings else 0.0
        ),
        "n_probes": probes_with_orderings,
    }


def random_baseline_rate(probes):
    """Expected hit-rate of a uniform random legal move over the probe set."""
    if not probes:
        return 0.0
    total = 0.0
    for board, _ in probes:
        n = len(legal_moves(board))
        total += 1.0 / n if n else 0.0
    return total / len(probes)
