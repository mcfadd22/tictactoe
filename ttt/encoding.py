from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple


def _flat_encode_move(cell: int) -> List[int]:
    return [cell]


def _flat_decode_move(tokens: Tuple[int, ...]) -> int:
    return tokens[0]


def _rowcol_encode_move(cell: int) -> List[int]:
    # ROW family = ids 0..2, COL family = ids 3..5 (distinct token families).
    return [cell // 3, 3 + cell % 3]


def _rowcol_decode_move(tokens: Tuple[int, ...]) -> int:
    row, col = tokens[0], tokens[1] - 3
    return row * 3 + col


@dataclass(frozen=True)
class Encoding:
    """How a move-path becomes a token sequence. Output is always BOS-prefixed.

    Holds module-level functions (not lambdas/closures) so instances pickle
    across the process-pool sweep workers.
    """
    name: str
    vocab_size: int
    max_len: int
    tokens_per_move: int
    bos_id: int
    pad_id: int
    encode_move: Callable[[int], List[int]]
    decode_move: Callable[[Tuple[int, ...]], int]

    def encode(self, path) -> List[int]:
        tokens = [self.bos_id]
        for cell in path:
            tokens.extend(self.encode_move(cell))
        return tokens

    def decode_path(self, tokens) -> Tuple[int, ...]:
        body = tokens[1:]  # strip BOS
        cells = []
        for i in range(0, len(body), self.tokens_per_move):
            chunk = tuple(body[i:i + self.tokens_per_move])
            cells.append(self.decode_move(chunk))
        return tuple(cells)


FLAT = Encoding(
    name="flat", vocab_size=11, max_len=10, tokens_per_move=1,
    bos_id=9, pad_id=10,
    encode_move=_flat_encode_move, decode_move=_flat_decode_move,
)

ROWCOL = Encoding(
    name="rowcol", vocab_size=8, max_len=19, tokens_per_move=2,
    bos_id=6, pad_id=7,
    encode_move=_rowcol_encode_move, decode_move=_rowcol_decode_move,
)

ENCODINGS = {"flat": FLAT, "rowcol": ROWCOL}
