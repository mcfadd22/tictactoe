# Composable Symmetry-Generalization Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four orthogonal, composable axes — input encoding, output head, horizontal-filter-as-row-set, and grid/depth — onto the existing tic-tac-toe transformer sweep so the symmetry-generalization follow-up experiments E0–E5b run as data-only `Condition` entries with no new code per experiment.

**Architecture:** Introduce a new `Encoding` value object (`flat`/`rowcol`) and thread it through dataset → model → probes → evaluate → sweep. Generalize the boolean horizontal filter to a `frozenset[int]` of rows to drop. Add a `head` selector (`flat9`/`tied`/`factored`) to the model. Wrap the existing process-pool sweep in a `Condition` dataclass + `run_condition`. **Every new parameter defaults to the current behavior (`FLAT` encoding, `flat9` head, drop-all filter), so the entire existing test suite stays green and E0 is a bit-for-bit re-baseline.**

**Tech Stack:** Python 3.12, PyTorch (CPU), NumPy, Matplotlib (Agg backend), pytest. Process-pool parallelism via `concurrent.futures`.

---

## Environment note (read first)

A real Python 3.12 lives at `C:\Users\parke\AppData\Local\Programs\Python\Python312\python.exe`, but the bare `python` command is shadowed by the Microsoft Store stub and **must not be used**. Task 0 creates a venv; **after Task 0, every command in this plan uses `.\.venv\Scripts\python.exe`** (PowerShell) for running tests and scripts. `torch` is **not yet installed** — Task 0 installs it.

## File Structure

| File | Responsibility | New/Modify |
|---|---|---|
| `ttt/encoding.py` | `Encoding` value object; `FLAT` and `ROWCOL` instances; `ENCODINGS` registry | **New** |
| `ttt/board.py` | Add `horizontal_row_completed_by_move`; re-express `move_completes_horizontal` on it | Modify |
| `ttt/dataset.py` | `build_examples(encoding, drop_horizontal_rows, …)` with legacy-boolean shim; `collate_fn(pad_id=…)` | Modify |
| `ttt/model.py` | `GPTConfig.head`; `flat9`/`tied`/`factored` readouts; `factored_logits` helper | Modify |
| `ttt/train.py` | Thread `encoding` so `collate_fn` uses `encoding.pad_id` | Modify |
| `ttt/probes.py` | `rows=` on both probe builders; `probe_prefixes(encoding=…)` | Modify |
| `ttt/evaluate.py` | Thread `encoding` into `evaluate_probes` → `probe_prefixes` | Modify |
| `ttt/sweep.py` | Per-row probe specs; random baselines; `Condition`; `run_condition`; `save_condition`; baseline/ceiling plot overlays | Modify |
| `run_experiment.py` | Conditions E0–E5b; `--condition` CLI; write `results/<name>/` | Modify |
| `tests/test_encoding.py` | Encoding round-trip and constants | **New** |
| `tests/test_*.py` (dataset, model, probes, evaluate, sweep) | New tests for each axis | Modify |

**Design invariant:** the model output is **always interpreted as a distribution over the 9 cells** (even for `factored`/`rowcol`), so `model_move` and the H_win metric are identical across all conditions. `model_move` needs no encoding argument — the prefix it receives is already encoded, and its output is always 9 cells.

---

## Task 0: Environment setup

**Files:** none (creates `.venv/`)

- [ ] **Step 1: Create the venv with the real interpreter**

Run (PowerShell):
```powershell
& "C:\Users\parke\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv
```
Expected: a `.venv\` directory appears; no output on success.

- [ ] **Step 2: Confirm `.gitignore` excludes the venv and results**

Read `.gitignore`. If it does not already contain `.venv/` and `results/`, append them:
```
.venv/
results/
```
(`results_prev_60ep/` is checked in and must stay tracked — only ignore `results/`.)

- [ ] **Step 3: Install dependencies**

Run:
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Expected: torch, numpy, matplotlib, pytest install successfully.

- [ ] **Step 4: Verify the toolchain and the existing suite are green**

Run:
```powershell
.\.venv\Scripts\python.exe -c "import torch, numpy, matplotlib; print('torch', torch.__version__)"
.\.venv\Scripts\python.exe -m pytest -q
```
Expected: torch version prints; **all existing tests pass**. This is the regression baseline every later task must preserve.

- [ ] **Step 5: Commit**

```powershell
git add .gitignore
git commit -m "chore: set up venv-ignore and confirm baseline test suite"
```

---

## Task 1: `board.py` — horizontal row helper

**Files:**
- Modify: `ttt/board.py:69-70`
- Test: `tests/test_board.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_board.py`:
```python
from ttt.board import (
    EMPTY, P1, P2, horizontal_row_completed_by_move, move_completes_horizontal,
)


def test_horizontal_row_completed_returns_row_index():
    # X to move (equal counts). Top row: X at 0,1 -> playing 2 completes row 0.
    board = (P1, P1, EMPTY, P2, P2, EMPTY, EMPTY, EMPTY, EMPTY)
    assert horizontal_row_completed_by_move(board, 2) == 0
    # Bottom row: P1 to move (equal counts), P1 at 6,7 -> playing 8 completes row 2.
    board2 = (P1, P2, EMPTY, P2, P2, EMPTY, P1, P1, EMPTY)
    assert horizontal_row_completed_by_move(board2, 8) == 2


def test_horizontal_row_completed_none_for_non_horizontal():
    # X at 0,3 -> playing 6 completes a vertical (col 0), not horizontal.
    board = (P1, P2, EMPTY, P1, P2, EMPTY, EMPTY, EMPTY, EMPTY)
    assert horizontal_row_completed_by_move(board, 6) is None
    assert move_completes_horizontal(board, 6) is False


def test_move_completes_horizontal_agrees_with_row_helper():
    board = (P1, P1, EMPTY, P2, P2, EMPTY, EMPTY, EMPTY, EMPTY)
    assert move_completes_horizontal(board, 2) is True
    assert horizontal_row_completed_by_move(board, 2) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_board.py -q`
Expected: FAIL with `ImportError: cannot import name 'horizontal_row_completed_by_move'`.

- [ ] **Step 3: Write minimal implementation**

In `ttt/board.py`, replace the final `move_completes_horizontal` function (lines 69-70) with:
```python
def horizontal_row_completed_by_move(board, cell):
    """Row index (0/1/2) of the horizontal line `cell` completes for the current
    player, or None if the move completes no horizontal line. Single source of
    truth for both the dataset row-set filter and per-row probes."""
    line = winning_line_for_move(board, cell)
    if line in HORIZONTAL_LINES:
        return HORIZONTAL_LINES.index(line)
    return None


def move_completes_horizontal(board, cell):
    return horizontal_row_completed_by_move(board, cell) is not None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_board.py tests/test_board_helpers.py -q`
Expected: PASS (new tests plus the unchanged existing board tests).

- [ ] **Step 5: Commit**

```powershell
git add ttt/board.py tests/test_board.py
git commit -m "feat(board): add horizontal_row_completed_by_move helper"
```

---

## Task 2: `ttt/encoding.py` — the Encoding value object

**Files:**
- Create: `ttt/encoding.py`
- Test: `tests/test_encoding.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_encoding.py`:
```python
from ttt.encoding import Encoding, FLAT, ROWCOL, ENCODINGS


def test_flat_constants():
    assert FLAT.name == "flat"
    assert FLAT.vocab_size == 11
    assert FLAT.max_len == 10
    assert FLAT.tokens_per_move == 1
    assert FLAT.bos_id == 9
    assert FLAT.pad_id == 10


def test_rowcol_constants():
    assert ROWCOL.name == "rowcol"
    assert ROWCOL.vocab_size == 8
    assert ROWCOL.max_len == 19
    assert ROWCOL.tokens_per_move == 2
    assert ROWCOL.bos_id == 6
    assert ROWCOL.pad_id == 7


def test_flat_encode_matches_legacy_encode_prefix():
    assert FLAT.encode((0, 4, 8)) == [9, 0, 4, 8]
    assert FLAT.encode(()) == [9]


def test_rowcol_encode_uses_distinct_row_and_col_families():
    # cell 0 -> (row 0, col 0) -> tokens [0, 3]; cell 8 -> (row 2, col 2) -> [2, 5]
    assert ROWCOL.encode((0, 8)) == [6, 0, 3, 2, 5]
    # cell 5 -> (row 1, col 2) -> [1, 5]
    assert ROWCOL.encode((5,)) == [6, 1, 5]


def test_encode_decode_round_trips_for_both_encodings():
    for enc in (FLAT, ROWCOL):
        for path in [(), (4,), (0, 4, 8), (2, 4, 6, 1, 3)]:
            assert enc.decode_path(enc.encode(path)) == path


def test_registry_contains_both():
    assert ENCODINGS["flat"] is FLAT
    assert ENCODINGS["rowcol"] is ROWCOL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_encoding.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ttt.encoding'`.

- [ ] **Step 3: Write minimal implementation**

Create `ttt/encoding.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_encoding.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```powershell
git add ttt/encoding.py tests/test_encoding.py
git commit -m "feat(encoding): add Encoding value object with flat and rowcol"
```

---

## Task 3: `dataset.py` — encoding + row-set filter

**Files:**
- Modify: `ttt/dataset.py:1-72`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_dataset.py`:
```python
from ttt.encoding import FLAT, ROWCOL
from ttt.board import horizontal_row_completed_by_move, EMPTY_BOARD, apply_move


def _board_of(input_ids, encoding):
    board = EMPTY_BOARD
    for cell in encoding.decode_path(input_ids):
        board = apply_move(board, cell)
    return board


def test_drop_all_equals_legacy_filter_true():
    legacy = build_examples(max_orderings=2, filter_horizontal=True)
    new = build_examples(FLAT, drop_horizontal_rows=frozenset({0, 1, 2}),
                         max_orderings=2)
    assert new == legacy


def test_drop_none_equals_legacy_filter_false():
    legacy = build_examples(max_orderings=2, filter_horizontal=False)
    new = build_examples(FLAT, drop_horizontal_rows=frozenset(), max_orderings=2)
    assert new == legacy


def test_keep_top_row_drops_only_rows_1_and_2():
    examples = build_examples(FLAT, drop_horizontal_rows=frozenset({1, 2}),
                              max_orderings=2)
    seen_row0 = False
    for input_ids, target in examples:
        board = _board_of(input_ids, FLAT)
        row = horizontal_row_completed_by_move(board, target)
        assert row not in (1, 2)          # rows 1,2 never appear as targets
        if row == 0:
            seen_row0 = True
    assert seen_row0, "kept top row should still produce row-0 horizontal targets"


def test_rowcol_encoding_emits_two_tokens_per_move():
    examples = build_examples(ROWCOL, drop_horizontal_rows=frozenset({0, 1, 2}),
                              max_orderings=2)
    assert len(examples) > 0
    for input_ids, _ in examples:
        # length-1 (drop BOS) must be even = 2 tokens/move
        assert (len(input_ids) - 1) % 2 == 0
        assert input_ids[0] == ROWCOL.bos_id


def test_collate_uses_given_pad_id():
    from ttt.dataset import collate_fn
    batch = [([6, 0, 3], 4), ([6, 1, 5, 2, 4], 7)]  # rowcol-style, ragged
    ids, lengths, targets = collate_fn(batch, pad_id=ROWCOL.pad_id)
    assert ids.shape == (2, 5)
    assert ids[0, 3].item() == ROWCOL.pad_id  # padded position
    assert lengths.tolist() == [3, 5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_dataset.py -q`
Expected: FAIL — `build_examples()` does not yet accept an encoding / `drop_horizontal_rows`.

- [ ] **Step 3: Write minimal implementation**

Replace the top of `ttt/dataset.py` (lines 1-43) with:
```python
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
```

Then update `collate_fn` (currently lines 58-72) to accept `pad_id`:
```python
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
```

(The `TTTDataset` class between them is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_dataset.py -q`
Expected: PASS (new tests plus the unchanged legacy ones — `filter_horizontal=True/False` still work via the shim).

- [ ] **Step 5: Commit**

```powershell
git add ttt/dataset.py tests/test_dataset.py
git commit -m "feat(dataset): encoding-aware build_examples with row-set filter"
```

---

## Task 4: `model.py` — output-head selector

**Files:**
- Modify: `ttt/model.py:9-83`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_model.py`:
```python
from ttt.model import GPTConfig, TTTGPT, factored_logits
import torch as _torch


def test_factored_logits_maps_row_and_col_to_cells():
    # cell index = row*3 + col; logit[cell] = row_logit[row] + col_logit[col]
    row = _torch.tensor([[10.0, 0.0, 0.0]])   # favors row 0
    col = _torch.tensor([[0.0, 0.0, 5.0]])    # favors col 2
    out = factored_logits(row, col)
    assert out.shape == (1, 9)
    # cell 2 = (row 0, col 2) should be the max: 10 + 5
    assert out.argmax(dim=-1).item() == 2
    assert abs(out[0, 2].item() - 15.0) < 1e-6
    assert abs(out[0, 4].item() - 0.0) < 1e-6  # (row1,col1) = 0 + 0


def test_each_head_outputs_nine_logits():
    ids = _torch.tensor([[9, 0, 4]])
    lengths = _torch.tensor([3])
    for head in ("flat9", "tied", "factored"):
        cfg = GPTConfig(n_layer=1, n_head=1, d_model=16, head=head)
        model = TTTGPT(cfg)
        logits = model(ids, lengths)
        assert logits.shape == (1, 9), head


def test_unknown_head_raises():
    import pytest
    with pytest.raises(ValueError):
        TTTGPT(GPTConfig(n_layer=1, n_head=1, d_model=16, head="bogus"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_model.py -q`
Expected: FAIL — `cannot import name 'factored_logits'` and `GPTConfig` has no `head`.

- [ ] **Step 3: Write minimal implementation**

In `ttt/model.py`, add `head` to the config dataclass (after line 19, `n_out`):
```python
@dataclass
class GPTConfig:
    n_layer: int
    n_head: int
    d_model: int
    vocab_size: int = VOCAB_SIZE
    max_len: int = 10
    n_out: int = N_CELLS
    head: str = "flat9"
```

Add this module-level helper (place it just above `class TTTGPT`):
```python
def factored_logits(row_logit, col_logit):
    """Combine a (B,3) row distribution and a (B,3) col distribution into (B,9)
    cell logits, where cell index = row*3 + col and
    logit[r, c] = row_logit[r] + col_logit[c]."""
    grid = row_logit.unsqueeze(2) + col_logit.unsqueeze(1)  # (B, 3, 3) [b, r, c]
    return grid.reshape(row_logit.shape[0], N_CELLS)
```

Replace the `TTTGPT.__init__` head line (`self.head = nn.Linear(cfg.d_model, cfg.n_out)`, line 72) and add a readout method. The full updated `TTTGPT`:
```python
class TTTGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [Block(cfg.d_model, cfg.n_head) for _ in range(cfg.n_layer)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head_type = cfg.head
        if cfg.head == "flat9":
            self.head = nn.Linear(cfg.d_model, N_CELLS)
        elif cfg.head == "tied":
            # Readout ties to the input cell embeddings (token ids 0..8 under the
            # flat encoding); only a per-cell bias is learned here.
            self.tied_bias = nn.Parameter(torch.zeros(N_CELLS))
        elif cfg.head == "factored":
            self.row_head = nn.Linear(cfg.d_model, 3)
            self.col_head = nn.Linear(cfg.d_model, 3)
        else:
            raise ValueError(f"unknown head: {cfg.head}")

    def _readout(self, h):
        if self.head_type == "flat9":
            return self.head(h)
        if self.head_type == "tied":
            return h @ self.tok_emb.weight[:N_CELLS].t() + self.tied_bias
        return factored_logits(self.row_head(h), self.col_head(h))

    def forward(self, ids, lengths):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        last = (lengths - 1).clamp(min=0)  # gather the final real token
        gathered = x[torch.arange(B, device=ids.device), last]
        return self._readout(gathered)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
```

> Note: `tied` reads out along token ids 0..8. This is only meaningful when those ids are the 9 cells, i.e. the `flat` encoding. The `tied` + `rowcol` combination is rejected at `Condition` construction (Task 8), so the model never sees that pairing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_model.py -q`
Expected: PASS (new head tests plus the unchanged `flat9` shape/overfit tests).

- [ ] **Step 5: Commit**

```powershell
git add ttt/model.py tests/test_model.py
git commit -m "feat(model): add flat9/tied/factored output head selector"
```

---

## Task 5: `train.py` — thread the encoding into collate

**Files:**
- Modify: `ttt/train.py:1-31`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_train.py`:
```python
from ttt.encoding import ROWCOL
from ttt.model import GPTConfig as _GPTConfig


def test_train_runs_with_rowcol_encoding_and_pad_id():
    # rowcol prefixes: BOS=6, ROW/COL tokens; train must pad with ROWCOL.pad_id=7.
    examples = [([6, 0, 3, 1, 4], 8), ([6, 2, 5], 0)]
    cfg = _GPTConfig(n_layer=1, n_head=1, d_model=16,
                     vocab_size=ROWCOL.vocab_size, max_len=ROWCOL.max_len)
    model, history = train_model(
        cfg, examples, epochs=3, lr=1e-2, batch_size=2, seed=0, encoding=ROWCOL,
    )
    assert len(history) == 3  # ran without an index/pad error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_train.py -q`
Expected: FAIL — `train_model()` got an unexpected keyword argument `encoding`.

- [ ] **Step 3: Write minimal implementation**

Replace `ttt/train.py` lines 1-31 (imports + `train_model` signature + loader) with:
```python
from __future__ import annotations

import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from ttt.model import TTTGPT, GPTConfig
from ttt.dataset import TTTDataset, collate_fn
from ttt.encoding import FLAT


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(cfg: GPTConfig, examples, *, epochs, lr, batch_size,
                seed, encoding=FLAT, device="cpu"):
    """Train a model by next-move cross-entropy. Returns (model, loss_history)."""
    _set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        TTTDataset(examples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_id=encoding.pad_id),
        generator=g,
    )
```

(The training loop below line 31 is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_train.py -q`
Expected: PASS — new rowcol test plus the unchanged flat tests (default `encoding=FLAT` ⇒ `pad_id=10`, identical to before).

- [ ] **Step 5: Commit**

```powershell
git add ttt/train.py tests/test_train.py
git commit -m "feat(train): thread encoding so collate uses the right pad id"
```

---

## Task 6: `probes.py` — per-row probes + encoding-aware prefixes

**Files:**
- Modify: `ttt/probes.py:1-93`
- Test: `tests/test_probes.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_probes.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_probes.py -q`
Expected: FAIL — `win_available_probes()` got an unexpected keyword argument `rows`.

- [ ] **Step 3: Write minimal implementation**

In `ttt/probes.py`, change the import line 8 from:
```python
from ttt.dataset import encode_prefix
```
to:
```python
from ttt.encoding import FLAT
```

Replace `win_available_probes` (lines 34-59) with a `rows`-aware version:
```python
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
```

Replace `block_needed_probes` (lines 62-88) with a `rows`-aware version:
```python
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
```

Replace `probe_prefixes` (lines 91-93) with:
```python
def probe_prefixes(board, paths, encoding=FLAT, max_orderings=4):
    """Encoded prefixes ([BOS, ...]) for the orderings that reach `board`."""
    return [encoding.encode(p) for p in paths.get(board, [])[:max_orderings]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_probes.py -q`
Expected: PASS — new `rows`/encoding tests plus the unchanged probe-purity tests (which call the builders with no `rows`, so behavior is identical).

- [ ] **Step 5: Commit**

```powershell
git add ttt/probes.py tests/test_probes.py
git commit -m "feat(probes): per-row probe filtering and encoding-aware prefixes"
```

---

## Task 7: `evaluate.py` — thread the encoding through

**Files:**
- Modify: `ttt/evaluate.py:21-30`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_evaluate.py`:
```python
from ttt.encoding import ROWCOL


def test_evaluate_probes_accepts_encoding_kwarg():
    probes = win_available_probes("vertical")[:5]
    paths = reachable_paths(max_orderings=4)
    # _AlwaysCellModel ignores its inputs, so the rate is well-defined regardless
    # of encoding; the point is that threading ROWCOL through does not error.
    res = evaluate_probes(_AlwaysCellModel(4), probes, paths,
                          encoding=ROWCOL, max_orderings=4)
    assert 0.0 <= res["rate"] <= 1.0
    assert res["n_probes"] >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluate.py -q`
Expected: FAIL — `evaluate_probes()` got an unexpected keyword argument `encoding`.

- [ ] **Step 3: Write minimal implementation**

In `ttt/evaluate.py`, add the import (after line 6 `from ttt.probes import probe_prefixes`):
```python
from ttt.encoding import FLAT
```

Change the `evaluate_probes` signature (line 21) and its `probe_prefixes` call (line 30):
```python
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
```

(The rest of the loop and `random_baseline_rate` are unchanged. `model_move` needs no change — its input is already an encoded prefix and its output is always 9 cells.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluate.py -q`
Expected: PASS — new encoding test plus the unchanged flat tests.

- [ ] **Step 5: Commit**

```powershell
git add ttt/evaluate.py tests/test_evaluate.py
git commit -m "feat(evaluate): thread encoding into probe evaluation"
```

---

## Task 8: `sweep.py` — per-row metrics, random baselines, Condition, run_condition

**Files:**
- Modify: `ttt/sweep.py` (whole file)
- Test: `tests/test_sweep.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sweep.py`:
```python
import json
import pytest
from ttt.encoding import FLAT, ROWCOL
from ttt.sweep import (
    Condition, run_condition, save_condition, compute_random_baselines,
    STANDARD_GRID, DEEP_GRID,
)


def test_standard_grid_excludes_indivisible_configs():
    for (L, H, D) in STANDARD_GRID:
        assert D % H == 0
    assert (1, 1, 16) in STANDARD_GRID
    # n_head=4 with d_model=16 is divisible (16%4==0); (H=4,D=32) etc included
    assert all(D % H == 0 for (L, H, D) in DEEP_GRID)
    assert any(L == 8 for (L, H, D) in DEEP_GRID)  # depth grid adds L8


def test_condition_rejects_tied_with_rowcol():
    with pytest.raises(ValueError):
        Condition("bad", encoding=ROWCOL, head="tied")


def test_condition_rejects_unknown_head():
    with pytest.raises(ValueError):
        Condition("bad", head="nonsense")


def test_compute_random_baselines_has_horizontal_win_in_range():
    base = compute_random_baselines()
    assert 0.0 < base["horizontal_win"] < 1.0
    assert "horizontal_win_row0" in base


def test_run_condition_smoke_produces_per_row_metrics():
    cond = Condition(
        "smoke", encoding=FLAT, head="flat9",
        drop_horizontal_rows=frozenset({0, 1, 2}),
        grid=((1, 1, 16),), seeds=(0,),
        epochs=2, lr=1e-2, batch_size=256, max_orderings=2,
    )
    raw = run_condition(cond, n_workers=1)
    assert len(raw) == 1
    metrics = raw[0]["metrics"]
    for key in ("horizontal_win", "horizontal_win_row0",
                "horizontal_win_row1", "horizontal_win_row2", "vertical_win"):
        assert key in metrics


def test_save_condition_writes_artifacts(tmp_path):
    cond = Condition(
        "smoke", grid=((1, 1, 16),), seeds=(0,),
        epochs=2, lr=1e-2, batch_size=256, max_orderings=2,
    )
    raw = run_condition(cond, n_workers=1)
    save_condition(cond, raw, str(tmp_path))
    for fname in ("raw.json", "agg.json", "baselines.json", "capacity.png"):
        assert (tmp_path / fname).exists()
    base = json.loads((tmp_path / "baselines.json").read_text())
    assert "horizontal_win" in base
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_sweep.py -q`
Expected: FAIL — cannot import `Condition`, `run_condition`, `save_condition`, `compute_random_baselines`, `STANDARD_GRID`, `DEEP_GRID`.

- [ ] **Step 3: Write minimal implementation**

Edit `ttt/sweep.py`. First, update the imports/constants block (lines 1-27) to add the new imports, the per-row `PROBE_SPECS` (now 3-tuples carrying `rows`), and the grids:
```python
from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from ttt.model import GPTConfig
from ttt.train import train_model
from ttt.dataset import build_examples
from ttt.enumerate import reachable_paths
from ttt.probes import win_available_probes, block_needed_probes
from ttt.evaluate import evaluate_probes, random_baseline_rate
from ttt.encoding import FLAT

# Metrics evaluated per trained model. (kind, line_type, rows-or-None)
PROBE_SPECS = {
    "horizontal_win": ("win", "horizontal", None),
    "horizontal_block": ("block", "horizontal", None),
    "vertical_win": ("win", "vertical", None),
    "vertical_block": ("block", "vertical", None),
    "diagonal_win": ("win", "diagonal", None),
    "diagonal_block": ("block", "diagonal", None),
    # Per-row horizontal wins drive the translational (E3) seen-vs-held-out split.
    "horizontal_win_row0": ("win", "horizontal", frozenset({0})),
    "horizontal_win_row1": ("win", "horizontal", frozenset({1})),
    "horizontal_win_row2": ("win", "horizontal", frozenset({2})),
}

STANDARD_GRID = tuple(
    (L, H, D)
    for L in (1, 2, 4) for H in (1, 2, 4) for D in (16, 32, 64)
    if D % H == 0
)
# Depth experiment (E4) appends L8; same width/head sub-grid.
DEEP_GRID = STANDARD_GRID + tuple(
    (8, H, D) for H in (1, 2, 4) for D in (16, 32, 64) if D % H == 0
)
```

Update `_build_probe_sets` (lines 34-41) to unpack the 3-tuple and pass `rows`:
```python
def _build_probe_sets():
    sets = {}
    for name, (kind, line_type, rows) in PROBE_SPECS.items():
        if kind == "win":
            sets[name] = win_available_probes(line_type, rows=rows)
        else:
            sets[name] = block_needed_probes(line_type, rows=rows)
    return sets
```

Update `_train_and_eval` (lines 44-65) to thread `encoding` and `head` (defaults preserve the legacy `run_sweep`):
```python
def _train_and_eval(config, seed, examples, paths, probe_sets, *,
                    epochs, lr, batch_size, max_orderings,
                    encoding=FLAT, head="flat9"):
    """Train one (config, seed) and evaluate all probe metrics. Returns a raw row.

    Shared by run_sweep, run_sweep_parallel, and run_condition so all three
    produce identically shaped rows.
    """
    n_layer, n_head, d_model = config
    cfg = GPTConfig(
        n_layer=n_layer, n_head=n_head, d_model=d_model,
        vocab_size=encoding.vocab_size, max_len=encoding.max_len, head=head,
    )
    model, _ = train_model(
        cfg, examples, epochs=epochs, lr=lr, batch_size=batch_size,
        seed=seed, encoding=encoding,
    )
    metrics = {
        name: evaluate_probes(model, probes, paths, encoding=encoding,
                              max_orderings=max_orderings)["rate"]
        for name, probes in probe_sets.items()
    }
    return {
        "config": config_name(n_layer, n_head, d_model),
        "n_params": model.num_params(),
        "seed": seed,
        "metrics": metrics,
    }
```

> `run_sweep`, `run_sweep_parallel`, `aggregate`, and `save_results` stay as they are — they call `_train_and_eval` with the new defaults (`FLAT`, `flat9`), which is bit-identical to before. `aggregate` already iterates `rows[0]["metrics"].keys()`, so it picks up the new per-row keys automatically.

Now append the `Condition` machinery at the end of the file (after `save_results`):
```python
# --- Conditions: one composable experiment per dataclass instance -----------

@dataclass(frozen=True)
class Condition:
    """A single experiment = a setting of the four orthogonal axes plus run
    hyperparameters. Combining axes needs no new code, only a new Condition."""
    name: str
    encoding: object = FLAT
    head: str = "flat9"
    drop_horizontal_rows: frozenset = frozenset({0, 1, 2})
    grid: tuple = STANDARD_GRID
    seeds: tuple = (0, 1, 2, 3, 4)
    epochs: int = 150
    lr: float = 1e-3
    batch_size: int = 256
    max_orderings: int = 4

    def __post_init__(self):
        if self.head not in ("flat9", "tied", "factored"):
            raise ValueError(f"unknown head: {self.head}")
        if self.head == "tied" and self.encoding.name != "flat":
            raise ValueError("tied head requires the flat encoding (1:1 cell<->token)")


_COND_WORKER = {}


def _cond_worker_init(encoding, drop_horizontal_rows, max_orderings):
    import torch
    torch.set_num_threads(1)
    _COND_WORKER["examples"] = build_examples(
        encoding, drop_horizontal_rows=drop_horizontal_rows,
        max_orderings=max_orderings,
    )
    _COND_WORKER["paths"] = reachable_paths(max_orderings=max_orderings)
    _COND_WORKER["probe_sets"] = _build_probe_sets()
    _COND_WORKER["encoding"] = encoding
    _COND_WORKER["max_orderings"] = max_orderings


def _cond_worker_task(args):
    config, seed, head, epochs, lr, batch_size = args
    return _train_and_eval(
        config, seed,
        _COND_WORKER["examples"], _COND_WORKER["paths"], _COND_WORKER["probe_sets"],
        epochs=epochs, lr=lr, batch_size=batch_size,
        max_orderings=_COND_WORKER["max_orderings"],
        encoding=_COND_WORKER["encoding"], head=head,
    )


def run_condition(cond: Condition, *, n_workers=None, progress=False):
    """Train the condition's (config, seed) grid across a process pool and
    evaluate every probe metric. Returns raw rows (same shape as run_sweep)."""
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    tasks = [
        (config, seed, cond.head, cond.epochs, cond.lr, cond.batch_size)
        for config in cond.grid
        for seed in cond.seeds
    ]
    raw = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_cond_worker_init,
        initargs=(cond.encoding, cond.drop_horizontal_rows, cond.max_orderings),
    ) as pool:
        futures = [pool.submit(_cond_worker_task, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            raw.append(fut.result())
            if progress:
                print(f"  [{cond.name} {i}/{len(tasks)}] done", flush=True)
    return raw


def compute_random_baselines():
    """Uniform-random-legal-move hit-rate per probe set (the 'chance' line)."""
    probe_sets = _build_probe_sets()
    return {name: random_baseline_rate(probes)
            for name, probes in probe_sets.items()}


def save_condition(cond: Condition, raw, out_dir, ceiling=None):
    """Aggregate, persist, and plot one condition into out_dir/.

    `ceiling`, if given, is a {n_params: horizontal_win_mean} map (e.g. from the
    positive-control E1) drawn as an upper reference on the plot.
    """
    os.makedirs(out_dir, exist_ok=True)
    agg = aggregate(raw)
    baselines = compute_random_baselines()
    save_results(raw, agg,
                 os.path.join(out_dir, "raw.json"),
                 os.path.join(out_dir, "agg.json"))
    with open(os.path.join(out_dir, "baselines.json"), "w") as f:
        json.dump(baselines, f, indent=2)
    plot_capacity(agg, os.path.join(out_dir, "capacity.png"),
                  baselines=baselines, ceiling=ceiling)
    return agg, baselines


def horizontal_win_ceiling(agg):
    """{n_params: horizontal_win mean} from a (positive-control) condition's agg,
    suitable to pass as `save_condition(..., ceiling=...)`."""
    return {entry["n_params"]: entry["horizontal_win"]["mean"]
            for entry in agg.values() if "horizontal_win" in entry}
```

Finally, extend `plot_capacity` (lines 160-197) to accept optional `baselines` and `ceiling` overlays. Replace its signature and add the overlay lines just before `ax.set_xlabel`:
```python
def plot_capacity(agg, out_path, baselines=None, ceiling=None):
    """Horizontal win/block vs. capacity, with vertical/diagonal controls,
    an optional random-baseline 'chance' line, and an optional positive-control
    ceiling overlay."""
    from collections import defaultdict

    groups = defaultdict(list)
    for cfg, entry in agg.items():
        groups[entry["n_params"]].append(entry)
    xs = sorted(groups)
    any_entry = next(iter(agg.values()))

    fig, ax = plt.subplots(figsize=(7, 5))
    for metric, label in [
        ("horizontal_win", "horizontal win (held-out)"),
        ("horizontal_block", "horizontal block"),
        ("vertical_win", "vertical win (control)"),
        ("diagonal_win", "diagonal win (control)"),
    ]:
        if metric not in any_entry:
            continue
        means, stds = [], []
        for x in xs:
            vals = [e[metric]["mean"] for e in groups[x]]
            means.append(statistics.fmean(vals))
            stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=label)
    if baselines and "horizontal_win" in baselines:
        ax.axhline(baselines["horizontal_win"], ls="--", color="gray",
                   label="random baseline (H-win)")
    if ceiling:
        cxs = sorted(ceiling)
        ax.plot(cxs, [ceiling[x] for x in cxs], ls=":", color="green",
                marker="^", label="positive-control ceiling (H-win)")
    ax.set_xlabel("model parameters")
    ax.set_ylabel("probe success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Horizontal-win generalization vs. model capacity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_sweep.py -q`
Expected: PASS — new Condition/run_condition/baseline tests plus the unchanged `aggregate`/`plot_capacity`/parallel-vs-sequential tests (the extra metric keys flow through but the seq/par comparison still matches because both paths produce them identically).

- [ ] **Step 5: Run the FULL suite to confirm no regression**

Run: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: PASS — entire suite green.

- [ ] **Step 6: Commit**

```powershell
git add ttt/sweep.py tests/test_sweep.py
git commit -m "feat(sweep): Condition runner, per-row metrics, random baselines"
```

---

## Task 9: `run_experiment.py` — define E0–E5b and the CLI

**Files:**
- Modify: `run_experiment.py` (whole file)
- Test: `tests/test_run_experiment.py` (**new**)

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_experiment.py`:
```python
import pytest
from run_experiment import CONDITIONS, PHASE1


def test_phase1_has_all_seven_conditions():
    assert PHASE1 == ["E0", "E1", "E2", "E3", "E4", "E5a", "E5b"]
    for name in PHASE1:
        assert name in CONDITIONS
        assert CONDITIONS[name].name == name


def test_condition_axes_match_the_design_matrix():
    assert CONDITIONS["E0"].drop_horizontal_rows == frozenset({0, 1, 2})
    assert CONDITIONS["E1"].drop_horizontal_rows == frozenset()       # positive control
    assert CONDITIONS["E2"].encoding.name == "rowcol"                  # encoding axis
    assert CONDITIONS["E3"].drop_horizontal_rows == frozenset({1, 2})  # translational
    assert any(L == 8 for (L, H, D) in CONDITIONS["E4"].grid)          # depth
    assert CONDITIONS["E5a"].head == "tied"
    assert CONDITIONS["E5b"].head == "factored"


def test_all_phase1_conditions_use_five_seeds():
    for name in PHASE1:
        assert len(CONDITIONS[name].seeds) == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_run_experiment.py -q`
Expected: FAIL — `cannot import name 'CONDITIONS' from 'run_experiment'`.

- [ ] **Step 3: Write minimal implementation**

Replace the whole of `run_experiment.py` with:
```python
"""Run symmetry-generalization conditions and write results to ./results/<name>/.

Each experiment is a composable `Condition` (four orthogonal axes). Run one with
`--condition E2`, or all Phase-1 conditions with `--condition all` (default).
Override worker count with `--workers` or the TTT_WORKERS env var.
"""
from __future__ import annotations

import argparse
import os
import time

from ttt.encoding import FLAT, ROWCOL
from ttt.sweep import (
    Condition, run_condition, save_condition, horizontal_win_ceiling,
    STANDARD_GRID, DEEP_GRID,
)

# Phase-1 matrix: each varies ONE axis off the baseline
# (flat encoding, flat9 head, drop-all filter, standard grid).
CONDITIONS = {
    "E0": Condition("E0", encoding=FLAT, head="flat9",
                    drop_horizontal_rows=frozenset({0, 1, 2}), grid=STANDARD_GRID),
    "E1": Condition("E1", encoding=FLAT, head="flat9",
                    drop_horizontal_rows=frozenset(), grid=STANDARD_GRID),
    "E2": Condition("E2", encoding=ROWCOL, head="flat9",
                    drop_horizontal_rows=frozenset({0, 1, 2}), grid=STANDARD_GRID),
    "E3": Condition("E3", encoding=FLAT, head="flat9",
                    drop_horizontal_rows=frozenset({1, 2}), grid=STANDARD_GRID),
    "E4": Condition("E4", encoding=FLAT, head="flat9",
                    drop_horizontal_rows=frozenset({0, 1, 2}), grid=DEEP_GRID),
    "E5a": Condition("E5a", encoding=FLAT, head="tied",
                     drop_horizontal_rows=frozenset({0, 1, 2}), grid=STANDARD_GRID),
    "E5b": Condition("E5b", encoding=FLAT, head="factored",
                     drop_horizontal_rows=frozenset({0, 1, 2}), grid=STANDARD_GRID),
}
PHASE1 = ["E0", "E1", "E2", "E3", "E4", "E5a", "E5b"]


def run_one(name, n_workers, ceiling=None):
    cond = CONDITIONS[name]
    out_dir = os.path.join("results", name)
    n_jobs = len(cond.grid) * len(cond.seeds)
    print(f"=== {name}: {n_jobs} jobs "
          f"({len(cond.grid)} configs x {len(cond.seeds)} seeds), "
          f"encoding={cond.encoding.name}, head={cond.head}, "
          f"drop={sorted(cond.drop_horizontal_rows)} ===")
    t0 = time.time()
    raw = run_condition(cond, n_workers=n_workers, progress=True)
    agg, _ = save_condition(cond, raw, out_dir, ceiling=ceiling)
    print(f"  {name} finished in {(time.time() - t0) / 60:.1f} min -> {out_dir}/")
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="all",
                        choices=["all"] + PHASE1)
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("TTT_WORKERS",
                                                   os.cpu_count() or 1)))
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    names = PHASE1 if args.condition == "all" else [args.condition]

    # When running the full set, compute the positive-control (E1) ceiling first
    # and overlay it on the other flat/standard-grid conditions' plots.
    ceiling = None
    aggs = {}
    if "E1" in names:
        aggs["E1"] = run_one("E1", args.workers)
        ceiling = horizontal_win_ceiling(aggs["E1"])
        names = [n for n in names if n != "E1"]

    for name in names:
        # Ceiling overlay only makes sense at matched params (flat, standard grid).
        use_ceiling = ceiling if CONDITIONS[name].encoding.name == "flat" \
            and CONDITIONS[name].grid == STANDARD_GRID else None
        aggs[name] = run_one(name, args.workers, ceiling=use_ceiling)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_run_experiment.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```powershell
git add run_experiment.py tests/test_run_experiment.py
git commit -m "feat(run): define conditions E0-E5b with a --condition CLI"
```

---

## Task 10: First-run diagnostic — the H_win floor vs. random baseline

This task produces the spec's first deliverable: confirm whether the held-out
H_win floor sits **below** the random-legal-move baseline (active softmax
suppression) or merely **at/around** it (failure-to-learn). It runs reduced-epoch
E0 + E1 as an end-to-end pipeline check before the full multi-hour sweep.

**Files:** none (produces `results/E0_diag/`, `results/E1_diag/`)

- [ ] **Step 1: Run a fast E0 (drop-all) + E1 (positive control) diagnostic**

Run (PowerShell — small grid, 3 seeds, 20 epochs ≈ a few minutes):
```powershell
.\.venv\Scripts\python.exe -c @'
from ttt.sweep import Condition, run_condition, save_condition, compute_random_baselines, STANDARD_GRID
from ttt.encoding import FLAT
small = ((1,1,16),(2,2,32),(4,4,64))
for nm, drop in [("E0_diag", frozenset({0,1,2})), ("E1_diag", frozenset())]:
    cond = Condition(nm, encoding=FLAT, head="flat9", drop_horizontal_rows=drop,
                     grid=small, seeds=(0,1,2), epochs=20, lr=1e-3, batch_size=256)
    raw = run_condition(cond, progress=True)
    agg, base = save_condition(cond, raw, f"results/{nm}")
    hw = [agg[c]["horizontal_win"]["mean"] for c in agg]
    print(nm, "H_win means:", [round(x,3) for x in hw],
          "| random baseline:", round(base["horizontal_win"],3))
'@
```
Expected: prints per-config H_win means for both conditions and the random
baseline. **Interpretation:** E0 H_win near/below the baseline confirms the
defense-only floor; E1 H_win rising toward the V/D ceiling confirms the filter
*caused* the gap. Both `results/E0_diag/` and `results/E1_diag/` contain
`raw.json`, `agg.json`, `baselines.json`, `capacity.png`.

- [ ] **Step 2: Verify the artifacts and baseline were written**

Run:
```powershell
.\.venv\Scripts\python.exe -c "import json; b=json.load(open('results/E0_diag/baselines.json')); print('baseline keys:', sorted(b)); print('H-win baseline:', b['horizontal_win'])"
```
Expected: lists all probe-set keys (incl. `horizontal_win_row0/1/2`) and a
baseline in (0, 1).

- [ ] **Step 3: Record the diagnostic finding**

Append a short note to `docs/design/specs/2026-06-02-composable-symmetry-experiments-design.md` under a new `## Diagnostic result (E0/E1 fast run)` heading: the E0 H_win means vs. baseline, the E1 means, and whether the floor is below baseline (suppression) or at it (failure-to-learn). Quote the printed numbers.

- [ ] **Step 4: Commit**

```powershell
git add docs/design/specs/2026-06-02-composable-symmetry-experiments-design.md
git commit -m "docs: record H_win-floor-vs-baseline diagnostic from fast E0/E1 run"
```

> **Full sweep (run when ready, not part of the green-tests loop):**
> `.\.venv\Scripts\python.exe run_experiment.py --condition all`
> This trains all Phase-1 conditions at 150 epochs × full grid × 5 seeds and
> writes `results/E0/ … results/E5b/`. E4 (+L8) and E2 (rowcol, ~2× sequence
> length) are the slowest; expect E4/D64 ≈ 2× the E0 wall-clock. Phase-2
> combinations (marquee = factored head × rowcol input) are added afterward as
> new `Condition` entries in `CONDITIONS` — no new code.

---

## Self-Review

**Spec coverage:**
- Axis 1 Encoding (`flat`/`rowcol`) → Task 2 (`encoding.py`), threaded in Tasks 3,5,6,7,8. ✓
- Axis 2 Output head (`flat9`/`tied`/`factored`) → Task 4; `tied`+`rowcol` rejected in Task 8 `Condition.__post_init__`. ✓
- Axis 3 Horizontal filter as row-set → Task 1 (`horizontal_row_completed_by_move`) + Task 3 (`build_examples`). ✓
- Axis 4 Grid/depth (+L8) → `DEEP_GRID` in Task 8, used by E4 in Task 9. ✓
- Per-row probes + seen/held-out split → Task 6 (`rows=`) + Task 8 (`horizontal_win_row0/1/2`). ✓
- Shared runner, 5 seeds, V/D controls, random baseline, positive-control ceiling, persisted results → Task 8. ✓
- Diagnostic (H_win floor vs. random baseline) as first deliverable → Task 10. ✓
- Experiment matrix E0–E5b + `--condition` selector → Task 9. ✓
- Regression: E0 == legacy `filter_horizontal=True` → Task 3 `test_drop_all_equals_legacy_filter_true`. ✓
- Edge cases: `tied`+`rowcol` rejected (Task 8); `max_len=19` for rowcol (Task 2); empty per-row probe sets → `evaluate_probes`/`random_baseline_rate` already return 0.0 (existing code). ✓
- Prerequisite (no real Python on PATH) → Task 0. ✓
- Decisions: seeds=5 (Condition default & Task 9 test); L16 deferred (only L8 in `DEEP_GRID`). ✓

**Type consistency:** `Encoding` fields (`name`, `vocab_size`, `max_len`, `tokens_per_move`, `bos_id`, `pad_id`, `encode`, `decode_path`) are used consistently across Tasks 2–9. `Condition` fields match between Task 8 definition and Task 9 construction. `factored_logits(row_logit, col_logit)` is defined in Task 4 and called from `_readout`. `horizontal_row_completed_by_move` (Task 1) is consumed by `build_examples` (Task 3) and the probe tests (Task 6). `PROBE_SPECS` 3-tuple shape (Task 8) matches `_build_probe_sets` unpacking. No placeholders.

---

## Execution Handoff

**Plan complete and saved to `docs/design/plans/2026-06-02-composable-symmetry-experiments.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
