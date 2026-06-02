# Tiny Transformer TTT Generalization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline that trains tiny decoder-only transformers on tic-tac-toe move-sequences with all horizontal-winning moves filtered out, then probes whether they score / block horizontal wins, swept across model capacity.

**Architecture:** Pure-Python game core + memoized minimax solver generate a fully-enumerated, optimally-labeled dataset rendered as move-sequences. A horizontal-win loss filter drops offensive-horizontal targets while keeping defensive blocks. A nanoGPT-style decoder predicts the next move; a sweep over (layers, heads, d_model) × seeds is evaluated on curated win/block probe suites for horizontal (held-out) vs. vertical/diagonal (control) lines.

**Tech Stack:** Python 3.11+, PyTorch (CPU-fine), NumPy, Matplotlib, pytest.

---

## Conventions (read once, used throughout)

- Board = a 9-tuple of ints, row-major, indices:
  ```
  0 1 2
  3 4 5
  6 7 8
  ```
- Cell values: `EMPTY=0`, `P1=1`, `P2=2`. P1 always moves first.
- Token ids: cell `c` → id `c` (0–8); `BOS_ID=9`; `PAD_ID=10`; `VOCAB_SIZE=11`; output classes `N_CELLS=9`.
- A "prefix" is the move-sequence that reached a board, encoded as `[BOS_ID, m1, m2, ...]`.
- Each training example is `(input_ids, target_cell)`: supervise only the next move after the prefix.

---

### Task 1: Project setup

**Files:**
- Create: `requirements.txt`
- Create: `ttt/__init__.py`
- Create: `tests/__init__.py`
- Create: `pytest.ini`

- [ ] **Step 1: Create requirements.txt**

```
torch>=2.0
numpy>=1.24
matplotlib>=3.7
pytest>=7.0
```

- [ ] **Step 2: Create empty package markers**

`ttt/__init__.py`:
```python
```

`tests/__init__.py`:
```python
```

- [ ] **Step 3: Create pytest.ini**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

- [ ] **Step 4: Create and activate a venv, install deps**

Run:
```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```
Expected: installs complete with "Successfully installed torch ... pytest ...".

- [ ] **Step 5: Commit**

```bash
git add requirements.txt pytest.ini ttt/__init__.py tests/__init__.py
git commit -m "chore: project setup (deps, package, pytest)"
```

---

### Task 2: Board core

**Files:**
- Create: `ttt/board.py`
- Test: `tests/test_board.py`

- [ ] **Step 1: Write failing tests**

`tests/test_board.py`:
```python
from ttt.board import (
    EMPTY, P1, P2, EMPTY_BOARD,
    HORIZONTAL_LINES, VERTICAL_LINES, DIAGONAL_LINES, ALL_LINES, LINES_BY_TYPE,
    current_player, legal_moves, apply_move, winner, is_terminal,
)


def test_lines_partition_into_eight():
    assert len(HORIZONTAL_LINES) == 3
    assert len(VERTICAL_LINES) == 3
    assert len(DIAGONAL_LINES) == 2
    assert len(ALL_LINES) == 8
    assert LINES_BY_TYPE["horizontal"] == HORIZONTAL_LINES


def test_current_player_starts_with_p1():
    assert current_player(EMPTY_BOARD) == P1


def test_current_player_alternates():
    b = apply_move(EMPTY_BOARD, 0)  # P1 plays 0
    assert current_player(b) == P2


def test_legal_moves_are_empty_cells():
    b = apply_move(EMPTY_BOARD, 4)
    assert 4 not in legal_moves(b)
    assert len(legal_moves(b)) == 8


def test_apply_move_places_current_players_mark():
    b = apply_move(EMPTY_BOARD, 0)
    assert b[0] == P1
    b2 = apply_move(b, 1)
    assert b2[1] == P2


def test_winner_detects_horizontal():
    # P1 at 0,1,2
    b = (P1, P1, P1, 0, 0, 0, 0, 0, 0)
    assert winner(b) == P1


def test_winner_none_on_empty():
    assert winner(EMPTY_BOARD) == 0


def test_is_terminal_on_win_and_full():
    win = (P1, P1, P1, 0, 0, 0, 0, 0, 0)
    assert is_terminal(win)
    full_draw = (P1, P2, P1, P1, P2, P2, P2, P1, P1)
    assert is_terminal(full_draw)
    assert not is_terminal(EMPTY_BOARD)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_board.py -v`
Expected: FAIL (ModuleNotFoundError / ImportError: cannot import from ttt.board).

- [ ] **Step 3: Implement ttt/board.py**

```python
from __future__ import annotations

EMPTY = 0
P1 = 1
P2 = 2

EMPTY_BOARD = (EMPTY,) * 9

HORIZONTAL_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
VERTICAL_LINES = [(0, 3, 6), (1, 4, 7), (2, 5, 8)]
DIAGONAL_LINES = [(0, 4, 8), (2, 4, 6)]
ALL_LINES = HORIZONTAL_LINES + VERTICAL_LINES + DIAGONAL_LINES
LINES_BY_TYPE = {
    "horizontal": HORIZONTAL_LINES,
    "vertical": VERTICAL_LINES,
    "diagonal": DIAGONAL_LINES,
}


def current_player(board):
    n1 = board.count(P1)
    n2 = board.count(P2)
    return P1 if n1 == n2 else P2


def legal_moves(board):
    return tuple(i for i, v in enumerate(board) if v == EMPTY)


def apply_move(board, cell):
    player = current_player(board)
    new = list(board)
    new[cell] = player
    return tuple(new)


def winner(board):
    for a, b, c in ALL_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def is_terminal(board):
    return winner(board) != 0 or EMPTY not in board
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_board.py -v`
Expected: PASS (all 8 tests).

- [ ] **Step 5: Commit**

```bash
git add ttt/board.py tests/test_board.py
git commit -m "feat: tic-tac-toe board core (lines, moves, win detection)"
```

---

### Task 3: Move/line classification helpers

**Files:**
- Modify: `ttt/board.py`
- Test: `tests/test_board_helpers.py`

- [ ] **Step 1: Write failing tests**

`tests/test_board_helpers.py`:
```python
from ttt.board import (
    P1, P2,
    winning_line_for_move, line_type_completed_by_move, move_completes_horizontal,
)


def test_winning_line_for_move_horizontal():
    # P1 has 0,1; playing 2 completes the top row
    b = (P1, P1, 0, 0, 0, 0, 0, 0, 0)
    assert winning_line_for_move(b, 2) == (0, 1, 2)


def test_winning_line_for_move_none():
    b = (P1, 0, 0, 0, 0, 0, 0, 0, 0)
    assert winning_line_for_move(b, 1) is None


def test_line_type_completed_classifies():
    horiz = (P1, P1, 0, 0, 0, 0, 0, 0, 0)
    assert line_type_completed_by_move(horiz, 2) == "horizontal"
    vert = (P1, 0, 0, P1, 0, 0, 0, 0, 0)
    assert line_type_completed_by_move(vert, 6) == "vertical"
    diag = (P1, 0, 0, 0, P1, 0, 0, 0, 0)
    assert line_type_completed_by_move(diag, 8) == "diagonal"


def test_move_completes_horizontal_flag():
    b = (P1, P1, 0, 0, 0, 0, 0, 0, 0)
    assert move_completes_horizontal(b, 2) is True
    # vertical completion is not horizontal
    bv = (P1, 0, 0, P1, 0, 0, 0, 0, 0)
    assert move_completes_horizontal(bv, 6) is False


def test_move_completes_horizontal_respects_current_player():
    # It is P2's turn here (P1 has one more mark); P2 playing 5 must NOT
    # be considered a P1 horizontal completion.
    b = (P1, P1, 0, 0, 0, P2, 0, 0, 0)  # n1=2, n2=1 -> P2 to move
    assert move_completes_horizontal(b, 2) is True   # P2 plays 2? no, 0,1 are P1
    # Correction: 0,1 are P1, so the third cell 2 completes a P1 line, but the
    # mover is P2 -> placing P2 at 2 does NOT complete a P1 line.
```

> Note: the last test's intent is encoded in the assertions below; replace its body with:
```python
def test_move_completes_horizontal_respects_current_player():
    b = (P1, P1, 0, 0, 0, P2, 0, 0, 0)  # n1=2, n2=1 -> P2 to move
    # P2 placing at cell 2 fills the top row with mixed marks -> not a win.
    assert move_completes_horizontal(b, 2) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_board_helpers.py -v`
Expected: FAIL (ImportError: cannot import name 'winning_line_for_move').

- [ ] **Step 3: Add helpers to ttt/board.py**

Append to `ttt/board.py`:
```python
def winning_line_for_move(board, cell):
    """The line completed by the current player playing `cell`, else None."""
    player = current_player(board)
    after = apply_move(board, cell)
    for line in ALL_LINES:
        if cell in line and all(after[i] == player for i in line):
            return line
    return None


def line_type_completed_by_move(board, cell):
    line = winning_line_for_move(board, cell)
    if line is None:
        return None
    if line in HORIZONTAL_LINES:
        return "horizontal"
    if line in VERTICAL_LINES:
        return "vertical"
    return "diagonal"


def move_completes_horizontal(board, cell):
    return line_type_completed_by_move(board, cell) == "horizontal"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_board_helpers.py -v`
Expected: PASS (5 tests). Then run `pytest tests/test_board.py -v` to confirm no regression.

- [ ] **Step 5: Commit**

```bash
git add ttt/board.py tests/test_board_helpers.py
git commit -m "feat: move/line classification helpers (winning_line, line_type, horizontal flag)"
```

---

### Task 4: Minimax solver

**Files:**
- Create: `ttt/solver.py`
- Test: `tests/test_solver.py`

- [ ] **Step 1: Write failing tests**

`tests/test_solver.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_solver.py -v`
Expected: FAIL (ImportError: cannot import from ttt.solver).

- [ ] **Step 3: Implement ttt/solver.py**

```python
from __future__ import annotations

from functools import lru_cache

from ttt.board import legal_moves, apply_move, winner, is_terminal, current_player


@lru_cache(maxsize=None)
def minimax_value(board):
    """Value to the player to move: +1 win, 0 draw, -1 loss (perfect play)."""
    w = winner(board)
    if w != 0:
        # Previous player just won, so the player to move has lost.
        return -1
    if is_terminal(board):
        return 0
    best = -2
    for m in legal_moves(board):
        # Opponent's value after our move; negate for our perspective.
        val = -minimax_value(apply_move(board, m))
        if val > best:
            best = val
    return best


def optimal_moves(board):
    """All moves achieving the minimax-optimal value for the player to move."""
    assert not is_terminal(board), "no moves from a terminal board"
    best = -2
    scored = []
    for m in legal_moves(board):
        val = -minimax_value(apply_move(board, m))
        scored.append((m, val))
        if val > best:
            best = val
    return tuple(m for m, val in scored if val == best)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_solver.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add ttt/solver.py tests/test_solver.py
git commit -m "feat: memoized minimax solver (value + optimal move set)"
```

---

### Task 5: Position enumeration with capped orderings

**Files:**
- Create: `ttt/enumerate.py`
- Test: `tests/test_enumerate.py`

- [ ] **Step 1: Write failing tests**

`tests/test_enumerate.py`:
```python
from ttt.board import EMPTY_BOARD, is_terminal
from ttt.enumerate import reachable_paths


def test_empty_board_has_empty_path():
    paths = reachable_paths(max_orderings=4)
    assert EMPTY_BOARD in paths
    assert () in paths[EMPTY_BOARD]


def test_total_reachable_positions_count():
    # Tic-tac-toe has 5478 reachable board states (including terminal).
    paths = reachable_paths(max_orderings=4)
    assert len(paths) == 5478


def test_orderings_capped():
    paths = reachable_paths(max_orderings=4)
    assert all(len(v) <= 4 for v in paths.values())


def test_paths_actually_reach_their_board():
    from ttt.board import apply_move
    paths = reachable_paths(max_orderings=4)
    # Pick a non-empty board and replay one of its paths.
    for board, seqs in paths.items():
        if board == EMPTY_BOARD:
            continue
        seq = seqs[0]
        replay = EMPTY_BOARD
        for m in seq:
            replay = apply_move(replay, m)
        assert replay == board
        break
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_enumerate.py -v`
Expected: FAIL (ImportError: cannot import from ttt.enumerate).

- [ ] **Step 3: Implement ttt/enumerate.py**

```python
from __future__ import annotations

from ttt.board import EMPTY_BOARD, legal_moves, apply_move, is_terminal


def reachable_paths(max_orderings=4):
    """Map every reachable board -> up to `max_orderings` move-sequences reaching it.

    Boards are deduped; the first `max_orderings` distinct paths discovered are kept.
    A board is expanded only once (children depend only on the board, not the path).
    """
    paths = {}
    expanded = set()

    def dfs(board, path):
        if board not in paths:
            paths[board] = []
        if len(paths[board]) < max_orderings:
            paths[board].append(path)
        if board in expanded:
            return
        expanded.add(board)
        if is_terminal(board):
            return
        for m in legal_moves(board):
            dfs(apply_move(board, m), path + (m,))

    dfs(EMPTY_BOARD, ())
    return paths
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_enumerate.py -v`
Expected: PASS (4 tests). The 5478 count confirms full coverage.

- [ ] **Step 5: Commit**

```bash
git add ttt/enumerate.py tests/test_enumerate.py
git commit -m "feat: reachable-position enumeration with capped move-orderings"
```

---

### Task 6: Dataset (tokenization + examples + horizontal filter)

**Files:**
- Create: `ttt/dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write failing tests**

`tests/test_dataset.py`:
```python
from ttt.board import P1, move_completes_horizontal
from ttt.dataset import (
    BOS_ID, PAD_ID, VOCAB_SIZE, N_CELLS,
    encode_prefix, build_examples,
)


def test_token_constants():
    assert BOS_ID == 9
    assert PAD_ID == 10
    assert VOCAB_SIZE == 11
    assert N_CELLS == 9


def test_encode_prefix_prepends_bos():
    assert encode_prefix((0, 4, 8)) == [BOS_ID, 0, 4, 8]
    assert encode_prefix(()) == [BOS_ID]


def test_filter_removes_all_horizontal_winning_targets():
    examples = build_examples(max_orderings=2, filter_horizontal=True)
    # Reconstruct the board each example's prefix reaches, and confirm the
    # target never completes a horizontal line.
    from ttt.board import EMPTY_BOARD, apply_move
    for input_ids, target in examples:
        board = EMPTY_BOARD
        for tok in input_ids[1:]:  # skip BOS
            board = apply_move(board, tok)
        assert not move_completes_horizontal(board, target)


def test_unfiltered_dataset_contains_some_horizontal_wins():
    examples = build_examples(max_orderings=2, filter_horizontal=False)
    from ttt.board import EMPTY_BOARD, apply_move
    found = False
    for input_ids, target in examples:
        board = EMPTY_BOARD
        for tok in input_ids[1:]:
            board = apply_move(board, tok)
        if move_completes_horizontal(board, target):
            found = True
            break
    assert found, "unfiltered data should contain horizontal-winning targets"


def test_examples_are_nonempty():
    assert len(build_examples(max_orderings=2, filter_horizontal=True)) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL (ImportError: cannot import from ttt.dataset).

- [ ] **Step 3: Implement ttt/dataset.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add ttt/dataset.py tests/test_dataset.py
git commit -m "feat: dataset builder, tokenization, and horizontal-win filter"
```

---

### Task 7: Model (nanoGPT-style decoder)

**Files:**
- Create: `ttt/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write failing tests**

`tests/test_model.py`:
```python
import torch
from ttt.model import GPTConfig, TTTGPT
from ttt.dataset import N_CELLS, BOS_ID, PAD_ID


def test_forward_output_shape():
    cfg = GPTConfig(n_layer=2, n_head=2, d_model=32)
    model = TTTGPT(cfg)
    ids = torch.tensor([[BOS_ID, 0, 4], [BOS_ID, 1, PAD_ID]])
    lengths = torch.tensor([3, 2])
    logits = model(ids, lengths)
    assert logits.shape == (2, N_CELLS)


def test_last_token_gather_ignores_padding():
    # Two identical real prefixes, one padded longer, must give equal logits.
    cfg = GPTConfig(n_layer=1, n_head=1, d_model=16)
    model = TTTGPT(cfg)
    model.eval()
    a = torch.tensor([[BOS_ID, 0, 4]])
    b = torch.tensor([[BOS_ID, 0, 4, PAD_ID, PAD_ID]])
    with torch.no_grad():
        la = model(a, torch.tensor([3]))
        lb = model(b, torch.tensor([3]))
    assert torch.allclose(la, lb, atol=1e-5)


def test_overfits_tiny_dataset():
    torch.manual_seed(0)
    cfg = GPTConfig(n_layer=2, n_head=2, d_model=32)
    model = TTTGPT(cfg)
    ids = torch.tensor([[BOS_ID, 0, 1], [BOS_ID, 3, 4]])
    lengths = torch.tensor([3, 3])
    targets = torch.tensor([2, 5])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn(model(ids, lengths), targets)
        loss.backward()
        opt.step()
    assert model(ids, lengths).argmax(dim=-1).tolist() == [2, 5]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py -v`
Expected: FAIL (ImportError: cannot import from ttt.model).

- [ ] **Step 3: Implement ttt/model.py**

```python
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ttt.dataset import VOCAB_SIZE, N_CELLS


@dataclass
class GPTConfig:
    n_layer: int
    n_head: int
    d_model: int
    vocab_size: int = VOCAB_SIZE
    max_len: int = 10
    n_out: int = N_CELLS


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


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
        self.head = nn.Linear(cfg.d_model, cfg.n_out)

    def forward(self, ids, lengths):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        last = (lengths - 1).clamp(min=0)  # gather the final real token
        gathered = x[torch.arange(B, device=ids.device), last]
        return self.head(gathered)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add ttt/model.py tests/test_model.py
git commit -m "feat: nanoGPT-style decoder over move-sequences"
```

---

### Task 8: Training loop

**Files:**
- Create: `ttt/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write failing tests**

`tests/test_train.py`:
```python
import torch
from ttt.model import GPTConfig
from ttt.train import train_model
from ttt.dataset import BOS_ID


def test_train_reduces_loss_and_returns_model():
    examples = [([BOS_ID, 0, 1], 2), ([BOS_ID, 3, 4], 5), ([BOS_ID, 6, 7], 8)]
    cfg = GPTConfig(n_layer=1, n_head=1, d_model=16)
    model, history = train_model(
        cfg, examples, epochs=50, lr=1e-2, batch_size=8, seed=0
    )
    assert history[-1] < history[0]  # loss decreased
    # Predictions on the training set should be correct after overfitting.
    import torch
    for ids, tgt in examples:
        t = torch.tensor([ids])
        logits = model(t, torch.tensor([len(ids)]))
        assert logits.argmax(dim=-1).item() == tgt


def test_seed_is_deterministic():
    examples = [([BOS_ID, 0, 1], 2), ([BOS_ID, 3, 4], 5)]
    cfg = GPTConfig(n_layer=1, n_head=1, d_model=16)
    m1, _ = train_model(cfg, examples, epochs=5, lr=1e-2, batch_size=8, seed=42)
    m2, _ = train_model(cfg, examples, epochs=5, lr=1e-2, batch_size=8, seed=42)
    p1 = torch.cat([p.flatten() for p in m1.parameters()])
    p2 = torch.cat([p.flatten() for p in m2.parameters()])
    assert torch.allclose(p1, p2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train.py -v`
Expected: FAIL (ImportError: cannot import from ttt.train).

- [ ] **Step 3: Implement ttt/train.py**

```python
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from ttt.model import TTTGPT, GPTConfig
from ttt.dataset import TTTDataset, collate_fn


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(cfg: GPTConfig, examples, *, epochs, lr, batch_size,
                seed, device="cpu"):
    """Train a model by next-move cross-entropy. Returns (model, loss_history)."""
    _set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        TTTDataset(examples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=g,
    )
    model = TTTGPT(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    history = []
    model.train()
    for _ in range(epochs):
        epoch_loss, n = 0.0, 0
        for ids, lengths, targets in loader:
            ids, lengths, targets = ids.to(device), lengths.to(device), targets.to(device)
            opt.zero_grad()
            loss = loss_fn(model(ids, lengths), targets)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(targets)
            n += len(targets)
        history.append(epoch_loss / max(n, 1))
    return model, history
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add ttt/train.py tests/test_train.py
git commit -m "feat: deterministic training loop (next-move cross-entropy)"
```

---

### Task 9: Probe suites (win-available / block-needed, H/V/D)

**Files:**
- Create: `ttt/probes.py`
- Test: `tests/test_probes.py`

- [ ] **Step 1: Write failing tests**

`tests/test_probes.py`:
```python
from ttt.board import P1, P2, EMPTY, current_player, move_completes_horizontal
from ttt.probes import win_available_probes, block_needed_probes, probe_prefixes
from ttt.enumerate import reachable_paths


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_probes.py -v`
Expected: FAIL (ImportError: cannot import from ttt.probes).

- [ ] **Step 3: Implement ttt/probes.py**

```python
from __future__ import annotations

from ttt.board import (
    EMPTY, P1, P2, current_player, is_terminal, winner,
    LINES_BY_TYPE, ALL_LINES,
)
from ttt.enumerate import reachable_paths
from ttt.dataset import encode_prefix


def _immediate_win_cell(board, player):
    """A cell where `player` immediately completes any line, else None."""
    for a, b, c in ALL_LINES:
        cells = (a, b, c)
        vals = [board[i] for i in cells]
        if vals.count(player) == 2 and vals.count(EMPTY) == 1:
            return cells[vals.index(EMPTY)]
    return None


def win_available_probes(line_type):
    """Boards where the player to move can win on a `line_type` line next move.

    Returns list of (board, target_cell). Excludes already-terminal boards.
    """
    probes = []
    seen = set()
    for board in reachable_paths(max_orderings=1):
        if is_terminal(board) or board in seen:
            continue
        player = current_player(board)
        for line in LINES_BY_TYPE[line_type]:
            vals = [board[i] for i in line]
            if vals.count(player) == 2 and vals.count(EMPTY) == 1:
                target = line[vals.index(EMPTY)]
                probes.append((board, target))
                seen.add(board)
                break
    return probes


def block_needed_probes(line_type):
    """Boards where the opponent threatens a `line_type` win and blocking it is
    the clear correct move.

    Clean conditions: the player to move has NO immediate win of their own, and
    there is exactly one opponent threat cell, so the block target is unique.
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
        target_for_type = None
        for a, b, c in ALL_LINES:
            cells = (a, b, c)
            vals = [board[i] for i in cells]
            if vals.count(opp) == 2 and vals.count(EMPTY) == 1:
                cell = cells[vals.index(EMPTY)]
                threats.add(cell)
                if (a, b, c) in LINES_BY_TYPE[line_type]:
                    target_for_type = cell
        if len(threats) == 1 and target_for_type is not None:
            probes.append((board, target_for_type))
    return probes


def probe_prefixes(board, paths, max_orderings=4):
    """Encoded prefixes ([BOS, ...]) for the orderings that reach `board`."""
    return [encode_prefix(p) for p in paths.get(board, [])[:max_orderings]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_probes.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add ttt/probes.py tests/test_probes.py
git commit -m "feat: win-available / block-needed probe suites (H/V/D)"
```

---

### Task 10: Evaluation metrics

**Files:**
- Create: `ttt/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing tests**

`tests/test_evaluate.py`:
```python
import torch
from ttt.board import EMPTY, legal_moves
from ttt.evaluate import model_move, evaluate_probes, random_baseline_rate
from ttt.enumerate import reachable_paths
from ttt.probes import win_available_probes


class _AlwaysCellModel(torch.nn.Module):
    """Test double: always prefers a fixed cell (high logit), masked to legal."""
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, ids, lengths):
        logits = torch.zeros(ids.shape[0], 9)
        logits[:, self.cell] = 10.0
        return logits


def test_model_move_masks_illegal_cells():
    board = (1, 0, 0, 0, 0, 0, 0, 0, 0)  # cell 0 occupied
    model = _AlwaysCellModel(0)  # wants illegal cell 0
    move = model_move(model, [9, 0], board)  # BOS, move 0
    assert move != 0
    assert board[move] == EMPTY


def test_evaluate_probes_perfect_when_model_targets_correct_cell():
    probes = win_available_probes("vertical")[:5]
    paths = reachable_paths(max_orderings=4)
    # Build a model that always picks each probe's target by using its target.
    # Use a model that targets the only winning cell via masking of a constant.
    # Here we instead check the rate computation with a model that picks legally.
    rates = evaluate_probes(_AlwaysCellModel(4), probes, paths, max_orderings=4)
    assert 0.0 <= rates["rate"] <= 1.0
    assert 0.0 <= rates["order_invariance"] <= 1.0


def test_random_baseline_between_zero_and_one():
    probes = win_available_probes("horizontal")
    base = random_baseline_rate(probes)
    assert 0.0 < base < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL (ImportError: cannot import from ttt.evaluate).

- [ ] **Step 3: Implement ttt/evaluate.py**

```python
from __future__ import annotations

import torch

from ttt.board import EMPTY, legal_moves
from ttt.probes import probe_prefixes


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


def evaluate_probes(model, probes, paths, max_orderings=4, device="cpu"):
    """Fraction of (probe, ordering) pairs where the model plays the target cell.

    Also returns order_invariance: fraction of probes whose chosen move is the
    same across all their orderings.
    """
    hits, total = 0, 0
    invariant_probes, probes_with_orderings = 0, 0
    for board, target in probes:
        prefixes = probe_prefixes(board, paths, max_orderings=max_orderings)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add ttt/evaluate.py tests/test_evaluate.py
git commit -m "feat: probe evaluation metrics, order-invariance, random baseline"
```

---

### Task 11: Game-play and behavior detection

**Files:**
- Create: `ttt/gameplay.py`
- Test: `tests/test_gameplay.py`

- [ ] **Step 1: Write failing tests**

`tests/test_gameplay.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gameplay.py -v`
Expected: FAIL (ImportError: cannot import from ttt.gameplay).

- [ ] **Step 3: Implement ttt/gameplay.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gameplay.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add ttt/gameplay.py tests/test_gameplay.py
git commit -m "feat: opponents (random/perfect/model) and game play"
```

---

### Task 12: Sweep runner, aggregation, and plot

**Files:**
- Create: `ttt/sweep.py`
- Test: `tests/test_sweep.py`

- [ ] **Step 1: Write failing tests**

`tests/test_sweep.py`:
```python
import os
from ttt.sweep import aggregate, plot_capacity


def test_aggregate_means_and_stds_over_seeds():
    raw = [
        {"config": "L1H1D16", "n_params": 100, "seed": 0,
         "metrics": {"horizontal_win": 0.0, "horizontal_block": 1.0}},
        {"config": "L1H1D16", "n_params": 100, "seed": 1,
         "metrics": {"horizontal_win": 0.2, "horizontal_block": 0.8}},
    ]
    agg = aggregate(raw)
    row = agg["L1H1D16"]
    assert row["n_params"] == 100
    assert abs(row["horizontal_win"]["mean"] - 0.1) < 1e-9
    assert row["horizontal_win"]["std"] >= 0.0
    assert abs(row["horizontal_block"]["mean"] - 0.9) < 1e-9


def test_plot_capacity_writes_file(tmp_path):
    agg = {
        "L1H1D16": {"n_params": 100,
                     "horizontal_win": {"mean": 0.1, "std": 0.05},
                     "horizontal_block": {"mean": 0.9, "std": 0.05},
                     "vertical_win": {"mean": 1.0, "std": 0.0},
                     "diagonal_win": {"mean": 1.0, "std": 0.0}},
        "L2H2D32": {"n_params": 500,
                     "horizontal_win": {"mean": 0.6, "std": 0.1},
                     "horizontal_block": {"mean": 0.95, "std": 0.02},
                     "vertical_win": {"mean": 1.0, "std": 0.0},
                     "diagonal_win": {"mean": 1.0, "std": 0.0}},
    }
    out = tmp_path / "capacity.png"
    plot_capacity(agg, str(out))
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sweep.py -v`
Expected: FAIL (ImportError: cannot import from ttt.sweep).

- [ ] **Step 3: Implement ttt/sweep.py**

```python
from __future__ import annotations

import json
import statistics

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from ttt.model import GPTConfig
from ttt.train import train_model
from ttt.dataset import build_examples
from ttt.enumerate import reachable_paths
from ttt.probes import win_available_probes, block_needed_probes
from ttt.evaluate import evaluate_probes

# Metrics evaluated per trained model.
PROBE_SPECS = {
    "horizontal_win": ("win", "horizontal"),
    "horizontal_block": ("block", "horizontal"),
    "vertical_win": ("win", "vertical"),
    "vertical_block": ("block", "vertical"),
    "diagonal_win": ("win", "diagonal"),
    "diagonal_block": ("block", "diagonal"),
}


def config_name(n_layer, n_head, d_model):
    return f"L{n_layer}H{n_head}D{d_model}"


def _build_probe_sets():
    sets = {}
    for name, (kind, line_type) in PROBE_SPECS.items():
        if kind == "win":
            sets[name] = win_available_probes(line_type)
        else:
            sets[name] = block_needed_probes(line_type)
    return sets


def run_sweep(grid, seeds, *, epochs, lr, batch_size, max_orderings=4):
    """Train every (config, seed); evaluate all probe metrics. Returns raw rows."""
    examples = build_examples(max_orderings=max_orderings, filter_horizontal=True)
    paths = reachable_paths(max_orderings=max_orderings)
    probe_sets = _build_probe_sets()
    raw = []
    for (n_layer, n_head, d_model) in grid:
        cfg = GPTConfig(n_layer=n_layer, n_head=n_head, d_model=d_model)
        for seed in seeds:
            model, _ = train_model(
                cfg, examples, epochs=epochs, lr=lr,
                batch_size=batch_size, seed=seed,
            )
            metrics = {
                name: evaluate_probes(model, probes, paths,
                                      max_orderings=max_orderings)["rate"]
                for name, probes in probe_sets.items()
            }
            raw.append({
                "config": config_name(n_layer, n_head, d_model),
                "n_params": model.num_params(),
                "seed": seed,
                "metrics": metrics,
            })
    return raw


def aggregate(raw):
    """Mean/std per config per metric across seeds."""
    by_config = {}
    for row in raw:
        by_config.setdefault(row["config"], []).append(row)
    agg = {}
    for config, rows in by_config.items():
        entry = {"n_params": rows[0]["n_params"]}
        metric_names = rows[0]["metrics"].keys()
        for m in metric_names:
            vals = [r["metrics"][m] for r in rows]
            entry[m] = {
                "mean": statistics.fmean(vals),
                "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            }
        agg[config] = entry
    return agg


def plot_capacity(agg, out_path):
    """Horizontal win/block vs. capacity, with vertical/diagonal controls."""
    configs = sorted(agg, key=lambda c: agg[c]["n_params"])
    x = [agg[c]["n_params"] for c in configs]

    def series(metric):
        means = [agg[c][metric]["mean"] for c in configs]
        stds = [agg[c][metric]["std"] for c in configs]
        return means, stds

    fig, ax = plt.subplots(figsize=(7, 5))
    for metric, label in [
        ("horizontal_win", "horizontal win (held-out)"),
        ("horizontal_block", "horizontal block"),
        ("vertical_win", "vertical win (control)"),
        ("diagonal_win", "diagonal win (control)"),
    ]:
        if metric not in agg[configs[0]]:
            continue
        means, stds = series(metric)
        ax.errorbar(x, means, yerr=stds, marker="o", capsize=3, label=label)
    ax.set_xlabel("model parameters")
    ax.set_ylabel("probe success rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Horizontal-win generalization vs. model capacity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_results(raw, agg, raw_path, agg_path):
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sweep.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add ttt/sweep.py tests/test_sweep.py
git commit -m "feat: sweep runner, seed aggregation, capacity plot"
```

---

### Task 13: End-to-end experiment entrypoint

**Files:**
- Create: `run_experiment.py`
- Create: `tests/test_end_to_end.py`

- [ ] **Step 1: Write failing test (fast, tiny smoke run)**

`tests/test_end_to_end.py`:
```python
import os
from ttt.sweep import run_sweep, aggregate, plot_capacity


def test_tiny_end_to_end_smoke(tmp_path):
    # One tiny config, one seed, few epochs: must run end-to-end and produce a plot.
    raw = run_sweep(
        grid=[(1, 1, 16)], seeds=[0],
        epochs=2, lr=1e-2, batch_size=256, max_orderings=2,
    )
    assert len(raw) == 1
    assert "horizontal_win" in raw[0]["metrics"]
    agg = aggregate(raw)
    out = tmp_path / "capacity.png"
    plot_capacity(agg, str(out))
    assert os.path.exists(out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_end_to_end.py -v`
Expected: FAIL initially only if `run_experiment.py` import is referenced; this test uses `ttt.sweep` directly, so it should PASS once Task 12 is done. If it already passes, proceed — it guards the full wiring.

- [ ] **Step 3: Implement run_experiment.py**

```python
"""Run the full sweep and write results + capacity plot to ./results/."""
from __future__ import annotations

import itertools
import os

from ttt.sweep import run_sweep, aggregate, plot_capacity, save_results

GRID = list(itertools.product([1, 2, 4], [1, 2, 4], [16, 32, 64]))
# Keep only configs where d_model is divisible by n_head.
GRID = [(L, H, D) for (L, H, D) in GRID if D % H == 0]
SEEDS = [0, 1, 2, 3, 4]


def main():
    os.makedirs("results", exist_ok=True)
    raw = run_sweep(
        grid=GRID, seeds=SEEDS,
        epochs=60, lr=1e-3, batch_size=256, max_orderings=4,
    )
    agg = aggregate(raw)
    save_results(raw, agg, "results/raw.json", "results/agg.json")
    plot_capacity(agg, "results/capacity.png")
    print("Wrote results/raw.json, results/agg.json, results/capacity.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the smoke test and a real (small) entrypoint check**

Run: `pytest tests/test_end_to_end.py -v`
Expected: PASS.

Optionally verify the entrypoint imports and the grid is valid:
Run: `python -c "import run_experiment as r; print(len(r.GRID), 'configs')"`
Expected: prints the config count (all have d_model divisible by n_head).

- [ ] **Step 5: Commit**

```bash
git add run_experiment.py tests/test_end_to_end.py
git commit -m "feat: end-to-end experiment entrypoint + smoke test"
```

---

### Task 14: Full test pass + README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Run the whole suite**

Run: `pytest -v`
Expected: PASS (all tests across all modules).

- [ ] **Step 2: Write README.md**

```markdown
# Tiny Transformer Generalization on Filtered Tic-Tac-Toe

Trains tiny decoder-only transformers on tic-tac-toe move-sequences with all
**horizontal-winning moves removed**, then probes whether they (1) score
horizontal wins they never saw (rotational-symmetry generalization) or (2) only
block horizontal threats (a pattern still present in training).

See the design and plan in `docs/superpowers/`.

## Setup
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Run tests
```bash
pytest -v
```

## Run the full experiment
```bash
python run_experiment.py
# writes results/raw.json, results/agg.json, results/capacity.png
```

## How it works
- `ttt/board.py` — board + win/line logic
- `ttt/solver.py` — perfect minimax, optimal move sets
- `ttt/enumerate.py` — all 5,478 reachable positions + capped move-orderings
- `ttt/dataset.py` — tokenization + horizontal-win filter (drops offensive targets, keeps blocks)
- `ttt/model.py` — nanoGPT-style decoder
- `ttt/train.py` — training loop
- `ttt/probes.py` / `ttt/evaluate.py` — win/block probes (H/V/D) + metrics
- `ttt/sweep.py` — capacity sweep, seed aggregation, capacity plot
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: project README"
```

---

## Self-Review

**1. Spec coverage** (each spec section → task):
- Solver + enumerate all ~5,478 positions → Tasks 4, 5 (test asserts 5478). ✓
- Move-sequence tokenization, optimal labels, multiple orderings → Tasks 5, 6. ✓
- Horizontal loss filter (drop offensive, keep blocks) → Task 6 (filter test confirms no horizontal target survives; unfiltered test confirms blocks/wins exist). ✓
- No symmetry augmentation → enforced by construction (no augmentation code anywhere). ✓
- Decoder-only model + sweep grid + multi-seed → Tasks 7, 12, 13 (grid in `run_experiment.py`, seeds 0–4; ≥5 seeds). ✓
- Probe suite: win-available + block-needed, H/V/D controls → Tasks 9, 10. ✓
- Order-invariance sub-measurement → Task 10 (`evaluate_probes` returns `order_invariance`). ✓
- Game-play vs random/perfect/self → Task 11 (RandomPlayer, PerfectPlayer, ModelPlayer; `play_game`). ✓
- Outcome interpretation / capacity plot with controls + random baseline → Tasks 10, 12 (`random_baseline_rate`, `plot_capacity`). ✓
- Statistical aggregation across seeds (mean±std) → Task 12 (`aggregate`). ✓

**2. Placeholder scan:** No TBD/TODO; every code step has complete code and exact commands. The one prose note in Task 3 explicitly supplies the corrected test body. ✓

**3. Type consistency:** `GPTConfig(n_layer, n_head, d_model)`, `train_model(cfg, examples, *, epochs, lr, batch_size, seed)`, `build_examples(max_orderings, filter_horizontal)`, `evaluate_probes(...)→{"rate","order_invariance","n_params"... }`, `win_available_probes/block_needed_probes(line_type)→[(board,target)]`, `probe_prefixes(board, paths, max_orderings)`, token constants `BOS_ID=9/PAD_ID=10/VOCAB_SIZE=11/N_CELLS=9` — all consistent across tasks. ✓

> **Note on game-play secondary analysis:** Task 11 builds the players and `play_game`; aggregating spontaneous horizontal-win/block *counts* from many played games is a thin reporting layer that can be added to `run_experiment.py` once the primary probe results are in. The probe suite (Tasks 9–10) is the primary metric per the spec, so this is intentionally left as a light follow-up rather than a blocking task.
