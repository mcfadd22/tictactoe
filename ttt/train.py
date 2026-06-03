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
