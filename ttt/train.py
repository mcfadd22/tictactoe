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
                seed, encoding=FLAT, device="cpu", weight_decay=0.0,
                eval_every=0, eval_hook=None):
    """Train a model by next-move cross-entropy. Returns (model, loss_history).

    weight_decay (decoupled, AdamW) defaults to 0.0, where AdamW is numerically
    identical to the previous Adam path.

    eval_every / eval_hook: if both are provided and eval_every > 0, eval_hook
    is called every eval_every epochs AND on the final epoch. Signature:
        hook(model, epoch: int, train_loss: float) -> None
    Epochs are 1-indexed. The hook is run under torch.no_grad(); model.train()
    is restored afterward so subsequent epochs are unaffected.
    """
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
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss, n = 0.0, 0
        for ids, lengths, targets in loader:
            ids, lengths, targets = ids.to(device), lengths.to(device), targets.to(device)
            opt.zero_grad()
            loss = loss_fn(model(ids, lengths), targets)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(targets)
            n += len(targets)
        mean_loss = epoch_loss / max(n, 1)
        history.append(mean_loss)
        if eval_hook is not None and eval_every > 0 and (
            epoch % eval_every == 0 or epoch == epochs
        ):
            try:
                with torch.no_grad():
                    eval_hook(model, epoch, mean_loss)
            finally:
                model.train()  # hook may put the model in eval mode; always restore
    return model, history
