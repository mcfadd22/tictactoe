import torch
from ttt.model import GPTConfig
from ttt.train import train_model
from ttt.dataset import BOS_ID
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


def test_weight_decay_zero_matches_default_behavior():
    # AdamW(weight_decay=0) must reproduce the prior Adam path exactly (same seed).
    examples = [([BOS_ID, 0, 1], 2), ([BOS_ID, 3, 4], 5)]
    cfg = GPTConfig(n_layer=1, n_head=1, d_model=16)
    m1, _ = train_model(cfg, examples, epochs=5, lr=1e-2, batch_size=8, seed=7)
    m2, _ = train_model(cfg, examples, epochs=5, lr=1e-2, batch_size=8, seed=7,
                        weight_decay=0.0)
    p1 = torch.cat([p.flatten() for p in m1.parameters()])
    p2 = torch.cat([p.flatten() for p in m2.parameters()])
    assert torch.allclose(p1, p2)


def test_weight_decay_changes_trained_weights():
    examples = [([BOS_ID, 0, 1], 2), ([BOS_ID, 3, 4], 5)]
    cfg = GPTConfig(n_layer=1, n_head=1, d_model=16)
    m0, _ = train_model(cfg, examples, epochs=20, lr=1e-2, batch_size=8, seed=7,
                        weight_decay=0.0)
    mw, _ = train_model(cfg, examples, epochs=20, lr=1e-2, batch_size=8, seed=7,
                        weight_decay=1.0)
    p0 = torch.cat([p.flatten() for p in m0.parameters()])
    pw = torch.cat([p.flatten() for p in mw.parameters()])
    assert not torch.allclose(p0, pw)  # decay actually moved the weights
