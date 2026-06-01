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
