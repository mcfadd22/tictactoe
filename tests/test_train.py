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


def test_weight_decay_default_equals_explicit_zero():
    # Default weight_decay=0.0 must produce identical weights to weight_decay=0.0
    # passed explicitly (guards that the default is correctly wired).
    examples = [([BOS_ID, 0, 1], 2), ([BOS_ID, 3, 4], 5)]
    cfg = GPTConfig(n_layer=1, n_head=1, d_model=16)
    m1, _ = train_model(cfg, examples, epochs=5, lr=1e-2, batch_size=8, seed=7)
    m2, _ = train_model(cfg, examples, epochs=5, lr=1e-2, batch_size=8, seed=7,
                        weight_decay=0.0)
    p1 = torch.cat([p.flatten() for p in m1.parameters()])
    p2 = torch.cat([p.flatten() for p in m2.parameters()])
    assert torch.allclose(p1, p2)


def test_weight_decay_changes_trained_weights():
    # weight_decay=1.0 is large enough to visibly move weights within 20 epochs.
    examples = [([BOS_ID, 0, 1], 2), ([BOS_ID, 3, 4], 5)]
    cfg = GPTConfig(n_layer=1, n_head=1, d_model=16)
    m0, _ = train_model(cfg, examples, epochs=20, lr=1e-2, batch_size=8, seed=7,
                        weight_decay=0.0)
    mw, _ = train_model(cfg, examples, epochs=20, lr=1e-2, batch_size=8, seed=7,
                        weight_decay=1.0)
    p0 = torch.cat([p.flatten() for p in m0.parameters()])
    pw = torch.cat([p.flatten() for p in mw.parameters()])
    assert not torch.allclose(p0, pw)  # decay actually moved the weights


def test_eval_hook_called_at_expected_epochs():
    examples = [([BOS_ID, 0, 1], 2), ([BOS_ID, 3, 4], 5)]
    cfg = GPTConfig(n_layer=1, n_head=1, d_model=16)
    calls = []
    def hook(model, epoch, train_loss):
        calls.append((epoch, train_loss))
    train_model(cfg, examples, epochs=10, lr=1e-2, batch_size=8, seed=0,
                eval_every=4, eval_hook=hook)
    # epochs are 1-indexed in the hook; every 4th plus the final epoch (10)
    assert [e for e, _ in calls] == [4, 8, 10]
    assert all(isinstance(l, float) for _, l in calls)


def test_eval_hook_does_not_perturb_training():
    examples = [([BOS_ID, 0, 1], 2), ([BOS_ID, 3, 4], 5)]
    cfg = GPTConfig(n_layer=1, n_head=1, d_model=16)
    m_plain, _ = train_model(cfg, examples, epochs=10, lr=1e-2, batch_size=8, seed=0)
    m_hooked, _ = train_model(cfg, examples, epochs=10, lr=1e-2, batch_size=8, seed=0,
                              eval_every=2, eval_hook=lambda *a: None)
    p1 = torch.cat([p.flatten() for p in m_plain.parameters()])
    p2 = torch.cat([p.flatten() for p in m_hooked.parameters()])
    assert torch.allclose(p1, p2)


def test_no_eval_hook_is_default_noop():
    examples = [([BOS_ID, 0, 1], 2)]
    cfg = GPTConfig(n_layer=1, n_head=1, d_model=16)
    model, history = train_model(cfg, examples, epochs=3, lr=1e-2, batch_size=8, seed=0)
    assert len(history) == 3  # unchanged (model, history) return, no hook required


def test_eval_hook_eval_mode_is_restored():
    # A hook that flips the model to eval mode must not leave it there.
    examples = [([BOS_ID, 0, 1], 2)]
    cfg = GPTConfig(n_layer=1, n_head=1, d_model=16)
    def hook(model, epoch, train_loss):
        model.eval()
    model, _ = train_model(cfg, examples, epochs=4, lr=1e-2, batch_size=8, seed=0,
                           eval_every=2, eval_hook=hook)
    assert model.training is True
