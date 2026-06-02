import torch
from ttt.model import GPTConfig, TTTGPT, factored_logits
from ttt.dataset import N_CELLS, BOS_ID, PAD_ID
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
