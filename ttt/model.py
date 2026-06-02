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
    head: str = "flat9"


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


def factored_logits(row_logit, col_logit):
    """Combine a (B,3) row distribution and a (B,3) col distribution into (B,9)
    cell logits, where cell index = row*3 + col and
    logit[r, c] = row_logit[r] + col_logit[c]."""
    grid = row_logit.unsqueeze(2) + col_logit.unsqueeze(1)  # (B, 3, 3) [b, r, c]
    return grid.reshape(row_logit.shape[0], N_CELLS)


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
