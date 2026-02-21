from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class TransformerConfig:
    context_len: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    ff_mult: int = 4

class CausalTransformerForecaster(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.in_proj = nn.Linear(1, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.context_len, cfg.d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_mult * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.out_proj = nn.Linear(cfg.d_model, 1)

        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def _causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 1)
        B, L, _ = x.shape
        if L != self.cfg.context_len:
            raise ValueError(f"expected context_len={self.cfg.context_len}, got {L}")
        h = self.in_proj(x) + self.pos_emb[:, :L, :]
        h = self.encoder(h, mask=self._causal_mask(L, x.device))
        last = h[:, -1, :]
        return self.out_proj(last)