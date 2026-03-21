# models/transformer_forecaster.py
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    context_len: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    dropout: float = 0.1
    ff_mult: int = 4
    dense_supervision: bool = False  # if True, predict at every position not just last


class CausalTransformerForecaster(nn.Module):
    """
    Causal transformer encoder forecaster.

    Standard mode (dense_supervision=False):
        input:  (B, L, 1)
        output: (B, 1)         — prediction for position L only

    Dense supervision mode (dense_supervision=True):
        input:  (B, L, 1)
        output: (B, L, 1)      — prediction for every position simultaneously
        Position t predicts timestep t+1. All L predictions are supervised
        during training, giving 64x more gradient signal per forward pass.
        At eval time, only the last position output is used.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(1, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.context_len, cfg.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_mult * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.out_proj = nn.Linear(cfg.d_model, 1)

        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def _causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        # True means blocked — position i cannot attend to position j > i
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 1)
        B, L, _ = x.shape
        if L != self.cfg.context_len:
            raise ValueError(f"expected L={self.cfg.context_len}, got {L}")

        h = self.in_proj(x) + self.pos_emb[:, :L, :]
        h = self.encoder(h, mask=self._causal_mask(L, x.device))

        if self.cfg.dense_supervision:
            return self.out_proj(h)          # (B, L, 1) — all positions
        else:
            return self.out_proj(h[:, -1, :])  # (B, 1) — last position only

    def predict_last(self, x: torch.Tensor) -> torch.Tensor:
        """
        Always returns (B, 1) regardless of dense_supervision setting.
        Used during evaluation so eval_loop works the same in both modes.
        """
        B, L, _ = x.shape
        h = self.in_proj(x) + self.pos_emb[:, :L, :]
        h = self.encoder(h, mask=self._causal_mask(L, x.device))
        return self.out_proj(h[:, -1, :])    # (B, 1)