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


class CausalTransformerForecaster(nn.Module):
    """
    Decoder-only transformer for one-step-ahead forecasting.

    Architecture: GPT-style causally masked transformer. At every position t,
    the model predicts the next value y[t+1] using only y[0]...y[t] as context
    (enforced by causal mask). This is equivalent to a decoder-only language
    model with next-token prediction.

    Training (dense supervision):
        All L positions are supervised simultaneously. Position t predicts t+1,
        giving L times more gradient signal per forward pass than supervising
        only the final position. The loss is averaged over all L predictions.

    Inference:
        Only the last position's prediction is used — it has attended to the
        full context window and produces the forecast for the next timestep.

    input:  (B, L, 1)   — batch of L scalar observations
    output: (B, L, 1)   — next-step predictions at every position
                          use [:, -1, :] at inference for the forecast
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(1, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.context_len, cfg.d_model))

        dec_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_mult * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,   # pre-norm, more stable (GPT-2 style)
        )
        # nn.TransformerEncoder implements a stack of causally masked layers.
        # With the causal mask applied at every forward call this is functionally
        # identical to a GPT-style decoder stack.
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=cfg.n_layers)
        self.out_proj = nn.Linear(cfg.d_model, 1)

        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def _causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask: position i cannot attend to position j > i."""
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full decoder forward pass.

        Args:
            x: (B, L, 1) — context window of L scalar observations

        Returns:
            (B, L, 1) — predicted next value at every position.
                        During training all positions are supervised.
                        At inference use output[:, -1, :] for the forecast.
        """
        B, L, _ = x.shape
        if L != self.cfg.context_len:
            raise ValueError(f"expected L={self.cfg.context_len}, got {L}")

        h = self.in_proj(x) + self.pos_emb[:, :L, :]
        h = self.decoder(h, mask=self._causal_mask(L, x.device))
        return self.out_proj(h)   # (B, L, 1)

    def predict_next(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference-only: returns the single next-step forecast.

        Args:
            x: (B, L, 1)

        Returns:
            (B, 1) — forecast for timestep L+1
        """
        return self.forward(x)[:, -1, :]   # (B, 1)