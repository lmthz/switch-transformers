from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_forecaster import TransformerConfig, CausalTransformerForecaster


class ChangeAwareTransformer(CausalTransformerForecaster):
    """
    Extends CausalTransformerForecaster with a learned attention decay for
    positions before a detected change point.

    Adds one learnable scalar log_gamma. For positions j before the change
    point k, attention logits are reduced by gamma * (k - j):

        mask[i, j] = 0                       if j >= k  (post-switch, attend freely)
        mask[i, j] = -gamma * (k - j)        if j <  k  (pre-switch, penalised)

    where gamma = softplus(log_gamma) > 0.  Combined with the standard causal
    upper-triangular -inf mask, so causality is preserved.

    change_point_idx = 0  →  no decay (equivalent to base transformer)
    change_point_idx = k  →  positions 0..k-1 get distance-proportional penalty

    During training:  change_point_idx comes from ground-truth switch positions.
    During inference: change_point_idx comes from a CUSUM detector.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__(cfg)
        # Initialised small so training starts near the base transformer.
        # softplus(-5) ≈ 0.007, so initial decay is nearly zero.
        self.log_gamma = nn.Parameter(torch.full((1,), -5.0))

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.log_gamma)

    def _build_decay_mask(
        self,
        L: int,
        change_point_idx: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build a batched float attention mask combining causal -inf and distance
        decay for pre-switch positions.

        change_point_idx: (B,) int64 — per-example switch position in [0, L]
        Returns: (B * n_heads, L, L) float mask to be added to attention logits.
        """
        B = change_point_idx.shape[0]
        gamma = self.gamma  # scalar

        j = torch.arange(L, device=device, dtype=torch.float32)  # (L,)
        cp = change_point_idx.float().to(device)                  # (B,)

        # decay[b, j] = -gamma * max(0, cp[b] - j)
        decay = -gamma * torch.clamp(cp.unsqueeze(1) - j.unsqueeze(0), min=0.0)  # (B, L)

        # Same decay value for every query position i → broadcast to (B, L, L)
        decay_3d = decay.unsqueeze(1).expand(B, L, L)  # (B, L, L)

        # Causal part: -inf where j > i (identical for every example)
        causal = torch.triu(
            torch.full((L, L), float("-inf"), device=device), diagonal=1
        )  # (L, L)

        combined = causal.unsqueeze(0) + decay_3d  # (B, L, L)

        # nn.TransformerEncoder expects mask shape (B * n_heads, L, L).
        # .contiguous() is required before .reshape because .expand gives a
        # non-contiguous view (stride-0 along the nhead dim).
        nhead = self.cfg.n_heads
        return combined.unsqueeze(1).expand(B, nhead, L, L).contiguous().reshape(B * nhead, L, L)

    def forward(
        self,
        x: torch.Tensor,
        change_point_idx=None,
        return_hidden: bool = False,
    ):
        """
        Args:
            x:                (B, L, 1)
            change_point_idx: int, scalar tensor, (B,) int tensor, or None.
                              0 / None → standard causal mask, no decay.
            return_hidden:    if True, also return (B, L, d_model) hidden states.
        """
        B, L, _ = x.shape
        if L > self.cfg.context_len:
            raise ValueError(
                f"context_len {L} exceeds model context_len {self.cfg.context_len}"
            )

        h = self.in_proj(x) + self.pos_emb[:, -L:, :]

        # Resolve change_point_idx to (B,) int64
        if change_point_idx is None:
            cp = None
        elif isinstance(change_point_idx, (int, float)):
            cp = torch.full((B,), int(change_point_idx), dtype=torch.long, device=x.device)
        else:
            cp = change_point_idx.to(device=x.device, dtype=torch.long)
            if cp.dim() == 0:
                cp = cp.unsqueeze(0).expand(B)

        # Use float decay mask when cp is provided, bool causal mask otherwise.
        # Avoids a CPU-GPU sync (int(cp.max())) in the training hot path.
        if cp is None:
            mask = self._causal_mask(L, x.device)
        else:
            mask = self._build_decay_mask(L, cp, x.device)

        h = self.decoder(h, mask=mask)
        out = self.out_proj(h)

        if return_hidden:
            return out, h
        return out

    def predict_next_aware(
        self,
        x: torch.Tensor,
        change_point_idx=None,
    ) -> torch.Tensor:
        """
        Inference wrapper: returns (B, 1) next-step forecast with decay mask.
        Equivalent to forward(x, change_point_idx)[:, -1, :].
        """
        return self.forward(x, change_point_idx=change_point_idx)[:, -1, :]
