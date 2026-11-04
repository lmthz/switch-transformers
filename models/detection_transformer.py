"""
ComparisonDetectionTransformer — causal transformer with a comparison-based
detection head that scores how likely a regime switch occurred recently.

Architecture
------------
Backbone : CausalTransformerForecaster (unchanged)
Detection head : compares mean of last split_k hidden states ("recent") with
                 mean of earlier hidden states ("historical") using a 2-layer MLP
                 on [recent, hist, recent-hist, recent*hist].

The detection score is a raw scalar logit (apply sigmoid for probability).
During training it is supervised with soft-decayed targets via MSE loss.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from models.transformer_forecaster import CausalTransformerForecaster, TransformerConfig


class ComparisonDetectionTransformer(CausalTransformerForecaster):
    """
    Extends CausalTransformerForecaster with a comparison-based detection head.

    Parameters
    ----------
    cfg : TransformerConfig
    split_k : int
        Number of trailing context positions treated as "recent" (default 8).
        Remaining L-split_k positions are "historical".
        A switch that occurred <=split_k steps ago will appear in the recent
        window, making recent and historical encode different regimes.
    """

    def __init__(self, cfg: TransformerConfig, split_k: int = 8):
        super().__init__(cfg)
        self.split_k = split_k
        d = cfg.d_model
        self.detection_head = nn.Sequential(
            nn.Linear(4 * d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward_detection(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both next-step predictions and detection score.

        Parameters
        ----------
        x : (B, L, 1) input context

        Returns
        -------
        pred  : (B, L, 1) next-step predictions (same as base forward)
        score : (B, 1)    raw detection logit (sigmoid gives P(switch recently))
        """
        pred, h = super().forward(x, return_hidden=True)   # h: (B, L, d_model)

        B, L, d = h.shape
        k = min(self.split_k, L - 1)   # guard against very short contexts

        recent = h[:, -k:, :].mean(dim=1)          # (B, d)
        hist   = h[:, :-k, :].mean(dim=1)          # (B, d)

        feat  = torch.cat([recent, hist, recent - hist, recent * hist], dim=-1)  # (B, 4d)
        score = self.detection_head(feat)            # (B, 1)

        return pred, score

    def detection_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: returns sigmoid(score) — P(switch recently) in [0,1].
        Input: (B, L, 1). Output: (B, 1).
        """
        _, score = self.forward_detection(x)
        return torch.sigmoid(score)
