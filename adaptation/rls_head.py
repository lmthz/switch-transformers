from __future__ import annotations

import numpy as np


class RLSHead:
    """
    Online Recursive Least Squares prediction head.

    Maintains a weight vector theta and covariance matrix P that are
    updated after each observation using the RLS equations with a
    forgetting factor lambda. No gradient descent — closed-form update.

    Feature vector: transformer hidden state h (d_features,) with bias
    appended → x = [h, 1], shape (d_features+1,).

    Prediction: y_hat = theta @ x
    Update:
        K_t = P_{t-1} x_t / (lambda + x_t^T P_{t-1} x_t)
        theta_t = theta_{t-1} + K_t (y_t - x_t^T theta_{t-1})
        P_t = lambda^{-1} (P_{t-1} - K_t x_t^T P_{t-1})
    """

    def __init__(self, d_features: int, forgetting_factor: float = 0.95):
        self.lam = float(forgetting_factor)
        d = d_features + 1  # +1 for bias
        self.theta = np.zeros(d, dtype=np.float64)
        self.P = np.eye(d, dtype=np.float64) * 1000.0  # large initial uncertainty

    def predict(self, h: np.ndarray) -> float:
        """h: (d_features,) -> scalar prediction."""
        x = np.append(h.astype(np.float64), 1.0)
        return float(self.theta @ x)

    def update(self, h: np.ndarray, y_true: float) -> float:
        """
        Predict, then update weights given the true value.
        Returns the prediction made *before* seeing y_true.
        """
        x = np.append(h.astype(np.float64), 1.0)
        y_hat = float(self.theta @ x)

        # RLS update
        Px = self.P @ x
        denom = self.lam + float(x @ Px)
        K = Px / denom
        error = float(y_true) - y_hat
        self.theta = self.theta + K * error
        self.P = (self.P - np.outer(K, x @ self.P)) / self.lam

        return y_hat

    def reset_uncertainty(self):
        """
        Increase P to signal high uncertainty about current regime.
        Call this when a change point is detected so the head adapts
        faster to new dynamics.
        """
        d = len(self.theta)
        self.P = np.eye(d, dtype=np.float64) * 1000.0

    def reset(self):
        """Full reset: zero weights, maximum uncertainty."""
        d = len(self.theta)
        self.theta = np.zeros(d, dtype=np.float64)
        self.P = np.eye(d, dtype=np.float64) * 1000.0
