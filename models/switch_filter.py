"""
Bayesian switch filter (HMM forward algorithm) for regime detection.

Takes per-regime log-likelihoods at each step and maintains a posterior
distribution over which regime is currently active.

Also provides:
  ar_log_lik()           — exact AR(p) Gaussian one-step log-likelihood (oracle)
  transformer_log_lik()  — approximate Gaussian log-lik from transformer prediction
  filter_detection_scores() — convert posterior trajectories to [0,1] detection scores
"""
from __future__ import annotations
import numpy as np


class SwitchFilter:
    """
    Log-space HMM forward algorithm for K-regime switching.

    Maintains log α_t[k] = log P(z_t=k, x_{1:t}), normalized so
    exp(log_alpha) = P(z_t=k | x_{1:t}) after each update.

    Detection signal: 1 - max(posterior), high when the filter is uncertain
    about which regime is active, peaking immediately after a regime switch.
    """

    def __init__(self, log_T: np.ndarray):
        """
        log_T : (K, K) where log_T[i,j] = log P(z_t=j | z_{t-1}=i)
        """
        self.log_T = log_T
        K = log_T.shape[0]
        self.log_alpha = np.full(K, -np.log(K), dtype=np.float64)

    def update(self, log_liks: np.ndarray) -> np.ndarray:
        """
        Process one step's per-regime log-likelihoods.

        log_liks : (K,) log P(x_t | x_{1:t-1}, z_t=k) for each regime k.
        Returns posterior (K,) = P(z_t=k | x_{1:t}).
        """
        # Predict: Chapman-Kolmogorov
        # log_alpha_pred[j] = logsumexp_i(log_alpha[i] + log_T[i,j])
        # log_T shape (K,K), log_T[:, j] = column j = log P(z_t=j | z_{t-1}=i) for all i
        log_alpha_pred = np.logaddexp.reduce(
            self.log_alpha[:, None] + self.log_T, axis=0
        )  # (K,)

        # Update: Bayes' rule
        log_alpha_new = log_alpha_pred + log_liks

        # Normalize
        log_norm = np.logaddexp.reduce(log_alpha_new)
        self.log_alpha = log_alpha_new - log_norm

        return np.exp(self.log_alpha)

    def reset(self) -> None:
        K = self.log_T.shape[0]
        self.log_alpha = np.full(K, -np.log(K), dtype=np.float64)


def build_log_T(T: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Convert a stochastic transition matrix T to log scale."""
    return np.log(np.clip(T, eps, 1.0))


def ar_log_lik(
    y_t: float,
    context: np.ndarray,
    ar_coeffs: np.ndarray,
    sigma_std: float,
) -> float:
    """
    Exact AR(p) one-step Gaussian log-likelihood in the STANDARDIZED space.

    y_t       : standardized observed value at time t
    context   : recent standardized observations, len >= p; uses the last p
    ar_coeffs : AR coefficients for this regime, shape (p,)
    sigma_std : innovation std for this regime IN THE STANDARDIZED SPACE
                (= raw sigma / stdzr_std, since AR coeffs are scale-invariant)
    """
    p = len(ar_coeffs)
    # Generation convention: ar[0]*y[t-1] + ar[1]*y[t-2] (newest lag first).
    # context[-p:] is ordered oldest→newest, so reverse before dot-product.
    pred = float(ar_coeffs @ context[-p:][::-1])
    return float(
        -0.5 * np.log(2.0 * np.pi)
        - np.log(sigma_std)
        - (y_t - pred) ** 2 / (2.0 * sigma_std ** 2)
    )


def transformer_log_lik(
    y_t: float,
    pred_t: float,
    sigma_hat: float,
) -> float:
    """
    Approximate Gaussian log-likelihood using a transformer point prediction.

    y_t      : standardized observed value
    pred_t   : transformer's standardized one-step-ahead prediction
    sigma_hat: estimated noise std from training residuals (standardized space)
    """
    return float(
        -0.5 * np.log(2.0 * np.pi)
        - np.log(sigma_hat)
        - (y_t - pred_t) ** 2 / (2.0 * sigma_hat ** 2)
    )


def filter_detection_scores(posteriors: np.ndarray) -> np.ndarray:
    """
    Convert posterior trajectories (T, K) to detection scores (T,) in [0, 1].

    Score = 1 - max(posterior): high when the filter is uncertain about the
    active regime, which peaks immediately after a regime switch.
    """
    return 1.0 - posteriors.max(axis=1)
