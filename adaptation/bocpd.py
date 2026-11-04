"""
Bayesian Online Changepoint Detection (Adams & MacKay, 2007).

Uses a Normal-Gamma conjugate prior with a Gaussian likelihood model.
At each step maintains a distribution over run-lengths r_t (steps since
the last switch).

Detection signal: the negative log-predictive score
    score_t = -log P(x_t | x_{1:t-1})
This is the marginal predictive probability of x_t under the current
run-length distribution. It is HIGH when x_t is inconsistent with the
current model — i.e. right after a regime switch — and LOW during stable
segments. Using this instead of P(r_t=0), which is always mathematically
equal to the hazard rate H and carries no detection information.

Threshold: fire when score_t >= threshold (threshold is on the negative
log-predictive scale; a value of ~2 is a moderate alarm, ~5 is strong).
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional


class BOCPD:
    """
    Online Bayesian changepoint detector.

    Parameters
    ----------
    hazard_lambda : float
        Expected number of steps between switches (geometric hazard
        H(r) = 1 / hazard_lambda).
    mu0 : float
        Prior mean for the Gaussian likelihood.
    kappa0 : float
        Prior pseudo-count on the mean.
    alpha0 : float
        Prior shape for the precision (Gamma).
    beta0 : float
        Prior rate for the precision (Gamma).
    threshold : float
        Fire when -log P(x_t | x_{1:t-1}) >= threshold.
        Typical range: 1.5 (sensitive) to 5.0 (conservative).
    """

    def __init__(
        self,
        hazard_lambda: float = 100.0,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        threshold: float = 3.0,
    ):
        self.hazard_lambda = hazard_lambda
        self.threshold = threshold

        # Normal-Gamma hyperparameters — one set per run-length hypothesis.
        # At t=0 there is one hypothesis: r=0 (just switched), weight=1.
        self._mu    = np.array([mu0],    dtype=np.float64)
        self._kappa = np.array([kappa0], dtype=np.float64)
        self._alpha = np.array([alpha0], dtype=np.float64)
        self._beta  = np.array([beta0],  dtype=np.float64)
        self._log_R = np.array([0.0],    dtype=np.float64)  # log run-length probs

    # ------------------------------------------------------------------
    def _hazard(self, r: int) -> float:
        return 1.0 / self.hazard_lambda

    def _predictive_log_prob(self, x: float) -> np.ndarray:
        """
        Log predictive probability of x under each run-length hypothesis.
        Student-t predictive: t_{2*alpha}(mu, beta*(kappa+1)/(alpha*kappa))
        """
        df   = 2.0 * self._alpha
        loc  = self._mu
        scale = np.sqrt(self._beta * (self._kappa + 1.0) / (self._alpha * self._kappa))

        # log Student-t density
        from scipy.special import gammaln
        lp = (gammaln((df + 1.0) / 2.0)
              - gammaln(df / 2.0)
              - 0.5 * np.log(df * np.pi)
              - np.log(scale)
              - ((df + 1.0) / 2.0) * np.log1p(((x - loc) / scale) ** 2 / df))
        return lp

    def _update_params(self, x: float) -> None:
        """Update Normal-Gamma sufficient statistics for each hypothesis."""
        kappa_new = self._kappa + 1.0
        mu_new    = (self._kappa * self._mu + x) / kappa_new
        alpha_new = self._alpha + 0.5
        beta_new  = (self._beta
                     + (self._kappa * (x - self._mu) ** 2) / (2.0 * kappa_new))
        self._kappa = kappa_new
        self._mu    = mu_new
        self._alpha = alpha_new
        self._beta  = beta_new

    def update(self, x: float) -> tuple[float, bool]:
        """
        Process one observation x.

        Returns
        -------
        neg_log_pred : float
            -log P(x_t | x_{1:t-1}): the negative marginal log-predictive.
            HIGH when x is inconsistent with the current model (i.e. after a
            switch); LOW during stable segments.  Use this as the detection
            score and fire when it exceeds self.threshold.
        fired : bool
            True when neg_log_pred >= self.threshold.

        Note on P(r_t=0): the posterior probability of r_t=0 (just switched)
        is mathematically always equal to the prior hazard H = 1/hazard_lambda,
        regardless of the data, because the normalization cancels the marginal
        predictive.  It is NOT useful as a detection signal.
        """
        log_pred = self._predictive_log_prob(x)

        # Marginal log-predictive: used as detection score
        log_marginal = np.logaddexp.reduce(self._log_R + log_pred)
        neg_log_pred = float(-log_marginal)

        # Growth probabilities: existing run-lengths grow by 1
        log_R_grow = self._log_R + log_pred + np.log(1.0 - self._hazard(0))

        # Changepoint probability: all mass collapses to r=0
        log_cp = np.logaddexp.reduce(self._log_R + log_pred + np.log(self._hazard(0)))

        # New log run-length distribution (prepend r=0)
        self._log_R = np.concatenate([[log_cp], log_R_grow])

        # Update Normal-Gamma params for growth hypotheses; add fresh prior for r=0
        self._update_params(x)
        # Prepend fresh hypothesis for the new changepoint (r=0)
        self._mu    = np.concatenate([[0.0],    self._mu])
        self._kappa = np.concatenate([[1.0],    self._kappa])
        self._alpha = np.concatenate([[1.0],    self._alpha])
        self._beta  = np.concatenate([[1.0],    self._beta])

        # Normalise log_R
        log_norm = np.logaddexp.reduce(self._log_R)
        self._log_R -= log_norm

        # Trim to avoid unbounded growth (keep at most 1000 hypotheses)
        if len(self._log_R) > 1000:
            self._log_R = self._log_R[:1000]
            self._mu    = self._mu[:1000]
            self._kappa = self._kappa[:1000]
            self._alpha = self._alpha[:1000]
            self._beta  = self._beta[:1000]

        fired = neg_log_pred >= self.threshold
        return neg_log_pred, fired

    def reset(self) -> None:
        self._mu    = np.array([0.0])
        self._kappa = np.array([1.0])
        self._alpha = np.array([1.0])
        self._beta  = np.array([1.0])
        self._log_R = np.array([0.0])

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold


# ---------------------------------------------------------------------------
# Convenience functions matching the CUSUM interface
# ---------------------------------------------------------------------------

def bocpd_score_series(
    y: np.ndarray,
    hazard_lambda: float = 100.0,
    warmup: int = 0,
) -> np.ndarray:
    """
    Run BOCPD on a 1-D series and return per-timestep detection scores.

    Score is -log P(x_t | x_{1:t-1}): the negative marginal log-predictive.
    High when x_t is inconsistent with the current run-length model, i.e.
    immediately after a regime switch.

    Parameters
    ----------
    y : (T,) float array
    hazard_lambda : expected steps between switches
    warmup : first `warmup` steps are assigned score 0 (not used for detection)
    """
    detector = BOCPD(hazard_lambda=hazard_lambda)
    scores = np.zeros(len(y), dtype=np.float64)
    for t, yt in enumerate(y):
        neg_log_pred, _ = detector.update(float(yt))
        if t >= warmup:
            scores[t] = neg_log_pred
    return scores


def bocpd_detection_delays(
    y: np.ndarray,
    states: np.ndarray,
    hazard_lambda: float = 100.0,
    threshold: float = 0.5,
    max_delay: int = 50,
) -> List[Dict[str, Any]]:
    """
    Run BOCPD on series y and measure detection delay for each true switch.

    Parameters
    ----------
    y : (T,) observed series (val split)
    states : (T,) int array of ground-truth regime labels
    hazard_lambda : expected steps between switches
    threshold : fire when P(r_t=0) >= threshold
    max_delay : cap delay at this value if BOCPD never fires

    Returns
    -------
    List of dicts with keys: switch_position, fire_position, delay_steps
    """
    switch_pts = [t for t in range(1, len(states))
                  if states[t] != states[t - 1]]
    if not switch_pts:
        return []

    detector = BOCPD(hazard_lambda=hazard_lambda, threshold=threshold)
    fire_times: List[int] = []

    for t, yt in enumerate(y):
        neg_log_pred, fired = detector.update(float(yt))
        if fired:
            fire_times.append(t)

    results = []
    fire_idx = 0
    for s in switch_pts:
        # Advance fire_idx to first fire >= s
        while fire_idx < len(fire_times) and fire_times[fire_idx] < s:
            fire_idx += 1
        if fire_idx < len(fire_times) and fire_times[fire_idx] <= s + max_delay:
            fp = fire_times[fire_idx]
            results.append({
                "switch_position": s,
                "fire_position": fp,
                "delay_steps": fp - s,
            })
        else:
            results.append({
                "switch_position": s,
                "fire_position": None,
                "delay_steps": max_delay,
            })
    return results
