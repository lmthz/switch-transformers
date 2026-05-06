# ms_timeseries_generator.py
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

Array = np.ndarray


# ---------------- Regime definition ----------------
@dataclass
class RegimeARMA:
    ar: Array                      # nonseasonal AR coeffs (length p)
    ma: Optional[Array]            # nonseasonal MA coeffs (length q) or None
    sigma: float                   # innovation std
    name: Optional[str] = None
    # --- SARIMAX extras (all optional) ---
    sar: Optional[Array] = None    # seasonal AR (length P)
    sma: Optional[Array] = None    # seasonal MA (length Q)
    beta: Optional[Array] = None   # exogenous coefficients (length d_exog)


# ---------------- Generator ----------------
class MSSwitchGenerator:
    def __init__(self, save_dir: str = "generated_data", seed: Optional[int] = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)
        print(f"[init] save_dir={self.save_dir.resolve()} seed={seed}")

    # ---- helpers ----
    @staticmethod
    def _check_ar_ma_orders(regimes: List[RegimeARMA]) -> Tuple[int, int]:
        ps = {len(r.ar) for r in regimes}
        qs = {0 if r.ma is None else len(r.ma) for r in regimes}
        if len(ps) != 1 or len(qs) != 1:
            raise ValueError("All regimes must share the same nonseasonal AR/MA orders.")
        p, q = ps.pop(), qs.pop()
        print(f"[check] nonseasonal orders: p={p}, q={q}")
        return p, q

    def _sample_markov_chain(self, T: Array, n: int, init: Optional[int] = None) -> Array:
        print(f"[markov] sampling {n} states")
        T = np.asarray(T, dtype=float)
        if not np.allclose(T.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("Each row of T must sum to 1.")
        K = T.shape[0]
        # stationary start
        if init is None:
            eigvals, eigvecs = np.linalg.eig(T.T)
            stat = np.real(eigvecs[:, np.isclose(eigvals, 1.0)]).flatten()
            stat = np.abs(stat) / np.sum(np.abs(stat))
            k_prev = self.rng.choice(K, p=stat)
        else:
            k_prev = int(init)
        states = np.empty(n, dtype=int)
        for t in range(n):
            states[t] = k_prev
            k_prev = self.rng.choice(K, p=T[k_prev])
        print(f"[markov] done (K={K})")
        return states

    @staticmethod
    def _seasonal_diff(x: Array, s: int, D: int) -> Array:
        y = np.asarray(x, dtype=float)
        for _ in range(D):
            y = y[s:] - y[:-s]
        return y

    @staticmethod
    def _seasonal_integrate(z: Array, s: int, D: int, n_out: int) -> Array:
        """Invert seasonal differencing by repeated seasonal cumulative sums."""
        y = np.zeros(n_out, dtype=float)
        # Simple reconstruction: place z and accumulate seasonally
        y[:len(z)] = z
        for _ in range(D):
            for t in range(s, n_out):
                y[t] += y[t - s]
        return y

    # ---- plotting ----
    def _plot_series(self, y: Array, states: Array, title: str, png_name: str):
        plt.figure(figsize=(11, 4))
        plt.plot(y, label="series")
        # scale and overlay states
        scale = (np.max(np.abs(y)) or 1.0) * 0.5
        plt.scatter(range(len(states)), states * scale, c=states, s=6, label="state")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        out_path = self.save_dir / png_name
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"[plot] saved {out_path.resolve()}")

    # ---------------- core simulators ----------------
    def simulate_ar(
        self,
        regimes: List[RegimeARMA],
        T: Array,
        n: int,
        burn_in: int,
        save_name: str,
        plot: bool = True,
        init_state: Optional[int] = None,
        states_override: Optional[Array] = None,
    ):
        """Simulate pure AR(p) Markov-switching process."""
        p, q = self._check_ar_ma_orders(regimes)
        if q != 0:
            raise ValueError("Use simulate_arma() if MA terms are nonzero.")

        total = n + burn_in

        if states_override is not None:
            states = np.asarray(states_override, dtype=int)
            if len(states) != total:
                raise ValueError("states_override must have length n + burn_in.")
        else:
            states = self._sample_markov_chain(T, total, init=init_state)

        y = np.zeros(total)
        eps = np.zeros(total)

        start = p
        for t in range(start, total):
            r = regimes[states[t]]
            eps[t] = self.rng.normal(scale=r.sigma)
            ar_part = np.dot(r.ar, y[t - np.arange(1, p + 1)]) if p else 0.0
            y[t] = ar_part + eps[t]

        y = y[burn_in:]
        states = states[burn_in:]

        # save metadata for baseline/analysis
        ar_mat = np.stack([r.ar for r in regimes])
        sigma_vec = np.array([r.sigma for r in regimes])

        if save_name is not None:
            np.savez(
                self.save_dir / f"{save_name}.npz",
                y=y,
                states=states,
                T=T,
                ar=ar_mat,
                sigma=sigma_vec,
            )
            if plot:
                self._plot_series(y, states, save_name, f"{save_name}_plot.png")
            print(f"[save] {save_name}.npz done")
        else:
            if plot:
                self._plot_series(y, states, "AR_series", "ar_series_plot.png")

        return {"y": y, "states": states, "T": T}

    def simulate_arma(
        self,
        regimes: List[RegimeARMA],
        T: Array,
        n: int,
        burn_in: int,
        save_name: str,
        plot: bool = True,
    ):
        """Simulate ARMA(p,q) Markov-switching process."""
        p, q = self._check_ar_ma_orders(regimes)
        states = self._sample_markov_chain(T, n + burn_in)
        y = np.zeros(n + burn_in)
        eps = np.zeros(n + burn_in)
        start = max(p, q)
        for t in range(start, n + burn_in):
            r = regimes[states[t]]
            eps[t] = self.rng.normal(scale=r.sigma)
            ar_part = np.dot(r.ar, y[t - np.arange(1, p + 1)]) if p else 0.0
            ma_part = np.dot(r.ma, eps[t - np.arange(1, q + 1)]) if q else 0.0
            y[t] = ar_part + ma_part + eps[t]

        y = y[burn_in:]
        states = states[burn_in:]
        eps_eff = eps[burn_in:]

        ar_mat = np.stack([r.ar for r in regimes])
        sigma_vec = np.array([r.sigma for r in regimes])
        if q > 0:
            ma_mat = np.stack([r.ma for r in regimes])
        else:
            ma_mat = None

        if save_name is not None:
            save_kwargs: Dict[str, Any] = dict(
                y=y,
                states=states,
                T=T,
                ar=ar_mat,
                sigma=sigma_vec,
            )
            if ma_mat is not None:
                save_kwargs["ma"] = ma_mat
                save_kwargs["eps"] = eps_eff
            np.savez(self.save_dir / f"{save_name}.npz", **save_kwargs)
            if plot:
                self._plot_series(y, states, save_name, f"{save_name}_plot.png")
            print(f"[save] {save_name}.npz done")
        else:
            if plot:
                self._plot_series(y, states, "AR_series", "ar_series_plot.png")

        return {"y": y, "states": states, "T": T}

    def simulate_arima(self, regimes: List[RegimeARMA], T: Array,
                    n: int, d: int, burn_in: int, save_name: str, plot: bool = True):
        """Simulate ARIMA(p,d,q) by differencing/integrating ARMA innovations."""
        p, q = self._check_ar_ma_orders(regimes)
        states = self._sample_markov_chain(T, n + burn_in)
        z = np.zeros(n + burn_in)
        eps = np.zeros(n + burn_in)
        start = max(p, q)
        for t in range(start, n + burn_in):
            r = regimes[states[t]]
            eps[t] = self.rng.normal(scale=r.sigma)
            ar_part = np.dot(r.ar, z[t - np.arange(1, p + 1)]) if p else 0.0
            ma_part = np.dot(r.ma, eps[t - np.arange(1, q + 1)]) if q else 0.0
            z[t] = ar_part + ma_part + eps[t]

        # integrate to get y
        y = np.cumsum(z, axis=0)
        for _ in range(d - 1):
            y = np.cumsum(y, axis=0)

        # drop burn-in
        y = y[burn_in:]
        z_out = z[burn_in:]
        eps_out = eps[burn_in:]
        states = states[burn_in:]

        ar_mat = np.stack([r.ar for r in regimes])
        sigma_vec = np.array([r.sigma for r in regimes])

        # optional: stack ma if present (q>0)
        if q > 0:
            ma_mat = np.stack([r.ma for r in regimes])
        else:
            ma_mat = None

        if save_name is not None:
            save_kwargs = dict(
                y=y,
                z=z_out,
                eps=eps_out,
                states=states,
                T=T,
                ar=ar_mat,
                sigma=sigma_vec,
                d=d,
            )
            if ma_mat is not None:
                save_kwargs["ma"] = ma_mat

            np.savez(self.save_dir / f"{save_name}.npz", **save_kwargs)

            if plot:
                self._plot_series(y, states, save_name, f"{save_name}_plot.png")
            print(f"[save] {save_name}.npz done")
        else:
            if plot:
                self._plot_series(y, states, "AR_series", "ar_series_plot.png")

        return {"y": y, "states": states, "T": T, "d": d}

    def simulate_sarimax(
        self,
        regimes: List[RegimeARMA],
        T: Array,
        n: int,
        p: int,
        d: int,
        q: int,
        P: int,
        D: int,
        Q: int,
        s: int,
        X: Optional[Array],
        burn_in: int,
        save_name: str,
        plot: bool = True,
    ):
        """Simulate SARIMAX with optional exogenous regressors."""
        states = self._sample_markov_chain(T, n + burn_in)
        y = np.zeros(n + burn_in)
        eps = np.zeros(n + burn_in)
        start = max(p, q, s * P if P > 0 else 0, s * Q if Q > 0 else 0)
        for t in range(start, n + burn_in):
            r = regimes[states[t]]
            eps[t] = self.rng.normal(scale=r.sigma)
            ar_part  = np.dot(r.ar,  y[t - np.arange(1, p + 1)]) if p else 0.0
            ma_part  = np.dot(r.ma,  eps[t - np.arange(1, q + 1)]) if q else 0.0
            sar_part = np.dot(r.sar, y[t - s * np.arange(1, P + 1)]) if (r.sar is not None and P) else 0.0
            sma_part = np.dot(r.sma, eps[t - s * np.arange(1, Q + 1)]) if (r.sma is not None and Q) else 0.0
            x_part = 0.0
            if X is not None and r.beta is not None:
                x_curr = X[t] if X.ndim == 1 else X[t, :]
                x_part = float(np.dot(r.beta.ravel(), np.atleast_1d(x_curr)).sum())
            y[t] = ar_part + ma_part + sar_part + sma_part + x_part + eps[t]

        # integration if D>0 or d>0
        if D > 0:
            y = self._seasonal_integrate(y, s, D, len(y))
        if d > 0:
            for _ in range(d):
                y = np.cumsum(y)
        y = y[burn_in:]
        states = states[burn_in:]

        ar_mat = np.stack([r.ar for r in regimes])
        sigma_vec = np.array([r.sigma for r in regimes])

        if save_name is not None:
            np.savez(
                self.save_dir / f"{save_name}.npz",
                y=y,
                states=states,
                T=T,
                ar=ar_mat,
                sigma=sigma_vec,
                orders=(p, d, q, P, D, Q, s),
            )
            if plot:
                self._plot_series(y, states, save_name, f"{save_name}_plot.png")
            print(f"[save] {save_name}.npz done")
        else:
            if plot:
                self._plot_series(y, states, "AR_series", "ar_series_plot.png")

        return {"y": y, "states": states, "T": T, "orders": (p, d, q, P, D, Q, s)}

    # ---------------- dataset menu (extended) ----------------
    def make_datasets_menu(self, n: int = 1000, burn: int = 600, suffix: str = "_r0"):
        """
        Generate an extended suite of Markov-switching datasets. All artifacts are saved
        under self.save_dir with consistent names and plots.

        Scenarios:
          A1  AR(2) CoeffsOnly (easy)
          A2  AR(2) CoeffsOnly (harder / closer roots)
          A3  AR(2) Coeffs+Var (coefficients AND sigma change)
          B1  AR(2) VarianceOnly
          B2  AR(2) VarianceOnly (bigger contrast)
          C1  ARMA(2,1) Coeffs+Var
          D1  ARIMA(2,1,1)
          D2  ARIMA(2,2,1)
          D3  ARIMA(2,1,0) (integrated AR, no MA)
          E1  DriftOnly via exogenous constant (β switch)
          E2  LevelShiftOnly via exogenous constant (β changes sign)
          F1  Seasonal SARIMAX (2,0,1)x(1,1,1)s=24 (seasonal AR/MA change)
          F2  Seasonal+Exog (sin, cos) with regime-specific β
          G1  ExogenousOnly (no MA; AR identical; β on sin)
          H1  HighOrder AR(p=10) CoeffsOnly
          H2  Near-Unit-Root stress (AR(1) ~ 0.98 vs 0.90)
          S1  SparseSwitching (long sojourns)
          S2  FrequentSwitching (short sojourns)
          NS0/NS1  No-switch cases (always regime 0 or 1)
          SW1       Single-switch trajectory (half 0, half 1)
        """

        print("\n[menu] Generating extended dataset menu...\n")

        # Base transitions
        T_base = np.array([[0.97, 0.03],
                           [0.08, 0.92]])   # avg runs ~33 and ~12.5
        T_sparse   = np.array([[0.992, 0.008],
                               [0.02,  0.98 ]])  # very long runs
        T_frequent = np.array([[0.85, 0.15],
                               [0.20, 0.80]])    # frequent switching
        # No-switch transition
        T_no_switch = np.array([[1.0, 0.0],
                                [0.0, 1.0]])

        # ---------------- A1: AR(2) CoeffsOnly (easy, well-separated) ----------------
        regA1 = RegimeARMA(ar=np.array([0.6, -0.2]), ma=None, sigma=0.30, name="calm")
        regA2 = RegimeARMA(ar=np.array([1.2, -0.6]), ma=None, sigma=0.30, name="volatile")
        self.simulate_ar([regA1, regA2], T_base, n=n, burn_in=burn,
                         save_name=f"A1_ar2_coeffs_easy{suffix}", plot=True)

        # ---------------- A2: AR(2) CoeffsOnly (harder, closer) ----------------
        regA3 = RegimeARMA(ar=np.array([0.75, -0.25]), ma=None, sigma=0.30, name="r1")
        regA4 = RegimeARMA(ar=np.array([0.85, -0.30]), ma=None, sigma=0.30, name="r2")
        self.simulate_ar([regA3, regA4], T_base, n=n, burn_in=burn,
                         save_name=f"A2_ar2_coeffs_hard{suffix}", plot=True)

        # ---------------- A3: AR(2) Coeffs+Var (coefficients AND sigma change) ----------------
        regA5 = RegimeARMA(ar=np.array([0.65, -0.15]), ma=None, sigma=0.20, name="coeffs_var_low")
        regA6 = RegimeARMA(ar=np.array([1.05, -0.55]), ma=None, sigma=0.60, name="coeffs_var_high")
        self.simulate_ar([regA5, regA6], T_base, n=n, burn_in=burn,
                         save_name=f"A3_ar2_coeffs_plus_var{suffix}", plot=True)

        # ---------------- B1: AR(2) VarianceOnly ----------------
        regB1 = RegimeARMA(ar=np.array([0.8, -0.3]), ma=None, sigma=0.20, name="lowvar")
        regB2 = RegimeARMA(ar=np.array([0.8, -0.3]), ma=None, sigma=0.60, name="highvar")
        self.simulate_ar([regB1, regB2], T_base, n=n, burn_in=burn,
                         save_name=f"B1_ar2_variance{suffix}", plot=True)

        # ---------------- B2: AR(2) VarianceOnly (bigger contrast) ----------------
        regB3 = RegimeARMA(ar=np.array([0.8, -0.3]), ma=None, sigma=0.10, name="verylow")
        regB4 = RegimeARMA(ar=np.array([0.8, -0.3]), ma=None, sigma=0.80, name="veryhigh")
        self.simulate_ar([regB3, regB4], T_base, n=n, burn_in=burn,
                         save_name=f"B2_ar2_variance_big{suffix}", plot=True)

        # ---------------- C1: ARMA(2,1) Coeffs+Var ----------------
        regC1 = RegimeARMA(ar=np.array([0.6, -0.2]), ma=np.array([0.3]), sigma=0.25, name="c1")
        regC2 = RegimeARMA(ar=np.array([1.2, -0.6]), ma=np.array([0.2]), sigma=0.55, name="c2")
        self.simulate_arma([regC1, regC2], T_base, n=n, burn_in=burn,
                           save_name=f"C1_arma21_coeffs_var{suffix}", plot=True)

        # ---------------- D1: ARIMA(2,1,1) ----------------
        self.simulate_arima([regC1, regC2], T_base, n=n, d=1, burn_in=burn,
                            save_name=f"D1_arima211{suffix}", plot=True)

        # ---------------- D2: ARIMA(2,2,1) (more integration) ----------------
        self.simulate_arima([regC1, regC2], T_base, n=n, d=2, burn_in=burn,
                            save_name=f"D2_arima221{suffix}", plot=True)

        # ---------------- D3: ARIMA(2,1,0) (integrated AR, no MA) ----------------
        regD3_1 = RegimeARMA(ar=np.array([0.6, -0.2]), ma=None, sigma=0.30, name="ar_int_r1")
        regD3_2 = RegimeARMA(ar=np.array([1.0, -0.5]), ma=None, sigma=0.30, name="ar_int_r2")
        self.simulate_arima([regD3_1, regD3_2], T_base, n=n, d=1, burn_in=burn,
                            save_name=f"D3_arima210{suffix}", plot=True)

        # ---------------- E1: DriftOnly via X=1 (β switch, piecewise linear) ----------------
        regE1 = RegimeARMA(ar=np.array([0.8, -0.3]), ma=None, sigma=0.30,
                           beta=np.array([0.00]), name="flat")
        regE2 = RegimeARMA(ar=np.array([0.8, -0.3]), ma=None, sigma=0.30,
                           beta=np.array([0.02]), name="uptrend")
        X_const = np.ones(n + burn)
        self.simulate_sarimax([regE1, regE2], T_base, n=n,
                              p=2, d=0, q=0, P=0, D=0, Q=0, s=1,
                              X=X_const, burn_in=burn,
                              save_name=f"E1_drift_only{suffix}", plot=True)

        # ---------------- E2: LevelShiftOnly via X=1 (β changes sign) ----------------
        regE3 = RegimeARMA(ar=np.array([0.7, -0.25]), ma=None, sigma=0.30,
                           beta=np.array([+0.02]), name="pos")
        regE4 = RegimeARMA(ar=np.array([0.7, -0.25]), ma=None, sigma=0.30,
                           beta=np.array([-0.02]), name="neg")
        self.simulate_sarimax([regE3, regE4], T_base, n=n,
                              p=2, d=0, q=0, P=0, D=0, Q=0, s=1,
                              X=X_const, burn_in=burn,
                              save_name=f"E2_level_shift{suffix}", plot=True)

        # ---------------- F1: Seasonal SARIMAX (2,0,1)x(1,1,1)_{24} ----------------
        s = 24
        regF1 = RegimeARMA(ar=np.array([0.6, -0.2]), ma=np.array([0.3]), sigma=0.25,
                           sar=np.array([0.3]), sma=np.array([0.2]), name="seasonal_soft")
        regF2 = RegimeARMA(ar=np.array([0.6, -0.2]), ma=np.array([0.3]), sigma=0.25,
                           sar=np.array([0.1]), sma=np.array([0.4]), name="seasonal_strong")
        self.simulate_sarimax([regF1, regF2], T_base, n=n,
                              p=2, d=0, q=1, P=1, D=1, Q=1, s=s,
                              X=None, burn_in=burn,
                              save_name=f"F1_seasonal_sarimax{suffix}", plot=True)

        # ---------------- F2: Seasonal+Exog (sin, cos) with regime β ----------------
        t_full = np.arange(n + burn)
        X_trig = np.stack(
            [np.sin(2 * np.pi * t_full / s), np.cos(2 * np.pi * t_full / s)],
            axis=1
        )  # (n+burn, 2)
        # Same AR/MA across regimes; only beta differs. Shrink beta + sigma to avoid explosions.
        regF3 = RegimeARMA(ar=np.array([0.7, -0.25]), ma=np.array([0.2]), sigma=0.20,
                           sar=np.array([0.2]), sma=np.array([0.2]),
                           beta=np.array([0.5, 0.0]), name="sin-dominant")
        regF4 = RegimeARMA(ar=np.array([0.7, -0.25]), ma=np.array([0.2]), sigma=0.20,
                           sar=np.array([0.2]), sma=np.array([0.2]),
                           beta=np.array([0.0, 0.5]), name="cos-dominant")
        self.simulate_sarimax([regF3, regF4], T_base, n=n,
                              p=2, d=0, q=1, P=1, D=1, Q=1, s=s,
                              X=X_trig, burn_in=burn,
                              save_name=f"F2_seasonal_exog{suffix}", plot=True)

        # ---------------- G1: ExogenousOnly (AR same; β on sin only) ----------------
        X_sin = np.sin(2 * np.pi * t_full / s)  # (n+burn,)
        regG1 = RegimeARMA(ar=np.array([0.8, -0.3]), ma=None, sigma=0.25,
                           beta=np.array([0.0]), name="no-exog")
        regG2 = RegimeARMA(ar=np.array([0.8, -0.3]), ma=None, sigma=0.25,
                           beta=np.array([0.8]), name="exog-on")
        self.simulate_sarimax([regG1, regG2], T_base, n=n,
                              p=2, d=0, q=0, P=0, D=0, Q=0, s=1,
                              X=X_sin, burn_in=burn,
                              save_name=f"G1_exogenous_only{suffix}", plot=True)

        # ---------------- H1: HighOrder AR(p=10) CoeffsOnly ----------------
        ar10_r1 = np.array([0.45, -0.3, 0.2, -0.12, 0.08, -0.06, 0.05, -0.04, 0.03, -0.02])
        ar10_r2 = np.array([0.60, -0.40, 0.22, -0.14, 0.10, -0.08, 0.06, -0.05, 0.04, -0.03])
        regH1 = RegimeARMA(ar=ar10_r1, ma=None, sigma=0.30, name="p10_r1")
        regH2 = RegimeARMA(ar=ar10_r2, ma=None, sigma=0.30, name="p10_r2")
        self.simulate_ar([regH1, regH2], T_base, n=n, burn_in=burn,
                         save_name=f"H1_ar10_coeffs{suffix}", plot=True)

        # ---------------- H2: Near-Unit-Root stress (AR(1)) ----------------
        regH3 = RegimeARMA(ar=np.array([0.98]), ma=None, sigma=0.20, name="unitish")
        regH4 = RegimeARMA(ar=np.array([0.90]), ma=None, sigma=0.20, name="less-persistent")
        self.simulate_ar([regH3, regH4], T_base, n=n, burn_in=burn,
                         save_name=f"H2_ar1_near_unit_root{suffix}", plot=True)

        # ---------------- S1: SparseSwitching on AR(2) CoeffsOnly ----------------
        self.simulate_ar([regA1, regA2], T_sparse, n=n, burn_in=burn,
                         save_name=f"S1_sparse_switching{suffix}", plot=True)

        # ---------------- S2: FrequentSwitching on AR(2) CoeffsOnly ----------------
        self.simulate_ar([regA1, regA2], T_frequent, n=n, burn_in=burn,
                         save_name=f"S2_frequent_switching{suffix}", plot=True)

        # ---------------- NS0/NS1: No-switch cases (always regime 0 or 1) ----------------
        # Always in regime 0 (using A1 params)
        self.simulate_ar([regA1, regA2], T_no_switch, n=n, burn_in=burn,
                         save_name=f"NS0_A1_no_switch_regime0{suffix}", plot=True, init_state=0)
        # Always in regime 1 (using A1 params)
        self.simulate_ar([regA1, regA2], T_no_switch, n=n, burn_in=burn,
                         save_name=f"NS1_A1_no_switch_regime1{suffix}", plot=True, init_state=1)

        # ---------------- SW1: Single-switch trajectory (half 0, half 1) ----------------
        total = n + burn
        states_single = np.zeros(total, dtype=int)
        split = total // 2
        states_single[split:] = 1
        self.simulate_ar(
            [regA1, regA2],
            T_base,
            n=n,
            burn_in=burn,
            save_name=f"SW1_A1_single_switch{suffix}",
            plot=True,
            states_override=states_single,
        )

        print(f"\n[menu] All datasets (suffix={suffix}) saved to {self.save_dir.resolve()}\n")


# ---------------- driver (runs all and saves plots) ----------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_instances", type=int, default=30,
                    help="Number of instances per dataset type (default 30). "
                         "Each instance uses a different RNG seed, giving different "
                         "regime sequences but the same model parameters.")
    ap.add_argument("--n", type=int, default=1000, help="Series length (default 1000)")
    ap.add_argument("--burn", type=int, default=600, help="Burn-in (default 600)")
    args = ap.parse_args()

    # Seeds for each instance — well-separated so regime sequences differ clearly
    seeds = [42 + i * 100 for i in range(args.n_instances)]
    print(f"Generating {args.n_instances} instance(s) per dataset type")
    print(f"Seeds: {seeds}")
    print("Files named: <dataset>_r0.npz, <dataset>_r1.npz, ... (r0 = original)\n")

    for i, seed in enumerate(seeds):
        print(f"\n=== Instance r{i} (seed={seed}) ===")
        gen = MSSwitchGenerator(save_dir=f"generated_data", seed=seed)
        gen.make_datasets_menu(n=args.n, burn=args.burn, suffix=f"_r{i}")

if __name__ == "__main__":
    main()