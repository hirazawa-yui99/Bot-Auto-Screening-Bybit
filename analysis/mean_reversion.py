"""
analysis/mean_reversion.py
Mean-reversion quant suite: Zeta (ζ), Ornstein-Uhlenbeck, Hurst, ADF.

Formulas
--------
1. OU / AR(1) fit:  X_t = α + φ·X_{t−1} + ε
   θ = −ln(φ)              (reversion speed)
   HL = ln(2) / θ           (half-life in bars)
   μ = α / (1 − φ)          (long-run mean)

2. Zeta (reversion-adjusted Z-score):
   z = (X_t − μ) / σ
   ζ = z × (1 − φ) / (1 + φ)
   |ζ| > 1.5 → actionable reversion signal

3. Hurst exponent (R/S):  H < 0.45 MR, H > 0.55 trending

4. ADF test:  p < 0.10 → stationary → reversion valid
"""
import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MeanReversionResult:
    zeta: float = 0.0
    z_score: float = 0.0
    phi: float = 0.0
    theta: float = 0.0
    mu: float = 0.0
    sigma: float = 0.0
    half_life_bars: float = 0.0
    half_life_hours: float = 0.0
    hurst: float = 0.5
    regime: str = "random"
    adf_pvalue: float = 1.0
    is_stationary: bool = False
    directional_bias: str = "neutral"
    confidence_contribution: float = 0.0


class MeanReversionAnalyzer:
    def __init__(self, min_samples: int = 60):
        self.min_samples = min_samples

    def fit_ou(self, series: pd.Series) -> Tuple[float, float, float, float]:
        if len(series) < self.min_samples:
            return np.nan, np.nan, np.nan, np.nan
        x = series.values.astype(float)
        x_lag, x_now = x[:-1], x[1:]
        x_lag_mean, x_now_mean = x_lag.mean(), x_now.mean()
        denom = np.sum((x_lag - x_lag_mean) ** 2)
        if denom < 1e-12:
            return np.nan, np.nan, np.nan, np.nan
        phi = np.sum((x_lag - x_lag_mean) * (x_now - x_now_mean)) / denom
        alpha = x_now_mean - phi * x_lag_mean
        sigma = float((x_now - (alpha + phi * x_lag)).std())
        if phi >= 1.0 or phi <= 0.0:
            return float(phi), float(alpha), np.nan, sigma
        mu = alpha / (1.0 - phi)
        return float(phi), float(alpha), float(mu), sigma

    def compute_half_life(self, phi: float) -> float:
        if phi >= 1.0 or phi <= 0.0:
            return float("inf")
        theta = -np.log(phi)
        return float(np.log(2) / theta) if theta > 0 else float("inf")

    def compute_hurst(self, series: pd.Series, max_lag: int = 40) -> float:
        x = series.values.astype(float)
        n = len(x)
        if n < 40:
            return 0.5
        lags = np.unique(
            np.floor(np.logspace(
                np.log10(8), np.log10(min(max_lag, n // 2)), 12
            )).astype(int)
        )
        lags = lags[lags >= 8]
        if len(lags) < 4:
            return 0.5
        rs_values = []
        for lag in lags:
            n_chunks = n // lag
            if n_chunks < 1:
                continue
            rs_chunk = []
            for i in range(n_chunks):
                chunk = x[i * lag:(i + 1) * lag]
                dev = chunk - chunk.mean()
                cumdev = np.cumsum(dev)
                R = cumdev.max() - cumdev.min()
                S = chunk.std()
                if S > 0:
                    rs_chunk.append(R / S)
            if rs_chunk:
                rs_values.append((lag, np.mean(rs_chunk)))
        if len(rs_values) < 4:
            return 0.5
        lags_arr = np.log([v[0] for v in rs_values])
        rs_arr = np.log([v[1] for v in rs_values])
        slope, _ = np.polyfit(lags_arr, rs_arr, 1)
        return float(np.clip(slope, 0.0, 1.0))

    def adf_test(
        self, series: pd.Series, max_lag: int = None,
    ) -> Tuple[float, float]:
        x = series.values.astype(float)
        n = len(x)
        if n < 30:
            return 0.0, 1.0
        if max_lag is None:
            max_lag = int(np.floor(12 * (n / 100) ** 0.25))
        max_lag = max(1, min(max_lag, n // 4))
        delta_x = np.diff(x)
        x_lag = x[:-1]
        rows = n - 1 - max_lag
        if rows < 10:
            return 0.0, 1.0
        Y = delta_x[max_lag:]
        X = np.column_stack([
            np.ones(rows),
            x_lag[max_lag:],
            *[delta_x[max_lag - i - 1: -i - 1] if i > 0
              else delta_x[max_lag - 1: -1] for i in range(max_lag)]
        ])
        try:
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
            resid = Y - X @ beta
            dof = rows - X.shape[1]
            if dof <= 0:
                return 0.0, 1.0
            s2 = (resid @ resid) / dof
            cov = s2 * np.linalg.pinv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            t_stat = beta[1] / se[1] if se[1] > 0 else 0.0
        except Exception:
            return 0.0, 1.0
        if t_stat < -3.43:
            p = 0.01
        elif t_stat < -2.86:
            p = 0.05
        elif t_stat < -2.57:
            p = 0.10
        else:
            p = min(0.99, 0.10 + (t_stat + 2.57) * 0.1)
        return float(t_stat), float(p)

    def compute_zeta(
        self, series: pd.Series, phi: float, mu: float, sigma: float,
    ) -> Tuple[float, float]:
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
            return 0.0, 0.0
        x_last = float(series.iloc[-1])
        z = (x_last - mu) / sigma
        if not (0 < phi < 1):
            return float(z), 0.0
        factor = (1.0 - phi) / (1.0 + phi)
        return float(z), float(z * factor)

    def analyze(
        self, series: pd.Series, bar_interval_hours: float = 1.0,
    ) -> MeanReversionResult:
        result = MeanReversionResult()
        if len(series) < self.min_samples:
            return result

        phi, alpha, mu, sigma = self.fit_ou(series)
        if not np.isfinite(phi):
            return result

        result.phi = phi
        if 0 < phi < 1:
            result.theta = float(-np.log(phi))
            result.half_life_bars = self.compute_half_life(phi)
            result.half_life_hours = (
                result.half_life_bars * bar_interval_hours
            )
        result.mu = mu if np.isfinite(mu) else 0.0
        result.sigma = sigma if np.isfinite(sigma) else 0.0
        result.z_score, result.zeta = self.compute_zeta(
            series, phi, result.mu, result.sigma,
        )
        result.hurst = self.compute_hurst(series)
        _, result.adf_pvalue = self.adf_test(series)
        result.is_stationary = result.adf_pvalue < 0.10

        if result.hurst < 0.45 and result.is_stationary:
            result.regime = "mean_reverting"
        elif result.hurst > 0.55:
            result.regime = "trending"
        else:
            result.regime = "random"

        if result.regime == "mean_reverting" and result.is_stationary:
            if result.zeta > 1.5:
                result.directional_bias = "revert_short"
            elif result.zeta < -1.5:
                result.directional_bias = "revert_long"

        score = 0.0
        if result.regime == "mean_reverting":
            score += min(abs(result.zeta), 3.0) * 4.0 * np.sign(-result.zeta)
        elif result.regime == "trending":
            score += 3.0
        if not result.is_stationary:
            score *= 0.4
        result.confidence_contribution = float(np.clip(score, -15, 15))
        return result