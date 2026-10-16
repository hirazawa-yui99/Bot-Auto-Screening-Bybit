"""
analysis/volume_anomaly.py
Volume normalization and anomaly detection.

Metrics
-------
1. RVOL            = V_t / SMA(V, N)
2. Robust Z-score  = (V_t − median) / (1.4826 · MAD)
3. Percentile rank = P(V_t > baseline)
4. VWPD            = corr(sign(Δprice), Δvol_z)
   < −0.3 → weak rally (bearish divergence)
   > +0.3 and Z > 0 → strong rally (bullish)
"""
import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VolumeAnomalyResult:
    latest_volume: float = 0.0
    baseline_mean: float = 0.0
    baseline_median: float = 0.0
    baseline_mad: float = 0.0
    rvol: float = 1.0
    z_score: float = 0.0
    percentile_rank: float = 0.0
    anomaly_level: str = "normal"
    vwpd: float = 0.0
    divergence: str = "none"
    confidence_contribution: float = 0.0


class VolumeAnomalyAnalyzer:
    def __init__(self, baseline_window: int = 50, min_samples: int = 20):
        self.baseline_window = baseline_window
        self.min_samples = min_samples

    def compute_rvol(self, volumes: pd.Series) -> float:
        if len(volumes) < self.min_samples:
            return 1.0
        baseline = volumes.iloc[-(self.baseline_window + 1):-1]
        if len(baseline) == 0 or baseline.mean() <= 0:
            return 1.0
        return float(volumes.iloc[-1] / baseline.mean())

    def compute_robust_zscore(
        self, volumes: pd.Series,
    ) -> Tuple[float, float, float, float]:
        if len(volumes) < self.min_samples:
            return 0.0, 0.0, 0.0, 0.0
        baseline = volumes.iloc[-(self.baseline_window + 1):-1].astype(float)
        latest = float(volumes.iloc[-1])
        median = float(baseline.median())
        mad = float((baseline - median).abs().median())
        scaled_mad = 1.4826 * mad if mad > 0 else float(baseline.std())
        z = (latest - median) / scaled_mad if scaled_mad > 1e-9 else 0.0
        return float(z), median, mad, float(baseline.mean())

    def compute_percentile_rank(self, volumes: pd.Series) -> float:
        if len(volumes) < self.min_samples:
            return 0.5
        baseline = volumes.iloc[-(self.baseline_window + 1):-1].astype(float)
        latest = float(volumes.iloc[-1])
        return float((baseline < latest).mean())

    def compute_vwpd(
        self, prices: pd.Series, volumes: pd.Series, lookback: int = 10,
    ) -> float:
        if len(prices) < lookback + 5 or len(volumes) < lookback + 5:
            return 0.0
        price_chg = prices.diff().iloc[-lookback:].values
        vol = volumes.iloc[-lookback:].values
        vol_z = (vol - vol.mean()) / (vol.std() + 1e-9)
        if len(price_chg) != len(vol_z):
            n = min(len(price_chg), len(vol_z))
            price_chg = price_chg[-n:]
            vol_z = vol_z[-n:]
        if len(price_chg) < 4:
            return 0.0
        sign_p = np.sign(price_chg)
        if sign_p.std() < 1e-9 or vol_z.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(sign_p, vol_z)[0, 1])

    def analyze(self, df: pd.DataFrame) -> VolumeAnomalyResult:
        result = VolumeAnomalyResult()
        if df is None or len(df) < self.min_samples + 5:
            return result
        volumes = df["volume"].astype(float)
        prices = df["close"].astype(float)
        result.latest_volume = float(volumes.iloc[-1])
        result.rvol = self.compute_rvol(volumes)
        z, median, mad, mean = self.compute_robust_zscore(volumes)
        result.z_score = z
        result.baseline_median = median
        result.baseline_mad = mad
        result.baseline_mean = mean
        result.percentile_rank = self.compute_percentile_rank(volumes)
        result.vwpd = self.compute_vwpd(prices, volumes)

        if z >= 3.0 or result.percentile_rank >= 0.98:
            result.anomaly_level = "extreme_high"
        elif z >= 2.0 or result.percentile_rank >= 0.90:
            result.anomaly_level = "high"
        elif z <= -2.0 or result.percentile_rank <= 0.05:
            result.anomaly_level = "extreme_low"
        elif z <= -1.5 or result.percentile_rank <= 0.15:
            result.anomaly_level = "low"

        if result.vwpd < -0.3:
            result.divergence = "bearish"
        elif result.vwpd > 0.3 and result.z_score > 0:
            result.divergence = "bullish"

        score = 0.0
        if result.anomaly_level == "extreme_high":
            score += 6.0
        elif result.anomaly_level == "high":
            score += 4.0
        elif result.anomaly_level == "extreme_low":
            score -= 3.0
        elif result.anomaly_level == "low":
            score -= 2.0
        if result.divergence == "bullish":
            score += 4.0
        elif result.divergence == "bearish":
            score -= 4.0
        result.confidence_contribution = float(np.clip(score, -10, 10))
        return result