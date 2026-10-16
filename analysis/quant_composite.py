"""
analysis/quant_composite.py
Composite quantitative scoring engine.

Combines: mean reversion (Zeta/OU/Hurst/ADF), basis spread, volume
anomaly, derivatives (CVD/OI/L-S), and momentum into a single
normalized score in [−1, +1].
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.mean_reversion import MeanReversionAnalyzer, MeanReversionResult
from analysis.basis import BasisAnalyzer, BasisResult
from analysis.volume_anomaly import VolumeAnomalyAnalyzer, VolumeAnomalyResult

logger = logging.getLogger(__name__)


@dataclass
class CompositeQuantScore:
    mean_reversion: MeanReversionResult = field(default_factory=MeanReversionResult)
    basis: BasisResult = field(default_factory=BasisResult)
    volume: VolumeAnomalyResult = field(default_factory=VolumeAnomalyResult)
    derivative_score: float = 0.0
    total_score: float = 0.0
    direction: str = "neutral"
    conviction: str = "low"
    weights: Dict[str, float] = field(default_factory=dict)
    agreement_ratio: float = 0.0
    conflicting_components: List[str] = field(default_factory=list)


class CompositeQuantEngine:
    DEFAULT_WEIGHTS = {
        "mean_reversion": 0.25,
        "basis": 0.15,
        "volume": 0.15,
        "derivatives": 0.30,
        "momentum": 0.15,
    }
    NORMALIZERS = {
        "mean_reversion": 15.0,
        "basis": 10.0,
        "volume": 10.0,
        "derivatives": 25.0,
        "momentum": 10.0,
    }

    def __init__(self, client, weights: Optional[Dict[str, float]] = None):
        self.client = client
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.mr_analyzer = MeanReversionAnalyzer()
        self.basis_analyzer = BasisAnalyzer(client)
        self.vol_analyzer = VolumeAnomalyAnalyzer()
        s = sum(self.weights.values())
        if abs(s - 1.0) > 1e-6:
            for k in self.weights:
                self.weights[k] /= s

    def _momentum_contribution(self, indicators: Dict) -> float:
        score = 0.0
        ema_9 = indicators.get("ema_9", 0)
        ema_21 = indicators.get("ema_21", 0)
        rsi = indicators.get("rsi", 50)
        if ema_9 > ema_21:
            score += 4.0
        elif ema_9 < ema_21:
            score -= 4.0
        if 40 <= rsi <= 60:
            score += 1.0
        elif rsi < 30:
            score += 2.0
        elif rsi > 70:
            score -= 2.0
        rvol = indicators.get("rvol", 1.0)
        score += np.clip((rvol - 1.0) * 2, -4, 4)
        return float(np.clip(score, -10, 10))

    async def compute(
        self,
        symbol: str,
        df: pd.DataFrame,
        indicators: Dict,
        derivative_snapshot,
        bar_interval_hours: float = 1.0,
    ) -> CompositeQuantScore:
        out = CompositeQuantScore(weights=self.weights.copy())
        if df is None or len(df) < 60:
            return out
        closes = df["close"].astype(float)

        out.mean_reversion = self.mr_analyzer.analyze(
            closes, bar_interval_hours=bar_interval_hours,
        )
        try:
            out.basis = await self.basis_analyzer.compute(symbol)
        except Exception as e:
            logger.debug(f"Basis compute failed for {symbol}: {e}")
        out.volume = self.vol_analyzer.analyze(df)
        out.derivative_score = getattr(
            derivative_snapshot, "confidence_contribution", 0.0,
        )
        momentum_contrib = self._momentum_contribution(indicators)

        mr_norm = out.mean_reversion.confidence_contribution / self.NORMALIZERS["mean_reversion"]
        basis_norm = out.basis.confidence_contribution / self.NORMALIZERS["basis"]
        vol_norm = out.volume.confidence_contribution / self.NORMALIZERS["volume"]
        deriv_norm = out.derivative_score / self.NORMALIZERS["derivatives"]
        mom_norm = momentum_contrib / self.NORMALIZERS["momentum"]

        out.total_score = float(np.clip(
            self.weights["mean_reversion"] * mr_norm
            + self.weights["basis"] * basis_norm
            + self.weights["volume"] * vol_norm
            + self.weights["derivatives"] * deriv_norm
            + self.weights["momentum"] * mom_norm,
            -1.0, 1.0,
        ))

        if out.total_score > 0.20:
            out.direction = "long"
        elif out.total_score < -0.20:
            out.direction = "short"

        abs_score = abs(out.total_score)
        if abs_score >= 0.55:
            out.conviction = "high"
        elif abs_score >= 0.30:
            out.conviction = "medium"

        component_signs = [
            np.sign(mr_norm), np.sign(basis_norm), np.sign(vol_norm),
            np.sign(deriv_norm), np.sign(mom_norm),
        ]
        nonzero = [s for s in component_signs if s != 0]
        if nonzero:
            dominant = np.sign(out.total_score)
            agreeing = sum(1 for s in nonzero if s == dominant)
            out.agreement_ratio = agreeing / len(nonzero)

        names = ["mean_reversion", "basis", "volume", "derivatives", "momentum"]
        for name, s in zip(names, component_signs):
            if s != 0 and s != np.sign(out.total_score):
                out.conflicting_components.append(name)

        return out