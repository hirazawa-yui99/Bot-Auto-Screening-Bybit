"""
analysis/quant_composite.py
Composite quantitative scoring engine.

Combines: mean reversion, basis spread, volume anomaly,
derivatives (CVD/OI/L-S), funding rate, and momentum into a
single normalized score in [−1, +1].

Weights are merged over DEFAULT_WEIGHTS in __init__, so any new
component you add later cannot trigger a KeyError on old configs.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.mean_reversion import MeanReversionAnalyzer, MeanReversionResult
from analysis.basis import BasisAnalyzer, BasisResult
from analysis.volume_anomaly import VolumeAnomalyAnalyzer, VolumeAnomalyResult
from analysis.funding import FundingAnalyzer, FundingResult

logger = logging.getLogger(__name__)


@dataclass
class CompositeQuantScore:
    mean_reversion: MeanReversionResult = field(default_factory=MeanReversionResult)
    basis: BasisResult = field(default_factory=BasisResult)
    volume: VolumeAnomalyResult = field(default_factory=VolumeAnomalyResult)
    funding: FundingResult = field(default_factory=FundingResult)
    derivative_score: float = 0.0
    total_score: float = 0.0
    direction: str = "neutral"
    conviction: str = "low"
    weights: Dict[str, float] = field(default_factory=dict)
    agreement_ratio: float = 0.0
    conflicting_components: List[str] = field(default_factory=list)


class CompositeQuantEngine:
    DEFAULT_WEIGHTS = {
        "mean_reversion": 0.20,
        "basis": 0.10,
        "volume": 0.15,
        "derivatives": 0.25,
        "funding": 0.15,
        "momentum": 0.15,
    }
    NORMALIZERS = {
        "mean_reversion": 15.0,
        "basis": 10.0,
        "volume": 10.0,
        "derivatives": 25.0,
        "funding": 15.0,
        "momentum": 10.0,
    }

    def __init__(
        self,
        client,
        weights: Optional[Dict[str, float]] = None,
        funding_config: Optional[Dict] = None,
    ):
        self.client = client

        # Merge user-provided weights over defaults so any missing
        # key (e.g. "funding" on old configs) is backfilled silently.
        merged = self.DEFAULT_WEIGHTS.copy()
        if weights:
            for k, v in weights.items():
                if k in merged:
                    try:
                        merged[k] = float(v)
                    except (TypeError, ValueError):
                        logger.warning(
                            f"Ignoring invalid weight '{k}'='{v}'"
                        )
                else:
                    logger.warning(
                        f"Ignoring unknown weight key '{k}' (not in "
                        f"DEFAULT_WEIGHTS)"
                    )

        # Renormalize
        s = sum(merged.values())
        if s <= 0:
            logger.error("All composite weights are zero — using defaults")
            merged = self.DEFAULT_WEIGHTS.copy()
            s = sum(merged.values())
        if abs(s - 1.0) > 1e-6:
            logger.info(f"Composite weights sum to {s:.4f}, renormalizing")
            for k in merged:
                merged[k] /= s

        self.weights = merged

        self.mr_analyzer = MeanReversionAnalyzer()
        self.basis_analyzer = BasisAnalyzer(client)
        self.vol_analyzer = VolumeAnomalyAnalyzer()
        self.funding_analyzer = FundingAnalyzer(funding_config)

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

        # 1. Mean reversion
        try:
            out.mean_reversion = self.mr_analyzer.analyze(
                closes, bar_interval_hours=bar_interval_hours,
            )
        except Exception as e:
            logger.debug(f"{symbol}: mean reversion failed: {e}")

        # 2. Basis
        try:
            out.basis = await self.basis_analyzer.compute(symbol)
        except Exception as e:
            logger.debug(f"{symbol}: basis compute failed: {e}")

        # 3. Volume anomaly
        try:
            out.volume = self.vol_analyzer.analyze(df)
        except Exception as e:
            logger.debug(f"{symbol}: volume anomaly failed: {e}")

        # 4. Funding
        try:
            out.funding = await self.funding_analyzer.analyze(
                symbol, self.client,
            )
        except Exception as e:
            logger.debug(f"{symbol}: funding compute failed: {e}")

        # 5. Derivatives
        out.derivative_score = getattr(
            derivative_snapshot, "confidence_contribution", 0.0,
        )

        # 6. Momentum
        momentum_contrib = self._momentum_contribution(indicators)

        # Normalize to [-1, +1]
        mr_norm = (
            out.mean_reversion.confidence_contribution
            / self.NORMALIZERS["mean_reversion"]
        )
        basis_norm = (
            out.basis.confidence_contribution
            / self.NORMALIZERS["basis"]
        )
        vol_norm = (
            out.volume.confidence_contribution
            / self.NORMALIZERS["volume"]
        )
        deriv_norm = (
            out.derivative_score
            / self.NORMALIZERS["derivatives"]
        )
        funding_norm = (
            out.funding.confidence_contribution
            / self.NORMALIZERS["funding"]
        )
        mom_norm = (
            momentum_contrib
            / self.NORMALIZERS["momentum"]
        )

        out.total_score = float(np.clip(
            self.weights["mean_reversion"] * mr_norm
            + self.weights["basis"] * basis_norm
            + self.weights["volume"] * vol_norm
            + self.weights["derivatives"] * deriv_norm
            + self.weights["funding"] * funding_norm
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
            np.sign(mr_norm),
            np.sign(basis_norm),
            np.sign(vol_norm),
            np.sign(deriv_norm),
            np.sign(funding_norm),
            np.sign(mom_norm),
        ]
        nonzero = [s for s in component_signs if s != 0]
        if nonzero:
            dominant = np.sign(out.total_score)
            agreeing = sum(1 for s in nonzero if s == dominant)
            out.agreement_ratio = agreeing / len(nonzero)

        names = [
            "mean_reversion", "basis", "volume",
            "derivatives", "funding", "momentum",
        ]
        for name, s in zip(names, component_signs):
            if s != 0 and s != np.sign(out.total_score):
                out.conflicting_components.append(name)

        return out