"""
analysis/confidence.py
Calibrated confidence scorer with fallacy guards and bootstrap mode.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceBreakdown:
    technical_score: float = 0.0
    composite_score: float = 0.0
    derivative_score: float = 0.0
    sentiment_score: float = 0.0
    pattern_score: float = 0.0
    llm_score: float = 0.0
    raw_total: float = 0.0
    calibrated_confidence: float = 0.0
    is_high_confidence: bool = False
    gate_reasons: List[str] = field(default_factory=list)


class ConfidenceScorer:
    WEIGHT_TECHNICAL = 20
    WEIGHT_COMPOSITE = 20
    WEIGHT_DERIVATIVE = 20
    WEIGHT_SENTIMENT = 15
    WEIGHT_PATTERN = 15
    WEIGHT_LLM = 10

    def __init__(
        self,
        min_raw_threshold: float = 42.0,
        min_calibrated: float = 0.42,
        min_live_samples: int = 30,
        shrinkage_prior: float = 0.5,
    ):
        self.min_raw_threshold = min_raw_threshold
        self.min_calibrated = min_calibrated
        self.min_live_samples = min_live_samples
        self.shrinkage_prior = shrinkage_prior
        self.calibration_map: Dict[str, float] = {}
        self._outcome_history: List[Dict] = []

    def check_hard_gates(
        self,
        symbol: str,
        deriv_snapshot,
        klines_df,
        bias: str = "",
        funding_result=None,
    ) -> Tuple[bool, List[str]]:
        reasons = []

        # ---- Funding gate (direction-aware) ----
        if funding_result is not None:
            # Use the analyzer's directional gate
            from analysis.funding import FundingAnalyzer
            reason = FundingAnalyzer.get_gate_reason(funding_result, bias)
            if reason:
                reasons.append(reason)
        else:
            # Fallback: old behavior, but direction-aware
            rate = deriv_snapshot.funding_rate
            if bias == "long" and rate > 0.0005:
                reasons.append(
                    f"Funding {rate:.4%} extreme positive — "
                    f"longs crowded, long blocked"
                )
            elif bias == "short" and rate < -0.0005:
                reasons.append(
                    f"Funding {rate:.4%} extreme negative — "
                    f"shorts crowded, short blocked"
                )

        # ---- Liquidity gate ----
        if klines_df is not None and len(klines_df) >= 24:
            avg_vol = klines_df["volume"].tail(24).mean()
            if avg_vol < 1_000_000:
                reasons.append(f"Low liquidity: avg 24h vol = {avg_vol:,.0f}")

        # ---- RSI gate ----
        if klines_df is not None and len(klines_df) >= 14:
            closes = klines_df["close"].tail(15).values
            delta = np.diff(closes)
            gain = np.where(delta > 0, delta, 0).mean()
            loss = np.where(delta < 0, -delta, 0).mean()
            rsi = 100 - (100 / (1 + gain / loss)) if loss > 0 else 100
            if rsi > 85:
                reasons.append(f"RSI overbought: {rsi:.1f}")

        # ---- OI anomaly gate ----
        if abs(deriv_snapshot.oi_change_1h) > 15.0:
            reasons.append(
                f"OI anomaly: {deriv_snapshot.oi_change_1h:.1f}% in 1h"
            )

        return len(reasons) == 0, reasons

    def score_technical(self, klines_df, indicators: Dict) -> float:
        if klines_df is None or len(klines_df) < 50:
            return 0.0
        score = 0.0
        close = klines_df["close"].iloc[-1]
        ema_9 = indicators.get("ema_9", close)
        ema_21 = indicators.get("ema_21", close)
        ema_50 = indicators.get("ema_50", close)
        if ema_9 > ema_21 > ema_50:
            score += 8
        elif ema_9 < ema_21 < ema_50:
            score += 8
        elif ema_9 > ema_21:
            score += 4
        elif ema_9 < ema_21:
            score += 4
        rsi = indicators.get("rsi", 50)
        if 40 <= rsi <= 60:
            score += 5
        elif rsi < 30:
            score += 6
        elif rsi > 70:
            score += 3
        rvol = indicators.get("rvol", 1.0)
        if rvol >= 2.0:
            score += 7
        elif rvol >= 1.5:
            score += 4
        elif rvol >= 1.0:
            score += 2
        return float(np.clip(score, 0, self.WEIGHT_TECHNICAL))

    def score_composite_quant(self, composite) -> float:
        if composite is None:
            return 0.0
        magnitude = abs(composite.total_score)
        agreement = composite.agreement_ratio
        conviction_mult = {"low": 0.5, "medium": 0.8, "high": 1.0}[composite.conviction]
        score = magnitude * agreement * conviction_mult * self.WEIGHT_COMPOSITE
        return float(np.clip(score, 0, self.WEIGHT_COMPOSITE))

    def score_derivatives(self, deriv_snapshot) -> float:
        return float(np.clip(
            (deriv_snapshot.confidence_contribution + 25) / 50 * self.WEIGHT_DERIVATIVE,
            0, self.WEIGHT_DERIVATIVE,
        ))

    def score_sentiment(self, deriv_snapshot) -> float:
        score = 0.0
        if deriv_snapshot.account_ls_ratio > 2.0:
            score += 2
        elif deriv_snapshot.account_ls_ratio < 0.5:
            score += 2
        if deriv_snapshot.smart_money_bias in ("long", "short"):
            score += 10
        if deriv_snapshot.ls_ratio_divergence != "none":
            score += 8
        return float(np.clip(score, 0, self.WEIGHT_SENTIMENT))

    def score_patterns(self, patterns_detected: List[str], bias: str) -> float:
        if not patterns_detected:
            return 0.0
        strong = {
            "double_bottom": ("Long", 12), "double_top": ("Short", 12),
            "bull_flag": ("Long", 10), "bear_flag": ("Short", 10),
            "ascending_triangle": ("Long", 8), "descending_triangle": ("Short", 8),
            "head_and_shoulders": ("Short", 12),
            "inverse_head_and_shoulders": ("Long", 12),
        }
        score = 0.0
        for p in patterns_detected:
            if p in strong:
                d, pts = strong[p]
                if d.lower() == bias.lower():
                    score += pts
        return float(np.clip(score, 0, self.WEIGHT_PATTERN))

    def score_llm(self, llm_response: Dict) -> float:
        if not llm_response:
            return 0.0
        # Neutral placeholder from disabled LLM gets a neutral score
        if llm_response.get("_llm_unavailable"):
            return 5.0
        score = 5.0
        if llm_response.get("agrees_with_quant"):
            score += 3
        if llm_response.get("invalidation_conditions"):
            score += 2
        if llm_response.get("risk_notes"):
            score += 1
        if llm_response.get("direction") and llm_response.get("quant_direction"):
            if llm_response["direction"] != llm_response["quant_direction"]:
                score -= 3
        return float(np.clip(score, 0, self.WEIGHT_LLM))

    def _bucket_key(self, raw_score: float) -> str:
        return f"{int(raw_score // 10) * 10}-{int(raw_score // 10) * 10 + 10}"

    def calibrate(self, raw_score: float) -> float:
        """
        Calibrate raw score → confidence using historical outcomes.

        Bootstrap (n == 0): use raw_score / 100 capped at 0.85.
        Shrinkage (0 < n < 30): blend historical WR with raw proxy.
        Full (n >= 30): historical win rate only.
        """
        bucket = self._bucket_key(raw_score)
        bucket_outcomes = [
            o for o in self._outcome_history
            if self._bucket_key(o["raw_score"]) == bucket
        ]
        n = len(bucket_outcomes)
        raw_proxy = min(0.85, max(0.30, raw_score / 100.0))

        if n == 0:
            logger.debug(
                f"Calibration bootstrap (n=0): raw={raw_score:.1f} "
                f"→ confidence={raw_proxy:.2f}"
            )
            return raw_proxy

        historical_wr = sum(1 for o in bucket_outcomes if o["won"]) / n

        if n < self.min_live_samples:
            alpha = n / self.min_live_samples
            blended = alpha * historical_wr + (1 - alpha) * raw_proxy
            logger.debug(
                f"Calibration blend (n={n}): historical={historical_wr:.2f}, "
                f"raw_proxy={raw_proxy:.2f} → {blended:.2f}"
            )
            return blended

        return historical_wr

    def record_outcome(self, raw_score: float, won: bool, metadata: Dict = None):
        self._outcome_history.append({
            "raw_score": raw_score,
            "won": won,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        })

    def compute(
        self,
        symbol: str,
        klines_df,
        indicators: Dict,
        deriv_snapshot,
        patterns_detected: List[str],
        bias: str,
        llm_response: Optional[Dict] = None,
        composite=None,
        funding_result=None,           # NEW
    ) -> ConfidenceBreakdown:
        breakdown = ConfidenceBreakdown()

        passed, reasons = self.check_hard_gates(
            symbol, deriv_snapshot, klines_df,
            bias=bias, funding_result=funding_result,
        )
        breakdown.gate_reasons = reasons
        if not passed:
            breakdown.is_high_confidence = False
            return breakdown

        breakdown.technical_score = self.score_technical(klines_df, indicators)
        breakdown.composite_score = self.score_composite_quant(composite)
        breakdown.derivative_score = self.score_derivatives(deriv_snapshot)
        breakdown.sentiment_score = self.score_sentiment(deriv_snapshot)
        breakdown.pattern_score = self.score_patterns(patterns_detected, bias)
        breakdown.llm_score = self.score_llm(llm_response)

        breakdown.raw_total = (
            breakdown.technical_score + breakdown.composite_score
            + breakdown.derivative_score + breakdown.sentiment_score
            + breakdown.pattern_score + breakdown.llm_score
        )
        breakdown.calibrated_confidence = self.calibrate(breakdown.raw_total)

        breakdown.is_high_confidence = (
            breakdown.raw_total >= self.min_raw_threshold
            and breakdown.calibrated_confidence >= self.min_calibrated
            and len(breakdown.gate_reasons) == 0
        )
        return breakdown