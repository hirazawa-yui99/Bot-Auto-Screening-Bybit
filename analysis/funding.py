"""
analysis/funding.py
Funding-rate analysis with contrarian bias detection.

Bybit perpetual funding:
  - Baseline: 0.01% per 8h (paid 3x daily = 0.03% per day = ~10.95% annualized)
  - Positive: longs pay shorts → crowded longs → BEARISH contrarian
  - Negative: shorts pay longs → crowded shorts → BULLISH contrarian
  - Extreme: either side → squeeze risk → hard block on crowded direction

Tiers (rates shown in raw decimal, e.g. 0.0005 = 0.05%):

  neutral   | \|r\| < elevated_pct   | no penalty
  elevated  | elevated ≤ \|r\| < high| soft penalty/bonus
  high      | high ≤ \|r\| < extreme  | hard block on crowded direction
  extreme   | \|r\| ≥ extreme         | maximum penalty + hard block
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FundingResult:
    current_rate: float = 0.0            # raw decimal, e.g. 0.0005 = 0.05%
    rate_pct: float = 0.0                # multiplied by 100 for display
    annualized_pct: float = 0.0
    regime: str = "neutral"              # neutral | elevated_long | elevated_short | high_long | high_short | extreme_long | extreme_short
    contrarian_bias: str = "neutral"     # "long" | "short" | "neutral"
    bias_strength: float = 0.0           # 0.0 to 1.0
    should_block_long: bool = False
    should_block_short: bool = False
    confidence_contribution: float = 0.0  # -15 to +15 (positive = bullish for composite)
    reasoning: str = ""

    def __repr__(self):
        return (
            f"FundingResult(rate={self.current_rate:.5f}="
            f"{self.rate_pct:+.4f}%, annual={self.annualized_pct:+.1f}%, "
            f"regime={self.regime}, bias={self.contrarian_bias}, "
            f"block_long={self.should_block_long}, "
            f"block_short={self.should_block_short})"
        )


class FundingAnalyzer:
    """Converts funding rate into a contrarian trading signal."""

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.neutral_band = abs(
            float(cfg.get("neutral_band_pct", 0.01))
        ) / 100.0
        self.elevated = abs(
            float(cfg.get("elevated_pct", 0.02))
        ) / 100.0
        self.high = abs(
            float(cfg.get("high_pct", 0.05))
        ) / 100.0
        self.extreme = abs(
            float(cfg.get("extreme_pct", 0.10))
        ) / 100.0
        self.block_long_at = abs(
            float(cfg.get("block_long_at_pct", 0.05))
        ) / 100.0
        self.block_short_at = abs(
            float(cfg.get("block_short_at_pct", -0.05))
        ) / 100.0
        self.annualization_factor = (
            float(cfg.get("annualization_factor", 3)) * 365
        )

    # ------------------------------------------------------------------ #
    #  Main entry
    # ------------------------------------------------------------------ #
    async def analyze(self, symbol: str, client) -> FundingResult:
        """Fetch and analyze current funding rate for a symbol."""
        result = FundingResult()

        try:
            ticker = await client.get_funding_rate(symbol)
            rate_raw = ticker.get("fundingRate")
            if rate_raw is None:
                logger.debug(f"{symbol}: no funding rate in ticker")
                return result
            rate = float(rate_raw)
        except (TypeError, ValueError, Exception) as e:
            logger.debug(f"{symbol}: funding rate fetch failed: {e}")
            return result

        result.current_rate = rate
        result.rate_pct = rate * 100.0
        result.annualized_pct = rate * self.annualization_factor * 100.0

        self._classify(result)
        self._compute_blocks(result)
        self._compute_contribution(result)
        return result

    # ------------------------------------------------------------------ #
    #  Classification into tiers
    # ------------------------------------------------------------------ #
    def _classify(self, r: FundingResult):
        rate = r.current_rate
        abs_rate = abs(rate)
        is_positive = rate > 0

        # Neutral band
        if abs_rate < self.neutral_band:
            r.regime = "neutral"
            r.contrarian_bias = "neutral"
            r.bias_strength = 0.0
            r.reasoning = (
                f"Funding {r.rate_pct:+.4f}% is within neutral band "
                f"(±{self.neutral_band*100:.2f}%)"
            )
            return

        # Elevated tier (soft signal)
        if abs_rate < self.high:
            r.regime = "elevated_long" if is_positive else "elevated_short"
            r.contrarian_bias = "short" if is_positive else "long"
            # Scale 0.0 → 1.0 between neutral_band and high
            r.bias_strength = float(
                (abs_rate - self.neutral_band)
                / max(1e-9, self.high - self.neutral_band)
            )
            r.reasoning = (
                f"Funding {r.rate_pct:+.4f}% is elevated "
                f"({'longs' if is_positive else 'shorts'} paying). "
                f"Mild contrarian bias {'short' if is_positive else 'long'}."
            )
            return

        # High tier (hard block)
        if abs_rate < self.extreme:
            r.regime = "high_long" if is_positive else "high_short"
            r.contrarian_bias = "short" if is_positive else "long"
            r.bias_strength = 0.7 + 0.2 * float(
                (abs_rate - self.high)
                / max(1e-9, self.extreme - self.high)
            )
            r.reasoning = (
                f"Funding {r.rate_pct:+.4f}% is HIGH "
                f"({'longs' if is_positive else 'shorts'} crowded). "
                f"Blocking {'long' if is_positive else 'short'} signals."
            )
            return

        # Extreme tier (squeeze imminent)
        r.regime = "extreme_long" if is_positive else "extreme_short"
        r.contrarian_bias = "short" if is_positive else "long"
        r.bias_strength = 1.0
        r.reasoning = (
            f"Funding {r.rate_pct:+.4f}% is EXTREME "
            f"({r.annualized_pct:+.1f}% annualized). "
            f"Squeeze risk — contrarian {'short' if is_positive else 'long'}."
        )

    # ------------------------------------------------------------------ #
    #  Hard-block flags (used by risk manager)
    # ------------------------------------------------------------------ #
    def _compute_blocks(self, r: FundingResult):
        # Block LONG when funding is high/extreme positive
        if r.current_rate >= self.block_long_at:
            r.should_block_long = True
        # Block SHORT when funding is high/extreme negative
        if r.current_rate <= -self.block_short_at:
            r.should_block_short = True

    # ------------------------------------------------------------------ #
    #  Soft contribution to composite score
    # ------------------------------------------------------------------ #
    def _compute_contribution(self, r: FundingResult):
        """
        Positive contribution = bullish (funding negative / shorts crowded).
        Negative contribution = bearish (funding positive / longs crowded).
        Range: -15 to +15.
        """
        rate = r.current_rate
        abs_rate = abs(rate)
        if abs_rate < self.neutral_band:
            r.confidence_contribution = 0.0
            return

        # Scale linearly from neutral_band (0) to extreme (15)
        if abs_rate >= self.extreme:
            magnitude = 15.0
        elif abs_rate >= self.high:
            span = self.extreme - self.high
            frac = (abs_rate - self.high) / max(1e-9, span)
            magnitude = 10.0 + 5.0 * frac
        else:
            span = self.high - self.neutral_band
            frac = (abs_rate - self.neutral_band) / max(1e-9, span)
            magnitude = 3.0 + 7.0 * frac

        # Sign: negative funding = bullish, positive funding = bearish
        sign = -1.0 if rate > 0 else 1.0
        r.confidence_contribution = float(
            np.clip(sign * magnitude, -15.0, 15.0)
        )

    # ------------------------------------------------------------------ #
    #  Direction-aware gate (used by confidence scorer)
    # ------------------------------------------------------------------ #
    @staticmethod
    def get_gate_reason(result: FundingResult, bias: str) -> Optional[str]:
        """
        Return a gate reason if the signal direction is crowded,
        or None if the signal is aligned with the contrarian view.
        """
        if bias == "long" and result.should_block_long:
            return (
                f"Funding {result.rate_pct:+.4f}% extreme — "
                f"longs crowded, blocking long signal"
            )
        if bias == "short" and result.should_block_short:
            return (
                f"Funding {result.rate_pct:+.4f}% extreme — "
                f"shorts crowded, blocking short signal"
            )
        return None