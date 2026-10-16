"""
analysis/smc.py
Smart Money Concepts: Break of Structure (BOS), Change of Character (CHoCH),
Order Blocks (OB), Fair Value Gaps (FVG), Liquidity sweeps.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SMCResult:
    bos: str = "none"                 # "bullish" | "bearish" | "none"
    choch: str = "none"
    last_ob_high: float = 0.0
    last_ob_low: float = 0.0
    ob_direction: str = "none"
    fvg_bullish: bool = False
    fvg_bearish: bool = False
    liquidity_sweep: str = "none"     # "high" | "low" | "none"
    confidence_contribution: float = 0.0


class SMCAnalyzer:
    def __init__(self, swing_lookback: int = 5):
        self.swing_lookback = swing_lookback

    def _find_swings(self, df: pd.DataFrame):
        highs, lows = [], []
        n = self.swing_lookback
        for i in range(n, len(df) - n):
            window = df.iloc[i - n:i + n + 1]
            if df["high"].iloc[i] == window["high"].max():
                highs.append((i, df["high"].iloc[i]))
            if df["low"].iloc[i] == window["low"].min():
                lows.append((i, df["low"].iloc[i]))
        return highs, lows

    def analyze(self, df: pd.DataFrame) -> SMCResult:
        result = SMCResult()
        if df is None or len(df) < 30:
            return result

        highs, lows = self._find_swings(df)
        if len(highs) < 2 or len(lows) < 2:
            return result

        last_close = float(df["close"].iloc[-1])
        prev_high = highs[-1][1]
        prev_higher_high = highs[-2][1]
        prev_low = lows[-1][1]
        prev_lower_low = lows[-2][1]

        # BOS: price closes above previous swing high (bullish) or below previous swing low
        if last_close > prev_high and prev_high > prev_higher_high:
            result.bos = "bullish"
        elif last_close < prev_low and prev_low < prev_lower_low:
            result.bos = "bearish"

        # CHoCH: first break against the prevailing trend
        if result.bos == "bullish" and prev_low < prev_lower_low:
            result.choch = "bullish"
        elif result.bos == "bearish" and prev_high > prev_higher_high:
            result.choch = "bearish"

        # Order block: last down-candle before an up move (bullish OB) and vice versa
        for i in range(len(df) - 3, max(len(df) - 20, 0), -1):
            row = df.iloc[i]
            if row["close"] < row["open"] and df["close"].iloc[i + 1] > row["high"]:
                result.last_ob_low = float(row["low"])
                result.last_ob_high = float(row["high"])
                result.ob_direction = "bullish"
                break
            if row["close"] > row["open"] and df["close"].iloc[i + 1] < row["low"]:
                result.last_ob_low = float(row["low"])
                result.last_ob_high = float(row["high"])
                result.ob_direction = "bearish"
                break

        # FVG: gap between candle[i-1].high and candle[i+1].low
        for i in range(len(df) - 3, max(len(df) - 15, 1), -1):
            if df["high"].iloc[i - 1] < df["low"].iloc[i + 1]:
                result.fvg_bullish = True
                break
            if df["low"].iloc[i - 1] > df["high"].iloc[i + 1]:
                result.fvg_bearish = True
                break

        # Liquidity sweep: wick beyond prev swing but close back inside
        last = df.iloc[-1]
        if last["high"] > prev_high and last["close"] < prev_high:
            result.liquidity_sweep = "high"
        elif last["low"] < prev_low and last["close"] > prev_low:
            result.liquidity_sweep = "low"

        score = 0.0
        if result.bos == "bullish": score += 4
        elif result.bos == "bearish": score -= 4
        if result.choch == "bullish": score += 3
        elif result.choch == "bearish": score -= 3
        if result.fvg_bullish: score += 2
        if result.fvg_bearish: score -= 2
        if result.liquidity_sweep == "low": score += 2
        elif result.liquidity_sweep == "high": score -= 2
        result.confidence_contribution = float(np.clip(score, -12, 12))
        return result