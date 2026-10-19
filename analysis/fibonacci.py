"""
analysis/fibonacci.py
Fibonacci retracement entry zones.

For a long signal, we want price to be retracing DOWN into the
0.5-0.618 fib zone of the last upswing. For short, retracing UP
into 0.5-0.618 of the last downswing.

Returns entry_zone = (low, high) for limit placement.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


FIB_LEVELS = {
    "0.236": 0.236,
    "0.382": 0.382,
    "0.500": 0.500,
    "0.618": 0.618,
    "0.786": 0.786,
}


@dataclass
class FibResult:
    valid: bool = False
    direction: str = "neutral"
    swing_high: float = 0.0
    swing_low: float = 0.0
    levels: Dict[str, float] = None
    entry_zone_low: float = 0.0
    entry_zone_high: float = 0.0
    current_price: float = 0.0
    in_zone: bool = False
    zone_name: str = ""
    reason: str = ""

    def __post_init__(self):
        if self.levels is None:
            self.levels = {}


class FibonacciAnalyzer:
    def __init__(self, lookback_bars: int = 50):
        self.lookback_bars = lookback_bars

    def analyze(
        self, df: pd.DataFrame, direction: str,
    ) -> FibResult:
        """
        Compute fib retracement entry zone for `direction`.
        df: 1h klines (must have open/high/low/close)
        direction: "long" | "short"
        """
        r = FibResult(direction=direction)
        if df is None or len(df) < self.lookback_bars:
            r.reason = "insufficient data"
            return r
        if direction not in ("long", "short"):
            r.reason = f"invalid direction: {direction}"
            return r

        window = df.tail(self.lookback_bars)
        swing_high = float(window["high"].max())
        swing_low = float(window["low"].min())
        current = float(df["close"].iloc[-1])
        r.swing_high = swing_high
        r.swing_low = swing_low
        r.current_price = current

        fib_range = swing_high - swing_low
        if fib_range <= 0:
            r.reason = "zero range"
            return r

        if direction == "long":
            # Retracement measured down from swing_high
            for name, ratio in FIB_LEVELS.items():
                r.levels[name] = swing_high - fib_range * ratio
            zone_low = r.levels["0.618"]
            zone_high = r.levels["0.500"]
            r.zone_name = "0.500-0.618 retracement"
        else:
            # Retracement measured up from swing_low
            for name, ratio in FIB_LEVELS.items():
                r.levels[name] = swing_low + fib_range * ratio
            zone_low = r.levels["0.500"]
            zone_high = r.levels["0.618"]
            r.zone_name = "0.500-0.618 retracement"

        r.entry_zone_low = float(min(zone_low, zone_high))
        r.entry_zone_high = float(max(zone_low, zone_high))
        r.in_zone = r.entry_zone_low <= current <= r.entry_zone_high

        # Sanity: current price must be on the correct side of the swing
        if direction == "long" and current > swing_high:
            r.reason = "price above swing high (already broke out)"
            return r
        if direction == "short" and current < swing_low:
            r.reason = "price below swing low (already broke down)"
            return r

        r.valid = True
        r.reason = (
            f"fib {r.zone_name}, "
            f"zone [{r.entry_zone_low:.4f}-{r.entry_zone_high:.4f}], "
            f"current={current:.4f} "
            f"{'IN' if r.in_zone else 'out'}"
        )
        return r