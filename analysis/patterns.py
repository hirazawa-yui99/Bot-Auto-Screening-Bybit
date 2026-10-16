"""
analysis/patterns.py
Geometric chart pattern detection:
  - Double top / bottom
  - Head & shoulders / inverse
  - Triangles (ascending, descending, symmetric)
  - Flags (bull, bear)
  - Rectangles
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PatternResult:
    patterns: List[str] = field(default_factory=list)
    directions: Dict[str, str] = field(default_factory=dict)
    strengths: Dict[str, float] = field(default_factory=dict)


class PatternDetector:
    def __init__(self, min_bars: int = 40):
        self.min_bars = min_bars

    def _pivots(self, series: pd.Series, n: int = 5) -> Tuple[List[int], List[int]]:
        highs, lows = [], []
        arr = series.values
        for i in range(n, len(arr) - n):
            if arr[i] == max(arr[i - n:i + n + 1]):
                highs.append(i)
            if arr[i] == min(arr[i - n:i + n + 1]):
                lows.append(i)
        return highs, lows

    def _double_top(self, df: pd.DataFrame) -> Tuple[bool, float]:
        highs, _ = self._pivots(df["high"])
        if len(highs) < 2:
            return False, 0.0
        h1, h2 = highs[-2], highs[-1]
        if h2 - h1 < 5 or h2 - h1 > 60:
            return False, 0.0
        p1, p2 = df["high"].iloc[h1], df["high"].iloc[h2]
        if abs(p1 - p2) / p1 > 0.02:
            return False, 0.0
        # Neckline break
        neckline = df["low"].iloc[h1:h2 + 1].min()
        if df["close"].iloc[-1] < neckline:
            return True, 0.9
        return False, 0.0

    def _double_bottom(self, df: pd.DataFrame) -> Tuple[bool, float]:
        _, lows = self._pivots(df["low"])
        if len(lows) < 2:
            return False, 0.0
        l1, l2 = lows[-2], lows[-1]
        if l2 - l1 < 5 or l2 - l1 > 60:
            return False, 0.0
        p1, p2 = df["low"].iloc[l1], df["low"].iloc[l2]
        if abs(p1 - p2) / p1 > 0.02:
            return False, 0.0
        neckline = df["high"].iloc[l1:l2 + 1].max()
        if df["close"].iloc[-1] > neckline:
            return True, 0.9
        return False, 0.0

    def _head_shoulders(self, df: pd.DataFrame) -> Tuple[bool, float]:
        highs, _ = self._pivots(df["high"])
        if len(highs) < 3:
            return False, 0.0
        h1, h2, h3 = highs[-3], highs[-2], highs[-1]
        p1, p2, p3 = df["high"].iloc[h1], df["high"].iloc[h2], df["high"].iloc[h3]
        if p2 > p1 and p2 > p3 and abs(p1 - p3) / p1 < 0.04:
            neckline = df["low"].iloc[h1:h3 + 1].min()
            if df["close"].iloc[-1] < neckline:
                return True, 0.85
        return False, 0.0

    def _inverse_head_shoulders(self, df: pd.DataFrame) -> Tuple[bool, float]:
        _, lows = self._pivots(df["low"])
        if len(lows) < 3:
            return False, 0.0
        l1, l2, l3 = lows[-3], lows[-2], lows[-1]
        p1, p2, p3 = df["low"].iloc[l1], df["low"].iloc[l2], df["low"].iloc[l3]
        if p2 < p1 and p2 < p3 and abs(p1 - p3) / p1 < 0.04:
            neckline = df["high"].iloc[l1:l3 + 1].max()
            if df["close"].iloc[-1] > neckline:
                return True, 0.85
        return False, 0.0

    def _triangle(self, df: pd.DataFrame) -> Tuple[str, float]:
        highs, lows = self._pivots(df["high"]), self._pivots(df["low"])
        if len(highs) < 2 or len(lows) < 2:
            return "", 0.0
        h_slope = np.polyfit(
            [h for h in highs[-3:]],
            [df["high"].iloc[h] for h in highs[-3:]],
            1,
        )[0]
        l_slope = np.polyfit(
            [l for l in lows[-3:]],
            [df["low"].iloc[l] for l in lows[-3:]],
            1,
        )[0]

        if abs(h_slope) < 0.001 and l_slope > 0.001:
            return "ascending_triangle", 0.7
        if abs(l_slope) < 0.001 and h_slope < -0.001:
            return "descending_triangle", 0.7
        if h_slope < -0.001 and l_slope > 0.001:
            return "symmetric_triangle", 0.6
        return "", 0.0

    def _flag(self, df: pd.DataFrame) -> Tuple[str, float]:
        if len(df) < 30:
            return "", 0.0
        pole = (df["close"].iloc[-20] - df["close"].iloc[-30]) / df["close"].iloc[-30]
        recent = df["close"].iloc[-10:]
        recent_range = (recent.max() - recent.min()) / recent.mean()
        if pole > 0.05 and recent_range < 0.02:
            return "bull_flag", 0.75
        if pole < -0.05 and recent_range < 0.02:
            return "bear_flag", 0.75
        return "", 0.0

    def _rectangle(self, df: pd.DataFrame) -> Tuple[str, float]:
        if len(df) < 40:
            return "", 0.0
        recent = df.iloc[-30:]
        rng = (recent["high"].max() - recent["low"].min()) / recent["close"].mean()
        if rng < 0.02:
            mid = recent["close"].mean()
            if df["close"].iloc[-1] > mid:
                return "bullish_rectangle", 0.5
            return "bearish_rectangle", 0.5
        return "", 0.0

    def detect(self, df: pd.DataFrame) -> PatternResult:
        result = PatternResult()
        if df is None or len(df) < self.min_bars:
            return result

        checks = [
            ("double_top", self._double_top, "short"),
            ("double_bottom", self._double_bottom, "long"),
            ("head_and_shoulders", self._head_shoulders, "short"),
            ("inverse_head_and_shoulders", self._inverse_head_shoulders, "long"),
        ]
        for name, fn, direction in checks:
            detected, strength = fn(df)
            if detected:
                result.patterns.append(name)
                result.directions[name] = direction
                result.strengths[name] = strength

        tri_name, tri_strength = self._triangle(df)
        if tri_name:
            result.patterns.append(tri_name)
            result.directions[tri_name] = "long" if "ascending" in tri_name else "short"
            result.strengths[tri_name] = tri_strength

        flag_name, flag_strength = self._flag(df)
        if flag_name:
            result.patterns.append(flag_name)
            result.directions[flag_name] = "long" if "bull" in flag_name else "short"
            result.strengths[flag_name] = flag_strength

        rect_name, rect_strength = self._rectangle(df)
        if rect_name:
            result.patterns.append(rect_name)
            result.directions[rect_name] = "long" if "bullish" in rect_name else "short"
            result.strengths[rect_name] = rect_strength

        return result