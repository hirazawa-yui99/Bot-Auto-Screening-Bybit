"""
analysis/setup_classifier.py
Classifies a signal into one or more of: scalp, intraday, swing.

DIRECTION LOCK:
  All setups in a single signal share ONE direction (the "primary"
  bias). Each classifier computes its own timeframe bias as a
  sanity check — if it opposes primary, the setup is invalidated
  (never forced to primary). This prevents contradictory alerts
  like "[scalp long] + [swing short]" in the same message.

Primary direction is established once in classify_all() from:
  1. The `bias` argument (typically from composite or 1h EMA)
  2. Fallback to composite.direction if bias is "neutral"
  3. Fallback to 1h EMA if composite is "neutral"
  4. If still neutral → return empty (no directional conviction)
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SETUP_SCALP = "scalp"
SETUP_INTRADAY = "intraday"
SETUP_SWING = "swing"


@dataclass
class SetupCandidate:
    setup_type: str
    valid: bool = False
    confidence: float = 0.0
    timeframe: str = ""
    reason: str = ""
    entry_zone_low: float = 0.0
    entry_zone_high: float = 0.0
    atr: float = 0.0
    direction: str = "neutral"
    metadata: Dict = field(default_factory=dict)

    # Populated by main.py after classification
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0

    @property
    def label(self) -> str:
        return self.setup_type

    @property
    def entry_mid(self) -> float:
        return (self.entry_zone_low + self.entry_zone_high) / 2

    @property
    def has_plan(self) -> bool:
        return self.sl > 0 and self.tp1 > 0


@dataclass
class SetupCollection:
    setups: List[SetupCandidate] = field(default_factory=list)
    primary_direction: str = "neutral"

    @property
    def count(self) -> int:
        return len([s for s in self.setups if s.valid])

    @property
    def valid_setups(self) -> List[SetupCandidate]:
        return [s for s in self.setups if s.valid]

    def highest_confidence(self) -> Optional[SetupCandidate]:
        valid = self.valid_setups
        if not valid:
            return None
        return max(valid, key=lambda s: s.confidence)

    @property
    def labels(self) -> List[str]:
        return [s.setup_type for s in self.valid_setups]


class SetupClassifier:
    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.scalp_rvol_min = float(cfg.get("scalp_rvol_min", 1.5))
        self.scalp_atr_pct_min = float(cfg.get("scalp_atr_pct_min", 0.4))
        self.scalp_hl_max = float(cfg.get("scalp_half_life_max_hours", 4.0))
        self.intraday_rvol_min = float(cfg.get("intraday_rvol_min", 1.0))
        self.intraday_hl_min = float(cfg.get("intraday_half_life_min_hours", 2.0))
        self.intraday_hl_max = float(cfg.get("intraday_half_life_max_hours", 24.0))
        self.swing_rvol_min = float(cfg.get("swing_rvol_min", 0.7))
        self.swing_hl_min = float(cfg.get("swing_half_life_min_hours", 24.0))

    # ================================================================== #
    #  Timeframe helpers
    # ================================================================== #
    @staticmethod
    def _compute(df: pd.DataFrame) -> Dict[str, float]:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["volume"]
        ema9 = close.ewm(span=9).mean().iloc[-1]
        ema21 = close.ewm(span=21).mean().iloc[-1]
        ema50 = close.ewm(span=50).mean().iloc[-1]
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi = (100 - 100 / (1 + rs)).iloc[-1]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        price = close.iloc[-1]
        avg_vol = vol.rolling(20).mean().iloc[-1]
        rvol = vol.iloc[-1] / avg_vol if avg_vol > 0 else 1.0
        return {
            "ema_9": float(ema9),
            "ema_21": float(ema21),
            "ema_50": float(ema50),
            "rsi": float(rsi),
            "atr": float(atr),
            "atr_pct": float(atr / price * 100) if price > 0 else 0.0,
            "rvol": float(rvol),
            "price": float(price),
        }

    @staticmethod
    def _ema_bias(ind: Dict[str, float]) -> str:
        """Full EMA stack alignment. Returns 'long', 'short', or 'neutral'."""
        if ind["ema_9"] > ind["ema_21"] > ind["ema_50"]:
            return "long"
        if ind["ema_9"] < ind["ema_21"] < ind["ema_50"]:
            return "short"
        return "neutral"

    # ================================================================== #
    #  Scalp — 15m
    # ================================================================== #
    def _classify_scalp(
        self,
        df_15m: pd.DataFrame,
        deriv,
        primary: str,
    ) -> SetupCandidate:
        s = SetupCandidate(setup_type=SETUP_SCALP, timeframe="15m")
        if df_15m is None or len(df_15m) < 50:
            s.reason = "insufficient 15m data"
            return s

        ind = self._compute(df_15m)
        tf_bias = self._ema_bias(ind)

        # Direction lock — 15m must not oppose primary
        if tf_bias != "neutral" and tf_bias != primary:
            s.reason = (
                f"scalp rejected: 15m EMA {tf_bias} "
                f"contradicts primary {primary}"
            )
            return s

        s.direction = primary

        conditions = []
        if ind["rvol"] >= self.scalp_rvol_min:
            conditions.append("rvol")
        if ind["atr_pct"] >= self.scalp_atr_pct_min:
            conditions.append("atr_pct")
        if 30 <= ind["rsi"] <= 70:
            conditions.append("rsi_mid")
        if deriv is not None and deriv.cvd_divergence != "none":
            conditions.append("cvd_div")

        if len(conditions) < 2:
            s.reason = f"insufficient conditions: {conditions}"
            return s

        base = len(conditions) / 4.0
        rvol_boost = min(1.0, (ind["rvol"] - 1.0) / 2.0)
        s.confidence = float(np.clip(base * 0.7 + rvol_boost * 0.3, 0.3, 0.9))

        entry_mid = ind["price"]
        half_width = ind["atr"] * 0.5
        s.entry_zone_low = entry_mid - half_width
        s.entry_zone_high = entry_mid + half_width
        s.atr = ind["atr"]
        s.valid = True
        s.reason = (
            f"scalp: {', '.join(conditions)}, "
            f"RVOL={ind['rvol']:.2f}, dir={primary}"
        )
        s.metadata = {"indicators": ind, "tf_bias": tf_bias}
        return s

    # ================================================================== #
    #  Intraday — 1h
    # ================================================================== #
    def _classify_intraday(
        self,
        df_1h: pd.DataFrame,
        composite,
        primary: str,
    ) -> SetupCandidate:
        s = SetupCandidate(setup_type=SETUP_INTRADAY, timeframe="1h")
        if df_1h is None or len(df_1h) < 60:
            s.reason = "insufficient 1h data"
            return s

        ind = self._compute(df_1h)
        tf_bias = self._ema_bias(ind)

        # Direction lock — 1h must not oppose primary
        if tf_bias != "neutral" and tf_bias != primary:
            s.reason = (
                f"intraday rejected: 1h EMA {tf_bias} "
                f"contradicts primary {primary}"
            )
            return s

        s.direction = primary

        mr = composite.mean_reversion
        hl = mr.half_life_hours

        conditions = []
        if ind["rvol"] >= self.intraday_rvol_min:
            conditions.append("rvol")
        if abs(mr.zeta) >= 1.0:
            conditions.append("zeta")
        if self.intraday_hl_min <= hl <= self.intraday_hl_max:
            conditions.append("half_life")
        if mr.is_stationary:
            conditions.append("adf_stationary")
        if composite.direction == primary and primary != "neutral":
            conditions.append("composite_aligned")

        if len(conditions) < 3:
            s.reason = f"insufficient conditions: {conditions}"
            return s

        base = len(conditions) / 5.0
        zeta_boost = min(1.0, abs(mr.zeta) / 2.5)
        s.confidence = float(np.clip(base * 0.7 + zeta_boost * 0.3, 0.4, 0.92))

        entry_mid = ind["price"]
        half_width = ind["atr"] * 1.0
        s.entry_zone_low = entry_mid - half_width
        s.entry_zone_high = entry_mid + half_width
        s.atr = ind["atr"]
        s.valid = True
        s.reason = (
            f"intraday: {', '.join(conditions)}, "
            f"zeta={mr.zeta:+.2f}, hl={hl:.1f}h, dir={primary}"
        )
        s.metadata = {"indicators": ind, "zeta": mr.zeta, "tf_bias": tf_bias}
        return s

    # ================================================================== #
    #  Swing — 4h
    # ================================================================== #
    def _classify_swing(
        self,
        df_4h: pd.DataFrame,
        composite,
        funding,
        primary: str,
    ) -> SetupCandidate:
        s = SetupCandidate(setup_type=SETUP_SWING, timeframe="4h")
        if df_4h is None or len(df_4h) < 40:
            s.reason = "insufficient 4h data"
            return s

        ind = self._compute(df_4h)
        tf_bias = self._ema_bias(ind)

        # Direction lock — 4h EMA is the strongest check.
        # If 4h opposes primary, this is a REAL contradiction:
        # we're trying to trade against the higher timeframe trend.
        if tf_bias != "neutral" and tf_bias != primary:
            s.reason = (
                f"swing rejected: 4h EMA {tf_bias} "
                f"contradicts primary {primary}"
            )
            return s

        s.direction = primary

        conditions = ["ema_stack"]
        if ind["rvol"] >= self.swing_rvol_min:
            conditions.append("rvol")
        if composite.mean_reversion.half_life_hours >= self.swing_hl_min:
            conditions.append("long_half_life")
        if composite.mean_reversion.hurst > 0.5:
            conditions.append("trending_hurst")
        if (
            funding is not None
            and not funding.should_block_long
            and not funding.should_block_short
        ):
            conditions.append("funding_clear")
        if composite.basis.regime == "contango" and primary == "long":
            conditions.append("basis_aligned")
        elif composite.basis.regime == "backwardation" and primary == "short":
            conditions.append("basis_aligned")

        if len(conditions) < 3:
            s.reason = f"insufficient conditions: {conditions}"
            return s

        base = len(conditions) / 6.0
        s.confidence = float(np.clip(base * 0.8 + 0.2, 0.5, 0.95))

        entry_mid = ind["price"]
        half_width = ind["atr"] * 1.5
        s.entry_zone_low = entry_mid - half_width
        s.entry_zone_high = entry_mid + half_width
        s.atr = ind["atr"]
        s.valid = True
        s.reason = (
            f"swing: {', '.join(conditions)}, dir={primary}"
        )
        s.metadata = {
            "indicators": ind,
            "tf_bias": tf_bias,
        }
        return s

    # ================================================================== #
    #  Primary direction resolution
    # ================================================================== #
    def _resolve_primary(
        self,
        bias: str,
        composite,
        df_1h: pd.DataFrame,
    ) -> str:
        """
        Establish the single direction for all setups.

        Order of precedence:
          1. Passed-in `bias` (typically from composite or 1h EMA)
          2. composite.direction if bias is "neutral"
          3. 1h EMA if composite is "neutral"
          4. "neutral" — meaning no directional conviction → no setups
        """
        if bias in ("long", "short"):
            return bias

        if composite is not None and composite.direction in ("long", "short"):
            return composite.direction

        if df_1h is not None and len(df_1h) >= 50:
            ind = self._compute(df_1h)
            ema_bias = self._ema_bias(ind)
            if ema_bias in ("long", "short"):
                return ema_bias

        return "neutral"

    # ================================================================== #
    #  Public entry
    # ================================================================== #
    def classify_all(
        self,
        df_15m: Optional[pd.DataFrame],
        df_1h: pd.DataFrame,
        df_4h: Optional[pd.DataFrame],
        composite,
        deriv,
        bias: str,
    ) -> SetupCollection:
        # 1. Resolve primary direction once
        primary = self._resolve_primary(bias, composite, df_1h)

        collection = SetupCollection(primary_direction=primary)

        if primary == "neutral":
            logger.debug(
                "classify_all: no primary direction — returning empty"
            )
            return collection

        # 2. All classifiers must agree with primary or be invalidated
        try:
            collection.setups.append(
                self._classify_scalp(df_15m, deriv, primary)
            )
        except Exception as e:
            logger.debug(f"scalp classify failed: {e}")

        try:
            collection.setups.append(
                self._classify_intraday(df_1h, composite, primary)
            )
        except Exception as e:
            logger.debug(f"intraday classify failed: {e}")

        try:
            collection.setups.append(
                self._classify_swing(
                    df_4h,
                    composite,
                    composite.funding if composite else None,
                    primary,
                )
            )
        except Exception as e:
            logger.debug(f"swing classify failed: {e}")

        # 3. Log summary for observability
        valid = collection.valid_setups
        if valid:
            logger.debug(
                f"setups: primary={primary}, valid={[s.setup_type for s in valid]}"
            )
        return collection