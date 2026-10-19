"""
analysis/setup_classifier.py
Classifies a signal into one or more of: scalp, intraday, swing.

Each setup has its own timeframe, RVOL threshold, half-life window,
and entry-zone width. All three run on every symbol; a symbol can
match any subset.

Rules:
  SCALP     15m TF, RVOL >= 1.5, ATR/price >= 0.4%, half-life < 4h
  INTRADAY  1h TF, RVOL >= 1.0, Zeta active, 2h <= half-life < 24h
  SWING     4h TF, EMA stack aligned, funding aligned, half-life >= 24h
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    confidence: float = 0.0            # 0..1, drives position sizing
    timeframe: str = ""
    reason: str = ""
    entry_zone_low: float = 0.0        # limit placement window
    entry_zone_high: float = 0.0
    atr: float = 0.0
    direction: str = "neutral"          # long / short / neutral
    metadata: Dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.setup_type


@dataclass
class SetupCollection:
    setups: List[SetupCandidate] = field(default_factory=list)

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

    # ------------------------------------------------------------------ #
    #  Indicators per timeframe
    # ------------------------------------------------------------------ #
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
        if ind["ema_9"] > ind["ema_21"] > ind["ema_50"]:
            return "long"
        if ind["ema_9"] < ind["ema_21"] < ind["ema_50"]:
            return "short"
        return "neutral"

    # ------------------------------------------------------------------ #
    #  Scalp
    # ------------------------------------------------------------------ #
    def _classify_scalp(
        self, df_15m: pd.DataFrame, deriv, bias: str,
    ) -> SetupCandidate:
        s = SetupCandidate(
            setup_type=SETUP_SCALP, timeframe="15m",
        )
        if df_15m is None or len(df_15m) < 50:
            s.reason = "insufficient 15m data"
            return s

        ind = self._compute(df_15m)

        # Directional bias — prefer composite if available, else EMA
        s.direction = bias if bias != "neutral" else self._ema_bias(ind)

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

        # Confidence scales with how many conditions hit + RVOL strength
        base = len(conditions) / 4.0
        rvol_boost = min(1.0, (ind["rvol"] - 1.0) / 2.0)
        s.confidence = float(np.clip(base * 0.7 + rvol_boost * 0.3, 0.3, 0.9))

        # Entry zone = current ± 0.5 ATR (tight for scalp)
        entry_mid = ind["price"]
        half_width = ind["atr"] * 0.5
        s.entry_zone_low = entry_mid - half_width
        s.entry_zone_high = entry_mid + half_width
        s.atr = ind["atr"]
        s.valid = True
        s.reason = f"scalp: {', '.join(conditions)}, RVOL={ind['rvol']:.2f}"
        s.metadata = {"indicators": ind}
        return s

    # ------------------------------------------------------------------ #
    #  Intraday
    # ------------------------------------------------------------------ #
    def _classify_intraday(
        self,
        df_1h: pd.DataFrame,
        composite,
        bias: str,
    ) -> SetupCandidate:
        s = SetupCandidate(
            setup_type=SETUP_INTRADAY, timeframe="1h",
        )
        if df_1h is None or len(df_1h) < 60:
            s.reason = "insufficient 1h data"
            return s

        ind = self._compute(df_1h)
        s.direction = bias if bias != "neutral" else self._ema_bias(ind)

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
        if composite.direction == bias and bias != "neutral":
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
            f"zeta={mr.zeta:+.2f}, hl={hl:.1f}h"
        )
        s.metadata = {"indicators": ind, "zeta": mr.zeta}
        return s

    # ------------------------------------------------------------------ #
    #  Swing
    # ------------------------------------------------------------------ #
    def _classify_swing(
        self,
        df_4h: pd.DataFrame,
        composite,
        funding,
        bias: str,
    ) -> SetupCandidate:
        s = SetupCandidate(
            setup_type=SETUP_SWING, timeframe="4h",
        )
        if df_4h is None or len(df_4h) < 40:
            s.reason = "insufficient 4h data"
            return s

        ind = self._compute(df_4h)
        ema_bias = self._ema_bias(ind)
        s.direction = ema_bias if ema_bias != "neutral" else bias
        if s.direction == "neutral":
            s.reason = "4h EMA not stacked"
            return s

        conditions = ["ema_stack"]
        if ind["rvol"] >= self.swing_rvol_min:
            conditions.append("rvol")
        if composite.mean_reversion.half_life_hours >= self.swing_hl_min:
            conditions.append("long_half_life")
        if composite.mean_reversion.hurst > 0.5:
            conditions.append("trending_hurst")
        if funding is not None and not funding.should_block_long \
                and not funding.should_block_short:
            conditions.append("funding_clear")
        # Basis aligned with direction
        if composite.basis.regime == "contango" and s.direction == "long":
            conditions.append("basis_aligned")
        elif composite.basis.regime == "backwardation" and s.direction == "short":
            conditions.append("basis_aligned")

        if len(conditions) < 3:
            s.reason = f"insufficient conditions: {conditions}"
            return s

        base = len(conditions) / 6.0
        s.confidence = float(np.clip(base * 0.8 + 0.2, 0.5, 0.95))

        # Swing entry zone is wider — bigger ATR buffer
        entry_mid = ind["price"]
        half_width = ind["atr"] * 1.5
        s.entry_zone_low = entry_mid - half_width
        s.entry_zone_high = entry_mid + half_width
        s.atr = ind["atr"]
        s.valid = True
        s.reason = f"swing: {', '.join(conditions)}"
        s.metadata = {"indicators": ind, "ema_bias": ema_bias}
        return s

    # ------------------------------------------------------------------ #
    #  Public entry
    # ------------------------------------------------------------------ #
    def classify_all(
        self,
        df_15m: Optional[pd.DataFrame],
        df_1h: pd.DataFrame,
        df_4h: Optional[pd.DataFrame],
        composite,
        deriv,
        bias: str,
    ) -> SetupCollection:
        collection = SetupCollection()

        try:
            collection.setups.append(
                self._classify_scalp(df_15m, deriv, bias)
            )
        except Exception as e:
            logger.debug(f"scalp classify failed: {e}")

        try:
            collection.setups.append(
                self._classify_intraday(df_1h, composite, bias)
            )
        except Exception as e:
            logger.debug(f"intraday classify failed: {e}")

        try:
            collection.setups.append(
                self._classify_swing(
                    df_4h, composite, composite.funding, bias,
                )
            )
        except Exception as e:
            logger.debug(f"swing classify failed: {e}")

        return collection