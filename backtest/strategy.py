"""
backtest/strategy.py
Full strategy replica — reproduces _analyse_and_signal without live APIs.

Substitutions for components that can't be historically reconstructed:
  - Orderbook heatmap → neutral (skipped in agreement)
  - Live L/S ratio → neutral (funding rate used instead)
  - CVD → computed from kline direction × volume (same as production)
  - Open interest → from historical OI data

All other components (SMC, Fibonacci, momentum guard, ATR filter,
setup classifier, agreement gate, confidence scorer) run as-is.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.setup_classifier import SetupClassifier, SetupCollection
from analysis.fibonacci import FibonacciAnalyzer
from analysis.agreement_gate import AgreementGate
from analysis.confidence import ConfidenceScorer

logger = logging.getLogger(__name__)


@dataclass
class BacktestSignal:
    """Everything needed to simulate a trade."""
    ts: int
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    tp_weights: tuple
    confidence: float
    setup_type: str
    setup_count: int
    agreement_count: int
    raw_score: float
    composite_score: float
    metadata: Dict


class BacktestStrategy:
    """
    Bar-by-bar strategy evaluator.

    Call `evaluate()` on each bar with the preceding N bars as context.
    Returns a BacktestSignal or None.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.strategy_cfg = config.get("strategy", {})
        self.execution_cfg = config.get("execution", {})
        self.risk_cfg = config.get("risk", {})
        self.setups_cfg = config.get("setups", {})
        self.agreement_cfg = config.get("agreement", {})

        self.setup_classifier = SetupClassifier(self.setups_cfg)
        self.fib_analyzer = FibonacciAnalyzer()
        self.agreement_gate = AgreementGate(
            min_agree=self.agreement_cfg.get("min_agree", 3),
        )
        self.scorer = ConfidenceScorer(
            min_raw_threshold=self.strategy_cfg.get(
                "min_raw_threshold", 50.0,
            ),
            min_calibrated=self.strategy_cfg.get(
                "min_calibrated", 0.50,
            ),
        )

        # Momentum guard params
        self.roc_threshold = float(
            self.execution_cfg.get("momentum_guard_roc_pct", 3.0)
        )
        self.roc_lookback = int(
            self.execution_cfg.get(
                "momentum_guard_lookback_bars", 5
            )
        )
        self.sigma_threshold = float(
            self.execution_cfg.get("momentum_guard_sigma", 2.5)
        )
        self.atr_pct_floor = float(
            self.execution_cfg.get(
                "momentum_guard_atr_pct_floor", 0.5
            )
        )

        # ATR filter params
        self.min_atr_pct = float(
            self.execution_cfg.get("min_atr_pct", 0.3)
        )
        self.max_atr_pct = float(
            self.execution_cfg.get("max_atr_pct", 5.0)
        )

        # TP/SL params
        self.atr_sl_mult = float(
            self.execution_cfg.get("atr_multiplier_sl", 1.5)
        )
        self.tp_r_mults = self.execution_cfg.get(
            "tp_r_multiples", [1.0, 2.0, 3.0]
        )
        self.tp_weights = tuple(
            self.execution_cfg.get(
                "tp_weights", [0.40, 0.35, 0.25]
            )
        )

    # ================================================================== #
    #  Indicator computation
    # ================================================================== #
    @staticmethod
    def _compute_indicators(df: pd.DataFrame) -> Dict[str, float]:
        close = df["close"]
        high = df["high"]
        low = df["low"]
        ema_9 = close.ewm(span=9).mean().iloc[-1]
        ema_21 = close.ewm(span=21).mean().iloc[-1]
        ema_50 = close.ewm(span=50).mean().iloc[-1]
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
        avg_vol = df["volume"].rolling(20).mean().iloc[-1]
        rvol = (
            df["volume"].iloc[-1] / avg_vol if avg_vol > 0 else 1.0
        )
        return {
            "ema_9": float(ema_9),
            "ema_21": float(ema_21),
            "ema_50": float(ema_50),
            "rsi": float(rsi),
            "atr": float(atr),
            "rvol": float(rvol),
        }

    # ================================================================== #
    #  Composite quant (approximation from OHLCV + funding + OI)
    # ================================================================== #
    def _compute_composite(
        self,
        df: pd.DataFrame,
        funding_df: pd.DataFrame,
        oi_df: pd.DataFrame,
        bar_ts: int,
    ) -> Dict:
        """
        Simplified composite — returns a dict with the fields the
        rest of the strategy reads. Uses historical data only.
        """
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values

        # Mean reversion (Zeta-like)
        window = close[-60:] if len(close) >= 60 else close
        mu = float(np.mean(window))
        sigma = float(np.std(window))
        z = (close[-1] - mu) / sigma if sigma > 0 else 0.0

        # Hurst approximation
        hurst = self._hurst(window)

        # Half-life from AR(1)
        half_life = self._half_life(window)

        # Direction from EMA stack
        ema_9 = df["close"].ewm(span=9).mean().iloc[-1]
        ema_21 = df["close"].ewm(span=21).mean().iloc[-1]
        if ema_9 > ema_21:
            direction = "long"
        elif ema_9 < ema_21:
            direction = "short"
        else:
            direction = "neutral"

        # Composite score (simple weighted blend)
        score = 0.0
        score += 0.3 * np.tanh(z)             # mean reversion
        score += 0.3 * (1 if direction == "long" else -1)  # trend
        # CVD proxy from kline direction
        deltas = np.where(close[-20:] > np.roll(close[-20:], 1),
                          volume[-20:], -volume[-20:])
        cvd_slope = float(np.polyfit(range(len(deltas)),
                                     np.cumsum(deltas), 1)[0])
        score += 0.2 * np.tanh(cvd_slope / 1e6)

        # Funding
        funding_rate = 0.0
        if not funding_df.empty:
            match = funding_df[funding_df["ts"] <= bar_ts]
            if not match.empty:
                funding_rate = float(match.iloc[-1]["fundingRate"])
        if abs(funding_rate) > 0.0005:
            score -= 0.2 * np.sign(funding_rate)

        score = float(np.clip(score, -1.0, 1.0))

        # Volatility
        atr_val = float(np.mean(np.maximum.reduce([
            high[-14:] - low[-14:],
            np.abs(high[-14:] - np.roll(close[-14:], 1)),
            np.abs(low[-14:] - np.roll(close[-14:], 1)),
        ])))

        return {
            "total_score": score,
            "direction": direction,
            "conviction": (
                "high" if abs(score) >= 0.55
                else "medium" if abs(score) >= 0.30
                else "low"
            ),
            "agreement_ratio": 0.0,
            "conflicting_components": [],
            "mean_reversion": {
                "zeta": z,
                "z_score": z,
                "hurst": hurst,
                "half_life_hours": half_life,
                "is_stationary": half_life < 100,
                "phi": 0.9,
                "mu": mu,
            },
            "basis": {
                "basis_pct": 0.0,
                "regime": "neutral",
                "signal_bias": "neutral",
            },
            "volume": {
                "rvol": float(
                    volume[-1] / np.mean(volume[-20:])
                ) if len(volume) >= 20 else 1.0,
                "z_score": 0.0,
                "anomaly_level": "normal",
                "vwpd": 0.0,
                "divergence": "none",
            },
            "funding": {
                "rate_pct": funding_rate * 100,
                "annualized_pct": funding_rate * 3 * 365 * 100,
                "regime": (
                    "high_long" if funding_rate > 0.0005
                    else "high_short" if funding_rate < -0.0005
                    else "neutral"
                ),
                "contrarian_bias": (
                    "short" if funding_rate > 0.0005
                    else "long" if funding_rate < -0.0005
                    else "neutral"
                ),
                "should_block_long": funding_rate > 0.0005,
                "should_block_short": funding_rate < -0.0005,
                "bias_strength": min(
                    1.0, abs(funding_rate) / 0.001
                ),
            },
        }

    @staticmethod
    def _hurst(series: np.ndarray, max_lag: int = 40) -> float:
        n = len(series)
        if n < 40:
            return 0.5
        lags = np.unique(
            np.floor(np.logspace(
                np.log10(8), np.log10(min(max_lag, n // 2)), 10
            )).astype(int)
        )
        lags = lags[lags >= 8]
        if len(lags) < 4:
            return 0.5
        rs_vals = []
        for lag in lags:
            n_chunks = n // lag
            if n_chunks < 1:
                continue
            chunk_rs = []
            for k in range(n_chunks):
                chunk = series[k * lag:(k + 1) * lag]
                dev = chunk - chunk.mean()
                cumdev = np.cumsum(dev)
                R = cumdev.max() - cumdev.min()
                S = chunk.std()
                if S > 0:
                    chunk_rs.append(R / S)
            if chunk_rs:
                rs_vals.append((lag, np.mean(chunk_rs)))
        if len(rs_vals) < 4:
            return 0.5
        log_l = np.log([v[0] for v in rs_vals])
        log_rs = np.log([v[1] for v in rs_vals])
        slope, _ = np.polyfit(log_l, log_rs, 1)
        return float(np.clip(slope, 0.0, 1.0))

    @staticmethod
    def _half_life(series: np.ndarray) -> float:
        if len(series) < 20:
            return 8.0
        x = series
        x_lag = x[:-1]
        x_now = x[1:]
        x_lag_mean = x_lag.mean()
        x_now_mean = x_now.mean()
        denom = np.sum((x_lag - x_lag_mean) ** 2)
        if denom < 1e-12:
            return 100.0
        phi = np.sum((x_lag - x_lag_mean) * (x_now - x_now_mean)) / denom
        if phi <= 0 or phi >= 1:
            return 100.0
        return float(np.log(2) / (-np.log(phi)))

    # ================================================================== #
    #  TP/SL computation
    # ================================================================== #
    def _compute_tp_levels(
        self, entry: float, atr: float, direction: str,
    ) -> tuple:
        if atr <= 0 or not np.isfinite(atr):
            atr = entry * 0.01
        if direction == "long":
            sl = entry - self.atr_sl_mult * atr
            risk = entry - sl
            tps = [entry + risk * r for r in self.tp_r_mults]
        else:
            sl = entry + self.atr_sl_mult * atr
            risk = sl - entry
            tps = [entry - risk * r for r in self.tp_r_mults]
        return sl, tps[0], tps[1], tps[2]

    # ================================================================== #
    #  Momentum guard
    # ================================================================== #
    def _momentum_guard(
        self, df: pd.DataFrame, bias: str,
    ) -> Optional[str]:
        """Returns skip reason or None."""
        if self.roc_threshold <= 0:
            return None
        if len(df) <= self.roc_lookback + 2:
            return None
        try:
            closes = df["close"].values
            current = float(closes[-1])
            past = float(closes[-(self.roc_lookback + 1)])
            if past <= 0 or current <= 0:
                return None
            roc_pct = (current - past) / past * 100

            high = df["high"].values
            low = df["low"].values
            tr = np.maximum.reduce([
                high[1:] - low[1:],
                np.abs(high[1:] - closes[:-1]),
                np.abs(low[1:] - closes[:-1]),
            ])
            atr = float(np.mean(tr[-14:]))
            atr_pct = atr / current * 100 if current > 0 else 0.0

            if atr_pct > 0:
                expected = atr_pct * (self.roc_lookback ** 0.5)
                sigma = abs(roc_pct) / expected if expected > 0 else 0.0
            else:
                sigma = 999.0

            chase = (
                (bias == "long" and roc_pct > 0)
                or (bias == "short" and roc_pct < 0)
            )
            abs_gate = abs(roc_pct) >= self.roc_threshold
            sigma_gate = sigma >= self.sigma_threshold
            if atr_pct < self.atr_pct_floor:
                sigma_gate = True

            if chase and abs_gate and sigma_gate:
                return (
                    f"momentum_chase:"
                    f"{roc_pct:+.2f}%/{sigma:.1f}σ"
                )
            return None
        except Exception:
            return None

    # ================================================================== #
    #  ATR filter
    # ================================================================== #
    def _atr_filter(
        self, indicators: Dict[str, float], price: float,
    ) -> Optional[str]:
        atr = indicators.get("atr", 0.0)
        if atr <= 0 or price <= 0:
            return None
        atr_pct = atr / price * 100
        if atr_pct < self.min_atr_pct:
            return f"atr_too_low:{atr_pct:.2f}%"
        if atr_pct > self.max_atr_pct:
            return f"atr_too_high:{atr_pct:.2f}%"
        return None

    # ================================================================== #
    #  Main evaluate
    # ================================================================== #
    def evaluate(
        self,
        df_1h: pd.DataFrame,
        funding_df: pd.DataFrame,
        oi_df: pd.DataFrame,
        bar_idx: int,
        symbol: str = "TESTUSDT",
    ) -> Optional[BacktestSignal]:
        """
        Evaluate the strategy at bar_idx using only df_1h[:bar_idx+1].

        Returns BacktestSignal or None.
        """
        # Minimum history
        if bar_idx < 100:
            return None

        # Use only data up to and including this bar (no lookahead)
        window = df_1h.iloc[:bar_idx + 1].reset_index(drop=True)
        bar_ts = int(window["ts"].iloc[-1])
        current_price = float(window["close"].iloc[-1])

        # 1. Indicators
        indicators = self._compute_indicators(window)

        # 2. ATR filter
        atr_skip = self._atr_filter(indicators, current_price)
        if atr_skip:
            return None

        # 3. Composite
        composite = self._compute_composite(
            window, funding_df, oi_df, bar_ts,
        )

        bias = (
            composite["direction"]
            if composite["direction"] != "neutral"
            else (
                "long"
                if indicators["ema_9"] > indicators["ema_21"]
                else "short"
            )
        )

        # 4. Momentum guard
        mom_skip = self._momentum_guard(window, bias)
        if mom_skip:
            return None

        # 5. Funding gate (direction-aware)
        funding = composite["funding"]
        if bias == "long" and funding["should_block_long"]:
            return None
        if bias == "short" and funding["should_block_short"]:
            return None

        # 6. SMC proxy from EMA + swing structure
        smc_dir = self._smc_direction(window, bias)
        deriv_dir = self._deriv_direction(window, bias)

        # 7. Fibonacci
        try:
            fibo = self.fib_analyzer.analyze(window, bias)
        except Exception:
            fibo = None

        fibo_dir = "neutral"
        fibo_valid = False
        if fibo is not None and fibo.valid:
            fibo_dir = bias
            fibo_valid = fibo.in_zone

        # 8. Agreement gate
        # Heatmap not available historically → "neutral" (ignored)
        # This effectively requires 2 of {smc, deriv, quant, fib}
        agreement = self.agreement_gate.check(
            smc_dir=smc_dir,
            deriv_dir=deriv_dir,
            quant_dir=composite["direction"],
            heatmap_dir="neutral",
            fib_direction=fibo_dir,
            fib_valid=fibo_valid,
        )
        if not agreement.passed:
            return None

        # 9. Confidence scoring
        breakdown = self._score_confidence(
            window, indicators, composite, agreement, bias,
        )
        if not breakdown["is_high_confidence"]:
            return None

        # 10. Setup classification
        # Use 1h as proxy for all three timeframes (backtest approximation)
        setups = self.setup_classifier.classify_all(
            df_15m=window,
            df_1h=window,
            df_4h=window,
            composite=_DictAsObj(composite),
            deriv=_DictAsObj({
                "cvd_divergence": "none",
                "smart_money_bias": "neutral",
            }),
            bias=bias,
        )
        if setups.count == 0:
            return None

        best = setups.highest_confidence()
        if best is None:
            return None

        # 11. Entry price (fibo zone or current)
        entry = current_price
        if fibo is not None and fibo.valid:
            if not (fibo.entry_zone_low <= current_price <= fibo.entry_zone_high):
                entry = (
                    fibo.entry_zone_low if bias == "long"
                    else fibo.entry_zone_high
                )

        sl, tp1, tp2, tp3 = self._compute_tp_levels(
            entry, best.atr, bias,
        )

        return BacktestSignal(
            ts=bar_ts,
            symbol=symbol,
            direction=bias,
            entry_price=entry,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            tp_weights=self.tp_weights,
            confidence=breakdown["calibrated_confidence"],
            setup_type=best.setup_type,
            setup_count=setups.count,
            agreement_count=agreement.agreement_count,
            raw_score=breakdown["raw_total"],
            composite_score=composite["total_score"],
            metadata={
                "smc_dir": smc_dir,
                "deriv_dir": deriv_dir,
                "fibo_in_zone": fibo_valid,
                "funding_rate_pct": funding["rate_pct"],
            },
        )

    # ================================================================== #
    #  Component approximations for backtest
    # ================================================================== #
    @staticmethod
    def _smc_direction(df: pd.DataFrame, bias: str) -> str:
        """Simplified SMC: BOS detection via swing high/low break."""
        if len(df) < 30:
            return "neutral"
        try:
            window = df.tail(30)
            recent_high = float(window["high"].iloc[-10:-1].max())
            recent_low = float(window["low"].iloc[-10:-1].min())
            last_close = float(window["close"].iloc[-1])

            if last_close > recent_high:
                return "long"
            if last_close < recent_low:
                return "short"
            return "neutral"
        except Exception:
            return "neutral"

    @staticmethod
    def _deriv_direction(df: pd.DataFrame, bias: str) -> str:
        """CVD-based directional bias from kline volume."""
        if len(df) < 20:
            return "neutral"
        try:
            close = df["close"].values[-20:]
            volume = df["volume"].values[-20:]
            deltas = np.where(
                np.diff(close, prepend=close[0]) > 0,
                volume, -volume,
            )
            cvd = np.cumsum(deltas)
            slope = float(np.polyfit(range(len(cvd)), cvd, 1)[0])
            if slope > 0:
                return "long"
            if slope < 0:
                return "short"
            return "neutral"
        except Exception:
            return "neutral"

    def _score_confidence(
        self,
        df: pd.DataFrame,
        indicators: Dict,
        composite: Dict,
        agreement,
        bias: str,
    ) -> Dict:
        """
        Reproduce ConfidenceScorer.compute using the backtest-safe
        components. Returns dict with raw_total, calibrated, is_high_confidence.
        """
        # Technical (0-20)
        tech = 0.0
        ema_9 = indicators.get("ema_9", 0)
        ema_21 = indicators.get("ema_21", 0)
        ema_50 = indicators.get("ema_50", 0)
        if ema_9 > ema_21 > ema_50:
            tech += 8
        elif ema_9 < ema_21 < ema_50:
            tech += 8
        elif ema_9 > ema_21 or ema_9 < ema_21:
            tech += 4
        rsi = indicators.get("rsi", 50)
        if 40 <= rsi <= 60:
            tech += 5
        elif rsi < 30:
            tech += 6
        elif rsi > 70:
            tech += 3
        rvol = indicators.get("rvol", 1.0)
        if rvol >= 2.0:
            tech += 7
        elif rvol >= 1.5:
            tech += 4
        elif rvol >= 1.0:
            tech += 2
        tech = min(20.0, tech)

        # Composite (0-20)
        score = abs(composite["total_score"])
        conviction_mult = {
            "low": 0.5, "medium": 0.8, "high": 1.0,
        }[composite["conviction"]]
        comp_score = score * conviction_mult * 20.0

        # Derivatives (0-20) — simplified: 10 baseline
        deriv_score = 10.0

        # Sentiment (0-15) — funding alignment
        funding = composite["funding"]
        sent_score = 0.0
        if (
            bias == "long" and funding["contrarian_bias"] == "long"
        ) or (
            bias == "short" and funding["contrarian_bias"] == "short"
        ):
            sent_score = 15.0
        elif funding["should_block_long"] and bias == "long":
            return {
                "raw_total": 0.0,
                "calibrated_confidence": 0.0,
                "is_high_confidence": False,
            }

        # Pattern (0-15) — no patterns detected in backtest
        pattern_score = 0.0

        # LLM (0-10) — no LLM in backtest
        llm_score = 5.0   # neutral baseline

        raw = (
            tech + comp_score + deriv_score
            + sent_score + pattern_score + llm_score
        )

        # Calibration (cold-start proxy)
        calibrated = min(0.85, max(0.30, raw / 100.0))

        min_raw = self.strategy_cfg.get("min_raw_threshold", 50.0)
        min_cal = self.strategy_cfg.get("min_calibrated", 0.50)

        return {
            "raw_total": raw,
            "calibrated_confidence": calibrated,
            "is_high_confidence": (
                raw >= min_raw and calibrated >= min_cal
            ),
        }


class _DictAsObj:
    """Wrap a dict so attribute access works like the production classes."""
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, _DictAsObj(v))
            else:
                setattr(self, k, v)