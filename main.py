"""
main.py
Bybit Quant Bot v9.1 — Async orchestrator with FULL-PAIR scanning.

Wires together:
  - V5 client with get_all_linear_symbols() pagination
  - DerivativeAnalyzer (CVD, OI, L/S ratio)
  - CompositeQuantEngine (Zeta, basis, volume anomaly)
  - ConfidenceScorer (calibrated, high-confidence gate)
  - DeepSeekAgent (LLM analysis)
  - TradeExecutor (WS orders + position monitor)
  - AutoTuner + OutcomeTracker (07:00 daily improvement)
  - Discord notifications
"""
import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from core.bybit_client import BybitV5Client, BybitConfig
from analysis.derivatives import DerivativeAnalyzer
from analysis.confidence import ConfidenceScorer
from analysis.quant_composite import CompositeQuantEngine
from agent.deepseek_agent import DeepSeekAgent, DeepSeekConfig
from execution.trade_executor import TradeExecutor, TradeSignal
from execution.risk_manager import RiskManager
from learning.signal_journal import SignalJournal
from learning.outcome_tracker import OutcomeTracker
from learning.auto_tuner import AutoTuner
from notification.discord_bot import DiscordNotifier
from persistence.database import Database
from typing import Dict, List, Optional, Tuple
import math 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot_v9.log"),
    ],
)
logger = logging.getLogger("main")


class QuantBotV9:
    def __init__(self, config_path: str = "config.json"):
        with open(config_path) as f:
            self.config = json.load(f)

        self.client = BybitV5Client(BybitConfig(
            api_key=self.config["api"]["bybit_key"],
            api_secret=self.config["api"]["bybit_secret"],
            testnet=self.config["api"].get("testnet", False),
            category="linear",
        ))

        self.db = Database(self.config["database"])
        self.journal = SignalJournal(self.db)

        self.derivatives = DerivativeAnalyzer(self.client)
        self.scorer = ConfidenceScorer(
            min_raw_threshold=self.config["strategy"]["min_raw_threshold"],
            min_calibrated=self.config["strategy"]["min_calibrated"],
        )
        self.composite_engine = CompositeQuantEngine(
        self.client,
        weights=self.config.get("strategy", {}).get("composite_weights"),
        funding_config=self.config.get("funding"),
        )

        self.agent = DeepSeekAgent(DeepSeekConfig(
            api_key=self.config["deepseek"]["api_key"],
            model=self.config["deepseek"].get("model", "deepseek-chat"),
        ))

        self.risk_manager = RiskManager(self.config["risk"])
        self.executor = TradeExecutor(
            self.client, self.risk_manager, self.config["execution"],
        )

        self.outcome_tracker = OutcomeTracker(self.client, self.journal)
        self.auto_tuner = AutoTuner(self.config["auto_tune"], self.scorer)
        self.notifier = DiscordNotifier(self.config["api"])

        self._running = False
        self._daily_task = None
        self._scan_task = None

    async def start(self):
        logger.info("=" * 60)
        logger.info("Bybit Quant Bot v9.1 starting — FULL-PAIR SCANNER")
        logger.info("=" * 60)

        self._running = True
        await self.client.start()
        await self.db.connect()
        await self.executor.start_monitor()

        self._daily_task = asyncio.create_task(self._daily_scheduler())
        self._scan_task = asyncio.create_task(self._scan_loop())

        await self.client.subscribe_private(
            ["position", "order", "execution", "wallet"],
            self._on_private_message,
        )

        logger.info("All subsystems started.")
        try:
            await asyncio.gather(self._daily_task, self._scan_task)
        except asyncio.CancelledError:
            pass

    async def stop(self):
        logger.info("Shutting down...")
        self._running = False
        await self.executor.stop_monitor()
        await self.executor.close_all_positions()
        await self.client.stop()
        await self.agent.close()
        await self.db.close()
        logger.info("Shutdown complete.")

    def _compute_tp_levels(
        self, entry: float, atr: float, direction: str,
    ) -> Tuple[float, float, float, float]:
        """
        Compute SL and TP1/TP2/TP3 levels.

        Returns: (sl, tp1, tp2, tp3)

        Modes:
          - "r_multiple" (default): SL = entry - atr_mult * ATR (defines 1R)
                                    TP1 = entry + 1R, TP2 = entry + 2R, TP3 = entry + 3R
          - "atr":  SL = entry - sl_mult*ATR
                    TP1/TP2/TP3 = entry ± tp_mults[i]*ATR
        """
        cfg = self.config["execution"]
        mode = cfg.get("tp_mode", "r_multiple")
        sl_mult = cfg.get("atr_multiplier_sl", 1.5)
        r_mults = cfg.get("tp_r_multiples", [1.0, 2.0, 3.0])

        # Guard against zero/negative ATR
        if atr <= 0 or not math.isfinite(atr):
            atr = entry * 0.01  # fallback to 1%

        if direction == "long":
            sl = entry - sl_mult * atr
            risk = entry - sl
            if mode == "atr":
                tp_mults = cfg.get("tp_atr_multiples", [1.5, 2.5, 4.0])
                tps = [entry + m * atr for m in tp_mults]
            else:  # r_multiple
                tps = [entry + risk * r for r in r_mults]
        else:
            sl = entry + sl_mult * atr
            risk = sl - entry
            if mode == "atr":
                tp_mults = cfg.get("tp_atr_multiples", [1.5, 2.5, 4.0])
                tps = [entry - m * atr for m in tp_mults]
            else:
                tps = [entry - risk * r for r in r_mults]

        return sl, tps[0], tps[1], tps[2]

    # ================================================================== #
    #  FULL-PAIR SCAN LOOP
    # ================================================================== #
    async def _scan_loop(self):
        """
        Scans ALL Bybit USDT perpetual pairs.

        Each cycle:
          1. Fetch every active LinearPerpetual symbol via pagination.
          2. Filter by 24h turnover (default >5M USDT).
          3. Analyse each symbol concurrently (bounded by semaphore).
        """
        interval = self.config["system"].get("check_interval_hours", 1)
        scan_top_n = self.config["system"].get("scan_top_n", 0)
        min_turnover = self.config["system"].get(
            "min_turnover_24h", 1_000_000,
        )

        while self._running:
            try:
                logger.info("Fetching full Bybit symbol list...")
                all_symbols = await self.client.get_all_linear_symbols(
                    quote_coin="USDT",
                    only_perpetual=True,
                    min_turnover_24h=min_turnover,
                )

                if scan_top_n > 0:
                    symbols = all_symbols[:scan_top_n]
                else:
                    symbols = all_symbols

                logger.info(
                    f"Scanning {len(symbols)} symbols "
                    f"(of {len(all_symbols)} total USDT perps)"
                )

                semaphore = asyncio.Semaphore(
                    self.config["system"].get("max_concurrent", 10)
                )

                async def analyse_symbol(sym):
                    async with semaphore:
                        try:
                            await self._analyse_and_signal(sym)
                        except Exception as e:
                            logger.error(
                                f"Analysis failed for {sym}: {e}",
                                exc_info=False,
                            )

                await asyncio.gather(
                    *[analyse_symbol(s) for s in symbols]
                )
                logger.info(f"Scan complete. Next in {interval}h.")

            except Exception as e:
                logger.error(f"Scan loop error: {e}", exc_info=True)

            await asyncio.sleep(interval * 3600)

    # ================================================================== #
    #  Per-symbol analysis pipeline
    # ================================================================== #
    async def _analyse_and_signal(self, symbol: str):
        """Full analysis pipeline for a single symbol."""
        klines = await self.client.get_klines(
            symbol, interval="60", limit=200,
        )
        if not klines or len(klines) < 100:
            return

        df = pd.DataFrame(
            klines,
            columns=["ts", "open", "high", "low", "close", "volume", "turnover"],
        )
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("ts").reset_index(drop=True)

        indicators = self._compute_indicators(df)

        deriv = await self.derivatives.get_full_snapshot(symbol)

        # Composite quant engine (Zeta, basis, volume)
        composite = await self.composite_engine.compute(
            symbol=symbol,
            df=df,
            indicators=indicators,
            derivative_snapshot=deriv,
            bar_interval_hours=1.0,
        )

        bias = composite.direction if composite.direction != "neutral" else (
            "long" if indicators.get("ema_9", 0) > indicators.get("ema_21", 0)
            else "short"
        )

        patterns = self._detect_patterns(df)

        prelim = self.scorer.compute(
            symbol=symbol,
            klines_df=df,
            indicators=indicators,
            deriv_snapshot=deriv,
            patterns_detected=patterns,
            bias=bias,
            llm_response=None,
            composite=composite,
            funding_result=getattr(composite, "funding", None),
        )
        if prelim.raw_total < 40:
            await self.journal.log_signal({
                "symbol": symbol,
                "direction": bias,
                "timestamp": datetime.utcnow().isoformat(),
                "entry_price": float(df["close"].iloc[-1]),
                "raw_score": prelim.raw_total,
                "calibrated_confidence": prelim.calibrated_confidence,
                "is_high_confidence": False,
                "gate_reasons": prelim.gate_reasons or ["prelim_below_40"],
                "llm_response": {},
                "indicators": indicators,
                "composite_score": composite.total_score,
                "composite_direction": composite.direction,
                "composite_conviction": composite.conviction,
                "zeta": composite.mean_reversion.zeta,
                "hurst": composite.mean_reversion.hurst,
                "half_life_hours": composite.mean_reversion.half_life_hours,
                "basis_pct": composite.basis.basis_pct,
                "funding_annualized_pct": composite.basis.funding_annualized,
                "volume_zscore": composite.volume.z_score,
                "volume_anomaly_level": composite.volume.anomaly_level,
                "funding_rate_pct": getattr(_f, "rate_pct", 0.0) if _f else 0.0,
                "funding_annualized_pct": getattr(_f, "annualized_pct", 0.0) if _f else 0.0,
                "funding_regime": getattr(_f, "regime", "neutral") if _f else "neutral",
                "funding_contrarian_bias": getattr(_f, "contrarian_bias", "neutral") if _f else "neutral",
                "_breakdown": {
                    "technical_score": prelim.technical_score,
                    "composite_score": prelim.composite_score,
                    "derivative_score": prelim.derivative_score,
                    "sentiment_score": prelim.sentiment_score,
                    "pattern_score": prelim.pattern_score,
                    "llm_score": prelim.llm_score,
                },
            })
            return

        market_context = {
            "symbol": symbol,
            "price": float(df["close"].iloc[-1]),
            "bias": bias,
            "indicators": indicators,
            "derivatives": {
                "cvd": deriv.cvd,
                "cvd_slope": deriv.cvd_slope,
                "cvd_divergence": deriv.cvd_divergence,
                "oi": deriv.oi,
                "oi_change_1h": deriv.oi_change_1h,
                "oi_change_4h": deriv.oi_change_4h,
                "oi_divergence": deriv.oi_divergence,
                "funding_rate": deriv.funding_rate,
                "account_ls_ratio": deriv.account_ls_ratio,
                "top_trader_ls_ratio": deriv.top_trader_ls_ratio,
                "ls_ratio_divergence": deriv.ls_ratio_divergence,
                "smart_money_bias": deriv.smart_money_bias,
            },
            "quant_composite": {
                "total_score": composite.total_score,
                "direction": composite.direction,
                "conviction": composite.conviction,
                "agreement_ratio": composite.agreement_ratio,
                "conflicting_components": composite.conflicting_components,
                "mean_reversion": {
                    "zeta": composite.mean_reversion.zeta,
                    "z_score": composite.mean_reversion.z_score,
                    "phi": composite.mean_reversion.phi,
                    "half_life_hours": composite.mean_reversion.half_life_hours,
                    "hurst": composite.mean_reversion.hurst,
                    "regime": composite.mean_reversion.regime,
                    "adf_pvalue": composite.mean_reversion.adf_pvalue,
                    "is_stationary": composite.mean_reversion.is_stationary,
                },
                "basis": {
                    "basis_pct": composite.basis.basis_pct,
                    "funding_rate": composite.basis.funding_rate,
                    "funding_annualized_pct": composite.basis.funding_annualized,
                    "regime": composite.basis.regime,
                    "signal_bias": composite.basis.signal_bias,
                },
                "volume": {
                    "rvol": composite.volume.rvol,
                    "z_score": composite.volume.z_score,
                    "percentile_rank": composite.volume.percentile_rank,
                    "anomaly_level": composite.volume.anomaly_level,
                    "vwpd": composite.volume.vwpd,
                    "divergence": composite.volume.divergence,
                },
            },
            "patterns": patterns,
            "preliminary_score": prelim.raw_total,
        }

        llm_response = await self.agent.analyse(symbol, market_context)

        final = self.scorer.compute(
            symbol=symbol,
            klines_df=df,
            indicators=indicators,
            deriv_snapshot=deriv,
            patterns_detected=patterns,
            bias=bias,
            llm_response=llm_response,
            composite=composite,
            funding_result=composite.funding,   # NEW
        )

        signal_id = await self.journal.log_signal({
            "symbol": symbol,
            "direction": bias,
            "timestamp": datetime.utcnow().isoformat(),
            "entry_price": float(df["close"].iloc[-1]),
            "raw_score": final.raw_total,
            "calibrated_confidence": final.calibrated_confidence,
            "is_high_confidence": final.is_high_confidence,
            "gate_reasons": final.gate_reasons,
            "llm_response": llm_response,
            "indicators": indicators,
            "composite_score": composite.total_score,
            "composite_direction": composite.direction,
            "composite_conviction": composite.conviction,
            "zeta": composite.mean_reversion.zeta,
            "hurst": composite.mean_reversion.hurst,
            "half_life_hours": composite.mean_reversion.half_life_hours,
            "basis_pct": composite.basis.basis_pct,
            "funding_annualized_pct": composite.basis.funding_annualized,
            "volume_zscore": composite.volume.z_score,
            "volume_anomaly_level": composite.volume.anomaly_level,
            "_breakdown": {
                "technical_score": final.technical_score,
                "composite_score": final.composite_score,
                "derivative_score": final.derivative_score,
                "sentiment_score": final.sentiment_score,
                "pattern_score": final.pattern_score,
                "llm_score": final.llm_score,
            },
        })

        if not final.is_high_confidence:
            logger.debug(
                f"{symbol}: not high confidence "
                f"(raw={final.raw_total:.1f}, "
                f"cal={final.calibrated_confidence:.2f})"
            )
            return

        entry = float(df["close"].iloc[-1])
        atr = indicators.get("atr", entry * 0.01)
        if bias == "long":
            sl = entry - 1.5 * atr
            tp1 = entry + 3.0 * atr
        else:
            sl = entry + 1.5 * atr
            tp1 = entry - 3.0 * atr

        sl, tp1, tp2, tp3 = self._compute_tp_levels(
            entry=entry,
            atr=atr,
            direction=bias,
        )

        signal = TradeSignal(
            symbol=symbol,
            direction=bias,
            entry_price=entry,
            stop_loss=round(sl, 6),
            take_profit_1=round(tp1, 6),
            take_profit_2=round(tp2, 6),
            take_profit_3=round(tp3, 6),
            tp_weights=tuple(self.config["execution"].get(
                "tp_weights", [0.40, 0.35, 0.25]
            )),
            confidence=final.calibrated_confidence,
            leverage=self.config["risk"].get("default_leverage", 5),
            reason=(
                f"Raw={final.raw_total:.0f} "
                f"Composite={composite.total_score:.2f} "
                f"Funding={(getattr(_f, 'rate_pct', 0.0) if _f else 0.0):+.3f}%"
            ),
            metadata={
                "signal_id": signal_id,
                "llm": llm_response,
                # --- NEW: funding flags for risk manager ---
                "funding_rate_pct": getattr(_f, "rate_pct", 0.0) if _f else 0.0,
                "funding_block_long": getattr(_f, "should_block_long", False) if _f else False,
                "funding_block_short": getattr(_f, "should_block_short", False) if _f else False,
                "funding_regime": getattr(_f, "regime", "neutral") if _f else "neutral",
            },
        )

        result = await self.executor.execute_signal(signal)
        if result["status"] == "filled":
            await self.notifier.send_signal_alert(
                signal, final, llm_response,
            )
            await self.journal.mark_executed(signal_id, result)

    # ================================================================== #
    #  Indicator helpers
    # ================================================================== #
    def _compute_indicators(self, df) -> dict:
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
        rvol = df["volume"].iloc[-1] / avg_vol if avg_vol > 0 else 1.0
        return {
            "ema_9": float(ema_9),
            "ema_21": float(ema_21),
            "ema_50": float(ema_50),
            "rsi": float(rsi),
            "atr": float(atr),
            "rvol": float(rvol),
        }

    def _detect_patterns(self, df) -> list:
        patterns = []
        close = df["close"].values
        if len(close) >= 40:
            recent = close[-40:]
            lows = [
                i for i in range(1, len(recent) - 1)
                if recent[i] < recent[i - 1] and recent[i] < recent[i + 1]
            ]
            if len(lows) >= 2:
                if abs(recent[lows[-1]] - recent[lows[-2]]) / recent[lows[-2]] < 0.02:
                    patterns.append("double_bottom")
        return patterns

    # ================================================================== #
    #  Daily 07:00 scheduler
    # ================================================================== #
    async def _daily_scheduler(self):
        target_hour = self.config["auto_tune"].get("run_hour", 7)
        while self._running:
            now = datetime.utcnow()
            target = now.replace(
                hour=target_hour, minute=0, second=0, microsecond=0,
            )
            if target <= now:
                target += timedelta(days=1)
            wait_seconds = (target - now).total_seconds()
            logger.info(
                f"Next auto-improvement in {wait_seconds/3600:.1f}h"
            )
            await asyncio.sleep(wait_seconds)
            try:
                logger.info("DAILY AUTO-IMPROVEMENT STARTING")
                await self.outcome_tracker.run_daily_check()
                results = await self.auto_tuner.run_daily_improvement(
                    self.journal, None,
                )
                if results:
                    await self.notifier.send_tuning_report(results)
                logger.info(
                    f"Auto-improvement complete. "
                    f"{len(results)} adjustments applied."
                )
            except Exception as e:
                logger.error(
                    f"Daily improvement failed: {e}", exc_info=True,
                )

    async def _on_private_message(self, msg: dict):
        topic = msg.get("topic", "")
        if topic == "position":
            for pos in msg.get("data", []):
                logger.debug(
                    f"Position update: {pos.get('symbol')} "
                    f"size={pos.get('size')}"
                )
        elif topic == "execution":
            for ex in msg.get("data", []):
                logger.info(
                    f"Execution: {ex.get('symbol')} {ex.get('side')} "
                    f"{ex.get('execQty')}"
                )


async def main():
    bot = QuantBotV9()
    loop = asyncio.get_event_loop()

    def shutdown():
        asyncio.create_task(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown)

    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())