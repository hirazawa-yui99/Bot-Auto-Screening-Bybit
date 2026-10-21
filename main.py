"""
main.py
Bybit Quant Bot v9.3 — Async orchestrator with tiered real-time + scheduled scan.

Flow per symbol:
  Tier 1 (WS strict)  → strict execution → queued
                                       → skipped/invalid
  Tier 2 (WS relaxed) → relaxed execution → queued
                                         → skipped/invalid
  Tier 3 (scheduled)  → natural retry at next hourly scan

Scheduled scan covers all 261 symbols as safety net.
Real-time WS covers top 20 by turnover with sub-second latency.
"""
import os
import asyncio
import json
import logging
import math
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.bybit_client import BybitV5Client, BybitConfig
from core.realtime_monitor import RealtimeMonitor
from analysis.derivatives import DerivativeAnalyzer
from analysis.confidence import ConfidenceScorer
from analysis.quant_composite import CompositeQuantEngine
from analysis.setup_classifier import SetupClassifier
from analysis.fibonacci import FibonacciAnalyzer
from analysis.orderbook_heatmap import OrderbookHeatmap
from analysis.agreement_gate import AgreementGate
from analysis.smc import SMCAnalyzer
from agent.deepseek_agent import DeepSeekAgent, DeepSeekConfig
from execution.trade_executor import TradeExecutor, TradeSignal
from execution.risk_manager import RiskManager
from execution.signal_queue import SignalQueue
from learning.signal_journal import SignalJournal
from learning.outcome_tracker import OutcomeTracker
from learning.auto_tuner import AutoTuner
from notification.discord_bot import DiscordNotifier
from notification.chart_generator import ChartGenerator
from persistence.database import Database

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

        # ============================================================== #
        #  Core clients
        # ============================================================== #
        self.client = BybitV5Client(BybitConfig(
            api_key=self.config["api"]["bybit_key"],
            api_secret=self.config["api"]["bybit_secret"],
            testnet=self.config["api"].get("testnet", False),
            category="linear",
        ))

        self.db = Database(self.config["database"])
        self.journal = SignalJournal(self.db)

        # ============================================================== #
        #  Analysis
        # ============================================================== #
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

        self.setup_classifier = SetupClassifier(
            self.config.get("setups", {}),
        )
        self.fib_analyzer = FibonacciAnalyzer()
        self.heatmap = OrderbookHeatmap()
        self.agreement_gate = AgreementGate(
            min_agree=self.config.get("agreement", {}).get(
                "min_agree", 3,
            ),
        )
        self.smc_analyzer = SMCAnalyzer()

        # ============================================================== #
        #  LLM
        # ============================================================== #
        ds = self.config.get("deepseek", {})
        self.agent = DeepSeekAgent(DeepSeekConfig(
            api_key=ds.get("api_key", ""),
            base_url=ds.get("base_url", "https://api.deepseek.com/v1"),
            model=ds.get("model", "deepseek-v4-pro"),
            model_high_confidence=ds.get(
                "model_high_confidence", "deepseek-v4-pro",
            ),
            temperature=ds.get("temperature", 0.3),
            max_tokens=ds.get("max_tokens", 1500),
            timeout=ds.get("timeout", 45),
            error_cooldown_sec=ds.get("error_cooldown_sec", 900),
        ))

        # ============================================================== #
        #  Risk + Execution
        # ============================================================== #
        self.risk_manager = RiskManager(self.config["risk"])
        self.executor = TradeExecutor(
            self.client,
            self.risk_manager,
            self.config["execution"],
            on_position_closed=self._on_position_closed,
            on_limit_stalled=self._on_limit_stalled,
        )

        # ============================================================== #
        #  Learning
        # ============================================================== #
        self.outcome_tracker = OutcomeTracker(self.client, self.journal)
        self.auto_tuner = AutoTuner(self.config["auto_tune"], self.scorer)

        # ============================================================== #
        #  Notification
        # ============================================================== #
        self.notifier = DiscordNotifier(self.config["api"])
        self.chart_generator = ChartGenerator()

        # ============================================================== #
        #  Signal dedup (cooldown + recent enqueue)
        # ============================================================== #
        self._alerted_signals: Dict[str, Dict] = {}
        self._signal_cooldown_hours: float = float(
            self.config.get("execution", {}).get(
                "signal_cooldown_hours", 4,
            )
        )
        self._prune_alerted_after_hours: float = float(
            self.config.get("execution", {}).get(
                "prune_alerted_after_hours", 24,
            )
        )

        # Recent enqueue guard — prevents scheduled scan + real-time
        # from double-enqueueing the same symbol within 5 min
        self._recent_enqueues: Dict[str, float] = {}
        self._recent_enqueue_window_sec: float = 300.0

        # ============================================================== #
        #  Decoupled auto-trade queue
        # ============================================================== #
        self.signal_queue = SignalQueue(
            self.db, self.config.get("auto_trade", {}),
        )
        self.signal_queue.attach(
            self.executor,
            on_filled=self._on_trade_filled,
        )

        # ============================================================== #
        #  Real-time monitor (wired in start())
        # ============================================================== #
        self.realtime_monitor: Optional[RealtimeMonitor] = None

        # ============================================================== #
        #  Runtime
        # ============================================================== #
        self._running = False
        self._daily_task: Optional[asyncio.Task] = None
        self._scan_task: Optional[asyncio.Task] = None

    # ================================================================== #
    #  Lifecycle
    # ================================================================== #
    async def start(self):
        logger.info("=" * 60)
        logger.info("Bybit Quant Bot v9.3 starting — TIERED MONITOR")
        logger.info("=" * 60)

        self._running = True
        await self.client.start()
        await self.db.connect()
        await self.executor.start_monitor()
        await self.signal_queue.start()

        self._daily_task = asyncio.create_task(self._daily_scheduler())
        self._scan_task = asyncio.create_task(self._scan_loop())

        # Load persisted cooldowns
        try:
            rows = await self.db.fetch(
                "SELECT symbol, alerted_at, direction, confidence, signal_id "
                "FROM signal_cooldowns "
                "WHERE alerted_at > NOW() - INTERVAL '24 hours'"
            )
            for r in rows:
                self._alerted_signals[r["symbol"]] = {
                    "timestamp": r["alerted_at"].replace(tzinfo=None),
                    "direction": r["direction"],
                    "confidence": r["confidence"],
                    "signal_id": r["signal_id"],
                }
            logger.info(
                f"Loaded {len(self._alerted_signals)} signal cooldowns"
            )
        except Exception as e:
            logger.debug(f"Cooldown load failed: {e}")

        await self.client.subscribe_private(
            ["position", "order", "execution", "wallet"],
            self._on_private_message,
        )

        # Real-time monitor (top N symbols)
        rt_cfg = self.config.get("realtime", {})
        if rt_cfg.get("enabled", True):
            try:
                self.realtime_monitor = RealtimeMonitor(
                    rest_client=self.client,
                    interval="60",
                    on_candle_close=self._on_realtime_candle,
                    max_connections=int(
                        rt_cfg.get("max_connections", 2)
                    ),
                    symbols=list(rt_cfg.get("symbols") or []),
                    selection=rt_cfg.get(
                        "selection", "top_by_turnover",
                    ),
                    max_symbols=int(rt_cfg.get("max_symbols", 20)),
                    analysis_concurrency=int(
                        rt_cfg.get("analysis_concurrency", 3)
                    ),
                    callback_timeout=int(
                        rt_cfg.get("callback_timeout_sec", 180)
                    ),
                    gap_fill_recent_min=int(
                        rt_cfg.get("gap_fill_recent_minutes", 10)
                    ),
                    min_turnover=float(
                        self.config["system"].get(
                            "min_turnover_24h", 1_000_000,
                        )
                    ),
                )
                await self.realtime_monitor.start()
            except Exception as e:
                logger.error(
                    f"RealtimeMonitor failed to start: {e}",
                    exc_info=True,
                )
                self.realtime_monitor = None
        else:
            logger.info("RealtimeMonitor disabled in config")

        logger.info("All subsystems started.")
        try:
            await asyncio.gather(
                self._daily_task, self._scan_task,
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            pass

    async def stop(self):
        logger.info("Shutting down...")
        self._running = False

        # 1. Stop realtime monitor first (stops firing new analyses)
        if self.realtime_monitor is not None:
            try:
                await self.realtime_monitor.stop()
            except Exception as e:
                logger.warning(f"realtime stop error: {e}")

        # 2. Cancel scheduled tasks
        tasks_to_cancel = []
        for attr in ("_scan_task", "_daily_task"):
            task = getattr(self, attr, None)
            if task is not None and not task.done():
                task.cancel()
                tasks_to_cancel.append(task)
        if tasks_to_cancel:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                    timeout=15,
                )
                logger.info("Background tasks exited cleanly")
            except asyncio.TimeoutError:
                logger.warning(
                    "Background tasks did not exit within 15s"
                )

        # 3. Close resources in order
        await self.signal_queue.stop()
        await self.executor.stop_monitor()
        await self.executor.close_all_positions()
        await self.client.stop()
        await self.agent.close()
        await self.db.close()
        logger.info("Shutdown complete.")

    # ================================================================== #
    #  Scheduled scan loop
    # ================================================================== #
    async def _scan_loop(self):
        interval = self.config["system"].get("check_interval_hours", 1)
        scan_top_n = self.config["system"].get("scan_top_n", 0)
        min_turnover = self.config["system"].get(
            "min_turnover_24h", 1_000_000,
        )

        while self._running:
            try:
                self._prune_alerted_signals()

                logger.info("Fetching full Bybit symbol list...")
                all_symbols = await self.client.get_all_linear_symbols(
                    quote_coin="USDT",
                    only_perpetual=True,
                    min_turnover_24h=min_turnover,
                )
                symbols = (
                    all_symbols[:scan_top_n] if scan_top_n > 0
                    else all_symbols
                )
                logger.info(
                    f"Scanning {len(symbols)} symbols "
                    f"(of {len(all_symbols)} total USDT perps)"
                )

                sem = asyncio.Semaphore(
                    self.config["system"].get("max_concurrent", 3)
                )

                async def run(sym):
                    async with sem:
                        if not self._running:
                            return
                        try:
                            await self._analyse_and_signal(sym, tier=1)
                        except Exception as e:
                            logger.error(f"Analysis failed {sym}: {e}")

                await asyncio.gather(*[run(s) for s in symbols])
                logger.info(f"Scan complete. Next in {interval}h.")

            except asyncio.CancelledError:
                logger.info("Scan loop cancelled")
                return
            except Exception as e:
                logger.error(f"Scan loop error: {e}", exc_info=True)

            # Interruptible sleep
            remaining = interval * 3600
            while remaining > 0 and self._running:
                chunk = min(5.0, remaining)
                await asyncio.sleep(chunk)
                remaining -= chunk
            if not self._running:
                return

    # ================================================================== #
    #  Real-time callback — tiered retry
    # ================================================================== #
    async def _on_realtime_candle(self, symbol: str):
        """
        Tier 1: strict → queued or skipped/invalid
        Tier 2: relaxed → queued or skipped
        Tier 3: scheduled scan (natural fallback)
        """
        rt_cfg = self.config.get("realtime", {})
        tier2_enabled = rt_cfg.get("tier2_enabled", True)
        tier2_for_invalid = rt_cfg.get("tier2_for_invalid", False)

        # ---------- Tier 1 ----------
        try:
            r1 = await self._analyse_and_signal(symbol, tier=1)
        except Exception as e:
            logger.error(
                f"{symbol} tier1 exception: {type(e).__name__}: {e}"
            )
            return

        status1 = r1.get("status", "error")
        reason1 = r1.get("reason", "")

        if status1 == "queued":
            logger.info(
                f"{symbol} tier1 QUEUED (id={r1.get('signal_id')})"
            )
            return
        if status1 in ("cooldown", "error"):
            logger.debug(f"{symbol} tier1 {status1}: {reason1}")
            return
        if not tier2_enabled:
            logger.debug(
                f"{symbol} tier1 {status1} ({reason1}) — tier2 disabled"
            )
            return
        if status1 == "invalid" and not tier2_for_invalid:
            logger.debug(
                f"{symbol} tier1 INVALID ({reason1}) — no tier2 "
                f"(analysis rejected, quality preserved)"
            )
            return

        # ---------- Tier 2 ----------
        logger.info(
            f"{symbol} tier1 {status1.upper()} ({reason1}) "
            f"— attempting tier2"
        )
        try:
            r2 = await self._analyse_and_signal(
                symbol, tier=2, allow_recent_enqueue=True,
            )
        except Exception as e:
            logger.error(
                f"{symbol} tier2 exception: {type(e).__name__}: {e}"
            )
            return

        status2 = r2.get("status", "error")
        reason2 = r2.get("reason", "")
        if status2 == "queued":
            logger.info(
                f"{symbol} tier2 QUEUED (id={r2.get('signal_id')})"
            )
        else:
            logger.info(
                f"{symbol} tier2 {status2.upper()} ({reason2}) "
                f"— waiting for scheduled scan"
            )

    # ================================================================== #
    #  Main analysis pipeline
    # ================================================================== #
    async def _analyse_and_signal(
        self,
        symbol: str,
        tier: int = 1,
        allow_recent_enqueue: bool = False,
    ) -> Dict:
        """
        Returns status dict:
          queued | invalid | skipped | cooldown | error
        """
        # 1. Early skip
        skip_reason = self._should_skip_symbol(
            symbol, allow_recent_enqueue=allow_recent_enqueue,
        )
        if skip_reason:
            logger.debug(f"{symbol}: skip — {skip_reason}")
            return {"status": "cooldown", "reason": skip_reason,
                    "symbol": symbol}

        # 2. Multi-TF fetch
        df_15m, df_1h, df_4h = await self._fetch_multi_tf(symbol)
        if df_1h is None or len(df_1h) < 100:
            return {"status": "invalid", "reason": "insufficient_data",
                    "symbol": symbol}

        # 3. Indicators + deriv FIRST
        indicators = self._compute_indicators(df_1h)
        deriv = await self.derivatives.get_full_snapshot(symbol)

        # 4a. Momentum guard (direction-aware, ATR-normalized)
        momentum_cfg = self.config.get("execution", {})
        roc_threshold = float(
            momentum_cfg.get("momentum_guard_roc_pct", 3.0)
        )
        roc_lookback = int(
            momentum_cfg.get("momentum_guard_lookback_bars", 5)
        )
        sigma_threshold = float(
            momentum_cfg.get("momentum_guard_sigma", 2.5)
        )
        atr_pct_floor = float(
            momentum_cfg.get("momentum_guard_atr_pct_floor", 0.5)
        )

        ema_9 = indicators.get("ema_9", 0.0)
        ema_21 = indicators.get("ema_21", 0.0)
        if ema_9 > ema_21:
            prelim_bias = "long"
        elif ema_9 < ema_21:
            prelim_bias = "short"
        else:
            prelim_bias = "neutral"

        if roc_threshold > 0 and len(df_1h) > roc_lookback + 2:
            try:
                closes = df_1h["close"].values
                cur = float(closes[-1])
                past = float(closes[-(roc_lookback + 1)])
                if past > 0 and cur > 0:
                    roc_pct = (cur - past) / past * 100
                    atr_pct = (
                        float(indicators.get("atr", 0.0)) / cur * 100
                        if cur > 0 else 0.0
                    )
                    if atr_pct > 0:
                        expected = atr_pct * (roc_lookback ** 0.5)
                        sigma = (
                            abs(roc_pct) / expected
                            if expected > 0 else 0.0
                        )
                    else:
                        sigma = 999.0
                    chase = (
                        (prelim_bias == "long" and roc_pct > 0)
                        or (prelim_bias == "short" and roc_pct < 0)
                    )
                    abs_gate = abs(roc_pct) >= roc_threshold
                    sigma_gate = sigma >= sigma_threshold
                    if atr_pct < atr_pct_floor:
                        sigma_gate = True
                    if chase and abs_gate and sigma_gate:
                        logger.info(
                            f"{symbol}: momentum guard — "
                            f"bias={prelim_bias} ROC={roc_pct:+.2f}% "
                            f"ATR%={atr_pct:.2f}% σ={sigma:.1f}"
                        )
                        await self._log_rejected(
                            symbol=symbol, bias=prelim_bias,
                            indicators=indicators, deriv=deriv,
                            composite=None, agreement=None,
                            price=cur,
                            extra_gate=(
                                f"momentum_chase:"
                                f"{roc_pct:+.2f}%/{sigma:.1f}σ"
                            ),
                        )
                        return {"status": "invalid",
                                "reason": "momentum_guard",
                                "symbol": symbol}
            except Exception as e:
                logger.debug(f"momentum guard err {symbol}: {e}")

        # 4b. ATR filter
        atr_val = float(indicators.get("atr", 0.0))
        price_now = float(df_1h["close"].iloc[-1])
        if price_now > 0 and atr_val > 0:
            atr_pct_val = atr_val / price_now * 100
            min_atr = float(momentum_cfg.get("min_atr_pct", 0.3))
            max_atr = float(momentum_cfg.get("max_atr_pct", 5.0))
            if atr_pct_val < min_atr:
                logger.debug(
                    f"{symbol}: ATR%={atr_pct_val:.2f}% < {min_atr}%"
                )
                return {"status": "invalid", "reason": "atr_too_low",
                        "symbol": symbol}
            if atr_pct_val > max_atr:
                logger.info(
                    f"{symbol}: ATR%={atr_pct_val:.2f}% > {max_atr}%"
                )
                await self._log_rejected(
                    symbol=symbol, bias="neutral",
                    indicators=indicators, deriv=deriv,
                    composite=None, agreement=None, price=price_now,
                    extra_gate=f"atr_too_high:{atr_pct_val:.2f}%",
                )
                return {"status": "invalid", "reason": "atr_too_high",
                        "symbol": symbol}

        # 5. Composite
        composite = await self.composite_engine.compute(
            symbol=symbol,
            df=df_1h,
            indicators=indicators,
            derivative_snapshot=deriv,
            bar_interval_hours=1.0,
        )

        bias = (
            composite.direction
            if composite.direction != "neutral"
            else (
                "long"
                if indicators.get("ema_9", 0) > indicators.get("ema_21", 0)
                else "short"
            )
        )
        patterns = self._detect_patterns(df_1h)
        current_price = float(df_1h["close"].iloc[-1])

        # 6. SMC + Fibonacci + heatmap
        try:
            smc = self.smc_analyzer.analyze(df_1h)
        except Exception as e:
            logger.debug(f"SMC err {symbol}: {e}")
            smc = None
        try:
            fibo = self.fib_analyzer.analyze(df_1h, bias)
        except Exception as e:
            logger.debug(f"Fibo err {symbol}: {e}")
            fibo = None
        try:
            ob = await self.client.get_orderbook(symbol, limit=50)
            heatmap = self.heatmap.analyze(ob)
        except Exception as e:
            logger.debug(f"Heatmap err {symbol}: {e}")
            heatmap = None

        # 7. Directional contributions
        smc_dir = "neutral"
        if smc is not None:
            if smc.bos in ("bullish", "bearish"):
                smc_dir = "long" if smc.bos == "bullish" else "short"
            elif smc.choch in ("bullish", "bearish"):
                smc_dir = "long" if smc.choch == "bullish" else "short"

        deriv_dir = "neutral"
        if (deriv.cvd_divergence == "bullish"
                or deriv.smart_money_bias == "long"):
            deriv_dir = "long"
        elif (deriv.cvd_divergence == "bearish"
              or deriv.smart_money_bias == "short"):
            deriv_dir = "short"

        heatmap_dir = heatmap.direction if heatmap else "neutral"
        fibo_dir = "neutral"
        fibo_in_zone = False
        if fibo is not None and fibo.valid:
            fibo_dir = bias
            fibo_in_zone = fibo.in_zone

        # 8. Agreement gate
        agreement = self.agreement_gate.check(
            smc_dir=smc_dir,
            deriv_dir=deriv_dir,
            quant_dir=composite.direction,
            heatmap_dir=heatmap_dir,
            fib_direction=fibo_dir,
            fib_valid=fibo_in_zone,
        )
        if not agreement.passed:
            logger.debug(
                f"{symbol}: agreement failed — {agreement.reason}"
            )
            await self._log_rejected(
                symbol=symbol, bias=bias, indicators=indicators,
                deriv=deriv, composite=composite,
                agreement=agreement, price=current_price,
            )
            return {"status": "invalid",
                    "reason": "agreement_failed",
                    "symbol": symbol}

        # 9. Preliminary confidence
        prelim = self.scorer.compute(
            symbol=symbol,
            klines_df=df_1h,
            indicators=indicators,
            deriv_snapshot=deriv,
            patterns_detected=patterns,
            bias=bias,
            llm_response=None,
            composite=composite,
            funding_result=composite.funding,
        )
        if prelim.raw_total < 40:
            await self._log_rejected(
                symbol=symbol, bias=bias, indicators=indicators,
                deriv=deriv, composite=composite,
                agreement=agreement, prelim=prelim,
                price=current_price,
            )
            return {"status": "invalid",
                    "reason": "low_prelim",
                    "symbol": symbol}

        # 10. DeepSeek
        market_context = {
            "symbol": symbol,
            "price": current_price,
            "bias": bias,
            "indicators": indicators,
            "derivatives": {
                "cvd": deriv.cvd,
                "cvd_divergence": deriv.cvd_divergence,
                "oi_change_1h": deriv.oi_change_1h,
                "oi_change_4h": deriv.oi_change_4h,
                "oi_divergence": deriv.oi_divergence,
                "funding_rate": deriv.funding_rate,
                "smart_money_bias": deriv.smart_money_bias,
            },
            "smc": {
                "bos": smc.bos if smc else "none",
                "choch": smc.choch if smc else "none",
                "liquidity_sweep": (
                    smc.liquidity_sweep if smc else "none"
                ),
            },
            "fibo": {
                "zone": fibo.zone_name if fibo else "",
                "in_zone": fibo_in_zone,
                "entry_zone": (
                    [fibo.entry_zone_low, fibo.entry_zone_high]
                    if fibo else []
                ),
            },
            "heatmap": {
                "imbalance": heatmap.imbalance if heatmap else 0.0,
                "direction": heatmap_dir,
                "strong": (
                    getattr(heatmap, "strong", False) if heatmap else False
                ),
            },
            "agreement": {
                "direction": agreement.direction,
                "count": agreement.agreement_count,
                "details": agreement.details,
            },
            "composite": {
                "total_score": composite.total_score,
                "direction": composite.direction,
                "conviction": composite.conviction,
            },
            "preliminary_score": prelim.raw_total,
        }
        llm_response = await self.agent.analyse(symbol, market_context)

        # 11. Final confidence
        final = self.scorer.compute(
            symbol=symbol,
            klines_df=df_1h,
            indicators=indicators,
            deriv_snapshot=deriv,
            patterns_detected=patterns,
            bias=bias,
            llm_response=llm_response,
            composite=composite,
            funding_result=composite.funding,
        )

        llm_dir = llm_response.get("direction", "neutral")
        if not self._llm_agrees_with_quant_llm(
            llm_dir, bias, agreement.direction,
        ):
            logger.info(
                f"{symbol}: LLM {llm_dir} vs agreement {agreement.direction}"
            )
            await self._log_rejected(
                symbol=symbol, bias=bias, indicators=indicators,
                deriv=deriv, composite=composite,
                agreement=agreement, prelim=final,
                price=current_price,
                extra_gate=(
                    f"llm_mismatch:{llm_dir}!={agreement.direction}"
                ),
            )
            return {"status": "invalid",
                    "reason": "llm_mismatch",
                    "symbol": symbol}

        if not final.is_high_confidence:
            logger.debug(
                f"{symbol}: low confidence raw={final.raw_total:.1f} "
                f"cal={final.calibrated_confidence:.2f}"
            )
            await self._log_rejected(
                symbol=symbol, bias=bias, indicators=indicators,
                deriv=deriv, composite=composite,
                agreement=agreement, prelim=final,
                price=current_price,
            )
            return {"status": "invalid",
                    "reason": "low_confidence",
                    "symbol": symbol}

        # 12. Setup classification
        setups = self.setup_classifier.classify_all(
            df_15m=df_15m, df_1h=df_1h, df_4h=df_4h,
            composite=composite, deriv=deriv, bias=bias,
        )
        for setup in setups.valid_setups:
            try:
                entry_mid = (
                    setup.entry_zone_low + setup.entry_zone_high
                ) / 2
                _sl, _tp1, _tp2, _tp3 = self._compute_tp_levels(
                    entry=entry_mid, atr=setup.atr,
                    direction=setup.direction,
                )
                setup.sl = _sl
                setup.tp1 = _tp1
                setup.tp2 = _tp2
                setup.tp3 = _tp3
            except Exception as e:
                logger.debug(
                    f"{symbol}: sl/tp fail {setup.setup_type}: {e}"
                )

        if setups.count == 0:
            await self._log_rejected(
                symbol=symbol, bias=bias, indicators=indicators,
                deriv=deriv, composite=composite,
                agreement=agreement, prelim=final,
                price=current_price, extra_gate="no_setups",
            )
            return {"status": "invalid", "reason": "no_setups",
                    "symbol": symbol}

        best = setups.highest_confidence()
        if best is None:
            return {"status": "invalid", "reason": "no_best_setup",
                    "symbol": symbol}

        # 13. Entry price (Fibonacci zone if outside)
        entry = current_price
        if fibo is not None and fibo.valid:
            if (fibo.entry_zone_low <= current_price
                    <= fibo.entry_zone_high):
                entry = current_price
            else:
                entry = (
                    fibo.entry_zone_low if bias == "long"
                    else fibo.entry_zone_high
                )

        sl, tp1, tp2, tp3 = self._compute_tp_levels(
            entry=entry, atr=best.atr, direction=bias,
        )
        signal = TradeSignal(
            symbol=symbol,
            direction=bias,
            entry_price=round(entry, 6),
            stop_loss=round(sl, 6),
            take_profit_1=round(tp1, 6),
            take_profit_2=round(tp2, 6),
            take_profit_3=round(tp3, 6),
            tp_weights=tuple(self.config["execution"].get(
                "tp_weights", [0.40, 0.35, 0.25]
            )),
            confidence=final.calibrated_confidence,
            leverage=self.config["risk"].get("default_leverage", 10),
            reason=(
                f"tier={tier} setup={best.setup_type} "
                f"raw={final.raw_total:.0f} "
                f"agree={agreement.agreement_count}"
            ),
            metadata={
                "setup_type": best.setup_type,
                "setups": setups.labels,
                "agreement": agreement.details,
                "smc_dir": smc_dir,
                "deriv_dir": deriv_dir,
                "quant_dir": composite.direction,
                "heatmap_dir": heatmap_dir,
                "fibo_zone": fibo.zone_name if fibo else "",
                "fibo_in_zone": fibo_in_zone,
                "funding_block_long": (
                    composite.funding.should_block_long
                ),
                "funding_block_short": (
                    composite.funding.should_block_short
                ),
                "funding_rate_pct": composite.funding.rate_pct,
            },
        )

        # 14. Late skip
        late_skip = self._should_skip_symbol(
            symbol, allow_recent_enqueue=allow_recent_enqueue,
        )
        if late_skip:
            logger.info(
                f"{symbol}: late skip — {late_skip}"
            )
            return {"status": "skipped",
                    "reason": f"late:{late_skip}",
                    "symbol": symbol, "tier": tier}

        # 15. Plan (per-setup delay)
        delay_cfg = self.config.get("auto_trade", {})
        delay_by_setup = delay_cfg.get("delay_seconds_by_setup", {})
        default_delay = int(delay_cfg.get("delay_seconds", 30))
        delay_seconds = int(
            delay_by_setup.get(best.setup_type, default_delay)
        )
        plan = {
            "direction": bias,
            "entry_price": float(signal.entry_price),
            "stop_loss": float(signal.stop_loss),
            "take_profit_1": float(signal.take_profit_1),
            "take_profit_2": (
                float(signal.take_profit_2)
                if signal.take_profit_2 else None
            ),
            "take_profit_3": (
                float(signal.take_profit_3)
                if signal.take_profit_3 else None
            ),
            "tp_weights": list(signal.tp_weights),
            "leverage": int(signal.leverage),
            "delay_seconds": delay_seconds,
            "setup_type": best.setup_type,
            "setups": setups.labels,
            "agreement": agreement.details,
            "funding_block_long": (
                composite.funding.should_block_long
            ),
            "funding_block_short": (
                composite.funding.should_block_short
            ),
            "metadata": signal.metadata,
        }

        # 16. Log signal
        signal_dict = {
            "symbol": symbol,
            "direction": bias,
            "timestamp": datetime.utcnow().isoformat(),
            "entry_price": float(signal.entry_price),
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
            "half_life_hours": (
                composite.mean_reversion.half_life_hours
            ),
            "basis_pct": composite.basis.basis_pct,
            "funding_annualized_pct": (
                composite.funding.annualized_pct
            ),
            "volume_zscore": composite.volume.z_score,
            "volume_anomaly_level": composite.volume.anomaly_level,
            "funding_rate_pct": composite.funding.rate_pct,
            "funding_regime": composite.funding.regime,
            "funding_contrarian_bias": (
                composite.funding.contrarian_bias
            ),
            "_breakdown": {
                "technical_score": final.technical_score,
                "composite_score": final.composite_score,
                "derivative_score": final.derivative_score,
                "sentiment_score": final.sentiment_score,
                "pattern_score": final.pattern_score,
                "llm_score": final.llm_score,
            },
        }
        signal_id = await self.journal.log_signal(
            signal_dict, plan=plan, execution_status="pending",
        )
        if signal_id <= 0:
            return {"status": "error", "reason": "journal_failed",
                    "symbol": symbol}

        # Record enqueue timestamp
        self._recent_enqueues[symbol] = time.time()

        # 17. Execute (tier-aware)
        override = None
        if tier == 2:
            rt_cfg = self.config.get("realtime", {})
            dev_mult = float(
                rt_cfg.get("tier2_deviation_multiplier", 1.5)
            )
            tol_mult = float(
                rt_cfg.get("tier2_market_tolerance_multiplier", 100.0)
            )
            override = {
                "max_deviation_pct": (
                    self.executor.max_deviation_pct * dev_mult
                ),
                "cancel_on_rally_pct": 999.0,
                "market_tolerance_pct": (
                    self.executor.market_tolerance_pct * tol_mult
                ),
            }
            logger.info(
                f"{symbol} tier2 override: "
                f"max_dev={override['max_deviation_pct']:.2f}%, "
                f"market_tol={override['market_tolerance_pct']:.2f}%"
            )

        result = await self.executor.execute_signal(
            signal, override_params=override,
        )
        exec_status = result.get("status", "error")

        if exec_status in ("filled", "pending_limit"):
            # Discord dispatch — tier1 only by default
            rt_cfg = self.config.get("realtime", {})
            suppress_t2 = rt_cfg.get("suppress_tier2_alerts", True)
            if tier == 1 or not suppress_t2:
                await self._dispatch_multi_setup_alerts(
                    symbol=symbol,
                    setups=setups,
                    df_1h=df_1h,
                    composite=composite,
                    deriv=deriv,
                    indicators=indicators,
                    breakdown=final,
                    llm_response=llm_response,
                    agreement=agreement,
                    smc=smc,
                    fibo=fibo,
                    heatmap=heatmap,
                    signal_id=signal_id,
                    delay_seconds=delay_seconds,
                )
            logger.info(
                f"{symbol}: signal {signal_id} queued "
                f"(tier={tier}, setup={best.setup_type}, "
                f"delay={delay_seconds}s, "
                f"agreement={agreement.agreement_count})"
            )
            return {
                "status": "queued",
                "signal_id": signal_id,
                "symbol": symbol,
                "tier": tier,
            }
        elif exec_status in ("skipped", "rejected"):
            return {
                "status": "skipped",
                "reason": result.get("reason", ""),
                "symbol": symbol,
                "tier": tier,
            }
        else:
            return {
                "status": "error",
                "reason": result.get("reason", ""),
                "symbol": symbol,
                "tier": tier,
            }

    # ================================================================== #
    #  Helpers
    # ================================================================== #
    async def _fetch_multi_tf(self, symbol: str) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame],
    ]:
        async def _fetch(interval, limit):
            try:
                klines = await self.client.get_klines(
                    symbol, interval=interval, limit=limit,
                )
                if not klines or len(klines) < 40:
                    return None
                df = pd.DataFrame(
                    klines,
                    columns=["ts", "open", "high", "low", "close",
                             "volume", "turnover"],
                )
                for c in ("open", "high", "low", "close", "volume"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                return df.sort_values("ts").reset_index(drop=True)
            except Exception as e:
                logger.debug(f"fetch {interval} err {symbol}: {e}")
                return None

        df_15m, df_1h, df_4h = await asyncio.gather(
            _fetch("15", 100),
            _fetch("60", 200),
            _fetch("240", 100),
        )
        return df_15m, df_1h, df_4h

    def _compute_indicators(self, df) -> dict:
        close, high, low = df["close"], df["high"], df["low"]
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
                if recent[i] < recent[i - 1]
                and recent[i] < recent[i + 1]
            ]
            if len(lows) >= 2:
                if (abs(recent[lows[-1]] - recent[lows[-2]])
                        / recent[lows[-2]] < 0.02):
                    patterns.append("double_bottom")
        return patterns

    def _compute_tp_levels(
        self, entry: float, atr: float, direction: str,
    ) -> Tuple[float, float, float, float]:
        cfg = self.config["execution"]
        mode = cfg.get("tp_mode", "r_multiple")
        sl_mult = cfg.get("atr_multiplier_sl", 1.5)
        r_mults = cfg.get("tp_r_multiples", [1.0, 2.0, 3.0])

        if atr <= 0 or not math.isfinite(atr):
            atr = entry * 0.01

        if direction == "long":
            sl = entry - sl_mult * atr
            risk = entry - sl
            if mode == "atr":
                tp_mults = cfg.get("tp_atr_multiples", [1.5, 2.5, 4.0])
                tps = [entry + m * atr for m in tp_mults]
            else:
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

    def _llm_agrees_with_quant_llm(
        self, llm_dir: str, bias: str, agreement_dir: str,
    ) -> bool:
        if llm_dir == "neutral":
            return True
        return llm_dir == agreement_dir

    def _should_skip_symbol(
        self, symbol: str, allow_recent_enqueue: bool = False,
    ) -> Optional[str]:
        # 1. Active position
        if symbol in self.executor.active_positions:
            return "active_position"

        # 2. Pending entry
        pending = self.executor._pending_entries.get(symbol)
        if pending is not None and not pending.done():
            return "pending_entry"

        # 3. Cooldown window
        if self._signal_cooldown_hours > 0:
            entry = self._alerted_signals.get(symbol)
            if entry is not None:
                age = (
                    datetime.utcnow() - entry["timestamp"]
                ).total_seconds() / 3600
                if age < self._signal_cooldown_hours:
                    return (
                        f"cooldown:alerted_{age:.1f}h_ago"
                        f"(limit={self._signal_cooldown_hours}h)"
                    )

        # 4. Recent enqueue
        if not allow_recent_enqueue:
            now = time.time()
            last = self._recent_enqueues.get(symbol, 0.0)
            if now - last < self._recent_enqueue_window_sec:
                return f"recent_enqueue:{now - last:.0f}s_ago"

        # 5. SignalQueue pending symbol
        if self.signal_queue.is_pending(symbol):
            return "pending_execution_queue"

        return None

    def _record_alert(
        self, symbol: str, direction: str,
        confidence: float, signal_id: int,
    ):
        self._alerted_signals[symbol] = {
            "timestamp": datetime.utcnow(),
            "direction": direction,
            "confidence": confidence,
            "signal_id": signal_id,
        }
        asyncio.create_task(self._persist_alert(
            symbol, direction, confidence, signal_id,
        ))

    async def _persist_alert(
        self, symbol, direction, confidence, signal_id,
    ):
        try:
            await self.db.execute(
                "INSERT INTO signal_cooldowns "
                "(symbol, alerted_at, direction, confidence, signal_id) "
                "VALUES ($1, NOW(), $2, $3, $4) "
                "ON CONFLICT (symbol) DO UPDATE SET "
                "alerted_at = NOW(), direction = EXCLUDED.direction, "
                "confidence = EXCLUDED.confidence, "
                "signal_id = EXCLUDED.signal_id",
                symbol, direction, confidence, signal_id,
            )
        except Exception as e:
            logger.debug(f"persist alert failed {symbol}: {e}")

    def _prune_alerted_signals(self):
        if self._prune_alerted_after_hours > 0:
            cutoff = datetime.utcnow() - timedelta(
                hours=self._prune_alerted_after_hours,
            )
            self._alerted_signals = {
                s: info for s, info in self._alerted_signals.items()
                if info["timestamp"] >= cutoff
            }
        now = time.time()
        self._recent_enqueues = {
            s: t for s, t in self._recent_enqueues.items()
            if now - t < 600
        }

    async def _log_rejected(
        self,
        symbol: str,
        bias: str,
        indicators: Dict,
        deriv,
        composite,
        agreement,
        prelim=None,
        price: float = 0.0,
        extra_gate: str = "",
    ):
        if prelim is not None:
            raw = float(getattr(prelim, "raw_total", 0.0))
            cal = float(getattr(prelim, "calibrated_confidence", 0.0))
            gates = list(getattr(prelim, "gate_reasons", []) or [])
            bd = {
                "technical_score": float(
                    getattr(prelim, "technical_score", 0.0)
                ),
                "composite_score": float(
                    getattr(prelim, "composite_score", 0.0)
                ),
                "derivative_score": float(
                    getattr(prelim, "derivative_score", 0.0)
                ),
                "sentiment_score": float(
                    getattr(prelim, "sentiment_score", 0.0)
                ),
                "pattern_score": float(
                    getattr(prelim, "pattern_score", 0.0)
                ),
                "llm_score": float(
                    getattr(prelim, "llm_score", 0.0)
                ),
            }
        else:
            raw = cal = 0.0
            gates = []
            bd = {k: 0.0 for k in (
                "technical_score", "composite_score", "derivative_score",
                "sentiment_score", "pattern_score", "llm_score",
            )}

        if agreement is not None and not agreement.passed:
            gates.append(f"agreement:{agreement.reason}")
        if extra_gate:
            gates.append(extra_gate)
        if not gates:
            gates = ["rejected_unknown"]

        entry_price = float(price) if price > 0 else 0.0
        if entry_price <= 0 and composite is not None:
            try:
                entry_price = float(
                    composite.mean_reversion.mu or 0.0
                )
            except Exception:
                entry_price = 0.0

        try:
            await self.journal.log_signal(
                {
                    "symbol": symbol,
                    "direction": bias,
                    "timestamp": datetime.utcnow().isoformat(),
                    "entry_price": entry_price,
                    "raw_score": raw,
                    "calibrated_confidence": cal,
                    "is_high_confidence": False,
                    "gate_reasons": gates,
                    "llm_response": {},
                    "indicators": indicators or {},
                    "composite_score": (
                        float(getattr(composite, "total_score", 0.0))
                        if composite else 0.0
                    ),
                    "composite_direction": (
                        getattr(composite, "direction", "neutral")
                        if composite else "neutral"
                    ),
                    "composite_conviction": (
                        getattr(composite, "conviction", "low")
                        if composite else "low"
                    ),
                    "zeta": (
                        float(composite.mean_reversion.zeta)
                        if composite else 0.0
                    ),
                    "hurst": (
                        float(composite.mean_reversion.hurst)
                        if composite else 0.5
                    ),
                    "half_life_hours": (
                        float(composite.mean_reversion.half_life_hours)
                        if composite else 0.0
                    ),
                    "basis_pct": (
                        float(composite.basis.basis_pct)
                        if composite else 0.0
                    ),
                    "funding_annualized_pct": (
                        float(composite.funding.annualized_pct)
                        if composite else 0.0
                    ),
                    "volume_zscore": (
                        float(composite.volume.z_score)
                        if composite else 0.0
                    ),
                    "volume_anomaly_level": (
                        getattr(
                            composite.volume, "anomaly_level", "normal"
                        ) if composite else "normal"
                    ),
                    "funding_rate_pct": (
                        float(composite.funding.rate_pct)
                        if composite else 0.0
                    ),
                    "funding_regime": (
                        getattr(composite.funding, "regime", "neutral")
                        if composite else "neutral"
                    ),
                    "funding_contrarian_bias": (
                        getattr(
                            composite.funding, "contrarian_bias",
                            "neutral",
                        ) if composite else "neutral"
                    ),
                    "_breakdown": bd,
                },
                execution_status="skipped",
            )
        except Exception as e:
            logger.debug(f"_log_rejected failed {symbol}: {e}")

    async def _dispatch_multi_setup_alerts(
        self,
        symbol, setups, df_1h, composite, deriv, indicators,
        breakdown, llm_response, agreement, smc, fibo, heatmap,
        signal_id: int = 0, delay_seconds: int = 0,
    ):
        valid = setups.valid_setups
        if not valid:
            return

        chart_path = None
        try:
            chart_path = await self.chart_generator.render_signal_chart(
                df=df_1h,
                context={
                    "symbol": symbol,
                    "price": float(df_1h["close"].iloc[-1]),
                    "signal_info": {
                        "entry": valid[0].entry_zone_low,
                        "sl": valid[0].entry_zone_low * 0.98,
                        "tps": [],
                        "direction": valid[0].direction,
                    },
                    "composite": composite,
                    "breakdown": breakdown,
                    "deriv": deriv,
                    "indicators": indicators,
                },
            )
        except Exception as e:
            logger.error(f"Chart failed {symbol}: {e}")

        try:
            if len(valid) == 3:
                logger.info(
                    f"{symbol}: 3 setup alerts "
                    f"({[s.setup_type for s in valid]})"
                )
                for s in valid:
                    await self.notifier.send_signal_alert(
                        signal=self._setup_to_signal(
                            s, symbol, breakdown,
                        ),
                        breakdown=breakdown,
                        llm_response=llm_response,
                        chart_path=chart_path,
                        composite=composite,
                        deriv=deriv,
                        indicators=indicators,
                        setup_label=s.label,
                        signal_id=signal_id,
                        delay_seconds=delay_seconds,
                    )
            else:
                logger.info(
                    f"{symbol}: combined {len(valid)}-setup alert "
                    f"({[s.setup_type for s in valid]})"
                )
                await self.notifier.send_multi_setup_alert(
                    symbol=symbol,
                    setups=valid,
                    breakdown=breakdown,
                    llm_response=llm_response,
                    chart_path=chart_path,
                    composite=composite,
                    deriv=deriv,
                    indicators=indicators,
                    agreement=agreement,
                    smc=smc,
                    fibo=fibo,
                    heatmap=heatmap,
                    signal_id=signal_id,
                    delay_seconds=delay_seconds,
                )
        except Exception as e:
            logger.error(
                f"Discord dispatch failed {symbol}: {e}", exc_info=True
            )
        finally:
            if chart_path:
                try:
                    os.unlink(chart_path)
                except Exception:
                    pass

    def _setup_to_signal(self, setup, symbol, breakdown):
        direction = (
            setup.direction if setup.direction != "neutral" else "long"
        )
        entry = (setup.entry_zone_low + setup.entry_zone_high) / 2
        sl, tp1, tp2, tp3 = self._compute_tp_levels(
            entry=entry, atr=setup.atr, direction=direction,
        )
        return TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            confidence=setup.confidence,
            metadata={"setup_type": setup.setup_type},
        )

    # ================================================================== #
    #  Position / queue callbacks
    # ================================================================== #
    async def _on_trade_filled(self, signal, result, signal_id):
        self._record_alert(
            symbol=signal.symbol,
            direction=signal.direction,
            confidence=signal.confidence,
            signal_id=signal_id,
        )
        if not self.signal_queue.cfg.notify_on_execution:
            return
        try:
            await self.notifier.send_trade_executed(
                symbol=signal.symbol,
                direction=signal.direction,
                entry=signal.entry_price,
                stop_loss=signal.stop_loss,
                tps=signal.tp_list,
                qty=result.get("qty", 0.0),
                signal_id=signal_id,
            )
        except Exception as e:
            logger.error(f"Trade executed alert failed: {e}")

    async def _on_position_closed(
        self, symbol, pos, pnl_pct, filled_tps, exit_reason,
    ):
        chart_path = None
        try:
            klines = await self.client.get_klines(
                symbol, interval="60", limit=80,
            )
            if klines and len(klines) >= 20:
                df = pd.DataFrame(
                    klines,
                    columns=["ts", "open", "high", "low", "close",
                             "volume", "turnover"],
                )
                for c in ("open", "high", "low", "close", "volume"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.sort_values("ts").reset_index(drop=True)
                chart_path = (
                    await self.chart_generator.render_signal_chart(
                        df=df,
                        context={
                            "symbol": symbol,
                            "price": float(df["close"].iloc[-1]),
                            "signal_info": {
                                "entry": pos.entry_price,
                                "sl": pos.initial_stop_loss,
                                "tps": pos.tps,
                                "direction": (
                                    "long"
                                    if pos.side == "Buy" else "short"
                                ),
                            },
                            "composite": None,
                            "breakdown": None,
                            "deriv": None,
                            "indicators": {},
                        },
                    )
                )
        except Exception as e:
            logger.debug(f"Chart for closed pos failed: {e}")

        pnl_usd = (
            pos.unrealized_pnl
            if pos.unrealized_pnl != 0
            else pos.entry_price * pos.total_qty * pnl_pct / 100
        )
        try:
            await self.notifier.send_position_closed(
                symbol=symbol,
                side="buy" if pos.side == "Buy" else "sell",
                entry=pos.entry_price,
                exit_price=pos.entry_price * (1 + pnl_pct / 100),
                pnl_pct=pnl_pct,
                pnl_usd=pnl_usd,
                filled_tps=filled_tps,
                total_tps=len(pos.tps),
                exit_reason=exit_reason,
                chart_path=chart_path,
            )
        except Exception as e:
            logger.error(f"Discord close alert failed: {e}")
        finally:
            if chart_path:
                try:
                    os.unlink(chart_path)
                except Exception:
                    pass

    async def _on_limit_stalled(self, signal, result: Dict):
        sig_id = int((signal.metadata or {}).get("signal_id", 0))
        setup_type = str(
            (signal.metadata or {}).get("setup_type", "")
        ).lower()

        elapsed_sec = int(result.get("elapsed_sec", 0))
        elapsed_min = elapsed_sec // 60
        candles = float(result.get("candles", 0))
        stall_sec = int(result.get("stall_timeout_sec", 0))
        stall_timeout_min = stall_sec // 60

        try:
            self.signal_queue.release_symbol(signal.symbol)
        except Exception as e:
            logger.debug(f"release_symbol err: {e}")

        stall_cooldown_min = int(
            self.config.get("execution", {}).get(
                "stall_cooldown_minutes", 0,
            )
        )
        if stall_cooldown_min > 0 and self._signal_cooldown_hours > 0:
            self._alerted_signals[signal.symbol] = {
                "timestamp": datetime.utcnow() - timedelta(
                    hours=max(
                        0.0,
                        self._signal_cooldown_hours
                        - stall_cooldown_min / 60.0,
                    )
                ),
                "direction": signal.direction,
                "confidence": 0.0,
                "signal_id": sig_id,
                "reason": "stalled",
            }

        if sig_id > 0:
            try:
                await self.db.execute(
                    "UPDATE signals SET execution_status = 'stalled', "
                    "execution_result = $1, executed_at = NOW() "
                    "WHERE id = $2",
                    json.dumps({
                        "reason": "limit_stalled",
                        "elapsed_sec": elapsed_sec,
                        "elapsed_min": elapsed_min,
                        "candles": candles,
                        "stall_timeout_sec": stall_sec,
                        "setup_type": setup_type,
                    }),
                    sig_id,
                )
                logger.info(
                    f"Signal {sig_id} {signal.symbol} marked stalled"
                )
            except Exception as e:
                logger.error(f"mark-stalled failed {sig_id}: {e}")

        stall_notify = bool(
            self.config.get("execution", {}).get(
                "stall_notify_discord", True,
            )
        )
        if not stall_notify:
            return
        try:
            await self.notifier.send_limit_stalled(
                symbol=signal.symbol,
                direction=signal.direction,
                entry=float(signal.entry_price),
                elapsed_min=elapsed_min,
                candles=candles,
                signal_id=sig_id,
                setup_type=setup_type,
                stall_timeout_min=stall_timeout_min,
            )
            logger.info(
                f"Stalled alert sent for {signal.symbol}"
            )
        except Exception as e:
            logger.error(f"Stalled alert failed {signal.symbol}: {e}")

    # ================================================================== #
    #  Daily 07:00 auto-improvement
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
            remaining = wait_seconds
            while remaining > 0 and self._running:
                chunk = min(30.0, remaining)
                await asyncio.sleep(chunk)
                remaining -= chunk
            if not self._running:
                return

            try:
                logger.info("DAILY AUTO-IMPROVEMENT STARTING")
                await self.outcome_tracker.run_daily_check()
                results = await self.auto_tuner.run_daily_improvement(
                    self.journal, None,
                )
                if results:
                    await self.notifier.send_tuning_report(results)
                    await self.soft_reload()
                    logger.info("Config reloaded after tuning")
                logger.info(
                    f"Auto-improvement complete "
                    f"({len(results)} adjustments)"
                )
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(
                    f"Daily improvement failed: {e}", exc_info=True,
                )

    # ================================================================== #
    #  Soft reload (SIGHUP)
    # ================================================================== #
    async def soft_reload(self):
        logger.info("SOFT RELOAD initiated")
        try:
            with open("config.json") as f:
                new_cfg = json.load(f)
        except Exception as e:
            logger.error(f"Soft reload: invalid config — {e}")
            return

        try:
            self.scorer.min_raw_threshold = new_cfg["strategy"][
                "min_raw_threshold"
            ]
            self.scorer.min_calibrated = new_cfg["strategy"][
                "min_calibrated"
            ]

            new_weights = new_cfg.get("strategy", {}).get(
                "composite_weights", {},
            )
            if new_weights:
                merged = self.composite_engine.DEFAULT_WEIGHTS.copy()
                merged.update({
                    k: float(v) for k, v in new_weights.items()
                    if k in merged
                })
                s = sum(merged.values()) or 1.0
                self.composite_engine.weights = {
                    k: v / s for k, v in merged.items()
                }

            self.risk_manager.cfg = new_cfg["risk"]

            ec = new_cfg["execution"]
            self.executor.config = ec
            self.executor.trailing_activation_pct = ec.get(
                "trailing_activation_pct", 1.0,
            )
            self.executor.trailing_distance_pct = ec.get(
                "trailing_distance_pct", 0.5,
            )
            self.executor.emergency_sl_pct = ec.get(
                "emergency_sl_pct", 2.5,
            )

            self._signal_cooldown_hours = float(
                ec.get("signal_cooldown_hours", 4)
            )

            if "auto_trade" in new_cfg:
                self.signal_queue.update_config(new_cfg["auto_trade"])

            if "realtime" in new_cfg and self.realtime_monitor:
                rt = new_cfg["realtime"]
                self.realtime_monitor.callback_timeout = int(
                    rt.get("callback_timeout_sec", 180)
                )
                self.realtime_monitor.gap_fill_recent_min = int(
                    rt.get("gap_fill_recent_minutes", 10)
                )

            self.config = new_cfg
            logger.info(
                f"SOFT RELOAD complete. Positions preserved: "
                f"{list(self.executor.active_positions.keys())}"
            )
        except Exception as e:
            logger.error(f"Soft reload failed: {e}", exc_info=True)

    async def _on_private_message(self, msg: dict):
        topic = msg.get("topic", "")
        if topic == "position":
            for p in msg.get("data", []):
                logger.debug(
                    f"Position: {p.get('symbol')} "
                    f"size={p.get('size')}"
                )
        elif topic == "execution":
            for ex in msg.get("data", []):
                logger.info(
                    f"Execution: {ex.get('symbol')} {ex.get('side')} "
                    f"{ex.get('execQty')}"
                )


# ====================================================================== #
#  Entry point
# ====================================================================== #
async def main():
    bot = QuantBotV9()
    loop = asyncio.get_event_loop()

    def shutdown():
        asyncio.create_task(bot.stop())

    def soft_reload():
        logger.info("SIGHUP received — soft reloading")
        asyncio.create_task(bot.soft_reload())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown)
    loop.add_signal_handler(signal.SIGHUP, soft_reload)

    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()
    finally:
        try:
            await bot.db.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())