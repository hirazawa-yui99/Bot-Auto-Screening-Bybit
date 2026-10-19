"""
main.py
Bybit Quant Bot v9.1 — Async orchestrator with FULL-PAIR scanning.

Wires together:
  - V5 client with get_all_linear_symbols() pagination
  - DerivativeAnalyzer (CVD, OI, L/S ratio)
  - CompositeQuantEngine (Zeta, basis, volume anomaly, funding)
  - ConfidenceScorer (calibrated, high-confidence gate)
  - DeepSeekAgent (LLM analysis)
  - TradeExecutor (WS orders + position monitor + close callback)
  - AutoTuner + OutcomeTracker (07:00 daily improvement)
  - Discord notifications + chart generation
"""
import os
import asyncio
import json
import logging
import math
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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
from notification.chart_generator import ChartGenerator
from persistence.database import Database
from analysis.setup_classifier import SetupClassifier, SetupCollection
from analysis.fibonacci import FibonacciAnalyzer
from analysis.orderbook_heatmap import OrderbookHeatmap
from analysis.agreement_gate import AgreementGate
from analysis.smc import SMCAnalyzer


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

        # ---- Core clients ----
        self.client = BybitV5Client(BybitConfig(
            api_key=self.config["api"]["bybit_key"],
            api_secret=self.config["api"]["bybit_secret"],
            testnet=self.config["api"].get("testnet", False),
            category="linear",
        ))
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

        self.db = Database(self.config["database"])
        self.journal = SignalJournal(self.db)

        # ---- Analysis ----
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

        # ---- LLM ----
        self.agent = DeepSeekAgent(DeepSeekConfig(
            api_key=self.config["deepseek"]["api_key"],
            base_url=self.config["deepseek"].get(
                "base_url", "https://api.deepseek.com/v1",
            ),
            model=self.config["deepseek"].get(
                "model", "deepseek-v4-pro",
            ),
            model_high_confidence=self.config["deepseek"].get(
                "model_high_confidence", "deepseek-v4-pro",
            ),
            temperature=self.config["deepseek"].get("temperature", 0.3),
            max_tokens=self.config["deepseek"].get("max_tokens", 1500),
            timeout=self.config["deepseek"].get("timeout", 45),
            error_cooldown_sec=self.config["deepseek"].get(
                "error_cooldown_sec", 900,
            ),
        ))

        # ---- Risk + execution ----
        self.risk_manager = RiskManager(self.config["risk"])
        self.executor = TradeExecutor(
            self.client,
            self.risk_manager,
            self.config["execution"],
            on_position_closed=self._on_position_closed,
        )

        # ---- Learning ----
        self.outcome_tracker = OutcomeTracker(self.client, self.journal)
        self.auto_tuner = AutoTuner(self.config["auto_tune"], self.scorer)

        # ---- Notification ----
        self.notifier = DiscordNotifier(self.config["api"])
        self.chart_generator = ChartGenerator()

        # ---- Runtime ----
        self._running = False
        self._daily_task: Optional[asyncio.Task] = None
        self._scan_task: Optional[asyncio.Task] = None
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
        self.setup_classifier = SetupClassifier(
            self.config.get("setups", {}),
        )
        self.fib_analyzer = FibonacciAnalyzer()
        self.heatmap = OrderbookHeatmap()
        self.agreement_gate = AgreementGate(
            min_agree=self.config.get("agreement", {}).get(
                "min_agree", 2,
            ),
        )
        self.smc_analyzer = SMCAnalyzer()
    
    async def _fetch_multi_tf(self, symbol: str) -> Tuple[
        Optional[pd.DataFrame], pd.DataFrame, Optional[pd.DataFrame],
    ]:
        """Fetch 15m, 1h, 4h klines concurrently."""
        async def _fetch(interval: str, limit: int):
            try:
                klines = await self.client.get_klines(
                    symbol, interval=interval, limit=limit,
                )
                if not klines or len(klines) < 40:
                    return None
                df = pd.DataFrame(
                    klines,
                    columns=["ts", "open", "high", "low",
                             "close", "volume", "turnover"],
                )
                for col in ("open", "high", "low", "close", "volume"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                return df.sort_values("ts").reset_index(drop=True)
            except Exception as e:
                logger.debug(f"fetch {interval} failed for {symbol}: {e}")
                return None

        df_15m, df_1h, df_4h = await asyncio.gather(
            _fetch("15", 100),
            _fetch("60", 200),
            _fetch("240", 100),
        )
        return df_15m, df_1h, df_4h

    # ================================================================== #
    #  Position-close callback (called by TradeExecutor)
    # ================================================================== #
    async def _on_position_closed(
        self, symbol, pos, pnl_pct, filled_tps, exit_reason,
    ):
        """Called by TradeExecutor when a position closes."""
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
                for col in ("open", "high", "low", "close", "volume"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.sort_values("ts").reset_index(drop=True)

                chart_path = await self.chart_generator.render_signal_chart(
                    df=df,
                    context={
                        "symbol": symbol,
                        "price": float(df["close"].iloc[-1]),
                        "signal_info": {
                            "entry": pos.entry_price,
                            "sl": pos.initial_stop_loss,
                            "tps": pos.tps,
                            "direction": (
                                "long" if pos.side == "Buy" else "short"
                            ),
                        },
                        "composite": None,
                        "breakdown": None,
                        "deriv": None,
                        "indicators": {},
                    },
                )
        except Exception as e:
            logger.debug(f"Chart for closed position failed: {e}")

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
    
    def _should_skip_symbol(self, symbol: str) -> Optional[str]:
        """
        Return a reason string if the symbol should be skipped, else None.

        Guards:
          1. Active position already open
          2. Pending limit entry in progress
          3. Alerted on this symbol within signal_cooldown_hours
        """
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
                age_hours = (
                    datetime.utcnow() - entry["timestamp"]
                ).total_seconds() / 3600
                if age_hours < self._signal_cooldown_hours:
                    return (
                        f"cooldown:alerted_{age_hours:.1f}h_ago"
                        f"(limit={self._signal_cooldown_hours}h)"
                    )

        return None

    def _record_alert(self, symbol: str, direction: str, confidence: float, signal_id: int):
        """Record a successfully-alerted signal for cooldown tracking."""
        self._alerted_signals[symbol] = {
            "timestamp": datetime.utcnow(),
            "direction": direction,
            "confidence": confidence,
            "signal_id": signal_id,
        }
        asyncio.create_task(self._persist_alert(
            symbol, direction, confidence, signal_id,
        ))

    async def _persist_alert(self, symbol, direction, confidence, signal_id):
        try:
            await self.db.execute(
                "INSERT INTO signal_cooldowns "
                "(symbol, alerted_at, direction, confidence, signal_id) "
                "VALUES ($1, NOW(), $2, $3, $4) "
                "ON CONFLICT (symbol) DO UPDATE SET "
                "alerted_at = NOW(), direction = EXCLUDED.direction, "
                "confidence = EXCLUDED.confidence, signal_id = EXCLUDED.signal_id",
                symbol, direction, confidence, signal_id,
            )
        except Exception as e:
            logger.debug(f"Cooldown persist failed for {symbol}: {e}")

    def _prune_alerted_signals(self):
        """Remove cooldown entries older than prune_alerted_after_hours."""
        if self._prune_alerted_after_hours <= 0:
            return
        cutoff = datetime.utcnow() - timedelta(
            hours=self._prune_alerted_after_hours,
        )
        stale = [
            s for s, info in self._alerted_signals.items()
            if info["timestamp"] < cutoff
        ]
        for s in stale:
            del self._alerted_signals[s]
        if stale:
            logger.debug(
                f"Pruned {len(stale)} stale alert cooldowns"
            )
    # ================================================================== #
    #  Lifecycle
    # ================================================================== #
    async def start(self):
        logger.info("=" * 60)
        logger.info("Bybit Quant Bot v9.1 starting — FULL-PAIR SCANNER")
        logger.info("=" * 60)

        self._running = True
        await self.client.start()
        await self.db.connect()
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
                f"Loaded {len(self._alerted_signals)} signal cooldowns from DB"
            )
        except Exception as e:
            logger.warning(f"Could not load signal cooldowns: {e}")
        await self.executor.start_monitor()

        self._daily_task = asyncio.create_task(self._daily_scheduler())
        self._scan_task = asyncio.create_task(self._scan_loop())

        await self.client.subscribe_private(
            ["position", "order", "execution", "wallet"],
            self._on_private_message,
        )

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

        # 1. Cancel background tasks FIRST so they stop touching the DB
        tasks_to_cancel = []
        for attr in ("_scan_task", "_daily_task"):
            task = getattr(self, attr, None)
            if task is not None and not task.done():
                task.cancel()
                tasks_to_cancel.append(task)

        # 2. Wait for them to exit (max 15s)
        if tasks_to_cancel:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                    timeout=15,
                )
                logger.info("Background tasks exited cleanly")
            except asyncio.TimeoutError:
                logger.warning(
                    "Background tasks did not exit within 15s — "
                    "continuing shutdown"
                )

        # 3. Close resources in order
        await self.executor.stop_monitor()
        await self.executor.close_all_positions()
        await self.client.stop()
        await self.agent.close()
        await self.db.close()
        logger.info("Shutdown complete.")

    # ================================================================== #
    #  TP/SL computation
    # ================================================================== #
    def _compute_tp_levels(
        self, entry: float, atr: float, direction: str,
    ) -> Tuple[float, float, float, float]:
        """
        Compute SL and TP1/TP2/TP3 levels. Returns (sl, tp1, tp2, tp3).
        """
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

    # ================================================================== #
    #  Full-pair scan loop
    # ================================================================== #
    async def _scan_loop(self):
        interval = self.config["system"].get("check_interval_hours", 1)
        scan_top_n = self.config["system"].get("scan_top_n", 0)
        min_turnover = self.config["system"].get(
            "min_turnover_24h", 1_000_000,
        )

        while self._running:
            try:
                # Prune stale cooldown entries (memory housekeeping)
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

                semaphore = asyncio.Semaphore(
                    self.config["system"].get("max_concurrent", 3)
                )

                async def analyse_symbol(sym):
                    async with semaphore:
                        if not self._running:
                            return
                        try:
                            await self._analyse_and_signal(sym)
                        except Exception as e:
                            logger.error(
                                f"Analysis failed for {sym}: {e}",
                            )

                await asyncio.gather(
                    *[analyse_symbol(s) for s in symbols]
                )
                logger.info(f"Scan complete. Next in {interval}h.")

            except asyncio.CancelledError:
                logger.info("Scan loop cancelled")
                return
            except Exception as e:
                logger.error(f"Scan loop error: {e}", exc_info=True)

            # Interruptible sleep — checks _running every 5s
            remaining = interval * 3600
            while remaining > 0 and self._running:
                chunk = min(5.0, remaining)
                await asyncio.sleep(chunk)
                remaining -= chunk

            if not self._running:
                logger.info("Scan loop exiting (shutdown signaled)")
                return

    # ================================================================== #
    #  Per-symbol analysis pipeline
    # ================================================================== #
    async def _analyse_and_signal(self, symbol: str):
        """
        Full analysis pipeline for a single symbol.

        Flow:
          1. Cooldown check          → skip if recently alerted
          2. Multi-TF fetch (15m/1h/4h)
          3. Indicators + derivatives + composite
          4. SMC + Fibonacci + orderbook heatmap
          5. Agreement gate          → skip if contradiction
          6. Preliminary confidence  → skip if raw < 40
          7. DeepSeek analysis
          8. Final confidence + LLM agreement check
          9. Setup classification    → skip if 0 setups
          10. Build plan + log as 'pending'
          11. Dispatch Discord alert (includes queue badge)
          12. Return — queue executes after delay_seconds
        """
        # ============================================================ #
        # 1. Early skip (cooldown / active position / pending entry)
        # ============================================================ #
        skip_reason = self._should_skip_symbol(symbol)
        if skip_reason:
            logger.debug(f"{symbol}: skipping — {skip_reason}")
            return

        # ============================================================ #
        # 2. Multi-timeframe fetch
        # ============================================================ #
        df_15m, df_1h, df_4h = await self._fetch_multi_tf(symbol)
        if df_1h is None or len(df_1h) < 100:
            return

        # ============================================================ #
        # 3. Core analysis
        # ============================================================ #
        indicators = self._compute_indicators(df_1h)
        deriv = await self.derivatives.get_full_snapshot(symbol)

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

        # ============================================================ #
        # 4. SMC / Fibonacci / Heatmap (parallel)
        # ============================================================ #
        try:
            smc = self.smc_analyzer.analyze(df_1h)
        except Exception as e:
            logger.debug(f"SMC failed for {symbol}: {e}")
            smc = None

        try:
            fibo = self.fib_analyzer.analyze(df_1h, bias)
        except Exception as e:
            logger.debug(f"Fibo failed for {symbol}: {e}")
            fibo = None

        try:
            ob = await self.client.get_orderbook(symbol, limit=50)
            heatmap = self.heatmap.analyze(ob)
        except Exception as e:
            logger.debug(f"Heatmap failed for {symbol}: {e}")
            heatmap = None

        # ============================================================ #
        # 5. Directional contributions for agreement gate
        # ============================================================ #
        smc_dir = "neutral"
        if smc is not None:
            if smc.bos in ("bullish", "bearish"):
                smc_dir = "long" if smc.bos == "bullish" else "short"
            elif smc.choch in ("bullish", "bearish"):
                smc_dir = "long" if smc.choch == "bullish" else "short"

        deriv_dir = "neutral"
        if (
            deriv.cvd_divergence == "bullish"
            or deriv.smart_money_bias == "long"
        ):
            deriv_dir = "long"
        elif (
            deriv.cvd_divergence == "bearish"
            or deriv.smart_money_bias == "short"
        ):
            deriv_dir = "short"

        heatmap_dir = heatmap.direction if heatmap else "neutral"

        fibo_dir = "neutral"
        fibo_in_zone = False
        if fibo is not None and fibo.valid:
            fibo_dir = bias
            fibo_in_zone = fibo.in_zone

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
                f"{symbol}: agreement gate failed — {agreement.reason}"
            )
            await self._log_rejected(
                symbol=symbol,
                bias=bias,
                indicators=indicators,
                deriv=deriv,
                composite=composite,
                agreement=agreement,
                price=current_price,
            )
            return

        # ============================================================ #
        # 6. Preliminary confidence (no LLM)
        # ============================================================ #
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
                symbol=symbol,
                bias=bias,
                indicators=indicators,
                deriv=deriv,
                composite=composite,
                agreement=agreement,
                prelim=prelim,
                price=current_price,
            )
            return

        # ============================================================ #
        # 7. DeepSeek analysis
        # ============================================================ #
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
                "strong": heatmap.strong if heatmap else False,
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

        # ============================================================ #
        # 8. Final confidence + LLM agreement check
        # ============================================================ #
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
                f"{symbol}: LLM says {llm_dir}, "
                f"agreement says {agreement.direction} — refusing"
            )
            await self._log_rejected(
                symbol=symbol,
                bias=bias,
                indicators=indicators,
                deriv=deriv,
                composite=composite,
                agreement=agreement,
                prelim=final,
                price=current_price,
                extra_gate=f"llm_mismatch:{llm_dir}!={agreement.direction}",
            )
            return

        if not final.is_high_confidence:
            logger.debug(
                f"{symbol}: not high confidence "
                f"(raw={final.raw_total:.1f}, "
                f"cal={final.calibrated_confidence:.2f}, "
                f"gates={final.gate_reasons})"
            )
            await self._log_rejected(
                symbol=symbol,
                bias=bias,
                indicators=indicators,
                deriv=deriv,
                composite=composite,
                agreement=agreement,
                prelim=final,
                price=current_price,
            )
            return

        # ============================================================ #
        # 9. Setup classification
        # ============================================================ #
        setups = self.setup_classifier.classify_all(
            df_15m=df_15m,
            df_1h=df_1h,
            df_4h=df_4h,
            composite=composite,
            deriv=deriv,
            bias=bias,
        )

        if setups.count == 0:
            logger.debug(f"{symbol}: no valid setups")
            await self._log_rejected(
                symbol=symbol,
                bias=bias,
                indicators=indicators,
                deriv=deriv,
                composite=composite,
                agreement=agreement,
                prelim=final,
                price=current_price,
                extra_gate="no_setups",
            )
            return

        best = setups.highest_confidence()
        if best is None:
            return

        # ============================================================ #
        # 10. Entry price — use Fibonacci zone when applicable
        # ============================================================ #
        entry = current_price
        if fibo is not None and fibo.valid:
            if fibo.entry_zone_low <= current_price <= fibo.entry_zone_high:
                # Inside zone — use current
                entry = current_price
            else:
                # Outside zone — use near edge as limit
                entry = (
                    fibo.entry_zone_low
                    if bias == "long"
                    else fibo.entry_zone_high
                )
            logger.debug(
                f"{symbol}: fib entry={entry:.6f} "
                f"(zone {fibo.entry_zone_low:.6f}-{fibo.entry_zone_high:.6f}, "
                f"current={current_price:.6f})"
            )

        sl, tp1, tp2, tp3 = self._compute_tp_levels(
            entry=entry,
            atr=best.atr,
            direction=bias,
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
            leverage=self.config["risk"].get("default_leverage", 5),
            reason=(
                f"Setup={best.setup_type} "
                f"Raw={final.raw_total:.0f} "
                f"Agree={agreement.agreement_count} "
                f"Fibo={'in' if fibo_in_zone else 'out'}"
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
                "funding_block_long": composite.funding.should_block_long,
                "funding_block_short": composite.funding.should_block_short,
                "funding_rate_pct": composite.funding.rate_pct,
            },
        )

        # ============================================================ #
        # 11. Late skip check — mid-analysis race
        # ============================================================ #
        skip_reason = self._should_skip_symbol(symbol)
        if skip_reason:
            logger.info(
                f"{symbol}: blocked at execute — {skip_reason}"
            )
            return

        # ============================================================ #
        # 12. Build plan for queue
        # ============================================================ #
        delay_seconds = int(
            self.config.get("auto_trade", {}).get(
                "delay_seconds", 30,
            )
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

        # ============================================================ #
        # 13. Log as 'pending' — queue executes later
        # ============================================================ #
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
            signal_dict,
            plan=plan,
            execution_status="pending",
        )

        if signal_id <= 0:
            logger.error(
                f"{symbol}: signal logging failed — skipping alert"
            )
            return

        # ============================================================ #
        # 14. Dispatch Discord alerts
        # ============================================================ #
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

        # ============================================================ #
        # 15. Log queue status — queue handles execution
        # ============================================================ #
        auto_trade_state = (
            "on" if self.signal_queue.cfg.enabled else "off"
        )
        logger.info(
            f"{symbol}: signal {signal_id} queued — "
            f"execution in {delay_seconds}s "
            f"(auto_trade={auto_trade_state}, "
            f"setups={setups.labels}, "
            f"agreement={agreement.agreement_count})"
        )

    def _llm_agrees_with_quant_llm(
        self, llm_dir: str, bias: str, agreement_dir: str,
    ) -> bool:
        """LLM must not contradict the quant agreement direction."""
        if llm_dir == "neutral":
            return True
        return llm_dir == agreement_dir

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
        """
        Log a rejected signal so the auto-tuner has data.

        All rejection paths in _analyse_and_signal call this. Handles
        the case where `prelim` is None (rejected before confidence
        scoring, e.g. agreement gate failure).
        """
        # Extract confidence fields (default to 0 if prelim not run)
        if prelim is not None:
            raw = float(getattr(prelim, "raw_total", 0.0))
            cal = float(getattr(prelim, "calibrated_confidence", 0.0))
            gate_reasons = list(getattr(prelim, "gate_reasons", []) or [])
            breakdown_dict = {
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
            raw = 0.0
            cal = 0.0
            gate_reasons = []
            breakdown_dict = {
                "technical_score": 0.0,
                "composite_score": 0.0,
                "derivative_score": 0.0,
                "sentiment_score": 0.0,
                "pattern_score": 0.0,
                "llm_score": 0.0,
            }

        # Agreement gate reason
        if agreement is not None and not agreement.passed:
            gate_reasons.append(f"agreement:{agreement.reason}")

        # Extra gate (LLM mismatch, no setups, etc.)
        if extra_gate:
            gate_reasons.append(extra_gate)

        # Ensure at least one reason
        if not gate_reasons:
            gate_reasons = ["rejected_unknown"]

        # Entry price: prefer provided, else composite long-run mean
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
                    "gate_reasons": gate_reasons,
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
                        getattr(composite.volume, "anomaly_level", "normal")
                        if composite else "normal"
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
                            composite.funding,
                            "contrarian_bias",
                            "neutral",
                        ) if composite else "neutral"
                    ),
                    "_breakdown": breakdown_dict,
                },
                execution_status="skipped",
            )
        except Exception as e:
            logger.debug(f"_log_rejected failed for {symbol}: {e}")

    async def _dispatch_multi_setup_alerts(
        self,
        symbol, setups, df_1h, composite, deriv, indicators,
        breakdown, llm_response, agreement, smc, fibo, heatmap,
        signal_id: int = 0,
        delay_seconds: int = 0,
    ):
        valid = setups.valid_setups
        if not valid:
            logger.debug(f"{symbol}: no valid setups to dispatch")
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
            logger.error(f"Chart failed for {symbol}: {e}")

        try:
            if len(valid) == 3:
                logger.info(
                    f"{symbol}: dispatching 3 setup alerts "
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
                labels = [s.setup_type for s in valid]
                logger.info(
                    f"{symbol}: dispatching combined "
                    f"{len(valid)}-setup alert ({labels})"
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
                f"Discord dispatch failed for {symbol}: {e}",
                exc_info=True,
            )
        finally:
            if chart_path:
                try:
                    os.unlink(chart_path)
                except Exception:
                    pass

    def _setup_to_signal(self, setup, symbol, breakdown):
        """Build a TradeSignal-compatible object for one setup."""
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
    
    async def _on_trade_filled(self, signal, result, signal_id):
        """
        Called by SignalQueue when a queued signal executed successfully.
        Records cooldown and sends 'TRADE EXECUTED' alert.
        """
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

    # ================================================================== #
    #  Indicators & patterns
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
                if (
                    abs(recent[lows[-1]] - recent[lows[-2]])
                    / recent[lows[-2]] < 0.02
                ):
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

            # Interruptible sleep — checks _running every 30s
            remaining = wait_seconds
            while remaining > 0 and self._running:
                chunk = min(30.0, remaining)
                await asyncio.sleep(chunk)
                remaining -= chunk

            if not self._running:
                logger.info("Daily scheduler exiting (shutdown signaled)")
                return

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
            except asyncio.CancelledError:
                logger.info("Daily scheduler cancelled")
                return
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
    async def soft_reload(self):
        """
        Reload config without touching positions, database, or cooldowns.
        Triggered by SIGHUP. Applies new thresholds at next scan.
        """
        logger.info("SOFT RELOAD initiated")
        try:
            with open("config.json") as f:
                new_config = json.load(f)
        except Exception as e:
            logger.error(f"Soft reload failed — invalid config: {e}")
            return

        try:
            # In-memory thresholds
            self.scorer.min_raw_threshold = new_config["strategy"]["min_raw_threshold"]
            self.scorer.min_calibrated = new_config["strategy"]["min_calibrated"]

            # Composite weights
            new_weights = new_config.get("strategy", {}).get(
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

            # Risk config
            self.risk_manager.cfg = new_config["risk"]

            # Executor config
            self.executor.config = new_config["execution"]
            self.executor.trailing_activation_pct = new_config["execution"].get(
                "trailing_activation_pct", 1.0,
            )
            self.executor.trailing_distance_pct = new_config["execution"].get(
                "trailing_distance_pct", 0.5,
            )
            self.executor.emergency_sl_pct = new_config["execution"].get(
                "emergency_sl_pct", 2.5,
            )

            # Signal cooldown
            self._signal_cooldown_hours = float(
                new_config.get("execution", {}).get(
                    "signal_cooldown_hours", 4,
                )
            )

            # Preserve everything else — active positions, alerts,
            # DB connection, WS connections, chart generator
            self.config = new_config

            logger.info(
                f"SOFT RELOAD complete. "
                f"Active positions preserved: "
                f"{list(self.executor.active_positions.keys())}"
            )

        except Exception as e:
            logger.error(f"Soft reload failed: {e}", exc_info=True)


async def main():
    bot = QuantBotV9()
    loop = asyncio.get_event_loop()

    def shutdown():
        asyncio.create_task(bot.stop())

    def soft_reload():
        logger.info("SIGHUP received — soft reloading config")
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