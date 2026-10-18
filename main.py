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

    # ================================================================== #
    #  Position-close callback (called by TradeExecutor)
    # ================================================================== #
    async def _on_position_closed(
        self, symbol, pos, pnl_pct, filled_tps, exit_reason,
    ):
        """Called by executor when a position closes — sends Discord alert."""
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
        skip_reason = self._should_skip_symbol(symbol)
        if skip_reason:
            logger.debug(f"{symbol}: skipping — {skip_reason}")
            return

        klines = await self.client.get_klines(
            symbol, interval="60", limit=200,
        )
        if not klines or len(klines) < 100:
            return

        df = pd.DataFrame(
            klines,
            columns=["ts", "open", "high", "low", "close",
                     "volume", "turnover"],
        )
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("ts").reset_index(drop=True)

        indicators = self._compute_indicators(df)
        deriv = await self.derivatives.get_full_snapshot(symbol)

        composite = await self.composite_engine.compute(
            symbol=symbol,
            df=df,
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
            funding_result=composite.funding,
        )

        # ---- Prelim gate: log rejects so auto-tuner has data ----
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
                    "technical_score": prelim.technical_score,
                    "composite_score": prelim.composite_score,
                    "derivative_score": prelim.derivative_score,
                    "sentiment_score": prelim.sentiment_score,
                    "pattern_score": prelim.pattern_score,
                    "llm_score": prelim.llm_score,
                },
            })
            return

        # ---- DeepSeek analysis ----
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
                    "half_life_hours": (
                        composite.mean_reversion.half_life_hours
                    ),
                    "hurst": composite.mean_reversion.hurst,
                    "regime": composite.mean_reversion.regime,
                    "adf_pvalue": composite.mean_reversion.adf_pvalue,
                    "is_stationary": (
                        composite.mean_reversion.is_stationary
                    ),
                },
                "basis": {
                    "basis_pct": composite.basis.basis_pct,
                    "funding_rate": composite.basis.funding_rate,
                    "funding_annualized_pct": (
                        composite.basis.funding_annualized
                    ),
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
                "funding": {
                    "rate_pct": composite.funding.rate_pct,
                    "annualized_pct": composite.funding.annualized_pct,
                    "regime": composite.funding.regime,
                    "contrarian_bias": composite.funding.contrarian_bias,
                    "bias_strength": composite.funding.bias_strength,
                    "should_block_long": (
                        composite.funding.should_block_long
                    ),
                    "should_block_short": (
                        composite.funding.should_block_short
                    ),
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
            funding_result=composite.funding,
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
            "funding_annualized_pct": composite.funding.annualized_pct,
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
        })

        if not final.is_high_confidence:
            logger.debug(
                f"{symbol}: not high confidence "
                f"(raw={final.raw_total:.1f}, "
                f"cal={final.calibrated_confidence:.2f}, "
                f"gates={final.gate_reasons})"
            )
            return

        # ---- Build trade signal ----
        entry = float(df["close"].iloc[-1])
        atr = indicators.get("atr", entry * 0.01)

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
                f"Funding={composite.funding.rate_pct:+.3f}%"
            ),
            metadata={
                "signal_id": signal_id,
                "llm": llm_response,
                "funding_rate_pct": composite.funding.rate_pct,
                "funding_annualized_pct": (
                    composite.funding.annualized_pct
                ),
                "funding_regime": composite.funding.regime,
                "funding_contrarian_bias": (
                    composite.funding.contrarian_bias
                ),
                "funding_block_long": (
                    composite.funding.should_block_long
                ),
                "funding_block_short": (
                    composite.funding.should_block_short
                ),
            },
        )

        skip_reason = self._should_skip_symbol(symbol)
        if skip_reason:
            logger.info(
                f"{symbol}: signal blocked at execute — {skip_reason}"
            )
            return

        result = await self.executor.execute_signal(signal)
        if result["status"] != "filled":
            return

        # Record for cooldown — happens only after successful fill
        self._record_alert(
            symbol=symbol,
            direction=bias,
            confidence=final.calibrated_confidence,
            signal_id=signal_id,
        )

        # ---- Notify Discord (chart + embeds) ----
        chart_path = None
        try:
            chart_path = await self.chart_generator.render_signal_chart(
                df=df,
                context={
                    "symbol": symbol,
                    "price": entry,
                    "signal_info": {
                        "entry": entry,
                        "sl": signal.stop_loss,
                        "tps": signal.tp_list,
                        "direction": signal.direction,
                    },
                    "composite": composite,
                    "breakdown": final,
                    "deriv": deriv,
                    "indicators": indicators,
                },
            )
        except Exception as e:
            logger.error(f"Chart failed for {symbol}: {e}")

        try:
            await self.notifier.send_signal_alert(
                signal=signal,
                breakdown=final,
                llm_response=llm_response,
                chart_path=chart_path,
                composite=composite,
                deriv=deriv,
                indicators=indicators,
            )
        except Exception as e:
            logger.error(f"Discord alerts failed for {symbol}: {e}")
        finally:
            if chart_path:
                try:
                    os.unlink(chart_path)
                except Exception:
                    pass

        await self.journal.mark_executed(signal_id, result)

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
    finally:
        # Ensure DB is closed even on unexpected start() crash
        try:
            await bot.db.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())