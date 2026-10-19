"""
execution/trade_executor.py
WebSocket order placement with dual-path entry (market vs limit),
multi-target TP, SL management, rally cancel, stall detection,
and shutdown-safe order cancellation.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    tp_weights: Tuple[float, ...] = (0.40, 0.35, 0.25)
    confidence: float = 0.0
    position_size_usd: float = 0.0
    leverage: int = 5
    reason: str = ""
    metadata: Dict = field(default_factory=dict)

    @property
    def side(self) -> str:
        return "Buy" if self.direction == "long" else "Sell"

    @property
    def tp_list(self) -> List[float]:
        return [
            tp for tp in (
                self.take_profit_1,
                self.take_profit_2,
                self.take_profit_3,
            ) if tp is not None
        ]

    @property
    def risk_reward_tp1(self) -> float:
        risk = abs(self.entry_price - self.stop_loss)
        if risk <= 0:
            return 0.0
        return abs(self.take_profit_1 - self.entry_price) / risk

    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry_price - self.stop_loss)
        if risk <= 0 or not self.tp_list:
            return 0.0
        weights = list(self.tp_weights[:len(self.tp_list)])
        total_w = sum(weights) or 1.0
        weighted = sum(
            abs(tp - self.entry_price) / risk * w
            for tp, w in zip(self.tp_list, weights)
        )
        return weighted / total_w


@dataclass
class ActivePosition:
    symbol: str
    side: str
    entry_price: float
    total_qty: float
    remaining_qty: float
    stop_loss: float
    initial_stop_loss: float
    tps: List[float]
    tp_weights: List[float]
    tp_order_ids: Dict[float, str] = field(default_factory=dict)
    tp_filled: Dict[float, bool] = field(default_factory=dict)
    trailing_stop: Optional[float] = None
    opened_at: str = ""
    signal_confidence: float = 0.0
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    sl_moved_to_be: bool = False
    sl_moved_to_tp1: bool = False
    exit_reason: Optional[str] = None


class TradeExecutor:
    def __init__(
        self,
        client,
        risk_manager,
        config: Dict,
        on_position_closed=None,
        on_limit_stalled=None,
    ):
        self.client = client
        self.risk_manager = risk_manager
        self.config = config
        self.on_position_closed = on_position_closed
        self.on_limit_stalled = on_limit_stalled

        self.active_positions: Dict[str, ActivePosition] = {}
        self._pending_entries: Dict[str, asyncio.Task] = {}
        self._recently_closed: Dict[str, float] = {}
        self._pending_order_ids: Dict[str, str] = {}   # Fix A
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

        # SL/TP
        self.trailing_activation_pct = config.get(
            "trailing_activation_pct", 1.0,
        )
        self.trailing_distance_pct = config.get(
            "trailing_distance_pct", 0.5,
        )
        self.emergency_sl_pct = config.get("emergency_sl_pct", 2.5)
        self.max_hold_hours = config.get("max_hold_hours", 72)
        self.move_sl_be_after_tp1 = config.get(
            "move_sl_to_be_after_tp1", True,
        )
        self.move_sl_tp1_after_tp2 = config.get(
            "move_sl_to_tp1_after_tp2", True,
        )

        # Entry routing
        self.entry_order_type = config.get(
            "entry_order_type", "limit_preferred",
        )
        self.market_tolerance_pct = config.get(
            "market_tolerance_pct", 0.15,
        )
        self.max_deviation_pct = config.get("max_deviation_pct", 10.0)
        self.cancel_on_rally_pct = config.get(
            "cancel_on_rally_pct", 1.5,
        )
        self.limit_expiry_min = config.get(
            "limit_order_expiry_minutes", 1440,
        )
        self.limit_check_interval_sec = config.get(
            "limit_check_interval_sec", 20,
        )
        self.tp_placement_retries = config.get(
            "tp_placement_retries", 2,
        )
        self.tp_placement_retry_delay = config.get(
            "tp_placement_retry_delay_sec", 1.5,
        )
        self.tp_min_position_warn_usd = config.get(
            "tp_min_position_warn_usd", 30.0,
        )

        # Stall detection
        self.stall_timeout_candles = int(
            config.get("stall_timeout_candles", 8)
        )
        self.stall_candle_seconds_by_setup = dict(
            config.get(
                "stall_candle_seconds_by_setup",
                {"scalp": 900, "intraday": 3600, "swing": 14400},
            )
        )
        self.stall_notify_discord = bool(
            config.get("stall_notify_discord", True)
        )

    # ================================================================== #
    #  Helpers
    # ================================================================== #
    async def _get_current_price(self, symbol: str) -> float:
        try:
            ticker = await self.client.get_ticker(symbol)
            if ticker:
                return float(
                    ticker.get("lastPrice")
                    or ticker.get("markPrice")
                    or ticker.get("indexPrice")
                    or 0
                )
        except Exception as e:
            logger.warning(f"Price fetch failed for {symbol}: {e}")
        return 0.0

    @staticmethod
    def _rally_pct(direction: str, entry: float, current: float) -> float:
        if entry <= 0 or current <= 0:
            return 0.0
        if direction == "long":
            return max(0.0, (current - entry) / entry * 100)
        return max(0.0, (entry - current) / entry * 100)

    def _rally_cancel_threshold(
        self, signal: TradeSignal,
    ) -> Optional[float]:
        if self.cancel_on_rally_pct <= 0:
            return None
        if signal.direction == "long":
            return signal.entry_price * (
                1 + self.cancel_on_rally_pct / 100
            )
        return signal.entry_price * (
            1 - self.cancel_on_rally_pct / 100
        )

    def _candle_seconds(self, signal: TradeSignal) -> int:
        """Candle duration in seconds based on the setup type."""
        meta = signal.metadata or {}
        setup = str(meta.get("setup_type", "")).lower()
        return int(self.stall_candle_seconds_by_setup.get(setup, 3600))

    def _stall_timeout_seconds(self, signal: TradeSignal) -> int:
        """Total stall timeout in seconds = candles × candle duration."""
        return self.stall_timeout_candles * self._candle_seconds(signal)

    # ================================================================== #
    #  Entry routing
    # ================================================================== #
    async def execute_signal(self, signal: TradeSignal) -> Dict:
        approved, size_usd, reason = self.risk_manager.approve_trade(
            signal,
        )
        if not approved:
            logger.warning(f"Trade rejected by risk manager: {reason}")
            return {"status": "rejected", "reason": reason}

        qty = size_usd / signal.entry_price
        signal.position_size_usd = size_usd

        existing = self._pending_entries.get(signal.symbol)
        if existing is not None and not existing.done():
            logger.info(
                f"{signal.symbol}: pending entry already in progress"
            )
            return {
                "status": "pending_limit",
                "reason": "duplicate_pending",
                "symbol": signal.symbol,
            }

        if signal.symbol in self.active_positions:
            logger.info(
                f"{signal.symbol}: already have active position"
            )
            return {
                "status": "rejected",
                "reason": "position_exists",
            }

        current_price = await self._get_current_price(signal.symbol)
        if current_price <= 0:
            logger.warning(
                f"{signal.symbol}: no price — defaulting to market"
            )
            return await self._execute_market_entry(signal, qty)

        deviation_pct = (
            abs(current_price - signal.entry_price)
            / signal.entry_price * 100
        )
        rally_pct = self._rally_pct(
            signal.direction, signal.entry_price, current_price,
        )

        logger.info(
            f"{signal.symbol}: entry={signal.entry_price:.6f} "
            f"current={current_price:.6f} "
            f"dev={deviation_pct:.3f}% rally={rally_pct:.3f}%"
        )

        if deviation_pct > self.max_deviation_pct:
            logger.info(
                f"{signal.symbol}: deviation {deviation_pct:.2f}% > "
                f"max {self.max_deviation_pct}% — skipping stale signal"
            )
            return {"status": "skipped", "reason": "stale_entry"}

        if self.entry_order_type == "market":
            return await self._execute_market_entry(signal, qty)

        if rally_pct > self.cancel_on_rally_pct:
            logger.info(
                f"{signal.symbol}: price already rallied "
                f"{rally_pct:.2f}% > cancel threshold "
                f"{self.cancel_on_rally_pct}% — skipping"
            )
            return {"status": "skipped", "reason": "rally_too_far"}

        if deviation_pct <= self.market_tolerance_pct:
            logger.info(
                f"{signal.symbol}: within tolerance "
                f"({deviation_pct:.3f}%) — MARKET order"
            )
            return await self._execute_market_entry(signal, qty)

        if self.entry_order_type == "limit_only":
            logger.info(f"{signal.symbol}: limit_only — LIMIT")
        else:
            logger.info(
                f"{signal.symbol}: dev {deviation_pct:.3f}% > "
                f"tolerance {self.market_tolerance_pct}% — LIMIT at "
                f"{signal.entry_price:.6f}"
            )
        return await self._spawn_limit_entry(signal, qty)

    # ================================================================== #
    #  Market entry
    # ================================================================== #
    async def _execute_market_entry(
        self, signal: TradeSignal, qty: float,
    ) -> Dict:
        try:
            await self.client.set_leverage(
                signal.symbol, signal.leverage,
            )
        except Exception as e:
            logger.debug(f"Leverage set: {e}")

        rounded_qty, reason = self.client.round_qty(signal.symbol, qty)
        if rounded_qty <= 0:
            logger.error(
                f"Qty rounding failed for {signal.symbol}: "
                f"{reason} (raw={qty})"
            )
            return {
                "status": "error",
                "reason": f"qty_invalid: {reason}",
            }

        ok, notional_reason = self.client.validate_notional(
            signal.symbol, rounded_qty, signal.entry_price,
        )
        if not ok:
            logger.error(
                f"Notional check failed for {signal.symbol}: "
                f"{notional_reason}"
            )
            return {"status": "error", "reason": notional_reason}

        try:
            response = await self.client.place_order_ws(
                symbol=signal.symbol,
                side=signal.side,
                qty=rounded_qty,
                order_type="Market",
                take_profit=None,
                stop_loss=signal.stop_loss,
                tp_trigger_by="LastPrice",
                sl_trigger_by="LastPrice",
                tpsl_mode="Full",
            )
        except Exception as e:
            logger.error(f"Market entry failed: {e}")
            return {"status": "error", "reason": str(e)}

        ret_code = response.get("retCode")
        if ret_code is not None and ret_code != 0:
            logger.error(
                f"Market entry rejected: retCode={ret_code} "
                f"retMsg={response.get('retMsg')}"
            )
            return {
                "status": "error",
                "reason": response.get("retMsg", ""),
                "retCode": ret_code,
            }

        tp_order_ids = await self._place_partial_tps(
            signal, rounded_qty, signal.entry_price,
        )

        pos = ActivePosition(
            symbol=signal.symbol,
            side=signal.side,
            entry_price=signal.entry_price,
            total_qty=rounded_qty,
            remaining_qty=rounded_qty,
            stop_loss=signal.stop_loss,
            initial_stop_loss=signal.stop_loss,
            tps=signal.tp_list,
            tp_weights=list(
                signal.tp_weights[:len(signal.tp_list)]
            ),
            tp_order_ids=tp_order_ids,
            tp_filled={tp: False for tp in signal.tp_list},
            opened_at=datetime.utcnow().isoformat(),
            signal_confidence=signal.confidence,
            highest_price=signal.entry_price,
            lowest_price=signal.entry_price,
        )
        self.active_positions[signal.symbol] = pos
        self.risk_manager.register_position_open()

        logger.info(
            f"POSITION OPENED (market): {signal.symbol} {signal.side} "
            f"qty={rounded_qty} entry={signal.entry_price:.6f} "
            f"SL={signal.stop_loss:.6f} "
            f"TPs={[round(t, 6) for t in signal.tp_list]} "
            f"tp_orders={len(tp_order_ids)}"
        )

        return {
            "status": "filled",
            "entry_type": "market",
            "symbol": signal.symbol,
            "side": signal.side,
            "qty": rounded_qty,
            "entry": signal.entry_price,
            "sl": signal.stop_loss,
            "tps": signal.tp_list,
            "tp_order_ids": tp_order_ids,
            "bybit_response": response,
        }

    # ================================================================== #
    #  Limit entry — background tracker
    # ================================================================== #
    async def _spawn_limit_entry(
        self, signal: TradeSignal, qty: float,
    ) -> Dict:
        task = asyncio.create_task(
            self._track_limit_entry(signal, qty)
        )
        self._pending_entries[signal.symbol] = task
        return {
            "status": "pending_limit",
            "entry_type": "limit",
            "symbol": signal.symbol,
            "entry": signal.entry_price,
            "qty": qty,
        }

    async def _track_limit_entry(
        self, signal: TradeSignal, qty: float,
    ):
        try:
            await self._execute_limit_entry(signal, qty)
        except Exception as e:
            logger.error(
                f"Limit entry tracker failed for {signal.symbol}: {e}",
                exc_info=True,
            )
        finally:
            self._pending_entries.pop(signal.symbol, None)
            self._pending_order_ids.pop(signal.symbol, None)

    async def _execute_limit_entry(
        self, signal: TradeSignal, qty: float,
    ):
        try:
            await self.client.set_leverage(
                signal.symbol, signal.leverage,
            )
        except Exception as e:
            logger.debug(f"Leverage set: {e}")

        rounded_qty, reason = self.client.round_qty(signal.symbol, qty)
        if rounded_qty <= 0:
            logger.error(
                f"Limit qty rounding failed for {signal.symbol}: "
                f"{reason} (raw={qty})"
            )
            return

        ok, notional_reason = self.client.validate_notional(
            signal.symbol, rounded_qty, signal.entry_price,
        )
        if not ok:
            logger.error(
                f"Limit notional check failed for {signal.symbol}: "
                f"{notional_reason}"
            )
            return

        limit_price = self.client.round_price(
            signal.symbol, signal.entry_price,
        )
        sl = self.client.round_price(
            signal.symbol, signal.stop_loss,
        )

        try:
            response = await self.client.place_order_ws(
                symbol=signal.symbol,
                side=signal.side,
                qty=rounded_qty,
                order_type="Limit",
                price=limit_price,
                take_profit=None,
                stop_loss=sl,
                tp_trigger_by="LastPrice",
                sl_trigger_by="LastPrice",
                tpsl_mode="Full",
            )
        except Exception as e:
            logger.error(
                f"Limit placement failed for {signal.symbol}: {e}"
            )
            return

        ret_code = response.get("retCode")
        if ret_code is not None and ret_code != 0:
            logger.error(
                f"Limit entry rejected for {signal.symbol}: "
                f"retCode={ret_code} retMsg={response.get('retMsg')}"
            )
            return

        order_id = self._extract_order_id(response)
        if not order_id:
            logger.error(
                f"Limit entry placed but no orderId returned: {response}"
            )
            return

        # Fix A: track order ID for shutdown cancellation
        self._pending_order_ids[signal.symbol] = order_id

        logger.info(
            f"LIMIT ORDER PLACED: {signal.symbol} {signal.side} "
            f"qty={rounded_qty} @ {limit_price:.6f} "
            f"SL={sl:.6f} orderId={order_id[:10]}"
        )

        fill = await self._wait_for_fill(
            signal, order_id, limit_price, rounded_qty,
        )

        status = fill["status"]

        if status == "filled":
            logger.info(
                f"LIMIT FILLED: {signal.symbol} "
                f"qty={fill['filled_qty']} @ "
                f"{fill['avg_price']:.6f} (planned {limit_price:.6f})"
            )
            await self._register_filled_position(
                signal, fill["filled_qty"], fill["avg_price"], sl,
            )
            return

        if status == "partial":
            logger.warning(
                f"LIMIT PARTIAL FILL: {signal.symbol} "
                f"{fill['filled_qty']}/{rounded_qty} @ "
                f"{fill['avg_price']:.6f} — cancelling remainder"
            )
            try:
                await self.client.cancel_order_ws(
                    signal.symbol, order_id,
                )
            except Exception as e:
                logger.warning(f"Cancel remainder failed: {e}")
            if fill["filled_qty"] > 0:
                await self._register_filled_position(
                    signal, fill["filled_qty"], fill["avg_price"], sl,
                )
            return

        if status == "cancelled_on_rally":
            logger.info(
                f"LIMIT CANCELLED (rally): {signal.symbol} "
                f"orderId={order_id[:10]} — price rallied away"
            )
            return

        if status == "stalled":
            logger.info(
                f"LIMIT STALLED: {signal.symbol} "
                f"orderId={order_id[:10]} — "
                f"{fill.get('candles', 0)} candles "
                f"({fill.get('elapsed_sec', 0)}s) elapsed, "
                f"order cancelled"
            )
            if self.on_limit_stalled:
                try:
                    await self.on_limit_stalled(signal, fill)
                except Exception as e:
                    logger.error(
                        f"on_limit_stalled callback error: {e}"
                    )
            return

        if status == "cancelled":
            logger.info(
                f"LIMIT CANCELLED (external): {signal.symbol} "
                f"orderId={order_id[:10]}"
            )
            return

        if status == "expired":
            logger.info(
                f"LIMIT EXPIRED (unfilled): {signal.symbol} "
                f"orderId={order_id[:10]} — signal missed"
            )
            try:
                await self.client.cancel_order_ws(
                    signal.symbol, order_id,
                )
            except Exception:
                pass
            return

    async def _wait_for_fill(
        self,
        signal: TradeSignal,
        order_id: str,
        limit_price: float,
        total_qty: float,
    ) -> Dict:
        deadline = time.time() + self.limit_expiry_min * 60
        start_time = time.time()

        cancel_threshold = self._rally_cancel_threshold(signal)
        candle_sec = self._candle_seconds(signal)
        stall_timeout = self._stall_timeout_seconds(signal)
        last_logged_candle = 0
        last_check_pct = None

        logger.info(
            f"LIMIT MONITOR START: {signal.symbol} "
            f"(setup={signal.metadata.get('setup_type', '?')}, "
            f"stall_after={self.stall_timeout_candles}×{candle_sec}s="
            f"{stall_timeout//60}min)"
        )

        while time.time() < deadline:
            await asyncio.sleep(self.limit_check_interval_sec)

            elapsed = time.time() - start_time
            elapsed_candles = elapsed / max(1, candle_sec)

            if int(elapsed_candles) >= last_logged_candle + 2:
                last_logged_candle = int(elapsed_candles)
                logger.info(
                    f"LIMIT PENDING: {signal.symbol} — "
                    f"{elapsed_candles:.1f}/{self.stall_timeout_candles} "
                    f"candles ({elapsed/60:.0f}/{stall_timeout/60:.0f} min)"
                )

            try:
                order = await self.client.get_order_status(
                    signal.symbol, order_id,
                )
            except Exception as e:
                logger.warning(
                    f"Order status fetch failed for "
                    f"{signal.symbol}/{order_id[:10]}: {e}"
                )
                order = None

            if order:
                status = order.get("orderStatus", "")
                cum_qty = float(order.get("cumExecQty", 0) or 0)
                avg_price = float(order.get("avgPrice", 0) or 0)

                if status == "Filled":
                    return {
                        "status": "filled",
                        "filled_qty": cum_qty,
                        "avg_price": avg_price,
                    }

                if status in (
                    "Cancelled", "Rejected", "Deactivated",
                    "PartiallyFilledCanceled",
                ):
                    if cum_qty > 0:
                        return {
                            "status": "partial",
                            "filled_qty": cum_qty,
                            "avg_price": avg_price,
                        }
                    return {
                        "status": "cancelled",
                        "filled_qty": 0,
                        "avg_price": 0,
                    }

            # ---- Stall check ----
            if elapsed > stall_timeout:
                logger.warning(
                    f"LIMIT STALLED: {signal.symbol} — "
                    f"not filled after {elapsed_candles:.1f} candles "
                    f"({elapsed/60:.0f} min), cancelling order"
                )
                try:
                    cancel_resp = await self.client.cancel_order_ws(
                        signal.symbol, order_id,
                    )
                    cancel_code = cancel_resp.get("retCode")
                    if cancel_code != 0:
                        recheck = await self.client.get_order_status(
                            signal.symbol, order_id,
                        )
                        if recheck:
                            s2 = recheck.get("orderStatus", "")
                            cq = float(recheck.get("cumExecQty", 0) or 0)
                            ap = float(recheck.get("avgPrice", 0) or 0)
                            if s2 == "Filled" or cq > 0:
                                logger.info(
                                    f"{signal.symbol}: order filled "
                                    f"despite stall cancel"
                                )
                                return {
                                    "status": "filled",
                                    "filled_qty": cq,
                                    "avg_price": ap,
                                }
                except Exception as e:
                    logger.warning(f"Stall cancel failed: {e}")

                return {
                    "status": "stalled",
                    "filled_qty": 0,
                    "avg_price": 0,
                    "elapsed_sec": int(elapsed),
                    "candles": round(elapsed_candles, 1),
                    "stall_timeout_sec": stall_timeout,
                }

            # ---- Rally check ----
            if cancel_threshold is not None:
                current = await self._get_current_price(signal.symbol)
                if current > 0:
                    rally_pct = self._rally_pct(
                        signal.direction, signal.entry_price, current,
                    )
                    rallied_away = (
                        (signal.direction == "long"
                         and current >= cancel_threshold)
                        or (signal.direction == "short"
                            and current <= cancel_threshold)
                    )

                    if rallied_away and last_check_pct != "rally":
                        logger.info(
                            f"{signal.symbol}: price rallied "
                            f"{rally_pct:.2f}% past entry — "
                            f"cancelling limit"
                        )
                        last_check_pct = "rally"

                    if rallied_away:
                        try:
                            cancel_resp = (
                                await self.client.cancel_order_ws(
                                    signal.symbol, order_id,
                                )
                            )
                            cancel_code = cancel_resp.get("retCode")
                            if cancel_code != 0:
                                recheck = (
                                    await self.client.get_order_status(
                                        signal.symbol, order_id,
                                    )
                                )
                                if recheck:
                                    s2 = recheck.get("orderStatus", "")
                                    cq = float(
                                        recheck.get("cumExecQty", 0) or 0
                                    )
                                    ap = float(
                                        recheck.get("avgPrice", 0) or 0
                                    )
                                    if s2 == "Filled" or cq > 0:
                                        logger.info(
                                            f"{signal.symbol}: order "
                                            f"filled despite rally cancel"
                                        )
                                        return {
                                            "status": "filled",
                                            "filled_qty": cq,
                                            "avg_price": ap,
                                        }
                        except Exception as e:
                            logger.warning(
                                f"Rally cancel failed: {e}"
                            )

                        return {
                            "status": "cancelled_on_rally",
                            "filled_qty": 0,
                            "avg_price": 0,
                            "rally_pct": rally_pct,
                        }

        # Absolute expiry
        try:
            order = await self.client.get_order_status(
                signal.symbol, order_id,
            )
            cum_qty = (
                float(order.get("cumExecQty", 0) or 0)
                if order else 0
            )
            avg_price = (
                float(order.get("avgPrice", 0) or 0)
                if order else 0
            )
        except Exception:
            cum_qty = 0
            avg_price = 0

        try:
            await self.client.cancel_order_ws(signal.symbol, order_id)
        except Exception:
            pass

        if cum_qty > 0:
            return {
                "status": "partial",
                "filled_qty": cum_qty,
                "avg_price": avg_price,
            }
        return {
            "status": "expired",
            "filled_qty": 0,
            "avg_price": 0,
            "elapsed_sec": int(time.time() - start_time),
        }

    async def _register_filled_position(
        self,
        signal: TradeSignal,
        filled_qty: float,
        fill_price: float,
        sl: float,
    ):
        if signal.symbol in self.active_positions:
            logger.warning(
                f"{signal.symbol}: position already tracked, "
                f"skipping re-register"
            )
            return

        tp_order_ids = await self._place_partial_tps(
            signal, filled_qty, fill_price,
        )

        pos = ActivePosition(
            symbol=signal.symbol,
            side=signal.side,
            entry_price=fill_price,
            total_qty=filled_qty,
            remaining_qty=filled_qty,
            stop_loss=sl,
            initial_stop_loss=sl,
            tps=signal.tp_list,
            tp_weights=list(
                signal.tp_weights[:len(signal.tp_list)]
            ),
            tp_order_ids=tp_order_ids,
            tp_filled={tp: False for tp in signal.tp_list},
            opened_at=datetime.utcnow().isoformat(),
            signal_confidence=signal.confidence,
            highest_price=fill_price,
            lowest_price=fill_price,
        )
        self.active_positions[signal.symbol] = pos
        self.risk_manager.register_position_open()

        logger.info(
            f"POSITION OPENED (limit): {signal.symbol} {signal.side} "
            f"qty={filled_qty} entry={fill_price:.6f} "
            f"SL={sl:.6f} "
            f"TPs={[round(t, 6) for t in signal.tp_list]} "
            f"tp_orders={len(tp_order_ids)}"
        )

    async def _place_partial_tps(
        self,
        signal: TradeSignal,
        filled_qty: float,
        fill_price: float,
    ) -> Dict[float, str]:
        tp_order_ids: Dict[float, str] = {}

        notional = filled_qty * fill_price
        if notional < self.tp_min_position_warn_usd:
            logger.warning(
                f"{signal.symbol}: position notional ${notional:.2f} "
                f"< ${self.tp_min_position_warn_usd} — "
                f"some TPs may round to zero"
            )

        tps = signal.tp_list
        weights = list(signal.tp_weights[:len(tps)])

        for i, tp_price in enumerate(tps):
            w = weights[i] if i < len(weights) else 0.0
            tp_qty = filled_qty * w
            if tp_qty <= 0:
                continue

            rounded_tp_qty, _ = self.client.round_qty(
                signal.symbol, tp_qty,
            )
            if rounded_tp_qty <= 0:
                logger.warning(
                    f"TP{i+1} qty too small for {signal.symbol}: "
                    f"raw={tp_qty}"
                )
                continue

            rounded_tp_price = self.client.round_price(
                signal.symbol, tp_price,
            )

            resp = None
            for attempt in range(self.tp_placement_retries + 1):
                try:
                    resp = await self.client.place_order_ws(
                        symbol=signal.symbol,
                        side="Sell" if signal.side == "Buy" else "Buy",
                        qty=rounded_tp_qty,
                        order_type="Limit",
                        price=rounded_tp_price,
                        reduce_only=True,
                    )
                    if resp.get("retCode") == 0:
                        break
                    if attempt < self.tp_placement_retries:
                        logger.warning(
                            f"TP{i+1} attempt {attempt+1} failed: "
                            f"retCode={resp.get('retCode')} "
                            f"retMsg={resp.get('retMsg')} — retrying"
                        )
                        await asyncio.sleep(
                            self.tp_placement_retry_delay,
                        )
                except Exception as e:
                    if attempt < self.tp_placement_retries:
                        logger.warning(
                            f"TP{i+1} attempt {attempt+1} exception: "
                            f"{e} — retrying"
                        )
                        await asyncio.sleep(
                            self.tp_placement_retry_delay,
                        )
                    else:
                        logger.error(
                            f"TP{i+1} failed after "
                            f"{self.tp_placement_retries + 1} "
                            f"attempts: {e}"
                        )

            if not resp or resp.get("retCode") != 0:
                continue

            order_id = self._extract_order_id(resp)
            if order_id:
                tp_order_ids[rounded_tp_price] = order_id
                logger.info(
                    f"TP{i+1} placed: {signal.symbol} "
                    f"qty={rounded_tp_qty} "
                    f"({w*100:.0f}% of {filled_qty}) "
                    f"@ {rounded_tp_price:.6f} "
                    f"orderId={order_id[:10]}"
                )

        return tp_order_ids

    @staticmethod
    def _extract_order_id(resp: Dict) -> Optional[str]:
        data = resp.get("data", {})
        if isinstance(data, dict):
            lst = data.get("list", [])
            if lst and isinstance(lst[0], dict):
                return lst[0].get("orderId")
            if "orderId" in data:
                return data.get("orderId")
        result = resp.get("result", {})
        if isinstance(result, dict):
            return result.get("orderId")
        return None

    # ================================================================== #
    #  Position monitor
    # ================================================================== #
    async def start_monitor(self):
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def _monitor_loop(self):
        while self._running:
            try:
                cutoff = time.time() - 300
                self._recently_closed = {
                    s: t for s, t in self._recently_closed.items()
                    if t > cutoff
                }

                if not self.active_positions:
                    await asyncio.sleep(5)
                    continue

                positions = await self.client.get_positions()
                live_map = {
                    p["symbol"]: p for p in positions
                    if abs(float(p.get("size", 0) or 0)) > 0
                }

                for sym in list(self.active_positions.keys()):
                    if sym not in live_map:
                        await self._handle_position_closed(sym)

                for sym, pos in self.active_positions.items():
                    if sym not in live_map:
                        continue
                    p = live_map[sym]
                    live_qty = abs(float(p.get("size", 0) or 0))
                    current_price = float(
                        p.get("markPrice", p.get("lastPrice", 0)) or 0
                    )
                    pos.unrealized_pnl = float(
                        p.get("unrealisedPnl", 0) or 0
                    )

                    if pos.side == "Buy":
                        pos.highest_price = max(
                            pos.highest_price, current_price,
                        )
                    else:
                        pos.lowest_price = min(
                            pos.lowest_price or current_price,
                            current_price,
                        )

                    await self._detect_tp_fills(
                        pos, live_qty, current_price,
                    )

                    if pos.side == "Buy":
                        adverse = (
                            (pos.entry_price - current_price)
                            / pos.entry_price * 100
                        )
                    else:
                        adverse = (
                            (current_price - pos.entry_price)
                            / pos.entry_price * 100
                        )
                    if adverse >= self.emergency_sl_pct:
                        logger.critical(
                            f"EMERGENCY SL {sym}: "
                            f"adverse {adverse:.2f}%"
                        )
                        pos.exit_reason = "emergency_sl"
                        await self._emergency_close(pos)

                    if pos.opened_at:
                        held = (
                            datetime.utcnow()
                            - datetime.fromisoformat(pos.opened_at)
                        ).total_seconds() / 3600
                        if held > self.max_hold_hours:
                            logger.warning(
                                f"Max hold exceeded for {sym}"
                            )
                            pos.exit_reason = "max_hold"
                            await self._emergency_close(pos)

            except Exception as e:
                logger.error(
                    f"Monitor loop error: {e}", exc_info=True,
                )

            await asyncio.sleep(3)

    async def _detect_tp_fills(
        self,
        pos: ActivePosition,
        live_qty: float,
        current_price: float,
    ):
        cumulative = []
        acc = 0.0
        for w in pos.tp_weights[:len(pos.tps)]:
            acc += w
            cumulative.append(acc)
        tolerance = pos.total_qty * 0.01

        for i, tp in enumerate(pos.tps):
            if pos.tp_filled.get(tp):
                continue
            expected_remaining = pos.total_qty * (1 - cumulative[i])
            if live_qty <= expected_remaining + tolerance:
                price_ok = (
                    (pos.side == "Buy"
                     and current_price >= tp * 0.995)
                    or (pos.side == "Sell"
                        and current_price <= tp * 1.005)
                )
                if price_ok:
                    pos.tp_filled[tp] = True
                    pos.remaining_qty = live_qty
                    logger.info(
                        f"{pos.symbol}: TP{i+1} @ {tp:.6f} FILLED "
                        f"(remaining qty {live_qty:.6f})"
                    )
                    await self._on_tp_filled(pos, i)

    async def _on_tp_filled(
        self, pos: ActivePosition, tp_index: int,
    ):
        if tp_index == 0 and self.move_sl_be_after_tp1:
            await self._move_sl_to_breakeven(pos)
        elif tp_index == 1 and self.move_sl_tp1_after_tp2:
            await self._move_sl_to_tp1(pos)

    async def _move_sl_to_breakeven(self, pos: ActivePosition):
        if pos.sl_moved_to_be:
            return
        new_sl = pos.entry_price
        if abs(new_sl - pos.stop_loss) < 1e-9:
            pos.sl_moved_to_be = True
            return
        try:
            await self.client.set_trading_stop(
                symbol=pos.symbol, stop_loss=round(new_sl, 6),
            )
            pos.stop_loss = new_sl
            pos.sl_moved_to_be = True
            logger.info(
                f"{pos.symbol}: SL moved to BREAKEVEN "
                f"({new_sl:.6f})"
            )
        except Exception as e:
            logger.error(
                f"SL-to-BE failed for {pos.symbol}: {e}"
            )

    async def _move_sl_to_tp1(self, pos: ActivePosition):
        if pos.sl_moved_to_tp1 or not pos.tps:
            return
        tp1 = pos.tps[0]
        try:
            await self.client.set_trading_stop(
                symbol=pos.symbol, stop_loss=round(tp1, 6),
            )
            pos.stop_loss = tp1
            pos.sl_moved_to_tp1 = True
            logger.info(
                f"{pos.symbol}: SL moved to TP1 ({tp1:.6f})"
            )
        except Exception as e:
            logger.error(
                f"SL-to-TP1 failed for {pos.symbol}: {e}"
            )

    async def _handle_position_closed(self, symbol: str):
        pos = self.active_positions.get(symbol)
        if not pos:
            return

        del self.active_positions[symbol]

        now = time.time()
        last_closed = self._recently_closed.get(symbol, 0.0)
        if now - last_closed < 60:
            logger.debug(
                f"{symbol}: close already processed "
                f"{now - last_closed:.1f}s ago — skipping duplicate"
            )
            return
        self._recently_closed[symbol] = now

        try:
            pnl_pct = (
                pos.unrealized_pnl / (pos.entry_price * pos.total_qty) * 100
                if pos.entry_price * pos.total_qty > 0 else 0.0
            )
            filled_tps = [
                tp for tp, f in pos.tp_filled.items() if f
            ]
            total_tps = len(pos.tps)

            if pos.exit_reason:
                exit_reason = pos.exit_reason
            elif len(filled_tps) == total_tps and total_tps > 0:
                exit_reason = "tp_all"
            elif pnl_pct < 0:
                exit_reason = "sl_hit"
            else:
                exit_reason = "manual"

            logger.info(
                f"POSITION CLOSED: {symbol} "
                f"PnL={pos.unrealized_pnl:.2f} ({pnl_pct:.2f}%) "
                f"TPs hit: {len(filled_tps)}/{total_tps} "
                f"reason={exit_reason}"
            )

            self.risk_manager.register_position_close(pnl_pct)

            if self.on_position_closed:
                try:
                    await self.on_position_closed(
                        symbol=symbol,
                        pos=pos,
                        pnl_pct=pnl_pct,
                        filled_tps=filled_tps,
                        exit_reason=exit_reason,
                    )
                except Exception as e:
                    logger.error(
                        f"Position closed callback error for "
                        f"{symbol}: {e}"
                    )

            for tp_price, order_id in pos.tp_order_ids.items():
                if not pos.tp_filled.get(tp_price):
                    try:
                        await self.client.cancel_order_ws(
                            symbol=symbol, order_id=order_id,
                        )
                        logger.info(
                            f"Canceled unfilled TP order "
                            f"{order_id[:10]} for {symbol}"
                        )
                    except Exception as e:
                        logger.debug(
                            f"TP cancel failed {order_id}: {e}"
                        )

        except Exception as e:
            logger.error(
                f"_handle_position_closed error for {symbol}: {e}",
                exc_info=True,
            )

    async def _emergency_close(self, pos: ActivePosition):
        try:
            if pos.exit_reason is None:
                pos.exit_reason = "emergency_sl"
            await self.client.close_position_market(
                symbol=pos.symbol,
                side=pos.side,
                qty=pos.remaining_qty,
            )
            logger.info(f"Emergency close {pos.symbol}")
        except Exception as e:
            logger.error(
                f"Emergency close FAILED {pos.symbol}: {e}"
            )

    # ================================================================== #
    #  Lifecycle (Fix A: cancel Bybit orders before tasks die)
    # ================================================================== #
    async def close_all_positions(self):
        """
        Shutdown-safe close: cancel live Bybit limit orders FIRST,
        then cancel Python tasks, then close open positions.
        """
        # 1. Cancel live Bybit limit orders before tasks die
        for sym, order_id in list(self._pending_order_ids.items()):
            try:
                await self.client.cancel_order_ws(sym, order_id)
                logger.info(
                    f"Shutdown: cancelled live limit {order_id[:10]} "
                    f"for {sym}"
                )
            except Exception as e:
                logger.error(
                    f"Shutdown: cancel failed for {sym}/{order_id[:10]}: {e}"
                )
        self._pending_order_ids.clear()

        # 2. Cancel Python tasks
        for sym, task in list(self._pending_entries.items()):
            if not task.done():
                task.cancel()
        self._pending_entries.clear()

        # 3. Close open positions
        for sym in list(self.active_positions.keys()):
            await self._emergency_close(self.active_positions[sym])
        self.active_positions.clear()

    async def stop_monitor(self):
        self._running = False

        # 1. Cancel live Bybit orders
        for sym, order_id in list(self._pending_order_ids.items()):
            try:
                await self.client.cancel_order_ws(sym, order_id)
                logger.info(
                    f"Stop: cancelled live limit {order_id[:10]} for {sym}"
                )
            except Exception as e:
                logger.error(f"Stop: cancel failed {sym}: {e}")
        self._pending_order_ids.clear()

        # 2. Cancel Python tasks
        for task in self._pending_entries.values():
            if not task.done():
                task.cancel()
        self._pending_entries.clear()

        # 3. Stop monitor loop
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass