"""
execution/trade_executor.py
WebSocket order placement + real-time position monitoring with
multi-target take-profit (TP1/TP2/TP3) and SL management.

Strategy:
  - Entry: market order + native TP1 + native SL (exchange-protected)
  - After fill: TP2 and TP3 placed as reduce-only limit orders
  - On TP1 fill: SL moves to breakeven
  - On TP2 fill: SL moves to TP1 level (locks in profit)
  - On TP3 fill: position fully closed
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    symbol: str
    direction: str              # "long" | "short"
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
        """R:R if only TP1 is hit."""
        risk = abs(self.entry_price - self.stop_loss)
        if risk <= 0:
            return 0.0
        return abs(self.take_profit_1 - self.entry_price) / risk

    @property
    def risk_reward(self) -> float:
        """Weighted-average R:R across all TP targets."""
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


class TradeExecutor:
    def __init__(self, client, risk_manager, config: Dict):
        self.client = client
        self.risk_manager = risk_manager
        self.config = config
        self.active_positions: Dict[str, ActivePosition] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

        self.trailing_activation_pct = config.get("trailing_activation_pct", 1.0)
        self.trailing_distance_pct = config.get("trailing_distance_pct", 0.5)
        self.emergency_sl_pct = config.get("emergency_sl_pct", 2.5)
        self.max_hold_hours = config.get("max_hold_hours", 72)
        self.move_sl_be_after_tp1 = config.get("move_sl_to_be_after_tp1", True)
        self.move_sl_tp1_after_tp2 = config.get("move_sl_to_tp1_after_tp2", True)

    # ------------------------------------------------------------------ #
    #  Entry
    # ------------------------------------------------------------------ #
    async def execute_signal(self, signal: TradeSignal) -> Dict:
        approved, size_usd, reason = self.risk_manager.approve_trade(signal)
        if not approved:
            logger.warning(f"Trade rejected by risk manager: {reason}")
            return {"status": "rejected", "reason": reason}

        qty = size_usd / signal.entry_price
        signal.position_size_usd = size_usd

        try:
            await self.client.set_leverage(signal.symbol, signal.leverage)
        except Exception as e:
            logger.debug(f"Leverage set: {e}")

        # ---- Step 1: entry with native TP1 + SL ------------------------
        try:
            entry_resp = await self.client.place_order_ws(
                symbol=signal.symbol,
                side=signal.side,
                qty=round(qty, 6),
                order_type="Market",
                take_profit=signal.take_profit_1,
                stop_loss=signal.stop_loss,
                tp_trigger_by="LastPrice",
                sl_trigger_by="LastPrice",
                tpsl_mode="Full",
            )
        except Exception as e:
            logger.error(f"Entry placement failed: {e}")
            return {"status": "error", "reason": str(e)}

        # Verify entry succeeded
        ret_code = entry_resp.get("retCode")
        if ret_code is not None and ret_code != 0:
            logger.error(
                f"Entry rejected by Bybit: retCode={ret_code} "
                f"retMsg={entry_resp.get('retMsg')}"
            )
            return {"status": "error", "reason": entry_resp.get("retMsg", "")}

        # ---- Step 2: TP2 and TP3 as reduce-only limits ----------------
        tp_order_ids: Dict[float, str] = {}
        tps = signal.tp_list
        weights = list(signal.tp_weights[:len(tps)])

        # TP1 is native; place TP2 and TP3 as limit orders
        for i, tp_price in enumerate(tps):
            if i == 0:
                continue  # TP1 handled natively
            w = weights[i] if i < len(weights) else 0.0
            tp_qty = qty * w
            if tp_qty <= 0:
                continue
            try:
                tp_resp = await self.client.place_order_ws(
                    symbol=signal.symbol,
                    side="Sell" if signal.side == "Buy" else "Buy",
                    qty=round(tp_qty, 6),
                    order_type="Limit",
                    price=tp_price,
                    reduce_only=True,
                )
                order_id = self._extract_order_id(tp_resp)
                if order_id:
                    tp_order_ids[tp_price] = order_id
                    logger.info(
                        f"TP{i+1} placed @ {tp_price:.6f} "
                        f"qty={tp_qty:.6f} id={order_id[:8]}..."
                    )
                else:
                    logger.warning(
                        f"TP{i+1} placed but no orderId in response: "
                        f"{tp_resp}"
                    )
            except Exception as e:
                logger.warning(f"TP{i+1} placement failed: {e}")

        # ---- Step 3: register position --------------------------------
        pos = ActivePosition(
            symbol=signal.symbol,
            side=signal.side,
            entry_price=signal.entry_price,
            total_qty=round(qty, 6),
            remaining_qty=round(qty, 6),
            stop_loss=signal.stop_loss,
            initial_stop_loss=signal.stop_loss,
            tps=tps,
            tp_weights=weights,
            tp_order_ids=tp_order_ids,
            tp_filled={tp: False for tp in tps},
            opened_at=datetime.utcnow().isoformat(),
            signal_confidence=signal.confidence,
            highest_price=signal.entry_price,
            lowest_price=signal.entry_price,
        )
        self.active_positions[signal.symbol] = pos
        self.risk_manager.register_position_open()

        logger.info(
            f"POSITION OPENED: {signal.symbol} {signal.side} "
            f"qty={qty:.6f} entry={signal.entry_price:.6f} "
            f"SL={signal.stop_loss:.6f} "
            f"TPs={[round(t, 6) for t in tps]}"
        )

        return {
            "status": "filled",
            "symbol": signal.symbol,
            "side": signal.side,
            "qty": qty,
            "entry": signal.entry_price,
            "sl": signal.stop_loss,
            "tps": tps,
            "tp_weights": weights,
            "tp_order_ids": tp_order_ids,
            "bybit_response": entry_resp,
        }

    @staticmethod
    def _extract_order_id(resp: Dict) -> Optional[str]:
        """Extract order ID from Bybit WS/REST order.create response."""
        data = resp.get("data", {})
        if isinstance(data, dict):
            lst = data.get("list", [])
            if lst and isinstance(lst[0], dict):
                return lst[0].get("orderId")
            if "orderId" in data:
                return data.get("orderId")
        # Some SDKs flatten: {"result": {"orderId": "..."}}
        result = resp.get("result", {})
        if isinstance(result, dict):
            return result.get("orderId")
        return None

    # ------------------------------------------------------------------ #
    #  Monitor
    # ------------------------------------------------------------------ #
    async def start_monitor(self):
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def _monitor_loop(self):
        while self._running:
            try:
                if not self.active_positions:
                    await asyncio.sleep(5)
                    continue

                positions = await self.client.get_positions()
                live_map = {
                    p["symbol"]: p for p in positions
                    if abs(float(p.get("size", 0) or 0)) > 0
                }

                # Handle fully-closed positions
                for sym in list(self.active_positions.keys()):
                    if sym not in live_map:
                        await self._handle_position_closed(sym)

                # Process still-open positions
                for sym, pos in self.active_positions.items():
                    if sym not in live_map:
                        continue
                    p = live_map[sym]
                    live_qty = abs(float(p.get("size", 0) or 0))
                    current_price = float(
                        p.get("markPrice", p.get("lastPrice", 0)) or 0
                    )
                    pos.unrealized_pnl = float(p.get("unrealisedPnl", 0) or 0)

                    # Track high/low for trailing
                    if pos.side == "Buy":
                        pos.highest_price = max(pos.highest_price, current_price)
                    else:
                        pos.lowest_price = min(
                            pos.lowest_price or current_price, current_price
                        )

                    # Detect TP fills by comparing size
                    await self._detect_tp_fills(pos, live_qty, current_price)

                    # Emergency SL check
                    if pos.side == "Buy":
                        adverse = (pos.entry_price - current_price) / pos.entry_price * 100
                    else:
                        adverse = (current_price - pos.entry_price) / pos.entry_price * 100
                    if adverse >= self.emergency_sl_pct:
                        logger.critical(
                            f"EMERGENCY SL {sym}: adverse {adverse:.2f}%"
                        )
                        await self._emergency_close(pos)

                    # Max hold
                    if pos.opened_at:
                        held = (
                            datetime.utcnow()
                            - datetime.fromisoformat(pos.opened_at)
                        ).total_seconds() / 3600
                        if held > self.max_hold_hours:
                            logger.warning(f"Max hold exceeded for {sym}")
                            await self._emergency_close(pos)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}", exc_info=True)

            await asyncio.sleep(3)

    async def _detect_tp_fills(
        self, pos: ActivePosition, live_qty: float, current_price: float,
    ):
        """Detect TP fills by observing position size reduction."""
        # Cumulative weights: how much should be closed by each TP
        cumulative = []
        acc = 0.0
        for w in pos.tp_weights[:len(pos.tps)]:
            acc += w
            cumulative.append(acc)

        tolerance = pos.total_qty * 0.01  # 1% tolerance

        for i, tp in enumerate(pos.tps):
            if pos.tp_filled.get(tp):
                continue
            expected_remaining = pos.total_qty * (1 - cumulative[i])
            if live_qty <= expected_remaining + tolerance:
                # Verify price is near the TP (avoid false positives)
                price_ok = (
                    (pos.side == "Buy" and current_price >= tp * 0.995)
                    or (pos.side == "Sell" and current_price <= tp * 1.005)
                )
                if price_ok:
                    pos.tp_filled[tp] = True
                    pos.remaining_qty = live_qty
                    logger.info(
                        f"{pos.symbol}: TP{i+1} @ {tp:.6f} FILLED "
                        f"(remaining qty {live_qty:.6f})"
                    )
                    await self._on_tp_filled(pos, i)

    async def _on_tp_filled(self, pos: ActivePosition, tp_index: int):
        """Run SL management logic after a TP fills."""
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
                f"{pos.symbol}: SL moved to BREAKEVEN ({new_sl:.6f})"
            )
        except Exception as e:
            logger.error(f"SL-to-BE failed for {pos.symbol}: {e}")

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
            logger.error(f"SL-to-TP1 failed for {pos.symbol}: {e}")

    async def _handle_position_closed(self, symbol: str):
        pos = self.active_positions.get(symbol)
        if not pos:
            return

        pnl_pct = (
            pos.unrealized_pnl / (pos.entry_price * pos.total_qty) * 100
            if pos.entry_price * pos.total_qty > 0 else 0.0
        )
        filled_tps = [tp for tp, f in pos.tp_filled.items() if f]
        logger.info(
            f"POSITION CLOSED: {symbol} "
            f"PnL={pos.unrealized_pnl:.2f} ({pnl_pct:.2f}%) "
            f"TPs hit: {len(filled_tps)}/{len(pos.tps)}"
        )
        self.risk_manager.register_position_close(pnl_pct)

        # Cancel any outstanding TP orders (Bybit does not auto-cancel)
        for tp_price, order_id in pos.tp_order_ids.items():
            if not pos.tp_filled.get(tp_price):
                try:
                    await self.client.cancel_order_ws(
                        symbol=symbol, order_id=order_id,
                    )
                    logger.info(
                        f"Canceled unfilled TP order {order_id[:8]} "
                        f"for {symbol}"
                    )
                except Exception as e:
                    logger.debug(f"TP cancel failed {order_id}: {e}")

        del self.active_positions[symbol]

    async def _emergency_close(self, pos: ActivePosition):
        try:
            await self.client.close_position_market(
                symbol=pos.symbol, side=pos.side, qty=pos.remaining_qty,
            )
            logger.info(f"Emergency close {pos.symbol}")
        except Exception as e:
            logger.error(f"Emergency close FAILED {pos.symbol}: {e}")

    async def close_all_positions(self):
        for sym in list(self.active_positions.keys()):
            await self._emergency_close(self.active_positions[sym])
        self.active_positions.clear()

    async def stop_monitor(self):
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass