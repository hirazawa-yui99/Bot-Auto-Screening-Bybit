"""
execution/trade_executor.py
WebSocket order placement + real-time position monitoring.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

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
    confidence: float = 0.0
    position_size_usd: float = 0.0
    leverage: int = 5
    reason: str = ""
    metadata: Dict = field(default_factory=dict)

    @property
    def side(self) -> str:
        return "Buy" if self.direction == "long" else "Sell"

    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit_1 - self.entry_price)
        return reward / risk if risk > 0 else 0.0


@dataclass
class ActivePosition:
    symbol: str
    side: str
    entry_price: float
    qty: float
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    opened_at: str = ""
    signal_confidence: float = 0.0
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0


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

        try:
            result = await self.client.place_order_ws(
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
            logger.error(f"Order placement failed: {e}")
            return {"status": "error", "reason": str(e)}

        pos = ActivePosition(
            symbol=signal.symbol,
            side=signal.side,
            entry_price=signal.entry_price,
            qty=round(qty, 6),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit_1,
            opened_at=datetime.utcnow().isoformat(),
            signal_confidence=signal.confidence,
            highest_price=signal.entry_price,
            lowest_price=signal.entry_price,
        )
        self.active_positions[signal.symbol] = pos
        self.risk_manager.register_position_open()

        logger.info(
            f"POSITION OPENED: {signal.symbol} {signal.side} "
            f"qty={qty:.6f} entry={signal.entry_price} "
            f"SL={signal.stop_loss} TP={signal.take_profit_1}"
        )
        return {
            "status": "filled",
            "symbol": signal.symbol,
            "side": signal.side,
            "qty": qty,
            "entry": signal.entry_price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit_1,
            "bybit_response": result,
        }

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
                live_symbols = {
                    p["symbol"] for p in positions
                    if float(p.get("size", 0) or 0) != 0
                }

                for sym in list(self.active_positions.keys()):
                    if sym not in live_symbols:
                        logger.info(f"Position {sym} closed (TP/SL/Manual)")
                        pos = self.active_positions[sym]
                        pnl_pct = pos.unrealized_pnl / (
                            pos.entry_price * pos.qty
                        ) * 100 if pos.entry_price * pos.qty else 0.0
                        self.risk_manager.register_position_close(pnl_pct)
                        del self.active_positions[sym]

                for p in positions:
                    sym = p["symbol"]
                    if sym not in self.active_positions:
                        continue
                    pos = self.active_positions[sym]
                    current_price = float(p.get("markPrice", p.get("lastPrice", 0)) or 0)
                    pos.unrealized_pnl = float(p.get("unrealisedPnl", 0) or 0)

                    if pos.side == "Buy":
                        pos.highest_price = max(pos.highest_price, current_price)
                        await self._update_trailing_long(pos, current_price)
                        await self._check_emergency_long(pos, current_price)
                    else:
                        pos.lowest_price = min(pos.lowest_price or current_price, current_price)
                        await self._update_trailing_short(pos, current_price)
                        await self._check_emergency_short(pos, current_price)

                    if pos.opened_at:
                        held = (
                            datetime.utcnow()
                            - datetime.fromisoformat(pos.opened_at)
                        ).total_seconds() / 3600
                        if held > self.max_hold_hours:
                            logger.warning(f"Max hold exceeded for {sym}. Closing.")
                            await self._emergency_close(pos)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}", exc_info=True)

            await asyncio.sleep(3)

    async def _update_trailing_long(self, pos, current_price):
        profit_pct = (current_price - pos.entry_price) / pos.entry_price * 100
        if profit_pct >= self.trailing_activation_pct:
            new_sl = current_price * (1 - self.trailing_distance_pct / 100)
            if new_sl > pos.stop_loss:
                pos.stop_loss = new_sl
                pos.trailing_stop = new_sl
                try:
                    await self.client.set_trading_stop(
                        symbol=pos.symbol, stop_loss=round(new_sl, 6),
                    )
                    logger.info(f"Trailing SL {pos.symbol}: {new_sl:.6f}")
                except Exception as e:
                    logger.error(f"Trailing update failed: {e}")

    async def _update_trailing_short(self, pos, current_price):
        profit_pct = (pos.entry_price - current_price) / pos.entry_price * 100
        if profit_pct >= self.trailing_activation_pct:
            new_sl = current_price * (1 + self.trailing_distance_pct / 100)
            if pos.stop_loss is None or new_sl < pos.stop_loss:
                pos.stop_loss = new_sl
                pos.trailing_stop = new_sl
                try:
                    await self.client.set_trading_stop(
                        symbol=pos.symbol, stop_loss=round(new_sl, 6),
                    )
                    logger.info(f"Trailing SL {pos.symbol}: {new_sl:.6f}")
                except Exception as e:
                    logger.error(f"Trailing update failed: {e}")

    async def _check_emergency_long(self, pos, current_price):
        adverse = (pos.entry_price - current_price) / pos.entry_price * 100
        if adverse >= self.emergency_sl_pct:
            logger.critical(f"EMERGENCY SL {pos.symbol}: {adverse:.2f}%")
            await self._emergency_close(pos)

    async def _check_emergency_short(self, pos, current_price):
        adverse = (current_price - pos.entry_price) / pos.entry_price * 100
        if adverse >= self.emergency_sl_pct:
            logger.critical(f"EMERGENCY SL {pos.symbol}: {adverse:.2f}%")
            await self._emergency_close(pos)

    async def _emergency_close(self, pos):
        try:
            await self.client.close_position_market(
                symbol=pos.symbol, side=pos.side, qty=pos.qty,
            )
            logger.info(f"Emergency close {pos.symbol}")
        except Exception as e:
            logger.error(f"Emergency close FAILED {pos.symbol}: {e}")

    async def close_all_positions(self):
        for sym, pos in list(self.active_positions.items()):
            await self._emergency_close(pos)
        self.active_positions.clear()

    async def stop_monitor(self):
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass