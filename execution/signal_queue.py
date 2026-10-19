"""
execution/signal_queue.py
Decoupled signal-to-execution pipeline.

Signals are enqueued with status='pending' and execute_after timestamp.
A worker polls pending signals every N seconds and executes them once
the delay has elapsed. This separates signal generation (Discord alert,
DB record) from order placement (by the executor).

Modes:
  auto_trade.enabled = true   → worker executes pending signals
  auto_trade.enabled = false  → worker runs but does nothing (observation)

States:
  pending    → waiting for execute_after
  executing  → order placed, awaiting fill (limit case)
  filled     → position opened
  skipped    → rejected by risk/cooldown at exec time
  failed     → execution error
  expired    → execute_after passed but not executed within expiry window
  stalled    → limit placed but not filled within stall timeout
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class SignalQueueConfig:
    enabled: bool = True
    delay_seconds: int = 30
    execution_interval_seconds: int = 5
    expiry_seconds: int = 300
    notify_on_execution: bool = True


class SignalQueue:
    def __init__(self, db, config: Dict):
        cfg = config or {}
        self.cfg = SignalQueueConfig(
            enabled=bool(cfg.get("enabled", True)),
            delay_seconds=int(cfg.get("delay_seconds", 30)),
            execution_interval_seconds=int(
                cfg.get("execution_interval_seconds", 5),
            ),
            expiry_seconds=int(cfg.get("expiry_seconds", 300)),
            notify_on_execution=bool(
                cfg.get("notify_on_execution", True),
            ),
        )
        self.db = db
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._executor = None
        self._on_filled: Optional[Callable] = None
        self._pending_symbols: Set[str] = set()

    # ------------------------------------------------------------------ #
    #  Wiring
    # ------------------------------------------------------------------ #
    def attach(self, executor, on_filled: Optional[Callable] = None):
        self._executor = executor
        self._on_filled = on_filled

    def is_pending(self, symbol: str) -> bool:
        """Called by _should_skip_symbol to detect in-flight executions."""
        return symbol in self._pending_symbols

    def release_symbol(self, symbol: str):
        """
        Called when a background limit resolves (filled, stalled,
        rallied, expired). Removes from _pending_symbols so the
        symbol is eligible for new signals.
        """
        self._pending_symbols.discard(symbol)
        logger.debug(f"Queue released symbol {symbol}")

    def update_config(self, new_cfg: Dict):
        """Called during soft reload."""
        self.cfg.enabled = bool(new_cfg.get("enabled", self.cfg.enabled))
        self.cfg.delay_seconds = int(
            new_cfg.get("delay_seconds", self.cfg.delay_seconds)
        )
        self.cfg.execution_interval_seconds = int(
            new_cfg.get(
                "execution_interval_seconds",
                self.cfg.execution_interval_seconds,
            )
        )
        self.cfg.expiry_seconds = int(
            new_cfg.get("expiry_seconds", self.cfg.expiry_seconds)
        )
        self.cfg.notify_on_execution = bool(
            new_cfg.get(
                "notify_on_execution", self.cfg.notify_on_execution,
            )
        )
        logger.info(
            f"SignalQueue config updated: enabled={self.cfg.enabled}, "
            f"delay={self.cfg.delay_seconds}s"
        )

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #
    async def start(self):
        if not self.cfg.enabled:
            logger.info(
                "SignalQueue: auto_trade disabled — worker started but idle"
            )
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info(
            f"SignalQueue worker started "
            f"(delay={self.cfg.delay_seconds}s, "
            f"interval={self.cfg.execution_interval_seconds}s, "
            f"enabled={self.cfg.enabled})"
        )

    async def stop(self):
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------ #
    #  Worker loop
    # ------------------------------------------------------------------ #
    async def _worker_loop(self):
        while self._running:
            try:
                if self.cfg.enabled:
                    await self._process_pending()
            except Exception as e:
                logger.error(
                    f"SignalQueue worker error: {e}", exc_info=True,
                )
            await asyncio.sleep(self.cfg.execution_interval_seconds)

    async def _process_pending(self):
        if self._executor is None:
            return

        rows = await self.db.fetch(
            """
            SELECT * FROM signals
            WHERE execution_status = 'pending'
              AND execute_after <= NOW()
              AND timestamp > NOW() - INTERVAL '10 minutes'
            ORDER BY execute_after ASC
            LIMIT 10
            """
        )

        for row in rows:
            await self._execute_one(dict(row))

    # ------------------------------------------------------------------ #
    #  Execute one
    # ------------------------------------------------------------------ #
    async def _execute_one(self, sig: Dict):
        sig_id = sig["id"]
        symbol = sig["symbol"]
        self._pending_symbols.add(symbol)

        # Tracks whether we hand off to the executor's background tracker.
        # If True, we KEEP the symbol in _pending_symbols until
        # on_limit_stalled / on_filled / rally-cancel fires.
        handed_off = False

        try:
            # Expiry check
            age = (
                datetime.utcnow()
                - sig["timestamp"].replace(tzinfo=None)
            ).total_seconds()
            if age > self.cfg.delay_seconds + self.cfg.expiry_seconds:
                await self._mark_status(
                    sig_id, "expired",
                    {"reason": "signal_too_old", "age_sec": age},
                )
                logger.info(
                    f"Signal {sig_id} {symbol} EXPIRED (age={age:.0f}s)"
                )
                return

            # Parse plan
            plan = sig.get("plan")
            if isinstance(plan, str):
                plan = json.loads(plan)
            if not plan or "stop_loss" not in plan:
                await self._mark_status(
                    sig_id, "failed", {"reason": "missing_plan"},
                )
                logger.warning(
                    f"Signal {sig_id} {symbol} has no plan — skipping"
                )
                return

            # Atomic claim
            claimed = await self.db.execute(
                """
                UPDATE signals SET execution_status = 'executing'
                WHERE id = $1 AND execution_status = 'pending'
                """,
                sig_id,
            )
            if claimed != "UPDATE 1":
                return

            # Build TradeSignal
            from execution.trade_executor import TradeSignal
            signal = TradeSignal(
                symbol=symbol,
                direction=plan.get("direction", sig["direction"]),
                entry_price=float(plan.get(
                    "entry_price", sig["entry_price"],
                )),
                stop_loss=float(plan["stop_loss"]),
                take_profit_1=float(plan["take_profit_1"]),
                take_profit_2=(
                    float(plan["take_profit_2"])
                    if plan.get("take_profit_2") else None
                ),
                take_profit_3=(
                    float(plan["take_profit_3"])
                    if plan.get("take_profit_3") else None
                ),
                tp_weights=tuple(plan.get(
                    "tp_weights", [0.40, 0.35, 0.25],
                )),
                confidence=float(sig.get(
                    "calibrated_confidence", 0.0,
                )),
                leverage=int(plan.get("leverage", 5)),
                reason=f"queued_signal_id={sig_id}",
                metadata={
                    **(plan.get("metadata") or {}),
                    "signal_id": sig_id,
                    "setup_type": plan.get("setup_type", ""),
                    "setups": plan.get("setups", []),
                    "agreement": plan.get("agreement", {}),
                    "funding_block_long": plan.get(
                        "funding_block_long", False,
                    ),
                    "funding_block_short": plan.get(
                        "funding_block_short", False,
                    ),
                    "funding_rate_pct": sig.get(
                        "funding_rate_pct", 0.0,
                    ),
                },
            )

            # Execute
            try:
                result = await self._executor.execute_signal(signal)
            except Exception as e:
                await self._mark_status(
                    sig_id, "failed", {"reason": f"execute_error: {e}"},
                )
                logger.error(
                    f"Execution failed for {sig_id} {symbol}: {e}"
                )
                return

            status = result.get("status", "failed")

            if status == "filled":
                await self._mark_status(sig_id, "filled", result)
                logger.info(
                    f"Signal {sig_id} {symbol} FILLED via queue"
                )
                if self._on_filled:
                    try:
                        await self._on_filled(signal, result, sig_id)
                    except Exception as e:
                        logger.error(
                            f"on_filled callback error: {e}"
                        )

            elif status == "pending_limit":
                # Executor spawned a background tracker. Keep the
                # symbol locked so _should_skip_symbol rejects new
                # signals until the limit resolves (fill/stall/rally).
                await self._mark_status(sig_id, "executing", result)
                handed_off = True
                logger.info(
                    f"Signal {sig_id} {symbol} → LIMIT PLACED, "
                    f"awaiting fill/stall"
                )

            else:
                mapped = (
                    "skipped"
                    if status in ("rejected", "skipped")
                    else "failed"
                )
                await self._mark_status(sig_id, mapped, result)
                logger.info(
                    f"Signal {sig_id} {symbol} {status.upper()}: "
                    f"{result.get('reason', '')}"
                )

        finally:
            if not handed_off:
                self._pending_symbols.discard(symbol)

    async def _mark_status(self, sig_id: int, status: str, result: Dict):
        try:
            await self.db.execute(
                """
                UPDATE signals
                SET execution_status = $1,
                    execution_result = $2,
                    execution_attempts = execution_attempts + 1,
                    executed_at = NOW()
                WHERE id = $3
                """,
                status,
                json.dumps(result, default=str),
                sig_id,
            )
        except Exception as e:
            logger.error(f"mark_status failed for {sig_id}: {e}")