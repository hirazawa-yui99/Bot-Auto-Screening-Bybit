"""
core/realtime_monitor.py
Event-driven candle-close monitor for Bybit V5.

Design:
  - Backfills 200 bars per symbol via REST at startup
  - Opens N WebSocket connections (10 kline topics each)
  - Heartbeat every 20s via {"op": "ping"}
  - On reconnect: gap-fills from last stored candle, resubscribes
  - Fires on_candle_close(symbol) ONLY on confirm=true

Never triggers on confirm=false — rVOL and RSI would be distorted
by incomplete volume.
"""
import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional

import pandas as pd
import websockets

logger = logging.getLogger(__name__)

WS_PUBLIC = "wss://stream.bybit.com/v5/public/linear"
HEARTBEAT_SEC = 20
MAX_TOPICS_PER_CONN = 10
STORE_MAXLEN = 300


@dataclass
class Candle:
    start: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float
    confirm: bool


class CandleStore:
    """Bounded in-memory OHLCV store."""
    def __init__(self, maxlen: int = STORE_MAXLEN):
        self.maxlen = maxlen
        self._data: Dict[str, Deque[Candle]] = {}

    def append(self, symbol: str, candle: Candle):
        if symbol not in self._data:
            self._data[symbol] = deque(maxlen=self.maxlen)
        self._data[symbol].append(candle)

    def last_start(self, symbol: str) -> int:
        d = self._data.get(symbol)
        return d[-1].start if d else 0

    def get_df(self, symbol: str) -> pd.DataFrame:
        d = self._data.get(symbol)
        if not d:
            return pd.DataFrame()
        rows = [{
            "ts": c.start, "open": c.open, "high": c.high,
            "low": c.low, "close": c.close,
            "volume": c.volume, "turnover": c.turnover,
        } for c in d]
        return pd.DataFrame(rows)

    def size(self, symbol: str) -> int:
        return len(self._data.get(symbol, []))


class RealtimeMonitor:
    def __init__(
        self,
        rest_client,
        interval: str = "60",
        on_candle_close: Optional[Callable] = None,
        max_connections: int = 2,
        symbols: Optional[List[str]] = None,
        selection: str = "top_by_turnover",
        max_symbols: int = 20,
        analysis_concurrency: int = 3,
        callback_timeout: int = 180,
        gap_fill_recent_min: int = 10,
        min_turnover: float = 1_000_000,
    ):
        self.rest = rest_client
        self.interval = interval
        self.on_candle_close = on_candle_close
        self.max_connections = max_connections
        self.symbols = symbols or []
        self.selection = selection
        self.max_symbols = max_symbols
        self.gap_fill_recent_min = gap_fill_recent_min
        self.min_turnover = min_turnover
        self.callback_timeout = callback_timeout

        self.store = CandleStore()
        self._running = False
        self._ws_tasks: List[asyncio.Task] = []
        self._analysis_sem = asyncio.Semaphore(analysis_concurrency)
        self._chunks: List[List[str]] = []
        self._trigger_count = 0
        self._error_count = 0

    # ================================================================== #
    #  Public API
    # ================================================================== #
    async def start(self):
        self._running = True

        if not self.symbols:
            self.symbols = await self._select_symbols()
        if not self.symbols:
            logger.warning("RealtimeMonitor: no symbols — not starting")
            self._running = False
            return

        cap = self.max_connections * MAX_TOPICS_PER_CONN
        self.symbols = self.symbols[:self.max_symbols][:cap]
        logger.info(
            f"RealtimeMonitor: {len(self.symbols)} symbols across "
            f"{self.max_connections} connection(s)"
        )

        await self._backfill_all()

        chunks = [
            self.symbols[i:i + MAX_TOPICS_PER_CONN]
            for i in range(0, len(self.symbols), MAX_TOPICS_PER_CONN)
        ][:self.max_connections]
        self._chunks = chunks

        for idx, chunk in enumerate(chunks):
            task = asyncio.create_task(self._connection_loop(idx, chunk))
            self._ws_tasks.append(task)

        logger.info(f"RealtimeMonitor started ({len(chunks)} tasks)")

    async def stop(self):
        self._running = False
        for t in self._ws_tasks:
            t.cancel()
        await asyncio.gather(*self._ws_tasks, return_exceptions=True)
        self._ws_tasks.clear()
        logger.info("RealtimeMonitor stopped")

    def stats(self) -> Dict:
        return {
            "running": self._running,
            "symbols": len(self.symbols),
            "connections": len(self._chunks),
            "triggers": self._trigger_count,
            "errors": self._error_count,
            "store_sizes": {
                s: self.store.size(s) for s in self.symbols[:5]
            },
        }

    # ================================================================== #
    #  Symbol selection
    # ================================================================== #
    async def _select_symbols(self) -> List[str]:
        if self.selection == "explicit":
            return self.symbols
        try:
            syms = await self.rest.get_all_linear_symbols(
                quote_coin="USDT",
                only_perpetual=True,
                min_turnover_24h=self.min_turnover,
            )
            logger.info(
                f"RealtimeMonitor: selected {len(syms)} symbols by turnover"
            )
            return syms
        except Exception as e:
            logger.error(f"Symbol selection failed: {e}")
            return []

    # ================================================================== #
    #  Backfill
    # ================================================================== #
    async def _backfill_all(self):
        logger.info(f"Backfilling {len(self.symbols)} symbols...")
        sem = asyncio.Semaphore(5)
        done = 0

        async def _one(sym):
            nonlocal done
            async with sem:
                try:
                    klines = await self._fetch_klines_rest(sym, limit=200)
                    for k in reversed(klines):
                        self.store.append(sym, Candle(
                            start=int(k[0]),
                            open=float(k[1]),
                            high=float(k[2]),
                            low=float(k[3]),
                            close=float(k[4]),
                            volume=float(k[5]),
                            turnover=float(k[6]) if len(k) > 6 else 0.0,
                            confirm=True,
                        ))
                    done += 1
                except Exception as e:
                    logger.warning(f"Backfill failed {sym}: {e}")

        await asyncio.gather(*[_one(s) for s in self.symbols])
        logger.info(f"Backfilled {done}/{len(self.symbols)} symbols")

    async def _fetch_klines_rest(
        self, symbol: str, limit: int = 200, start_ms: Optional[int] = None,
    ) -> List[List]:
        """Direct REST fetch — no start filter for backfill, start for gap-fill."""
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": self.interval,
            "limit": limit,
        }
        if start_ms:
            params["start"] = start_ms
        try:
            data = await self.rest._rest_request(
                "GET", "/v5/market/kline", params=params,
            )
            return data.get("result", {}).get("list", [])
        except Exception as e:
            logger.debug(f"REST kline failed {symbol}: {e}")
            return []

    # ================================================================== #
    #  Per-connection loop (reconnect + heartbeat + message handling)
    # ================================================================== #
    async def _connection_loop(self, conn_id: int, chunk: List[str]):
        backoff = 2
        topics = [f"kline.{self.interval}.{s}" for s in chunk]

        while self._running:
            try:
                async with websockets.connect(
                    WS_PUBLIC,
                    ping_interval=HEARTBEAT_SEC,
                    ping_timeout=10,
                    max_size=2**20,
                ) as ws:
                    backoff = 2
                    logger.info(
                        f"[conn{conn_id}] connected, subscribing "
                        f"{len(topics)} topics"
                    )

                    await ws.send(json.dumps({
                        "op": "subscribe", "args": topics,
                    }))

                    await self._gap_fill(chunk, conn_id)

                    hb_task = asyncio.create_task(
                        self._heartbeat(ws, conn_id),
                    )
                    try:
                        async for raw in ws:
                            if not self._running:
                                break
                            try:
                                msg = json.loads(raw)
                            except Exception:
                                continue
                            await self._handle_message(
                                conn_id, chunk, msg,
                            )
                    finally:
                        hb_task.cancel()
                        try:
                            await hb_task
                        except asyncio.CancelledError:
                            pass

            except asyncio.CancelledError:
                return
            except Exception as e:
                self._error_count += 1
                logger.warning(
                    f"[conn{conn_id}] disconnected: "
                    f"{type(e).__name__}: {e} — retry in {backoff}s"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _heartbeat(self, ws, conn_id: int):
        while self._running:
            try:
                await asyncio.sleep(HEARTBEAT_SEC)
                await ws.send(json.dumps({"op": "ping"}))
            except Exception:
                return

    async def _handle_message(
        self, conn_id: int, chunk: List[str], msg: dict,
    ):
        if msg.get("op") in ("ping", "pong", "subscribe"):
            if msg.get("op") == "subscribe" and not msg.get("success", True):
                logger.error(
                    f"[conn{conn_id}] subscribe failed: {msg}"
                )
            return

        topic = msg.get("topic", "")
        if not topic.startswith("kline."):
            return

        data = msg.get("data", [])
        if not data:
            return

        k = data[0]
        confirm = bool(k.get("confirm", False))
        if not confirm:
            return

        try:
            parts = topic.split(".")
            if len(parts) < 3:
                return
            symbol = parts[2]
        except Exception:
            return

        candle = Candle(
            start=int(k["start"]),
            open=float(k["open"]),
            high=float(k["high"]),
            low=float(k["low"]),
            close=float(k["close"]),
            volume=float(k["volume"]),
            turnover=float(k.get("turnover", 0)),
            confirm=True,
        )

        if candle.start <= self.store.last_start(symbol):
            return  # already stored

        self.store.append(symbol, candle)
        logger.debug(
            f"[conn{conn_id}] {symbol} closed @ {candle.close}"
        )
        asyncio.create_task(self._trigger(symbol, reason="live_close"))

    # ================================================================== #
    #  Gap-fill on (re)connect
    # ================================================================== #
    async def _gap_fill(self, chunk: List[str], conn_id: int):
        """After reconnect, fetch any candles missed while offline."""
        now_ms = int(time.time() * 1000)
        gap_window_ms = self.gap_fill_recent_min * 60 * 1000
        recent_cutoff = now_ms - gap_window_ms

        for sym in chunk:
            last = self.store.last_start(sym)
            if last == 0:
                continue
            try:
                klines = await self._fetch_klines_rest(
                    sym, limit=20, start_ms=last + 1,
                )
            except Exception:
                continue

            added = 0
            latest_confirm = None
            for k in reversed(klines):
                s = int(k[0])
                if s <= last:
                    continue
                c = Candle(
                    start=s, open=float(k[1]), high=float(k[2]),
                    low=float(k[3]), close=float(k[4]),
                    volume=float(k[5]),
                    turnover=float(k[6]) if len(k) > 6 else 0.0,
                    confirm=True,
                )
                self.store.append(sym, c)
                latest_confirm = c
                added += 1

            if added:
                logger.info(
                    f"[conn{conn_id}] gap-filled {sym}: {added} candles"
                )
                # Trigger only if the missed close is recent (not a long outage)
                if latest_confirm and latest_confirm.start >= recent_cutoff:
                    asyncio.create_task(
                        self._trigger(sym, reason="gap_fill")
                    )

    # ================================================================== #
    #  Analysis trigger with concurrency + timeout
    # ================================================================== #
    async def _trigger(self, symbol: str, reason: str = "live_close"):
        if not self.on_candle_close:
            return
        async with self._analysis_sem:
            self._trigger_count += 1
            t0 = time.time()
            try:
                await asyncio.wait_for(
                    self.on_candle_close(symbol),
                    timeout=self.callback_timeout,
                )
                logger.debug(
                    f"trigger {symbol} ({reason}) done in "
                    f"{time.time()-t0:.1f}s"
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"trigger {symbol} TIMEOUT after {self.callback_timeout}s"
                )
            except Exception as e:
                logger.error(
                    f"trigger {symbol} failed: {type(e).__name__}: {e}"
                )