"""
core/bybit_client.py
Bybit V5 Unified Trading API client.

Handles REST (market data, account, instruments) and WebSocket
(order execution via Trade WS, private streams via Private WS).

Includes per-symbol instrument filter caching (qtyStep, tickSize,
minOrderQty, minNotionalValue) so orders are rounded correctly.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp
import websockets

logger = logging.getLogger(__name__)


def _ws_is_open(ws) -> bool:
    """Return True if a websockets connection is open (v11 or v12 API)."""
    if ws is None:
        return False
    if hasattr(ws, "closed"):
        try:
            return not ws.closed
        except Exception:
            return False
    state = getattr(ws, "state", None)
    if state is not None:
        name = getattr(state, "name", str(state))
        return name == "OPEN"
    return False


@dataclass
class BybitConfig:
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    category: str = "linear"
    recv_window: int = 5000
    max_retries: int = 3
    retry_backoff: float = 1.5


@dataclass
class InstrumentFilter:
    """Per-symbol constraints from /v5/market/instruments-info."""
    symbol: str
    qty_step: float = 0.001
    min_order_qty: float = 0.001
    max_order_qty: float = 1e9
    min_notional: float = 5.0         # USDT
    tick_size: float = 0.01
    price_scale: int = 2
    qty_scale: int = 3


class BybitV5Client:
    """
    Unified Bybit V5 client with full-symbol scanning and instrument
    filter caching.
    """

    REST_BASE = "https://api.bybit.com"
    REST_TESTNET = "https://api-testnet.bybit.com"
    WS_PUBLIC = "wss://stream.bybit.com/v5/public/{category}"
    WS_PRIVATE = "wss://stream.bybit.com/v5/private"
    WS_TRADE = "wss://stream.bybit.com/v5/trade"

    def __init__(self, cfg: BybitConfig):
        self.cfg = cfg
        self.base_url = self.REST_TESTNET if cfg.testnet else self.REST_BASE
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_public = None
        self._ws_private = None
        self._ws_trade = None
        self._subscriptions: Dict[str, Callable] = {}
        self._running = False
        self._rate_limiter = asyncio.Semaphore(3)
        self._ws_trade_lock = asyncio.Lock()
        self._instrument_filters: Dict[str, InstrumentFilter] = {}

    # ================================================================== #
    #  REST helpers
    # ================================================================== #
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session

    def _sign(self, params: str, timestamp: int) -> str:
        payload = (
            f"{timestamp}{self.cfg.api_key}"
            f"{self.cfg.recv_window}{params}"
        )
        return hmac.new(
            self.cfg.api_secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

    async def _rest_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False,
    ) -> Dict[str, Any]:
        async with self._rate_limiter:
            await asyncio.sleep(0.1)      # pacing to avoid 403 rate-limits

            params = params or {}
            url = f"{self.base_url}{endpoint}"
            headers = {"Content-Type": "application/json"}
            use_params: Optional[Dict] = None
            use_json: Optional[Dict] = None

            if signed:
                ts = int(time.time() * 1000)
                if method.upper() == "GET":
                    query = "&".join(
                        f"{k}={v}" for k, v in sorted(params.items())
                    )
                    sig = self._sign(query, ts)
                    if query:
                        url += f"?{query}"
                    headers.update({
                        "X-BAPI-API-KEY": self.cfg.api_key,
                        "X-BAPI-TIMESTAMP": str(ts),
                        "X-BAPI-SIGN": sig,
                        "X-BAPI-RECV-WINDOW": str(self.cfg.recv_window),
                    })
                    use_params = None
                    use_json = None
                else:
                    body = json.dumps(params)
                    sig = self._sign(body, ts)
                    headers.update({
                        "X-BAPI-API-KEY": self.cfg.api_key,
                        "X-BAPI-TIMESTAMP": str(ts),
                        "X-BAPI-SIGN": sig,
                        "X-BAPI-RECV-WINDOW": str(self.cfg.recv_window),
                    })
                    use_params = None
                    use_json = params
            else:
                if method.upper() == "GET":
                    use_params = params
                    use_json = None
                else:
                    use_params = None
                    use_json = params

            session = await self._get_session()
            last_error: Optional[Exception] = None

            for attempt in range(self.cfg.max_retries):
                try:
                    async with session.request(
                        method,
                        url,
                        headers=headers,
                        params=use_params,
                        json=use_json,
                    ) as resp:
                        data = await resp.json()

                        if data.get("retCode") == 10006:
                            wait = self.cfg.retry_backoff ** (attempt + 1)
                            logger.warning(f"Rate limited. Waiting {wait:.1f}s")
                            await asyncio.sleep(wait)
                            continue

                        code = data.get("retCode")
                        if code != 0:
                            # 10001 = invalid param / symbol
                            # 10002 = invalid api key
                            # 110043 = leverage not modified (idempotent)
                            if code in (10001, 10002, 110043):
                                logger.debug(
                                    f"Bybit retCode {code} for "
                                    f"endpoint={endpoint}: "
                                    f"{data.get('retMsg')}"
                                )
                            else:
                                logger.error(
                                    f"Bybit API error {code}: "
                                    f"{data.get('retMsg')} | "
                                    f"endpoint={endpoint}"
                                )
                        return data

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_error = e
                    wait = self.cfg.retry_backoff ** (attempt + 1)
                    logger.warning(
                        f"Request failed ({e}). "
                        f"Retry {attempt + 1}/{self.cfg.max_retries} "
                        f"in {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)

            raise ConnectionError(
                f"Bybit REST failed after {self.cfg.max_retries} retries: "
                f"{last_error}"
            )

    # ================================================================== #
    #  INSTRUMENT FILTER CACHE
    # ================================================================== #
    async def load_instrument_filters(self, symbols: Optional[List[str]] = None):
        """
        Fetch and cache qtyStep, tickSize, minOrderQty, minNotional
        for all (or specified) linear symbols.

        Call once at startup and periodically to refresh.
        """
        all_filters: Dict[str, InstrumentFilter] = {}
        cursor: Optional[str] = None
        page = 0

        while True:
            params: Dict[str, Any] = {"category": self.cfg.category, "limit": 500}
            if cursor:
                params["cursor"] = cursor

            data = await self._rest_request(
                "GET", "/v5/market/instruments-info", params=params,
            )
            result = data.get("result", {})
            batch = result.get("list", [])
            if not batch:
                break

            for inst in batch:
                sym = inst.get("symbol")
                if not sym:
                    continue
                if symbols and sym not in symbols:
                    continue

                lot = inst.get("lotSizeFilter", {}) or {}
                price = inst.get("priceFilter", {}) or {}

                try:
                    qty_step = float(lot.get("qtyStep", 0.001) or 0.001)
                    min_qty = float(lot.get("minOrderQty", 0.001) or 0.001)
                    max_qty = float(lot.get("maxOrderQty", 1e9) or 1e9)
                    min_notional = float(
                        lot.get("minNotionalValue", 5.0) or 5.0
                    )
                    tick_size = float(price.get("tickSize", 0.01) or 0.01)

                    qty_scale = max(0, -int(math.floor(math.log10(qty_step))))
                    price_scale = max(0, -int(math.floor(math.log10(tick_size))))

                    all_filters[sym] = InstrumentFilter(
                        symbol=sym,
                        qty_step=qty_step,
                        min_order_qty=min_qty,
                        max_order_qty=max_qty,
                        min_notional=min_notional,
                        tick_size=tick_size,
                        price_scale=price_scale,
                        qty_scale=qty_scale,
                    )
                except (TypeError, ValueError) as e:
                    logger.debug(f"Filter parse failed for {sym}: {e}")

            cursor = result.get("nextPageCursor")
            page += 1
            if not cursor:
                break
            await asyncio.sleep(0.15)

        self._instrument_filters = all_filters
        logger.info(
            f"Loaded instrument filters for {len(all_filters)} symbols"
        )

    def get_filter(self, symbol: str) -> Optional[InstrumentFilter]:
        """Return cached filter, or None if not loaded."""
        return self._instrument_filters.get(symbol)

    def round_qty(self, symbol: str, qty: float) -> Tuple[float, str]:
        """
        Round qty DOWN to the nearest qtyStep. Returns (rounded_qty, reason).
        Returns (0.0, reason) if the qty is invalid (below min / above max).
        """
        f = self._instrument_filters.get(symbol)
        if f is None:
            # Fallback: round to 3 decimals if filters not loaded
            rounded = math.floor(qty * 1000) / 1000
            return rounded, "no_filter_fallback"

        if qty < f.min_order_qty:
            return 0.0, f"below_min_qty({qty:.6f}<{f.min_order_qty})"
        if qty > f.max_order_qty:
            qty = f.max_order_qty

        # Round DOWN to nearest qtyStep
        steps = math.floor(qty / f.qty_step)
        rounded = steps * f.qty_step
        rounded = round(rounded, f.qty_scale)

        if rounded < f.min_order_qty:
            return 0.0, f"rounded_below_min({rounded}<{f.min_order_qty})"

        return rounded, "ok"

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to the nearest tickSize."""
        f = self._instrument_filters.get(symbol)
        if f is None:
            return round(price, 6)
        tick = f.tick_size
        rounded = round(price / tick) * tick
        return round(rounded, f.price_scale)

    def validate_notional(
        self, symbol: str, qty: float, price: float,
    ) -> Tuple[bool, str]:
        """Check that qty × price >= minNotionalValue."""
        f = self._instrument_filters.get(symbol)
        if f is None:
            return True, "no_filter"
        notional = qty * price
        if notional < f.min_notional:
            return False, (
                f"notional_too_low({notional:.2f}<{f.min_notional})"
            )
        return True, "ok"

    # ================================================================== #
    #  FULL-PAIR SCANNER
    # ================================================================== #
    async def get_all_linear_symbols(
        self,
        quote_coin: str = "USDT",
        only_perpetual: bool = True,
        min_turnover_24h: float = 5_000_000,
    ) -> List[str]:
        """Fetch all active linear USDT perpetuals with turnover filtering."""
        all_instruments: List[Dict] = []
        cursor: Optional[str] = None
        page = 0

        while True:
            params: Dict[str, Any] = {"category": "linear", "limit": 500}
            if cursor:
                params["cursor"] = cursor

            data = await self._rest_request(
                "GET", "/v5/market/instruments-info", params=params,
            )
            result = data.get("result", {})
            batch = result.get("list", [])
            if not batch:
                break

            all_instruments.extend(batch)
            cursor = result.get("nextPageCursor")
            page += 1
            logger.info(
                f"Instruments page {page}: {len(batch)} symbols "
                f"(total so far: {len(all_instruments)})"
            )
            if not cursor:
                break
            await asyncio.sleep(0.15)

        usdt_perps: List[str] = []
        for inst in all_instruments:
            if inst.get("quoteCoin") != quote_coin:
                continue
            if inst.get("status") != "Trading":
                continue
            if only_perpetual and inst.get("contractType") != "LinearPerpetual":
                continue
            usdt_perps.append(inst["symbol"])

        logger.info(f"After filters: {len(usdt_perps)} USDT perpetuals")

        if not usdt_perps:
            return []

        tickers = await self.get_all_tickers(category="linear")
        logger.info(f"Fetched {len(tickers)} tickers")

        turnover_map: Dict[str, float] = {}
        zero_symbols: List[str] = []

        for t in tickers:
            sym = t.get("symbol")
            if not sym:
                continue
            try:
                turnover = float(t.get("turnover24h") or 0)
                volume = float(t.get("volume24h") or 0)
                last_price = float(t.get("lastPrice") or 0)
            except (TypeError, ValueError):
                turnover_map[sym] = 0.0
                continue

            if turnover > 0 and volume > 0 and last_price > 1.0:
                if turnover < volume:
                    turnover = volume * last_price
            elif turnover == 0 and volume > 0 and last_price > 0:
                turnover = volume * last_price

            turnover_map[sym] = turnover
            if turnover == 0.0 and sym in usdt_perps:
                zero_symbols.append(sym)

        # Per-symbol fallback for zero-turnover symbols
        if zero_symbols:
            logger.info(f"Fallback for {len(zero_symbols)} zero-turnover symbols")
            for sym in zero_symbols:
                try:
                    individual = await self.get_ticker(sym)
                    if not individual:
                        continue
                    turnover = float(individual.get("turnover24h") or 0)
                    volume = float(individual.get("volume24h") or 0)
                    last_price = float(individual.get("lastPrice") or 0)
                    if turnover > 0 and volume > 0 and last_price > 1.0:
                        if turnover < volume:
                            turnover = volume * last_price
                    elif turnover == 0 and volume > 0 and last_price > 0:
                        turnover = volume * last_price
                    if turnover > 0:
                        turnover_map[sym] = turnover
                except Exception as e:
                    logger.debug(f"Fallback failed for {sym}: {e}")

        symbols = [
            s for s in usdt_perps
            if turnover_map.get(s, 0) >= min_turnover_24h
        ]
        symbols.sort(key=lambda s: turnover_map.get(s, 0), reverse=True)

        logger.info(
            f"Filtered to {len(symbols)} {quote_coin} perpetuals with "
            f">{min_turnover_24h:,.0f} 24h turnover"
        )
        return symbols

    async def get_all_tickers(self, category: str = "linear") -> List[Dict]:
        data = await self._rest_request(
            "GET", "/v5/market/tickers", params={"category": category},
        )
        return data.get("result", {}).get("list", [])

    async def get_ticker(self, symbol: str) -> Dict:
        data = await self._rest_request(
            "GET", "/v5/market/tickers",
            params={"category": self.cfg.category, "symbol": symbol},
        )
        if data.get("retCode") != 0:
            return {}
        lst = data.get("result", {}).get("list", [])
        return lst[0] if lst else {}

    # ================================================================== #
    #  Market data
    # ================================================================== #
    async def get_klines(
        self, symbol: str, interval: str = "60", limit: int = 200,
        category: Optional[str] = None,
    ) -> List[List]:
        data = await self._rest_request(
            "GET", "/v5/market/kline",
            params={
                "category": category or self.cfg.category,
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
            },
        )
        return data.get("result", {}).get("list", [])

    async def get_open_interest(
        self, symbol: str, interval_time: str = "1h", limit: int = 50,
    ) -> List[Dict]:
        data = await self._rest_request(
            "GET", "/v5/market/open-interest",
            params={
                "category": self.cfg.category,
                "symbol": symbol,
                "intervalTime": interval_time,
                "limit": limit,
            },
        )
        return data.get("result", {}).get("list", [])

    async def get_account_ratio(
        self, symbol: str, period: str = "1h", limit: int = 50,
    ) -> List[Dict]:
        data = await self._rest_request(
            "GET", "/v5/market/account-ratio",
            params={
                "category": self.cfg.category,
                "symbol": symbol,
                "period": period,
                "limit": limit,
            },
        )
        return data.get("result", {}).get("list", [])

    async def get_funding_rate(self, symbol: str) -> Dict:
        data = await self._rest_request(
            "GET", "/v5/market/tickers",
            params={"category": self.cfg.category, "symbol": symbol},
        )
        if data.get("retCode") != 0:
            return {}
        lst = data.get("result", {}).get("list", [])
        return lst[0] if lst else {}

    async def get_orderbook(self, symbol: str, limit: int = 50) -> Dict:
        data = await self._rest_request(
            "GET", "/v5/market/orderbook",
            params={
                "category": self.cfg.category,
                "symbol": symbol,
                "limit": limit,
            },
        )
        return data.get("result", {})
        
    async def get_order_status(
        self, symbol: str, order_id: str,
    ) -> Optional[Dict]:
        """
        Fetch current status of a single order.

        Returns the order dict or None if not found.
        Useful fields: orderStatus, cumExecQty, leavesQty, avgPrice.
        """
        data = await self._rest_request(
            "GET", "/v5/order/realtime",
            params={
                "category": self.cfg.category,
                "symbol": symbol,
                "orderId": order_id,
            },
            signed=True,
        )
        if data.get("retCode") != 0:
            logger.debug(
                f"get_order_status failed for {symbol}/{order_id}: "
                f"retCode={data.get('retCode')} retMsg={data.get('retMsg')}"
            )
            return None
        lst = data.get("result", {}).get("list", [])
        return lst[0] if lst else None
    # ================================================================== #
    #  Account / Position
    # ================================================================== #
    async def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict:
        data = await self._rest_request(
            "GET", "/v5/account/wallet-balance",
            params={"accountType": account_type}, signed=True,
        )
        return data.get("result", {})

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        params: Dict[str, Any] = {
            "category": self.cfg.category,
            "settleCoin": "USDT",
        }
        if symbol:
            params["symbol"] = symbol
        data = await self._rest_request(
            "GET", "/v5/position/list", params=params, signed=True,
        )
        return data.get("result", {}).get("list", [])

    async def set_leverage(self, symbol: str, leverage: int) -> Dict:
        return await self._rest_request(
            "POST", "/v5/position/set-leverage",
            params={
                "category": self.cfg.category,
                "symbol": symbol,
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage),
            },
            signed=True,
        )

    # ================================================================== #
    #  Order execution
    # ================================================================== #
    async def place_order_ws(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "Market",
        price: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        tp_trigger_by: str = "LastPrice",
        sl_trigger_by: str = "LastPrice",
        tpsl_mode: str = "Full",
        reduce_only: bool = False,
    ) -> Dict:
        """Place order via WS Trade API with automatic qty/price rounding."""
        if not _ws_is_open(self._ws_trade):
            await self._connect_ws_trade()

        # ---- Round qty to symbol's qtyStep ----
        rounded_qty, reason = self.round_qty(symbol, qty)
        if rounded_qty <= 0:
            logger.error(f"Qty rounding failed for {symbol}: {reason} (raw={qty})")
            return {"retCode": -1, "retMsg": f"qty_invalid: {reason}"}

        # ---- Round prices to tickSize ----
        tp_rounded = self.round_price(symbol, take_profit) if take_profit else None
        sl_rounded = self.round_price(symbol, stop_loss) if stop_loss else None
        price_rounded = self.round_price(symbol, price) if price else None

        req_id = f"order_{int(time.time() * 1000)}"
        now_ms = str(int(time.time() * 1000))

        payload = {
            "reqId": req_id,
            "header": {
                "X-BAPI-TIMESTAMP": now_ms,
                "X-BAPI-RECV-WINDOW": str(self.cfg.recv_window),
            },
            "op": "order.create",
            "args": [{
                "category": self.cfg.category,
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(rounded_qty),
                "timeInForce": "IOC" if order_type == "Market" else "GTC",
                "reduceOnly": reduce_only,
            }],
        }

        if tp_rounded is not None:
            payload["args"][0]["takeProfit"] = str(tp_rounded)
            payload["args"][0]["tpTriggerBy"] = tp_trigger_by
            payload["args"][0]["tpOrderType"] = "Market"
        if sl_rounded is not None:
            payload["args"][0]["stopLoss"] = str(sl_rounded)
            payload["args"][0]["slTriggerBy"] = sl_trigger_by
            payload["args"][0]["slOrderType"] = "Market"
        if tp_rounded is not None or sl_rounded is not None:
            payload["args"][0]["tpslMode"] = tpsl_mode
        if price_rounded is not None and order_type == "Limit":
            payload["args"][0]["price"] = str(price_rounded)

        logger.info(f"WS place_order: {json.dumps(payload)}")

        async with self._ws_trade_lock:
            await self._ws_trade.send(json.dumps(payload))
            response = await asyncio.wait_for(
                self._ws_trade.recv(), timeout=10,
            )
        return json.loads(response)

    async def cancel_order_ws(self, symbol: str, order_id: str) -> Dict:
        if not _ws_is_open(self._ws_trade):
            await self._connect_ws_trade()

        req_id = f"cancel_{int(time.time() * 1000)}"
        payload = {
            "reqId": req_id,
            "header": {
                "X-BAPI-TIMESTAMP": str(int(time.time() * 1000)),
                "X-BAPI-RECV-WINDOW": str(self.cfg.recv_window),
            },
            "op": "order.cancel",
            "args": [{
                "category": self.cfg.category,
                "symbol": symbol,
                "orderId": order_id,
            }],
        }
        async with self._ws_trade_lock:
            await self._ws_trade.send(json.dumps(payload))
            response = await asyncio.wait_for(self._ws_trade.recv(), timeout=10)
        return json.loads(response)

    async def set_trading_stop(
        self,
        symbol: str,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        tp_trigger_by: str = "LastPrice",
        sl_trigger_by: str = "LastPrice",
    ) -> Dict:
        params: Dict[str, Any] = {
            "category": self.cfg.category,
            "symbol": symbol,
            "tpslMode": "Full",
            "tpTriggerBy": tp_trigger_by,
            "slTriggerBy": sl_trigger_by,
        }
        if take_profit is not None:
            params["takeProfit"] = str(self.round_price(symbol, take_profit))
        if stop_loss is not None:
            params["stopLoss"] = str(self.round_price(symbol, stop_loss))
        if trailing_stop is not None:
            params["trailingStop"] = str(
                self.round_price(symbol, trailing_stop)
            )

        return await self._rest_request(
            "POST", "/v5/position/trading-stop",
            params=params, signed=True,
        )

    async def close_position_market(
        self, symbol: str, side: str, qty: float,
    ) -> Dict:
        close_side = "Sell" if side == "Buy" else "Buy"
        return await self.place_order_ws(
            symbol=symbol,
            side=close_side,
            qty=qty,
            order_type="Market",
            reduce_only=True,
        )

    # ================================================================== #
    #  WebSocket auth helpers
    # ================================================================== #
    async def _await_op_response(
        self, ws, expected_op: str, timeout: float = 10.0,
    ) -> dict:
        timeout = float(timeout)
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = max(0.5, deadline - time.time())
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            msg = json.loads(raw)
            if msg.get("op") == expected_op:
                return msg
        raise TimeoutError(
            f"No '{expected_op}' response within {timeout:.1f}s"
        )

    def _is_auth_ok(self, resp: dict) -> tuple:
        if "retCode" in resp:
            ok = resp.get("retCode") == 0
            return ok, "" if ok else resp.get("retMsg", "")
        if "success" in resp:
            ok = bool(resp.get("success"))
            return ok, "" if ok else resp.get("ret_msg", "")
        return False, f"unrecognised auth response: {resp}"

    async def _connect_ws_trade(self):
        self._ws_trade = await websockets.connect(
            self.WS_TRADE, ping_interval=20, ping_timeout=10,
        )
        expires = int((time.time() + 60) * 1000)
        signature = hmac.new(
            self.cfg.api_secret.encode(),
            f"GET/realtime{expires}".encode(),
            hashlib.sha256,
        ).hexdigest()
        await self._ws_trade.send(json.dumps({
            "op": "auth",
            "args": [self.cfg.api_key, expires, signature],
        }))
        auth_resp = await self._await_op_response(self._ws_trade, "auth")
        ok, err = self._is_auth_ok(auth_resp)
        if not ok:
            raise ConnectionError(
                f"WS Trade auth failed: {err} | raw={auth_resp}"
            )
        logger.info("WS Trade authenticated successfully")

    async def _connect_ws_private(self):
        self._ws_private = await websockets.connect(
            self.WS_PRIVATE, ping_interval=20, ping_timeout=10,
        )
        expires = int((time.time() + 60) * 1000)
        signature = hmac.new(
            self.cfg.api_secret.encode(),
            f"GET/realtime{expires}".encode(),
            hashlib.sha256,
        ).hexdigest()
        await self._ws_private.send(json.dumps({
            "op": "auth",
            "args": [self.cfg.api_key, expires, signature],
        }))
        auth_resp = await self._await_op_response(self._ws_private, "auth")
        ok, err = self._is_auth_ok(auth_resp)
        if not ok:
            raise ConnectionError(
                f"WS Private auth failed: {err} | raw={auth_resp}"
            )
        logger.info("WS Private authenticated successfully")

    async def subscribe_private(
        self, topics: List[str], callback: Callable,
    ):
        if not _ws_is_open(self._ws_private):
            await self._connect_ws_private()
        await self._ws_private.send(
            json.dumps({"op": "subscribe", "args": topics})
        )
        for topic in topics:
            self._subscriptions[topic] = callback
        asyncio.create_task(self._private_listener())

    async def _private_listener(self):
        while self._running and _ws_is_open(self._ws_private):
            try:
                msg = json.loads(await self._ws_private.recv())
                topic = msg.get("topic", "")
                if topic in self._subscriptions:
                    await self._subscriptions[topic](msg)
            except websockets.ConnectionClosed:
                logger.warning("Private WS closed. Reconnecting...")
                await asyncio.sleep(3)
                try:
                    await self._connect_ws_private()
                except Exception as e:
                    logger.error(f"Private WS reconnect failed: {e}")
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Private WS listener error: {e}")
                await asyncio.sleep(1)

    async def start(self):
        self._running = True
        await self._connect_ws_trade()
        await self.load_instrument_filters()   # ← load filters at startup

    async def stop(self):
        self._running = False
        for ws in (self._ws_public, self._ws_private, self._ws_trade):
            if _ws_is_open(ws):
                try:
                    await ws.close()
                except Exception:
                    pass
        if self._session is not None and not self._session.closed:
            await self._session.close()