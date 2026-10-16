"""
core/bybit_client.py
Bybit V5 Unified Trading API client.

Handles REST (market data, account, instrument info) and
WebSocket (order execution, private streams).

Reference: https://bybit-exchange.github.io/docs/v5/intro
"""
import asyncio
import hmac
import hashlib
import json
import time
import logging
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass

import aiohttp
import websockets

logger = logging.getLogger(__name__)


@dataclass
class BybitConfig:
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    category: str = "linear"
    recv_window: int = 5000
    max_retries: int = 3
    retry_backoff: float = 1.5


class BybitV5Client:
    """
    Unified Bybit V5 client with full-symbol scanning support.

    REST base: https://api.bybit.com  (testnet: https://api-testnet.bybit.com)
    WS public:  wss://stream.bybit.com/v5/public/{category}
    WS private: wss://stream.bybit.com/v5/private
    WS trade:   wss://stream.bybit.com/v5/trade

    Rate limits (per official docs):
      - HTTP IP limit: 600 req / 5s window across all api.bybit.com hosts
      - Tickers: ~10 req/s per IP
      - Kline: ~10 req/s per IP
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
        self._rate_limiter = asyncio.Semaphore(8)  # conservative: 8 concurrent

    # ------------------------------------------------------------------ #
    #  REST helpers
    # ------------------------------------------------------------------ #
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session

    def _sign(self, params: str, timestamp: int) -> str:
        payload = f"{timestamp}{self.cfg.api_key}{self.cfg.recv_window}{params}"
        return hmac.new(
            self.cfg.api_secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()

    async def _rest_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False,
    ) -> Dict[str, Any]:
        """Execute REST request with retry, rate-limit back-off, and semaphore."""
        async with self._rate_limiter:
            url = f"{self.base_url}{endpoint}"
            headers = {"Content-Type": "application/json"}

            if signed:
                ts = int(time.time() * 1000)
                if method.upper() == "GET":
                    query = "&".join(
                        f"{k}={v}" for k, v in sorted((params or {}).items())
                    )
                    sig = self._sign(query, ts)
                    url += f"?{query}" if query else ""
                    headers.update({
                        "X-BAPI-API-KEY": self.cfg.api_key,
                        "X-BAPI-TIMESTAMP": str(ts),
                        "X-BAPI-SIGN": sig,
                        "X-BAPI-RECV-WINDOW": str(self.cfg.recv_window),
                    })
                else:
                    body = json.dumps(params or {})
                    sig = self._sign(body, ts)
                    headers.update({
                        "X-BAPI-API-KEY": self.cfg.api_key,
                        "X-BAPI-TIMESTAMP": str(ts),
                        "X-BAPI-SIGN": sig,
                        "X-BAPI-RECV-WINDOW": str(self.cfg.recv_window),
                    })

            session = await self._get_session()
            last_error = None

            for attempt in range(self.cfg.max_retries):
                try:
                    async with session.request(
                        method, url,
                        headers=headers,
                        json=params if method.upper() != "GET" else None,
                    ) as resp:
                        data = await resp.json()

                        if data.get("retCode") == 10006:
                            wait = self.cfg.retry_backoff ** (attempt + 1)
                            logger.warning(f"Rate limited. Waiting {wait:.1f}s")
                            await asyncio.sleep(wait)
                            continue

                        if data.get("retCode") != 0:
                            logger.error(
                                f"Bybit API error {data.get('retCode')}: "
                                f"{data.get('retMsg')} | endpoint={endpoint}"
                            )
                        return data

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_error = e
                    wait = self.cfg.retry_backoff ** (attempt + 1)
                    logger.warning(
                        f"Request failed ({e}). Retry {attempt+1}/"
                        f"{self.cfg.max_retries} in {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)

            raise ConnectionError(
                f"Bybit REST failed after {self.cfg.max_retries} retries: "
                f"{last_error}"
            )

    # ================================================================== #
    #  FULL-PAIR SCANNER
    # ================================================================== #
    async def get_all_linear_symbols(
        self,
        quote_coin: str = "USDT",
        only_perpetual: bool = True,
        min_turnover_24h: float = 5_000_000,
    ) -> List[str]:
        """
        Retrieve ALL active linear USDT perpetual symbols on Bybit.

        This method paginates through /v5/market/instruments-info
        (default limit 500 per page, cursor-based) to collect the
        complete symbol list, then filters by quote coin, contract
        type, and 24h turnover.

        Args:
            quote_coin:      "USDT" (default) or "USDC"
            only_perpetual:  True → only LinearPerpetual contracts
                             False → include dated futures too
            min_turnover_24h: minimum 24h turnover in quote currency

        Returns:
            Sorted list of symbol strings (e.g. ["BTCUSDT", "ETHUSDT", ...])
        """
        all_instruments: List[Dict] = []
        cursor: Optional[str] = None
        page = 0

        while True:
            params: Dict[str, Any] = {
                "category": "linear",
                "limit": 500,
            }
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

            await asyncio.sleep(0.15)  # gentle pacing

        logger.info(f"Fetched {len(all_instruments)} total linear instruments")

        # Filter
        symbols = []
        for inst in all_instruments:
            if inst.get("quoteCoin") != quote_coin:
                continue
            if inst.get("status") != "Trading":
                continue
            if only_perpetual and inst.get("contractType") != "LinearPerpetual":
                continue
            symbols.append(inst["symbol"])

        # Enrich with turnover and filter by liquidity
        if min_turnover_24h > 0 and symbols:
            tickers = await self.get_all_tickers()
            turnover_map = {
                t["symbol"]: float(t.get("turnover24h", 0))
                for t in tickers
            }
            symbols = [
                s for s in symbols
                if turnover_map.get(s, 0) >= min_turnover_24h
            ]
            symbols.sort(key=lambda s: turnover_map.get(s, 0), reverse=True)
        else:
            symbols.sort()

        logger.info(
            f"Filtered to {len(symbols)} {quote_coin} perpetuals "
            f"with >{min_turnover_24h:,.0f} 24h turnover"
        )
        return symbols

    async def _await_op_response(ws, expected_op: str, timeout: float = 10.0) -> dict:
        """
        Read WebSocket messages until we see the response with the expected `op`.
        Skips pongs, subscription confirmations, and unrelated messages.

        Bybit V5 endpoints use two different auth response schemas:
          - /v5/trade   → {"retCode": 0, "retMsg": "OK", "op": "auth", ...}
          - /v5/private → {"success": true, "ret_msg": "", "op": "auth", ...}
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = max(0.5, deadline - time.time())
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            msg = json.loads(raw)
            if msg.get("op") == expected_op:
                return msg
        raise TimeoutError(f"No '{expected_op}' response within {timeout}s")


    def _is_auth_ok(resp: dict) -> tuple[bool, str]:
        """
        Check auth response from either WS endpoint.
        Returns (ok, error_message).
        """
        # Trade WS style
        if "retCode" in resp:
            ok = resp.get("retCode") == 0
            return ok, resp.get("retMsg", "") if not ok else ""

        # Private WS style
        if "success" in resp:
            ok = bool(resp.get("success"))
            return ok, resp.get("ret_msg", "") if not ok else ""

        # Neither — unknown schema
        return False, f"unrecognised auth response: {resp}"

    async def get_all_tickers(
        self, category: str = "linear"
    ) -> List[Dict]:
        """
        Fetch ALL tickers in one request (no symbol parameter).
        The V5 tickers endpoint returns the full list when symbol is omitted.
        """
        data = await self._rest_request(
            "GET", "/v5/market/tickers",
            params={"category": category},
        )
        return data.get("result", {}).get("list", [])

    # ================================================================== #
    #  Market Data (V5 public)
    # ================================================================== #
    async def get_klines(
        self, symbol: str, interval: str = "60", limit: int = 200,
        category: Optional[str] = None,
    ) -> List[List]:
        """
        GET /v5/market/kline
        interval: 1,3,5,15,30,60,120,240,360,720,D,M,W
        Returns [[startTime, open, high, low, close, volume, turnover], ...]
        """
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
        """GET /v5/market/open-interest — intervalTime: 5min,15min,30min,1h,4h,1d"""
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
        """
        GET /v5/market/account-ratio — Account Long/Short Ratio.
        period: 5min,15min,30min,1h,4h,1d
        """
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
        """GET /v5/market/tickers — extract funding fields."""
        data = await self._rest_request(
            "GET", "/v5/market/tickers",
            params={"category": self.cfg.category, "symbol": symbol},
        )
        tickers = data.get("result", {}).get("list", [])
        return tickers[0] if tickers else {}

    async def get_orderbook(self, symbol: str, limit: int = 50) -> Dict:
        """GET /v5/market/orderbook"""
        data = await self._rest_request(
            "GET", "/v5/market/orderbook",
            params={
                "category": self.cfg.category,
                "symbol": symbol,
                "limit": limit,
            },
        )
        return data.get("result", {})

    # ================================================================== #
    #  Account / Position (V5 private)
    # ================================================================== #
    async def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict:
        data = await self._rest_request(
            "GET", "/v5/account/wallet-balance",
            params={"accountType": account_type}, signed=True,
        )
        return data.get("result", {})

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        params = {"category": self.cfg.category, "settleCoin": "USDT"}
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
    #  Order Execution (WS Trade API)
    # ================================================================== #
    async def place_order_ws(
        self,
        symbol: str,
        side: str,                # "Buy" | "Sell"
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
        """
        Place order via V5 WebSocket Trade API.
        Native TP/SL is attached so the exchange protects the position
        even if the bot disconnects.
        """
        if self._ws_trade is None or self._ws_trade.closed:
            await self._connect_ws_trade()

        req_id = f"order_{int(time.time()*1000)}"

        payload = {
            "op": "order.create",
            "args": [{
                "category": self.cfg.category,
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(qty),
                "timeInForce": "IOC" if order_type == "Market" else "GTC",
                "reduceOnly": reduce_only,
            }],
            "reqId": req_id,
        }

        if take_profit is not None:
            payload["args"][0]["takeProfit"] = str(take_profit)
            payload["args"][0]["tpTriggerBy"] = tp_trigger_by
            payload["args"][0]["tpOrderType"] = "Market"
        if stop_loss is not None:
            payload["args"][0]["stopLoss"] = str(stop_loss)
            payload["args"][0]["slTriggerBy"] = sl_trigger_by
            payload["args"][0]["slOrderType"] = "Market"
        if take_profit or stop_loss:
            payload["args"][0]["tpslMode"] = tpsl_mode
        if price is not None and order_type == "Limit":
            payload["args"][0]["price"] = str(price)

        logger.info(f"WS place_order: {json.dumps(payload)}")
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
        """Update TP/SL for an existing position via REST."""
        params = {
            "category": self.cfg.category,
            "symbol": symbol,
            "tpslMode": "Full",
            "tpTriggerBy": tp_trigger_by,
            "slTriggerBy": sl_trigger_by,
        }
        if take_profit is not None:
            params["takeProfit"] = str(take_profit)
        if stop_loss is not None:
            params["stopLoss"] = str(stop_loss)
        if trailing_stop is not None:
            params["trailingStop"] = str(trailing_stop)

        return await self._rest_request(
            "POST", "/v5/position/trading-stop",
            params=params, signed=True,
        )

    async def close_position_market(
        self, symbol: str, side: str, qty: float,
    ) -> Dict:
        """Emergency close: market order in opposite direction."""
        close_side = "Sell" if side == "Buy" else "Buy"
        return await self.place_order_ws(
            symbol=symbol, side=close_side, qty=qty,
            order_type="Market", reduce_only=True,
        )

    # ================================================================== #
    #  WebSocket connections
    # ================================================================== #
    async def _connect_ws_trade(self):
        """Authenticate and connect to V5 Trade WS."""
        self._ws_trade = await websockets.connect(
            self.WS_TRADE, ping_interval=20, ping_timeout=10,
        )

        expires = int((time.time() + 60) * 1000)   # 60s buffer
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
        ok, err = _is_auth_ok(auth_resp)
        if not ok:
            raise ConnectionError(f"WS Trade auth failed: {err} | raw={auth_resp}")
        logger.info("WS Trade authenticated successfully")

    async def _connect_ws_private(self):
        """Authenticate and connect to V5 Private WS."""
        self._ws_private = await websockets.connect(
            self.WS_PRIVATE, ping_interval=20, ping_timeout=10,
        )

        expires = int((time.time() + 60) * 1000)   # 60s buffer
        signature = hmac.new(
            self.cfg.api_secret.encode(),
            f"GET/realtime{expires}".encode(),
            hashlib.sha256,
        ).hexdigest()

        await self._ws_private.send(json.dumps({
            "op": "auth",
            "args": [self.cfg.api_key, expires, signature],
        }))

        # NOTE: Private WS uses `success` / `ret_msg`, NOT `retCode` / `retMsg`.
        auth_resp = await _await_op_response(self._ws_private, "auth")
        ok, err = _is_auth_ok(auth_resp)
        if not ok:
            raise ConnectionError(f"WS Private auth failed: {err} | raw={auth_resp}")
        logger.info("WS Private authenticated successfully")

    async def subscribe_private(
        self, topics: List[str], callback: Callable,
    ):
        """Subscribe to private topics: position, order, execution, wallet."""
        if self._ws_private is None or self._ws_private.closed:
            await self._connect_ws_private()
        await self._ws_private.send(
            json.dumps({"op": "subscribe", "args": topics})
        )
        for topic in topics:
            self._subscriptions[topic] = callback
        asyncio.create_task(self._private_listener())

    async def _private_listener(self):
        while self._running and self._ws_private and not self._ws_private.closed:
            try:
                msg = json.loads(await self._ws_private.recv())
                topic = msg.get("topic", "")
                if topic in self._subscriptions:
                    await self._subscriptions[topic](msg)
            except websockets.ConnectionClosed:
                logger.warning("Private WS closed. Reconnecting...")
                await asyncio.sleep(3)
                await self._connect_ws_private()

    async def start(self):
        self._running = True
        await self._connect_ws_trade()

    async def stop(self):
        self._running = False
        for ws in [self._ws_public, self._ws_private, self._ws_trade]:
            if ws and not ws.closed:
                await ws.close()
        if self._session and not self._session.closed:
            await self._session.close()