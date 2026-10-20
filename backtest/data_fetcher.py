"""
backtest/data_fetcher.py
Fetch historical OHLCV, funding rate, and open interest from Bybit REST.

Caches results to backtest/cache/{symbol}_{interval}_{start}_{end}.parquet
to avoid re-fetching on subsequent runs.
"""
import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import aiohttp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
API_BASE = "https://api.bybit.com"

# Bybit kline interval codes
INTERVALS = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15",
    "30m": "30", "1h": "60", "2h": "120", "4h": "240",
    "6h": "360", "12h": "720", "1d": "D",
}


class BybitDataFetcher:
    def __init__(self, category: str = "linear"):
        self.category = category
        self._session: Optional[aiohttp.ClientSession] = None
        os.makedirs(CACHE_DIR, exist_ok=True)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------ #
    #  Kline fetch with pagination
    # ------------------------------------------------------------------ #
    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch full kline history between start and end.

        Bybit returns max 1000 bars per request and paginates backward
        via the `end` parameter. We loop until we cover [start, end].
        """
        interval_code = INTERVALS.get(interval, interval)

        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            start = end - timedelta(days=30)

        cache_key = (
            f"{symbol}_{interval}_{int(start.timestamp())}"
            f"_{int(end.timestamp())}"
        )
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.parquet")

        if use_cache and os.path.exists(cache_path):
            age_hours = (
                time.time() - os.path.getmtime(cache_path)
            ) / 3600
            if age_hours < 24:
                logger.info(f"Cache hit: {cache_path}")
                return pd.read_parquet(cache_path)

        logger.info(
            f"Fetching {symbol} {interval} from "
            f"{start.date()} to {end.date()}"
        )

        all_klines: List[List] = []
        cursor_end = int(end.timestamp() * 1000)
        start_ms = int(start.timestamp() * 1000)
        session = await self._get_session()
        page = 0

        while cursor_end > start_ms:
            page += 1
            url = f"{API_BASE}/v5/market/kline"
            params = {
                "category": self.category,
                "symbol": symbol,
                "interval": interval_code,
                "limit": 1000,
                "end": cursor_end,
            }
            try:
                async with session.get(url, params=params) as resp:
                    data = await resp.json()
            except Exception as e:
                logger.error(f"Fetch failed page {page}: {e}")
                await asyncio.sleep(2)
                continue

            if data.get("retCode") != 0:
                logger.error(
                    f"Bybit error: {data.get('retMsg')}"
                )
                break

            batch = data.get("result", {}).get("list", [])
            if not batch:
                break

            all_klines.extend(batch)
            oldest_ts = int(batch[-1][0])
            cursor_end = oldest_ts - 1
            logger.debug(
                f"Page {page}: {len(batch)} bars, "
                f"oldest={pd.to_datetime(oldest_ts, unit='ms')}"
            )
            await asyncio.sleep(0.1)

        if not all_klines:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_klines,
            columns=["ts", "open", "high", "low", "close",
                     "volume", "turnover"],
        )
        for col in ("open", "high", "low", "close",
                    "volume", "turnover"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ts"] = pd.to_numeric(df["ts"])
        df = df.sort_values("ts").reset_index(drop=True)
        df = df.drop_duplicates(subset=["ts"]).reset_index(drop=True)

        logger.info(
            f"Fetched {len(df)} bars for {symbol} "
            f"({df['ts'].iloc[0]} → {df['ts'].iloc[-1]})"
        )

        if use_cache:
            try:
                df.to_parquet(cache_path, index=False)
                logger.info(f"Cached to {cache_path}")
            except Exception as e:
                logger.debug(f"Cache write failed: {e}")

        return df

    # ------------------------------------------------------------------ #
    #  Funding rate history
    # ------------------------------------------------------------------ #
    async def fetch_funding_history(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch funding rate history. Funding is paid every 8h.
        Returns df with columns [ts, fundingRate].
        """
        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            start = end - timedelta(days=30)

        cache_key = (
            f"{symbol}_funding_{int(start.timestamp())}"
            f"_{int(end.timestamp())}"
        )
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.parquet")
        if os.path.exists(cache_path):
            return pd.read_parquet(cache_path)

        all_rows: List[Dict] = []
        cursor_end = int(end.timestamp() * 1000)
        start_ms = int(start.timestamp() * 1000)
        session = await self._get_session()

        while cursor_end > start_ms:
            url = f"{API_BASE}/v5/market/funding/history"
            params = {
                "category": self.category,
                "symbol": symbol,
                "limit": 200,
                "endTime": cursor_end,
            }
            try:
                async with session.get(url, params=params) as resp:
                    data = await resp.json()
            except Exception as e:
                logger.warning(f"Funding fetch failed: {e}")
                await asyncio.sleep(2)
                continue

            batch = data.get("result", {}).get("list", [])
            if not batch:
                break

            all_rows.extend(batch)
            oldest_ts = int(batch[-1].get("fundingRateTimestamp", 0))
            if oldest_ts <= 0:
                break
            cursor_end = oldest_ts - 1
            await asyncio.sleep(0.1)

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df["ts"] = pd.to_numeric(
            df["fundingRateTimestamp"], errors="coerce"
        )
        df["fundingRate"] = pd.to_numeric(
            df["fundingRate"], errors="coerce"
        )
        df = df[["ts", "fundingRate"]].dropna()
        df = df.sort_values("ts").reset_index(drop=True)

        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            pass

        logger.info(
            f"Fetched {len(df)} funding rows for {symbol}"
        )
        return df

    # ------------------------------------------------------------------ #
    #  Open interest history
    # ------------------------------------------------------------------ #
    async def fetch_oi_history(
        self,
        symbol: str,
        interval: str = "1h",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch open interest history. Returns [ts, openInterest].
        """
        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            start = end - timedelta(days=7)

        cache_key = (
            f"{symbol}_oi_{interval}_{int(start.timestamp())}"
            f"_{int(end.timestamp())}"
        )
        cache_path = os.path.join(CACHE_DIR, f"{cache_key}.parquet")
        if os.path.exists(cache_path):
            return pd.read_parquet(cache_path)

        interval_map = {
            "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "1h", "4h": "4h", "1d": "1d",
        }
        interval_time = interval_map.get(interval, "1h")

        all_rows: List[Dict] = []
        cursor_end = int(end.timestamp() * 1000)
        start_ms = int(start.timestamp() * 1000)
        session = await self._get_session()

        while cursor_end > start_ms:
            url = f"{API_BASE}/v5/market/open-interest"
            params = {
                "category": self.category,
                "symbol": symbol,
                "intervalTime": interval_time,
                "limit": 200,
                "endTime": cursor_end,
            }
            try:
                async with session.get(url, params=params) as resp:
                    data = await resp.json()
            except Exception:
                await asyncio.sleep(2)
                continue

            batch = data.get("result", {}).get("list", [])
            if not batch:
                break

            all_rows.extend(batch)
            oldest_ts = int(batch[-1].get("timestamp", 0))
            if oldest_ts <= 0:
                break
            cursor_end = oldest_ts - 1
            await asyncio.sleep(0.1)

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df["ts"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["openInterest"] = pd.to_numeric(
            df["openInterest"], errors="coerce"
        )
        df = df[["ts", "openInterest"]].dropna()
        df = df.sort_values("ts").reset_index(drop=True)

        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            pass

        return df

    # ------------------------------------------------------------------ #
    #  Convenience: fetch everything for a symbol
    # ------------------------------------------------------------------ #
    async def fetch_bundle(
        self,
        symbol: str,
        interval: str = "1h",
        days: int = 30,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch klines + funding + OI in parallel."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        klines_task = asyncio.create_task(
            self.fetch_klines(symbol, interval, start, end)
        )
        funding_task = asyncio.create_task(
            self.fetch_funding_history(symbol, start, end)
        )
        oi_task = asyncio.create_task(
            self.fetch_oi_history(symbol, interval, start, end)
        )

        klines, funding, oi = await asyncio.gather(
            klines_task, funding_task, oi_task,
            return_exceptions=True,
        )

        return {
            "klines": (
                klines if isinstance(klines, pd.DataFrame)
                else pd.DataFrame()
            ),
            "funding": (
                funding if isinstance(funding, pd.DataFrame)
                else pd.DataFrame()
            ),
            "oi": (
                oi if isinstance(oi, pd.DataFrame)
                else pd.DataFrame()
            ),
        }

    # ------------------------------------------------------------------ #
    #  Symbol discovery (used by run_all_pairs)
    # ------------------------------------------------------------------ #
    async def list_all_symbols(
        self,
        min_turnover_24h: float = 1_000_000,
        only_perpetual: bool = True,
        quote_coin: str = "USDT",
    ) -> list:
        """
        Fetch all active linear perpetual symbols, filtered by
        24h turnover. Mirrors BybitV5Client.get_all_linear_symbols()
        but without importing the trading client.
        """
        session = await self._get_session()

        # 1. All linear instruments (paginated)
        all_instruments = []
        cursor = None
        page = 0
        while True:
            params = {"category": self.category, "limit": 500}
            if cursor:
                params["cursor"] = cursor
            url = f"{API_BASE}/v5/market/instruments-info"
            try:
                async with session.get(url, params=params) as resp:
                    data = await resp.json()
            except Exception as e:
                logger.error(f"Instruments fetch failed: {e}")
                break
            result = data.get("result", {})
            batch = result.get("list", [])
            if not batch:
                break
            all_instruments.extend(batch)
            cursor = result.get("nextPageCursor")
            page += 1
            if not cursor:
                break
            await asyncio.sleep(0.15)

        logger.info(
            f"Fetched {len(all_instruments)} linear instruments"
        )

        # 2. Filter
        filtered = []
        for inst in all_instruments:
            if inst.get("quoteCoin") != quote_coin:
                continue
            if inst.get("status") != "Trading":
                continue
            if only_perpetual and inst.get("contractType") != "LinearPerpetual":
                continue
            filtered.append(inst["symbol"])

        if not filtered:
            return []

        # 3. Turnover filter
        url = f"{API_BASE}/v5/market/tickers"
        try:
            async with session.get(
                url, params={"category": self.category},
            ) as resp:
                data = await resp.json()
        except Exception as e:
            logger.error(f"Tickers fetch failed: {e}")
            return filtered

        tickers = data.get("result", {}).get("list", [])
        turnover_map = {}
        for t in tickers:
            sym = t.get("symbol")
            if not sym:
                continue
            try:
                turnover = float(t.get("turnover24h") or 0)
            except (TypeError, ValueError):
                turnover = 0.0
            turnover_map[sym] = turnover

        symbols = [
            s for s in filtered
            if turnover_map.get(s, 0) >= min_turnover_24h
        ]
        symbols.sort(key=lambda s: turnover_map.get(s, 0), reverse=True)

        logger.info(
            f"After filters: {len(symbols)} symbols with "
            f">= ${min_turnover_24h:,.0f} 24h turnover"
        )
        return symbols