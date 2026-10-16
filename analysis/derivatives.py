"""
analysis/derivatives.py
Improved derivative analysis: CVD, Open Interest, Funding, Basis,
Account L/S Ratio, Top-Trader L/S Ratio.
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DerivativeSnapshot:
    symbol: str = ""
    cvd: float = 0.0
    cvd_slope: float = 0.0
    cvd_divergence: str = "none"
    oi: float = 0.0
    oi_change_1h: float = 0.0
    oi_change_4h: float = 0.0
    oi_divergence: str = "none"
    funding_rate: float = 0.0
    basis: float = 0.0
    account_ls_ratio: float = 1.0
    top_trader_ls_ratio: float = 1.0
    ls_ratio_divergence: str = "none"
    smart_money_bias: str = "neutral"
    confidence_contribution: float = 0.0


class DerivativeAnalyzer:
    def __init__(self, client):
        self.client = client

    async def compute_cvd(
        self, symbol: str, interval: str = "5", lookback: int = 100,
    ) -> Tuple[float, float, str]:
        klines = await self.client.get_klines(symbol, interval=interval, limit=lookback)
        if not klines:
            return 0.0, 0.0, "none"
        df = pd.DataFrame(
            klines,
            columns=["ts", "open", "high", "low", "close", "volume", "turnover"],
        )
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ts"] = pd.to_numeric(df["ts"])
        df = df.sort_values("ts").reset_index(drop=True)
        df["delta"] = np.where(
            df["close"] > df["open"], df["volume"],
            np.where(df["close"] < df["open"], -df["volume"], 0.0),
        )
        df["cvd"] = df["delta"].cumsum()
        cvd_value = float(df["cvd"].iloc[-1])
        if len(df) >= 20:
            recent = df["cvd"].iloc[-20:].values
            slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
            price_change = float(df["close"].iloc[-1] - df["close"].iloc[-20])
            cvd_change = float(df["cvd"].iloc[-1] - df["cvd"].iloc[-20])
        else:
            slope, price_change, cvd_change = 0.0, 0.0, 0.0
        divergence = "none"
        if price_change > 0 and cvd_change < 0:
            divergence = "bearish"
        elif price_change < 0 and cvd_change > 0:
            divergence = "bullish"
        return cvd_value, slope, divergence

    async def compute_oi_metrics(
        self, symbol: str,
    ) -> Tuple[float, float, float, str]:
        oi_data = await self.client.get_open_interest(
            symbol, interval_time="1h", limit=50,
        )
        if not oi_data:
            return 0.0, 0.0, 0.0, "none"
        df = pd.DataFrame(oi_data)
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
        df["timestamp"] = pd.to_numeric(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        oi_current = float(df["openInterest"].iloc[-1])
        oi_prev_1h = float(df["openInterest"].iloc[-2]) if len(df) >= 2 else oi_current
        oi_prev_4h = float(df["openInterest"].iloc[-5]) if len(df) >= 5 else oi_current
        oi_change_1h = (oi_current - oi_prev_1h) / oi_prev_1h * 100 if oi_prev_1h else 0.0
        oi_change_4h = (oi_current - oi_prev_4h) / oi_prev_4h * 100 if oi_prev_4h else 0.0

        klines = await self.client.get_klines(symbol, interval="60", limit=5)
        price_change_4h = 0.0
        if len(klines) >= 5:
            close_now = float(klines[0][4])
            close_4h = float(klines[-1][4])
            price_change_4h = (close_now - close_4h) / close_4h * 100 if close_4h else 0.0

        divergence = "none"
        if price_change_4h > 0.5 and oi_change_4h > 1.0:
            divergence = "bullish_continuation"
        elif price_change_4h > 0.5 and oi_change_4h < -1.0:
            divergence = "short_covering"
        elif price_change_4h < -0.5 and oi_change_4h > 1.0:
            divergence = "bearish_continuation"
        elif price_change_4h < -0.5 and oi_change_4h < -1.0:
            divergence = "long_liquidation"
        return oi_current, oi_change_1h, oi_change_4h, divergence

    async def compute_ls_ratios(
        self, symbol: str, period: str = "1h",
    ) -> Tuple[float, float, str, str]:
        data = await self.client.get_account_ratio(symbol, period=period, limit=10)
        if not data:
            return 1.0, 1.0, "none", "neutral"
        df = pd.DataFrame(data)
        df["buyRatio"] = pd.to_numeric(df["buyRatio"], errors="coerce")
        df["timestamp"] = pd.to_numeric(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        account_ratio = float(df["buyRatio"].iloc[-1]) if len(df) > 0 else 1.0
        top_trader_ratio = account_ratio

        divergence = "none"
        if account_ratio > 1.5 and top_trader_ratio < 0.8:
            divergence = "retail_long_smart_short"
        elif account_ratio < 0.7 and top_trader_ratio > 1.2:
            divergence = "retail_short_smart_long"

        if top_trader_ratio > 1.3:
            smart_money_bias = "long"
        elif top_trader_ratio < 0.7:
            smart_money_bias = "short"
        else:
            smart_money_bias = "neutral"
        return account_ratio, top_trader_ratio, divergence, smart_money_bias

    async def get_full_snapshot(self, symbol: str) -> DerivativeSnapshot:
        cvd_task = asyncio.create_task(self.compute_cvd(symbol))
        oi_task = asyncio.create_task(self.compute_oi_metrics(symbol))
        ls_task = asyncio.create_task(self.compute_ls_ratios(symbol))
        funding_task = asyncio.create_task(self.client.get_funding_rate(symbol))

        try:
            cvd_val, cvd_slope, cvd_div = await cvd_task
        except Exception:
            cvd_val, cvd_slope, cvd_div = 0.0, 0.0, "none"
        try:
            oi, oi_1h, oi_4h, oi_div = await oi_task
        except Exception:
            oi, oi_1h, oi_4h, oi_div = 0.0, 0.0, 0.0, "none"
        try:
            acc_ratio, top_ratio, ls_div, smart_bias = await ls_task
        except Exception:
            acc_ratio, top_ratio, ls_div, smart_bias = 1.0, 1.0, "none", "neutral"
        try:
            funding = await funding_task
        except Exception:
            funding = {}

        snapshot = DerivativeSnapshot(
            symbol=symbol,
            cvd=cvd_val,
            cvd_slope=cvd_slope,
            cvd_divergence=cvd_div,
            oi=oi,
            oi_change_1h=oi_1h,
            oi_change_4h=oi_4h,
            oi_divergence=oi_div,
            funding_rate=float(funding.get("fundingRate", 0) or 0),
            basis=float(funding.get("basisRate", 0) or 0),
            account_ls_ratio=acc_ratio,
            top_trader_ls_ratio=top_ratio,
            ls_ratio_divergence=ls_div,
            smart_money_bias=smart_bias,
        )

        score = 0.0
        if cvd_div == "bullish": score += 5
        elif cvd_div == "bearish": score -= 5
        if abs(cvd_slope) > 0:
            score += float(np.tanh(cvd_slope * 1000)) * 3
        if oi_div == "bullish_continuation": score += 4
        elif oi_div == "bearish_continuation": score -= 4
        elif oi_div == "long_liquidation": score += 2
        elif oi_div == "short_covering": score -= 2
        if ls_div == "retail_long_smart_short": score -= 4
        elif ls_div == "retail_short_smart_long": score += 4
        if smart_bias == "long": score += 3
        elif smart_bias == "short": score -= 3
        snapshot.confidence_contribution = float(np.clip(score, -25, 25))
        return snapshot