"""
analysis/basis.py
Basis spread rate between spot and perpetual futures.

Basis %       = (Perp_mark − Spot_last) / Spot_last × 100
Annualized    = funding_rate × 3 × 365 × 100  (Bybit 8h funding)

Regime:
  basis > +0.05%  → contango (bullish carry)
  basis < −0.05%  → backwardation (bearish carry)

Signal bias (contrarian at extremes):
  High positive basis + positive funding → crowded longs → short bias
  High negative basis + negative funding → crowded shorts → long bias
"""
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BasisResult:
    spot_price: float = 0.0
    perp_mark: float = 0.0
    perp_index: float = 0.0
    basis_abs: float = 0.0
    basis_pct: float = 0.0
    basis_rate_perp: float = 0.0
    funding_rate: float = 0.0
    funding_annualized: float = 0.0
    next_funding_in_hours: float = 0.0
    regime: str = "neutral"
    signal_bias: str = "neutral"
    confidence_contribution: float = 0.0


class BasisAnalyzer:
    def __init__(self, client, funding_annualization_factor: float = 3 * 365):
        self.client = client
        self.factor = funding_annualization_factor

    async def _get_spot_ticker(self, symbol: str) -> Optional[Dict]:
        try:
            data = await self.client._rest_request(
                "GET", "/v5/market/tickers",
                params={"category": "spot", "symbol": symbol},
            )
            lst = data.get("result", {}).get("list", [])
            return lst[0] if lst else None
        except Exception:
            return None

    async def _get_perp_ticker(self, symbol: str) -> Optional[Dict]:
        try:
            data = await self.client._rest_request(
                "GET", "/v5/market/tickers",
                params={"category": "linear", "symbol": symbol},
            )
            lst = data.get("result", {}).get("list", [])
            return lst[0] if lst else None
        except Exception:
            return None

    def _annualize_funding(self, funding_rate: float) -> float:
        return funding_rate * self.factor * 100

    async def compute(self, symbol: str) -> BasisResult:
        result = BasisResult()
        spot = await self._get_spot_ticker(symbol)
        perp = await self._get_perp_ticker(symbol)

        if perp:
            result.perp_mark = float(perp.get("markPrice", 0) or 0)
            result.perp_index = float(perp.get("indexPrice", 0) or 0)
            result.funding_rate = float(perp.get("fundingRate", 0) or 0)
            result.basis_rate_perp = float(perp.get("basisRate", 0) or 0)
            nft = perp.get("nextFundingTime")
            if nft:
                now_ms = int(time.time() * 1000)
                result.next_funding_in_hours = max(
                    0.0, (int(nft) - now_ms) / 3_600_000
                )

        if spot:
            result.spot_price = float(spot.get("lastPrice", 0) or 0)

        if result.spot_price > 0 and result.perp_mark > 0:
            result.basis_abs = result.perp_mark - result.spot_price
            result.basis_pct = (
                result.basis_abs / result.spot_price
            ) * 100.0
        elif result.perp_mark > 0 and result.perp_index > 0:
            result.basis_abs = result.perp_mark - result.perp_index
            result.basis_pct = (
                result.basis_abs / result.perp_index
            ) * 100.0
            result.spot_price = result.perp_index

        result.funding_annualized = self._annualize_funding(
            result.funding_rate
        )

        if result.basis_pct > 0.05:
            result.regime = "contango"
        elif result.basis_pct < -0.05:
            result.regime = "backwardation"

        if result.basis_pct > 0.15 and result.funding_rate > 0.0003:
            result.signal_bias = "short_perp"
        elif result.basis_pct < -0.15 and result.funding_rate < -0.0003:
            result.signal_bias = "long_perp"

        score = 0.0
        if result.funding_rate > 0.0005:
            score -= 5
        elif result.funding_rate < -0.0005:
            score += 5
        if result.basis_pct > 0.30:
            score -= 3
        elif result.basis_pct < -0.30:
            score += 3
        result.confidence_contribution = float(np.clip(score, -10, 10))
        return result