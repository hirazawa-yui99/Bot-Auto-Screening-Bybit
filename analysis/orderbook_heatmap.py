"""
analysis/orderbook_heatmap.py
Orderbook depth-weighted heatmap.

Aggregates bid and ask volume with distance-based weighting:
   weight = 1 - (distance_from_mid / max_distance)
Closer levels count more. The result is a signed imbalance in [-1, +1]:
   > +0.15 → bullish (bid walls > ask walls)
   < -0.15 → bearish (ask walls > bid walls)
   |imbalance| > 0.30 → "strong" wall
"""
import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HeatmapResult:
    valid: bool = False
    imbalance: float = 0.0                # -1..+1
    bid_score: float = 0.0
    ask_score: float = 0.0
    bid_wall_price: float = 0.0
    ask_wall_price: float = 0.0
    bullish: bool = False                 # imbalance > +0.15
    bearish: bool = False                 # imbalance < -0.15
    strong: bool = False                  # |imbalance| > 0.30
    direction: str = "neutral"            # long | short | neutral
    reason: str = ""


class OrderbookHeatmap:
    def __init__(self, depth_levels: int = 50, max_distance_pct: float = 5.0):
        self.depth_levels = depth_levels
        self.max_distance_pct = max_distance_pct

    def analyze(self, orderbook: Dict) -> HeatmapResult:
        r = HeatmapResult()
        if not orderbook:
            r.reason = "empty orderbook"
            return r

        bids = orderbook.get("b", []) or []
        asks = orderbook.get("a", []) or []
        if not bids or not asks:
            r.reason = "missing bids/asks"
            return r

        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
        except (ValueError, IndexError):
            r.reason = "malformed top of book"
            return r

        mid = (best_bid + best_ask) / 2
        if mid <= 0:
            r.reason = "invalid mid"
            return r

        max_dist = self.max_distance_pct / 100.0

        bid_score = 0.0
        ask_score = 0.0
        bid_wall_price = 0.0
        bid_wall_size = 0.0
        ask_wall_price = 0.0
        ask_wall_size = 0.0

        for entry in bids[: self.depth_levels]:
            try:
                p = float(entry[0])
                s = float(entry[1])
            except (ValueError, IndexError):
                continue
            if p <= 0 or s <= 0:
                continue
            dist = abs(p - mid) / mid
            if dist > max_dist:
                continue
            weight = 1.0 - (dist / max_dist)
            weighted = s * weight
            bid_score += weighted
            if weighted > bid_wall_size:
                bid_wall_size = weighted
                bid_wall_price = p

        for entry in asks[: self.depth_levels]:
            try:
                p = float(entry[0])
                s = float(entry[1])
            except (ValueError, IndexError):
                continue
            if p <= 0 or s <= 0:
                continue
            dist = abs(p - mid) / mid
            if dist > max_dist:
                continue
            weight = 1.0 - (dist / max_dist)
            weighted = s * weight
            ask_score += weighted
            if weighted > ask_wall_size:
                ask_wall_size = weighted
                ask_wall_price = p

        total = bid_score + ask_score
        if total <= 0:
            r.reason = "no weighted volume"
            return r

        r.bid_score = bid_score
        r.ask_score = ask_score
        r.bid_wall_price = bid_wall_price
        r.ask_wall_price = ask_wall_price
        r.imbalance = float((bid_score - ask_score) / total)

        r.bullish = r.imbalance > 0.15
        r.bearish = r.imbalance < -0.15
        r.strong = abs(r.imbalance) > 0.30

        if r.bullish:
            r.direction = "long"
        elif r.bearish:
            r.direction = "short"

        r.valid = True
        r.reason = (
            f"imbalance={r.imbalance:+.3f} "
            f"({'strong' if r.strong else 'mild' if (r.bullish or r.bearish) else 'neutral'})"
        )
        return r