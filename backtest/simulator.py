"""
backtest/simulator.py
Bar-by-bar trade simulation with realistic fills, fees, and slippage.

Trade lifecycle:
  1. Signal fires at bar N close → entry attempted at bar N+1 open
  2. Limit vs market order routing based on deviation
  3. TP/SL checked bar-by-bar with intrabar high/low
  4. Conservative tie-break: SL wins when both hit in same bar
  5. Fees deducted on entry and each TP fill
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    signal_ts: int
    entry_ts: int
    entry_price: float
    exit_ts: int
    exit_price: float
    direction: str
    setup_type: str
    sl: float
    tps: List[float]
    tp_hits: int
    tp_weights: List[float]
    confidence: float
    raw_score: float
    gross_pnl_pct: float
    net_pnl_pct: float
    fees_pct: float
    bars_held: int
    exit_reason: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class SimulationConfig:
    fee_taker_pct: float = 0.055    # Bybit taker fee
    fee_maker_pct: float = 0.020    # Bybit maker fee
    slippage_pct: float = 0.02      # Conservative slippage on market orders
    market_tolerance_pct: float = 0.15
    max_deviation_pct: float = 4.0
    cancel_on_rally_pct: float = 2.5
    limit_expiry_bars: int = 8
    max_hold_bars: int = 72
    entry_delay_bars: int = 1       # signal at N close → entry at N+1 open


class TradeSimulator:
    def __init__(self, config: SimulationConfig):
        self.cfg = config

    # ------------------------------------------------------------------ #
    #  Entry execution
    # ------------------------------------------------------------------ #
    def _attempt_entry(
        self,
        signal,
        df: pd.DataFrame,
        signal_bar_idx: int,
    ) -> Optional[Dict]:
        """
        Try to enter at bar signal_bar_idx + entry_delay_bars.

        Returns dict with entry data, or None if entry skipped.
        """
        entry_bar = signal_bar_idx + self.cfg.entry_delay_bars
        if entry_bar >= len(df):
            return None

        bar = df.iloc[entry_bar]
        bar_open = float(bar["open"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bar_close = float(bar["close"])

        signal_entry = float(signal.entry_price)
        direction = signal.direction

        # Deviation check (production uses current market vs signal entry)
        deviation = (
            abs(bar_open - signal_entry) / signal_entry * 100
        )
        if deviation > self.cfg.max_deviation_pct:
            return {
                "status": "skipped",
                "reason": f"stale_entry:{deviation:.2f}%",
            }

        # Rally check
        if direction == "long":
            rally = max(0.0, (bar_open - signal_entry) / signal_entry * 100)
        else:
            rally = max(0.0, (signal_entry - bar_open) / signal_entry * 100)

        if rally > self.cfg.cancel_on_rally_pct:
            return {
                "status": "skipped",
                "reason": f"rally_too_far:{rally:.2f}%",
            }

        # Determine fill type
        is_market = deviation <= self.cfg.market_tolerance_pct

        if is_market:
            # Market order: fills at open + slippage
            if direction == "long":
                fill_price = bar_open * (1 + self.cfg.slippage_pct / 100)
            else:
                fill_price = bar_open * (1 - self.cfg.slippage_pct / 100)
            fee_pct = self.cfg.fee_taker_pct
            return {
                "status": "filled",
                "entry_bar": entry_bar,
                "entry_price": fill_price,
                "entry_fee_pct": fee_pct,
                "order_type": "market",
                "fill_ts": int(bar["ts"]),
            }

        # Limit order at signal_entry
        # Fill requires bar to cross the limit price
        if direction == "long":
            # Long limit at signal_entry: fills if bar_low <= signal_entry
            if bar_low <= signal_entry:
                fill_price = signal_entry
                fee_pct = self.cfg.fee_maker_pct
                return {
                    "status": "filled",
                    "entry_bar": entry_bar,
                    "entry_price": fill_price,
                    "entry_fee_pct": fee_pct,
                    "order_type": "limit",
                    "fill_ts": int(bar["ts"]),
                }
        else:
            # Short limit at signal_entry: fills if bar_high >= signal_entry
            if bar_high >= signal_entry:
                fill_price = signal_entry
                fee_pct = self.cfg.fee_maker_pct
                return {
                    "status": "filled",
                    "entry_bar": entry_bar,
                    "entry_price": fill_price,
                    "entry_fee_pct": fee_pct,
                    "order_type": "limit",
                    "fill_ts": int(bar["ts"]),
                }

        # Not filled on entry bar — check expiry window
        for offset in range(2, self.cfg.limit_expiry_bars + 1):
            check_bar = entry_bar + offset - 1
            if check_bar >= len(df):
                break
            b = df.iloc[check_bar]

            # Rally cancel check
            if direction == "long":
                rally_pct = (
                    (b["high"] - signal_entry) / signal_entry * 100
                )
            else:
                rally_pct = (
                    (signal_entry - b["low"]) / signal_entry * 100
                )

            if rally_pct > self.cfg.cancel_on_rally_pct:
                return {
                    "status": "cancelled",
                    "reason": f"rally_cancel:{rally_pct:.2f}%",
                    "bars_waited": offset,
                }

            # Fill check
            if direction == "long" and b["low"] <= signal_entry:
                return {
                    "status": "filled",
                    "entry_bar": check_bar,
                    "entry_price": signal_entry,
                    "entry_fee_pct": self.cfg.fee_maker_pct,
                    "order_type": "limit",
                    "fill_ts": int(b["ts"]),
                }
            if direction == "short" and b["high"] >= signal_entry:
                return {
                    "status": "filled",
                    "entry_bar": check_bar,
                    "entry_price": signal_entry,
                    "entry_fee_pct": self.cfg.fee_maker_pct,
                    "order_type": "limit",
                    "fill_ts": int(b["ts"]),
                }

        return {
            "status": "stalled",
            "reason": f"not_filled_in_{self.cfg.limit_expiry_bars}_bars",
        }

    # ------------------------------------------------------------------ #
    #  Trade exit walk
    # ------------------------------------------------------------------ #
    def _walk_exit(
        self,
        df: pd.DataFrame,
        entry_bar: int,
        direction: str,
        entry_price: float,
        sl: float,
        tps: List[float],
        tp_weights: List[float],
        entry_fee_pct: float,
    ) -> Dict:
        """
        Walk forward from entry_bar + 1, tracking TP/SL hits.

        Conservative: SL checked before TP within the same bar.
        """
        remaining = 1.0
        gross_pnl = 0.0
        tp_hits = 0
        total_fee_pct = entry_fee_pct

        max_exit = min(
            entry_bar + 1 + self.cfg.max_hold_bars, len(df)
        )

        for i in range(entry_bar + 1, max_exit):
            bar = df.iloc[i]
            high = float(bar["high"])
            low = float(bar["low"])

            # SL first (conservative)
            if direction == "long" and low <= sl:
                gross_pnl += remaining * (sl - entry_price) / entry_price * 100
                total_fee_pct += self.cfg.fee_taker_pct   # SL is market
                return {
                    "exit_bar": i,
                    "exit_price": sl,
                    "exit_ts": int(bar["ts"]),
                    "gross_pnl_pct": gross_pnl,
                    "total_fee_pct": total_fee_pct,
                    "tp_hits": tp_hits,
                    "exit_reason": "sl_hit",
                }
            if direction == "short" and high >= sl:
                gross_pnl += remaining * (entry_price - sl) / entry_price * 100
                total_fee_pct += self.cfg.fee_taker_pct
                return {
                    "exit_bar": i,
                    "exit_price": sl,
                    "exit_ts": int(bar["ts"]),
                    "gross_pnl_pct": gross_pnl,
                    "total_fee_pct": total_fee_pct,
                    "tp_hits": tp_hits,
                    "exit_reason": "sl_hit",
                }

            # TP checks
            for j, tp in enumerate(tps):
                if j >= len(tp_weights) or remaining <= 1e-9:
                    continue
                hit = (
                    (direction == "long" and high >= tp)
                    or (direction == "short" and low <= tp)
                )
                if not hit:
                    continue

                w = tp_weights[j]
                if direction == "long":
                    gross_pnl += w * (tp - entry_price) / entry_price * 100
                else:
                    gross_pnl += w * (entry_price - tp) / entry_price * 100

                # TP is limit order (maker fee)
                total_fee_pct += self.cfg.fee_maker_pct * w
                remaining -= w
                tp_hits += 1

                if remaining <= 1e-9:
                    return {
                        "exit_bar": i,
                        "exit_price": tp,
                        "exit_ts": int(bar["ts"]),
                        "gross_pnl_pct": gross_pnl,
                        "total_fee_pct": total_fee_pct,
                        "tp_hits": tp_hits,
                        "exit_reason": "tp_all",
                    }

        # Max hold reached — close at last close
        last_bar = df.iloc[max_exit - 1]
        exit_price = float(last_bar["close"])
        if direction == "long":
            gross_pnl += (
                remaining * (exit_price - entry_price) / entry_price * 100
            )
        else:
            gross_pnl += (
                remaining * (entry_price - exit_price) / entry_price * 100
            )
        total_fee_pct += self.cfg.fee_taker_pct * remaining
        return {
            "exit_bar": max_exit - 1,
            "exit_price": exit_price,
            "exit_ts": int(last_bar["ts"]),
            "gross_pnl_pct": gross_pnl,
            "total_fee_pct": total_fee_pct,
            "tp_hits": tp_hits,
            "exit_reason": "max_hold",
        }

    # ------------------------------------------------------------------ #
    #  Full trade simulation
    # ------------------------------------------------------------------ #
    def simulate(
        self,
        signal,
        df: pd.DataFrame,
        signal_bar_idx: int,
    ) -> Optional[Trade]:
        """
        Simulate one trade. Returns Trade or None if entry skipped.
        """
        entry = self._attempt_entry(signal, df, signal_bar_idx)
        if entry is None or entry.get("status") != "filled":
            if entry and entry.get("status") != "filled":
                logger.debug(
                    f"{signal.symbol} skip: {entry.get('reason')}"
                )
            return None

        exit_result = self._walk_exit(
            df=df,
            entry_bar=entry["entry_bar"],
            direction=signal.direction,
            entry_price=entry["entry_price"],
            sl=signal.stop_loss,
            tps=[
                signal.take_profit_1,
                signal.take_profit_2,
                signal.take_profit_3,
            ],
            tp_weights=list(signal.tp_weights),
            entry_fee_pct=entry["entry_fee_pct"],
        )

        net_pnl = (
            exit_result["gross_pnl_pct"]
            - exit_result["total_fee_pct"]
        )

        return Trade(
            signal_ts=signal.ts,
            entry_ts=entry["fill_ts"],
            entry_price=entry["entry_price"],
            exit_ts=exit_result["exit_ts"],
            exit_price=exit_result["exit_price"],
            direction=signal.direction,
            setup_type=signal.setup_type,
            sl=signal.stop_loss,
            tps=[
                signal.take_profit_1,
                signal.take_profit_2,
                signal.take_profit_3,
            ],
            tp_hits=exit_result["tp_hits"],
            tp_weights=list(signal.tp_weights),
            confidence=signal.confidence,
            raw_score=signal.raw_score,
            gross_pnl_pct=exit_result["gross_pnl_pct"],
            net_pnl_pct=net_pnl,
            fees_pct=exit_result["total_fee_pct"],
            bars_held=exit_result["exit_bar"] - entry["entry_bar"],
            exit_reason=exit_result["exit_reason"],
            metadata=signal.metadata,
        )