"""
learning/outcome_tracker.py
Compares signals to actual outcomes at 07:00 daily.
"""
import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OutcomeTracker:
    def __init__(self, client, signal_journal, max_hold_hours: int = 72):
        self.client = client
        self.journal = signal_journal
        self.max_hold_hours = max_hold_hours

    async def run_daily_check(self):
        unresolved = self.journal.get_unresolved_signals()
        logger.info(f"Outcome tracker: {len(unresolved)} unresolved signals")
        for sig in unresolved:
            try:
                outcome = await self._resolve_signal(sig)
                if outcome:
                    self.journal.record_outcome(sig["id"], outcome)
                    logger.info(
                        f"Resolved {sig['symbol']} {sig['direction']}: "
                        f"{outcome['result']}"
                    )
            except Exception as e:
                logger.error(f"Failed to resolve signal {sig['id']}: {e}")

    async def _resolve_signal(self, sig: Dict) -> Optional[Dict]:
        signal_time = sig["timestamp"]
        if isinstance(signal_time, str):
            signal_time = datetime.fromisoformat(signal_time)
        elapsed_hours = (datetime.utcnow() - signal_time).total_seconds() / 3600
        if elapsed_hours < 1:
            return None

        if elapsed_hours <= 24:
            interval = "5"
        elif elapsed_hours <= 72:
            interval = "15"
        else:
            interval = "60"

        limit = min(int(elapsed_hours * 12) + 10, 1000)
        klines = await self.client.get_klines(
            sig["symbol"], interval=interval, limit=limit,
        )
        if not klines:
            return None

        df = pd.DataFrame(
            klines,
            columns=["ts", "open", "high", "low", "close", "volume", "turnover"],
        )
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ts"] = pd.to_numeric(df["ts"])
        df = df.sort_values("ts").reset_index(drop=True)

        signal_ts = int(signal_time.timestamp() * 1000)
        df = df[df["ts"] >= signal_ts].reset_index(drop=True)
        if df.empty:
            return None

        entry = sig["entry_price"]
        sl = sig["stop_loss"]
        tp = sig["take_profit_1"]
        direction = sig["direction"]

        tp_hit, sl_hit = False, False
        tp_hit_time, sl_hit_time = None, None

        for _, row in df.iterrows():
            if direction == "long":
                if row["high"] >= tp and not tp_hit:
                    tp_hit = True
                    tp_hit_time = row["ts"]
                if row["low"] <= sl and not sl_hit:
                    sl_hit = True
                    sl_hit_time = row["ts"]
            else:
                if row["low"] <= tp and not tp_hit:
                    tp_hit = True
                    tp_hit_time = row["ts"]
                if row["high"] >= sl and not sl_hit:
                    sl_hit = True
                    sl_hit_time = row["ts"]
            if tp_hit and sl_hit:
                break

        if tp_hit and sl_hit:
            if tp_hit_time and sl_hit_time and tp_hit_time < sl_hit_time:
                result = "tp_hit"
            else:
                result = "sl_hit"
        elif tp_hit:
            result = "tp_hit"
        elif sl_hit:
            result = "sl_hit"
        else:
            if elapsed_hours > self.max_hold_hours:
                current_price = df["close"].iloc[-1]
                pnl = (
                    (current_price - entry) / entry * 100
                    if direction == "long"
                    else (entry - current_price) / entry * 100
                )
                result = "expired_profit" if pnl > 0 else "expired_loss"
            else:
                return None

        if result == "tp_hit":
            pnl_pct = abs(tp - entry) / entry * 100
        elif result == "sl_hit":
            pnl_pct = -abs(sl - entry) / entry * 100
        else:
            current_price = df["close"].iloc[-1]
            pnl_pct = (
                (current_price - entry) / entry * 100
                if direction == "long"
                else (entry - current_price) / entry * 100
            )

        perps_after = {}
        try:
            oi_data = await self.client.get_open_interest(
                sig["symbol"], interval_time="1h", limit=48,
            )
            if oi_data:
                perps_after["oi_latest"] = float(oi_data[-1].get("openInterest", 0))
            funding = await self.client.get_funding_rate(sig["symbol"])
            perps_after["funding_latest"] = float(funding.get("fundingRate", 0))
        except Exception:
            pass

        return {
            "result": result,
            "won": result in ("tp_hit", "expired_profit"),
            "pnl_pct": round(pnl_pct, 4),
            "resolved_at": datetime.utcnow().isoformat(),
            "elapsed_hours": round(elapsed_hours, 2),
            "perps_data": perps_after,
        }