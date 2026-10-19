"""
execution/risk_manager.py
Position sizing, drawdown control, kill switch, funding-aware
directional block.

All methods are SYNCHRONOUS. Balance and position fetches are done
by the caller (trade_executor) before invoking approve_trade.
"""
import logging
import time
from dataclasses import dataclass
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    daily_pnl_pct: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    open_positions: int = 0
    day_start_ts: float = 0.0


class RiskManager:
    def __init__(self, config: Dict):
        self.cfg = config
        self.state = RiskState(day_start_ts=time.time())
        self.kill_switch = False

    @staticmethod
    def _confidence_to_pct(confidence: float) -> float:
        """Map confidence 0..1 to position % of equity: 0.5% to 3.0%."""
        if confidence >= 0.85:
            return 3.0
        if confidence >= 0.75:
            return 2.0
        if confidence >= 0.65:
            return 1.5
        if confidence >= 0.55:
            return 1.0
        return 0.5

    def _maybe_reset_daily(self):
        now = time.time()
        if now - self.state.day_start_ts >= 86_400:
            self.state.day_start_ts = now
            self.state.daily_pnl_pct = 0.0
            logger.info("Daily risk state reset")

    def approve_trade(self, signal) -> Tuple[bool, float, str]:
        """
        Synchronous approval. Returns (approved, size_usd, reason).
        """
        self._maybe_reset_daily()

        if self.kill_switch:
            return False, 0.0, "kill_switch_active"

        if self.state.daily_pnl_pct <= -self.cfg.get("max_daily_loss_pct", 5.0):
            self.kill_switch = True
            return False, 0.0, "daily_loss_limit_hit"

        if self.state.peak_equity > 0:
            dd = (
                (self.state.peak_equity - self.state.current_equity)
                / self.state.peak_equity * 100
            )
            if dd >= self.cfg.get("max_drawdown_pct", 10.0):
                self.kill_switch = True
                return False, 0.0, "max_drawdown_hit"

        if self.state.open_positions >= self.cfg.get("max_open_positions", 5):
            return False, 0.0, "max_open_positions"

        # ---- Funding-aware directional block ----
        meta = getattr(signal, "metadata", {}) or {}
        if signal.direction == "long" and meta.get("funding_block_long"):
            rate = meta.get("funding_rate_pct", 0.0)
            return False, 0.0, f"funding_long_crowded:{rate:+.3f}%"
        if signal.direction == "short" and meta.get("funding_block_short"):
            rate = meta.get("funding_rate_pct", 0.0)
            return False, 0.0, f"funding_short_crowded:{rate:+.3f}%"

        # ---- Risk-reward gates ----
        if hasattr(signal, "risk_reward_tp1"):
            tp1_rr = round(signal.risk_reward_tp1, 2)
            if tp1_rr < 0.99:
                return False, 0.0, f"tp1_rr_too_low:{tp1_rr:.2f}"
        if hasattr(signal, "risk_reward") and signal.risk_reward < 1.5:
            return False, 0.0, f"weighted_rr_too_low:{signal.risk_reward:.2f}"

        # ---- Position sizing ----
        equity = self.state.current_equity or 100.0
        conf = getattr(signal, "confidence", 0.5)
        size_pct = self._confidence_to_pct(conf) / 100.0
        leverage = getattr(signal, "leverage", 5)

        # Margin at risk = size_pct × equity
        margin_usd = equity * size_pct

        # Notional exposure = margin × leverage
        notional_usd = margin_usd * leverage

        # Safety cap: notional should never exceed equity × leverage
        # (implicit since margin < equity, but explicit for clarity)
        max_notional = equity * leverage
        notional_usd = min(notional_usd, max_notional)

        return True, notional_usd, f"approved:{size_pct*100:.2f}%_margin"

    def update_equity(self, equity: float):
        self.state.current_equity = equity
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity

    def register_position_open(self):
        self.state.open_positions += 1

    def register_position_close(self, pnl_pct: float = 0.0):
        self.state.open_positions = max(0, self.state.open_positions - 1)
        self.state.daily_pnl_pct += pnl_pct

    def reset_kill_switch(self):
        self.kill_switch = False
        logger.warning("Kill switch manually reset")