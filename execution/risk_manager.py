"""
execution/risk_manager.py
Position sizing, drawdown control, kill switch.
"""
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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

    def _maybe_reset_daily(self):
        now = time.time()
        if now - self.state.day_start_ts >= 86_400:
            self.state.day_start_ts = now
            self.state.daily_pnl_pct = 0.0
            logger.info("Daily risk state reset")

    def approve_trade(self, signal) -> Tuple[bool, float, str]:
        """
        Returns: (approved, size_usd, reason)
        """
        self._maybe_reset_daily()

        if self.kill_switch:
            return False, 0.0, "kill_switch_active"

        # Daily loss check
        if self.state.daily_pnl_pct <= -self.cfg.get("max_daily_loss_pct", 5.0):
            self.kill_switch = True
            return False, 0.0, "daily_loss_limit_hit"

        # Max drawdown check
        if self.state.peak_equity > 0:
            dd = (self.state.peak_equity - self.state.current_equity) / self.state.peak_equity * 100
            if dd >= self.cfg.get("max_drawdown_pct", 10.0):
                self.kill_switch = True
                return False, 0.0, "max_drawdown_hit"

        # Max open positions
        if self.state.open_positions >= self.cfg.get("max_open_positions", 5):
            return False, 0.0, "max_open_positions"

        # Risk-reward gate
        # Risk-reward gate: require TP1 >= 1R and weighted avg >= 1.5R
        if hasattr(signal, "risk_reward_tp1"):
            tp1_rr = round(signal.risk_reward_tp1, 2)
            if tp1_rr < 1.0:
                return False, 0.0, f"tp1_rr_too_low:{tp1_rr:.2f}"
        if hasattr(signal, "risk_reward") and signal.risk_reward < 1.5:
            return False, 0.0, f"weighted_rr_too_low:{signal.risk_reward:.2f}"

        # Position sizing: fixed fractional of equity
        equity = self.state.current_equity or 10_000  # default 10k for paper mode
        max_pct = self.cfg.get("max_position_pct", 2.0) / 100.0
        size_usd = equity * max_pct

        # Scale by confidence (higher confidence → larger position, cap at max_pct)
        conf = getattr(signal, "confidence", 0.5)
        size_usd *= min(1.5, max(0.5, conf / 0.7))

        # Apply leverage cap
        leverage = getattr(signal, "leverage", 5)
        notional_cap = equity * leverage * max_pct
        size_usd = min(size_usd, notional_cap)

        return True, size_usd, "approved"

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