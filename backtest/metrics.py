"""
backtest/metrics.py
Performance statistics from a list of Trades.
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    win_rate: float = 0.0
    total_pnl_pct: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    avg_bars_held: float = 0.0
    max_consecutive_losses: int = 0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    by_setup: Dict = field(default_factory=dict)
    by_exit_reason: Dict = field(default_factory=dict)
    equity_curve: List[float] = field(default_factory=list)


def compute_metrics(trades: List) -> BacktestMetrics:
    m = BacktestMetrics()

    if not trades:
        return m

    pnls = np.array([t.net_pnl_pct for t in trades])
    m.total_trades = len(trades)
    m.wins = int(np.sum(pnls > 0))
    m.losses = int(np.sum(pnls < 0))
    m.breakeven = m.total_trades - m.wins - m.losses
    m.win_rate = m.wins / m.total_trades if m.total_trades else 0.0
    m.total_pnl_pct = float(pnls.sum())
    m.avg_pnl_pct = float(pnls.mean())

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    m.avg_win_pct = float(wins.mean()) if len(wins) else 0.0
    m.avg_loss_pct = float(losses.mean()) if len(losses) else 0.0

    if len(losses) and losses.sum() != 0:
        m.profit_factor = float(wins.sum() / abs(losses.sum()))
    else:
        m.profit_factor = float("inf") if len(wins) else 0.0

    # Equity curve (cumulative % return)
    equity = np.cumsum(pnls)
    m.equity_curve = equity.tolist()

    # Max drawdown
    if len(equity):
        running_max = np.maximum.accumulate(equity)
        drawdown = equity - running_max
        m.max_drawdown_pct = float(drawdown.min())

    # Sharpe (per-trade, annualized approximation)
    if len(pnls) > 1 and pnls.std() > 0:
        # Trades per day assumed proportional to bars_held
        avg_bars = np.mean([t.bars_held for t in trades])
        trades_per_year = 365 * 24 / max(avg_bars, 1)
        m.sharpe = float(
            (pnls.mean() / pnls.std()) * np.sqrt(trades_per_year)
        )

    # Sortino (downside deviation)
    if len(pnls) > 1:
        downside = pnls[pnls < 0]
        if len(downside) > 0 and downside.std() > 0:
            avg_bars = np.mean([t.bars_held for t in trades])
            trades_per_year = 365 * 24 / max(avg_bars, 1)
            m.sortino = float(
                (pnls.mean() / downside.std()) * np.sqrt(trades_per_year)
            )

    m.avg_bars_held = float(np.mean([t.bars_held for t in trades]))
    m.best_trade_pct = float(pnls.max())
    m.worst_trade_pct = float(pnls.min())

    # Max consecutive losses
    streak = 0
    max_streak = 0
    for p in pnls:
        if p < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    m.max_consecutive_losses = max_streak

    # By setup
    setups = {}
    for t in trades:
        s = t.setup_type
        if s not in setups:
            setups[s] = {
                "trades": 0, "wins": 0, "total_pnl": 0.0,
                "avg_pnl": 0.0,
            }
        setups[s]["trades"] += 1
        setups[s]["wins"] += 1 if t.net_pnl_pct > 0 else 0
        setups[s]["total_pnl"] += t.net_pnl_pct
    for s in setups:
        setups[s]["avg_pnl"] = (
            setups[s]["total_pnl"] / setups[s]["trades"]
        )
        setups[s]["win_rate"] = (
            setups[s]["wins"] / setups[s]["trades"]
        )
    m.by_setup = setups

    # By exit reason
    reasons = {}
    for t in trades:
        r = t.exit_reason
        reasons[r] = reasons.get(r, 0) + 1
    m.by_exit_reason = reasons

    return m

def compute_portfolio_metrics(result) -> dict:
    """
    Convert a PortfolioResult into summary stats for reporting.
    Includes per-symbol and per-setup aggregation of PORTFOLIO positions
    (i.e., only trades that actually opened given equity constraints).
    """
    if not result.positions:
        return {
            "started_with": result.started_with,
            "ended_with": result.ended_with,
            "total_return_pct": 0.0,
            "total_trades": 0,
            "ruined": False,
        }

    returns_pct = np.array([p.equity_delta_pct for p in result.positions])
    usd_pnls = np.array([p.equity_delta_usd for p in result.positions])

    # Per-symbol aggregates (from the TRADES that actually executed)
    by_symbol = {}
    for p in result.positions:
        s = p.symbol
        if s not in by_symbol:
            by_symbol[s] = {
                "trades": 0, "wins": 0,
                "total_pnl_pct": 0.0, "total_pnl_usd": 0.0,
            }
        by_symbol[s]["trades"] += 1
        by_symbol[s]["wins"] += 1 if p.equity_delta_usd > 0 else 0
        by_symbol[s]["total_pnl_pct"] += p.equity_delta_pct
        by_symbol[s]["total_pnl_usd"] += p.equity_delta_usd

    for s in by_symbol:
        by_symbol[s]["win_rate"] = (
            by_symbol[s]["wins"] / by_symbol[s]["trades"]
        )
        by_symbol[s]["avg_pnl_pct"] = (
            by_symbol[s]["total_pnl_pct"] / by_symbol[s]["trades"]
        )

    total_return_pct = (
        (result.ended_with - result.started_with)
        / result.started_with * 100
    )

    wins = int(np.sum(usd_pnls > 0))
    losses = int(np.sum(usd_pnls < 0))

    return {
        "started_with": result.started_with,
        "ended_with": result.ended_with,
        "total_return_pct": total_return_pct,
        "total_return_usd": result.ended_with - result.started_with,
        "total_trades": result.total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / max(1, result.total_trades),
        "avg_pnl_pct": float(returns_pct.mean()),
        "avg_pnl_usd": float(usd_pnls.mean()),
        "best_trade_usd": float(usd_pnls.max()),
        "worst_trade_usd": float(usd_pnls.min()),
        "peak_equity": result.peak_equity,
        "trough_equity": result.trough_equity,
        "max_drawdown_pct": result.max_drawdown_pct,
        "ruined": result.ruined,
        "ruin_ts": result.ruin_ts,
        "ruin_symbol": result.ruin_symbol,
        "skipped_due_to_margin": result.skipped_due_to_margin,
        "skipped_due_to_positions": result.skipped_due_to_positions,
        "skipped_due_to_drawdown": result.skipped_due_to_drawdown,
        "peak_open_positions": result.peak_open_positions,
        "avg_open_positions": result.avg_open_positions,
        "by_symbol": by_symbol,
    }