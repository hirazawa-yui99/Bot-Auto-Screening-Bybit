"""
backtest/portfolio_simulator.py
Multi-symbol portfolio simulator with equity tracking.

Models realistic margin constraints:
  - Each trade uses `max_position_pct%` of CURRENT equity as margin
  - Leverage multiplies notional: notional = margin × leverage
  - Total margin used cannot exceed equity
  - Positions are processed in signal-timestamp order
  - Equity compounds on PnL; if equity <= 0, portfolio is ruined

Reports:
  - Ending equity, peak equity, max drawdown
  - Ruin event (if any)
  - Per-symbol stats
  - Position allocation stats (how many concurrent trades typically held)
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    starting_equity_usd: float = 100.0
    max_position_pct: float = 2.0
    leverage: int = 10
    max_open_positions: int = 5
    max_daily_loss_pct: float = 100.0       # 100 = disabled
    max_drawdown_pct: float = 100.0         # 100 = disabled


@dataclass
class PortfolioPosition:
    symbol: str
    direction: str
    entry_ts: int
    exit_ts: int
    margin_usd: float
    notional_usd: float
    net_pnl_pct: float                      # % of notional
    equity_delta_pct: float                 # % of portfolio equity
    equity_delta_usd: float
    trade: object = None                    # original Trade object


@dataclass
class PortfolioResult:
    started_with: float = 0.0
    ended_with: float = 0.0
    peak_equity: float = 0.0
    trough_equity: float = 0.0
    max_drawdown_pct: float = 0.0
    ruined: bool = False
    ruin_ts: Optional[int] = None
    ruin_symbol: Optional[str] = None
    total_trades: int = 0
    skipped_due_to_margin: int = 0
    skipped_due_to_positions: int = 0
    skipped_due_to_drawdown: int = 0
    positions: List[PortfolioPosition] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
    daily_equity: Dict = field(default_factory=dict)
    peak_open_positions: int = 0
    avg_open_positions: float = 0.0


class PortfolioSimulator:
    def __init__(self, config: PortfolioConfig):
        self.cfg = config

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #
    def simulate(self, all_trades: List) -> PortfolioResult:
        """
        Merge trades from all symbols, sort by entry timestamp,
        process in equity-constrained order.

        Args:
            all_trades: flat list of Trade objects from all symbols
                        (already simulated per-symbol)
        """
        r = PortfolioResult(started_with=self.cfg.starting_equity_usd)
        equity = self.cfg.starting_equity_usd
        r.peak_equity = equity
        r.trough_equity = equity

        if not all_trades:
            r.ended_with = equity
            return r

        # Sort by entry timestamp
        trades = sorted(all_trades, key=lambda t: t.entry_ts)

        # Track open positions as list of dicts with exit_ts for freeing
        open_positions: List[Dict] = []
        margin_used = 0.0
        peak_positions = 0
        positions_count_sum = 0
        positions_count_n = 0
        day_start_equity = equity
        day_key = None

        for trade in trades:
            # Free any positions that closed before this entry
            still_open = []
            for p in open_positions:
                if p["exit_ts"] <= trade.entry_ts:
                    # Position closes: apply equity delta
                    equity += p["equity_delta_usd"]
                    margin_used -= p["margin_usd"]
                    # Record equity curve point
                    r.equity_curve.append({
                        "ts": p["exit_ts"],
                        "equity": equity,
                        "symbol": p["symbol"],
                        "pnl_usd": p["equity_delta_usd"],
                    })
                    # Update peak / trough
                    if equity > r.peak_equity:
                        r.peak_equity = equity
                    if equity < r.trough_equity:
                        r.trough_equity = equity
                    # Ruin check
                    if equity <= 0 and not r.ruined:
                        r.ruined = True
                        r.ruin_ts = p["exit_ts"]
                        r.ruin_symbol = p["symbol"]
                else:
                    still_open.append(p)
            open_positions = still_open

            if r.ruined:
                break

            # Update open position count stats
            n_open = len(open_positions)
            positions_count_sum += n_open
            positions_count_n += 1
            if n_open > peak_positions:
                peak_positions = n_open

            # Daily bookkeeping
            import datetime as _dt
            try:
                ts_dt = _dt.datetime.utcfromtimestamp(trade.entry_ts / 1000)
                day = ts_dt.strftime("%Y-%m-%d")
                if day != day_key:
                    day_key = day
                    day_start_equity = equity
                # Daily loss check
                day_loss_pct = (equity - day_start_equity) / day_start_equity * 100
                if -day_loss_pct >= self.cfg.max_daily_loss_pct:
                    r.skipped_due_to_drawdown += 1
                    continue
            except Exception:
                pass

            # Drawdown check
            if self.cfg.max_drawdown_pct < 100:
                dd = (r.peak_equity - equity) / r.peak_equity * 100
                if dd >= self.cfg.max_drawdown_pct:
                    r.skipped_due_to_drawdown += 1
                    continue

            # Position count cap
            if n_open >= self.cfg.max_open_positions:
                r.skipped_due_to_positions += 1
                continue

            # Margin calculation
            margin_needed = self.cfg.max_position_pct / 100.0 * equity
            if margin_needed <= 0:
                r.skipped_due_to_margin += 1
                continue
            if margin_used + margin_needed > equity:
                r.skipped_due_to_margin += 1
                continue

            notional = margin_needed * self.cfg.leverage

            # Equity delta from this trade
            # trade.net_pnl_pct is % of notional
            equity_delta_pct = (
                trade.net_pnl_pct * self.cfg.leverage
                * self.cfg.max_position_pct / 100.0
            )
            equity_delta_usd = margin_needed * (trade.net_pnl_pct / 100.0) * self.cfg.leverage
            # Sanity: equity_delta_usd should be notional × net_pnl_pct%
            expected_delta = notional * (trade.net_pnl_pct / 100.0)
            # equity_delta_usd computed above is the same value
            # (margin_needed × net_pnl_pct% × leverage = notional × net_pnl_pct%)
            equity_delta_usd = expected_delta

            pos = PortfolioPosition(
                symbol=trade.symbol if hasattr(trade, "symbol") else "?",
                direction=trade.direction,
                entry_ts=trade.entry_ts,
                exit_ts=trade.exit_ts,
                margin_usd=margin_needed,
                notional_usd=notional,
                net_pnl_pct=trade.net_pnl_pct,
                equity_delta_pct=equity_delta_pct,
                equity_delta_usd=equity_delta_usd,
                trade=trade,
            )
            open_positions.append({
                "symbol": pos.symbol,
                "exit_ts": trade.exit_ts,
                "margin_usd": margin_needed,
                "equity_delta_usd": equity_delta_usd,
            })
            margin_used += margin_needed

            r.positions.append(pos)
            r.total_trades += 1

        # Close remaining positions at their exit_ts (end of simulation)
        for p in open_positions:
            equity += p["equity_delta_usd"]
            margin_used -= p["margin_usd"]
            r.equity_curve.append({
                "ts": p["exit_ts"],
                "equity": equity,
                "symbol": p["symbol"],
                "pnl_usd": p["equity_delta_usd"],
            })
            if equity > r.peak_equity:
                r.peak_equity = equity
            if equity < r.trough_equity:
                r.trough_equity = equity

        r.ended_with = equity
        r.peak_open_positions = peak_positions
        r.avg_open_positions = (
            positions_count_sum / positions_count_n
            if positions_count_n else 0.0
        )
        if r.peak_equity > 0:
            r.max_drawdown_pct = (
                (r.trough_equity - r.peak_equity) / r.peak_equity * 100
            )

        return r

def print_portfolio_report(
    result,
    metrics: dict,
    start_date: str,
    end_date: str,
    num_symbols: int,
):
    """Print portfolio-level backtest report."""
    print()
    print("=" * 78)
    print(f"  PORTFOLIO BACKTEST REPORT")
    print(f"  {start_date} → {end_date}  ({num_symbols} symbols)")
    print("=" * 78)
    print()

    if metrics["total_trades"] == 0:
        print("  No trades executed.")
        print()
        print(f"  Skipped due to margin     : {result.skipped_due_to_margin}")
        print(f"  Skipped due to positions  : {result.skipped_due_to_positions}")
        print(f"  Skipped due to drawdown   : {result.skipped_due_to_drawdown}")
        print()
        return

    # Ruin warning front and center
    if metrics["ruined"]:
        print("  ╔══════════════════════════════════════════════════════════╗")
        print("  ║  ⚠️  PORTFOLIO RUINED                                    ║")
        print("  ║  Equity reached $0 — all subsequent signals skipped      ║")
        if metrics.get("ruin_symbol"):
            print(f"  ║  Killed by: {metrics['ruin_symbol']:<44}║")
        print("  ╚══════════════════════════════════════════════════════════╝")
        print()

    print("EQUITY")
    print("-" * 78)
    print(f"  Started with      : ${metrics['started_with']:,.2f}")
    print(f"  Ended with        : ${metrics['ended_with']:,.2f}")
    print(f"  Total return      : {metrics['total_return_pct']:+.2f}% "
          f"(${metrics['total_return_usd']:+,.2f})")
    print(f"  Peak equity       : ${metrics['peak_equity']:,.2f}")
    print(f"  Trough equity     : ${metrics['trough_equity']:,.2f}")
    print(f"  Max drawdown      : {metrics['max_drawdown_pct']:+.2f}%")
    print()

    print("TRADING")
    print("-" * 78)
    print(f"  Trades executed   : {metrics['total_trades']}")
    print(f"  Wins / Losses     : {metrics['wins']} / {metrics['losses']}")
    print(f"  Win rate          : {metrics['win_rate']:.1%}")
    print(f"  Avg trade USD     : ${metrics['avg_pnl_usd']:+.4f}")
    print(f"  Best trade        : ${metrics['best_trade_usd']:+.4f}")
    print(f"  Worst trade       : ${metrics['worst_trade_usd']:+.4f}")
    print()

    print("CONSTRAINTS")
    print("-" * 78)
    print(f"  Peak open positions    : {metrics['peak_open_positions']}")
    print(f"  Avg open positions     : {metrics['avg_open_positions']:.1f}")
    print(f"  Skipped — margin       : {metrics['skipped_due_to_margin']}")
    print(f"  Skipped — max positions: {metrics['skipped_due_to_positions']}")
    print(f"  Skipped — drawdown     : {metrics['skipped_due_to_drawdown']}")
    print()

    by_symbol = metrics.get("by_symbol", {})
    if by_symbol:
        # Top 10 winners
        sorted_by_pnl = sorted(
            by_symbol.items(),
            key=lambda x: x[1]["total_pnl_usd"],
            reverse=True,
        )
        print("TOP 10 SYMBOLS BY PnL")
        print("-" * 78)
        print(f"  {'Symbol':<16} {'Trades':>7} {'Win%':>7} "
              f"{'PnL USD':>12} {'PnL %':>9}")
        print("  " + "-" * 60)
        for sym, d in sorted_by_pnl[:10]:
            print(
                f"  {sym:<16} {d['trades']:>7} "
                f"{d['win_rate']:>6.1%} "
                f"{d['total_pnl_usd']:>+11.4f} "
                f"{d['total_pnl_pct']:>+8.2f}%"
            )
        print()

        # Worst 10
        print("BOTTOM 10 SYMBOLS BY PnL")
        print("-" * 78)
        print(f"  {'Symbol':<16} {'Trades':>7} {'Win%':>7} "
              f"{'PnL USD':>12} {'PnL %':>9}")
        print("  " + "-" * 60)
        for sym, d in sorted_by_pnl[-10:]:
            print(
                f"  {sym:<16} {d['trades']:>7} "
                f"{d['win_rate']:>6.1%} "
                f"{d['total_pnl_usd']:>+11.4f} "
                f"{d['total_pnl_pct']:>+8.2f}%"
            )
        print()

    # Equity curve text chart
    if result.equity_curve:
        print("EQUITY CURVE (USD)")
        print("-" * 78)
        eq_values = [p["equity"] for p in result.equity_curve]
        if len(eq_values) > 5:
            n_points = min(60, len(eq_values))
            step = max(1, len(eq_values) // n_points)
            sampled = eq_values[::step]
            mn, mx = min(sampled), max(sampled)
            if mx == mn:
                mx = mn + 1
            rows = []
            for row in range(10, 0, -1):
                threshold = mn + (mx - mn) * row / 10
                line = "".join(
                    "█" if v >= threshold else " "
                    for v in sampled
                )
                rows.append(f"  ${threshold:>8.2f} │{line}")
            print("\n".join(rows))
            print("             └" + "─" * len(sampled))
        print()

    print("=" * 78)