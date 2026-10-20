"""
backtest/report.py
Human-readable and machine-readable backtest reports.
"""
import json
from datetime import datetime
from typing import List, Optional

from backtest.metrics import BacktestMetrics


def print_report(
    metrics: BacktestMetrics,
    trades: List,
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
):
    """Print a complete backtest report to stdout."""
    print("=" * 76)
    print(f"  BACKTEST REPORT — {symbol} {interval}")
    print(f"  {start_date} → {end_date}")
    print(f"  Generated: {datetime.utcnow().isoformat()} UTC")
    print("=" * 76)
    print()

    if metrics.total_trades == 0:
        print("  No trades executed.")
        print()
        print("  Possible reasons:")
        print("    - All signals filtered by momentum guard / ATR / agreement")
        print("    - Data too short (need >= 100 bars before first signal)")
        print("    - min_raw_threshold or min_calibrated too strict")
        print()
        return

    print("PERFORMANCE")
    print("-" * 76)
    print(f"  Total trades      : {metrics.total_trades}")
    print(f"  Wins / Losses     : {metrics.wins} / {metrics.losses}"
          f" (+{metrics.breakeven} breakeven)")
    print(f"  Win rate          : {metrics.win_rate:.1%}")
    print(f"  Total PnL         : {metrics.total_pnl_pct:+.2f}%")
    print(f"  Avg PnL per trade : {metrics.avg_pnl_pct:+.2f}%")
    print(f"  Avg win           : {metrics.avg_win_pct:+.2f}%")
    print(f"  Avg loss          : {metrics.avg_loss_pct:+.2f}%")
    if metrics.profit_factor == float("inf"):
        print(f"  Profit factor     : ∞ (no losses)")
    else:
        print(f"  Profit factor     : {metrics.profit_factor:.2f}")
    print()

    print("RISK")
    print("-" * 76)
    print(f"  Max drawdown      : {metrics.max_drawdown_pct:+.2f}%")
    print(f"  Sharpe (ann.)     : {metrics.sharpe:.2f}")
    print(f"  Sortino (ann.)    : {metrics.sortino:.2f}")
    print(f"  Max consec losses : {metrics.max_consecutive_losses}")
    print(f"  Best trade        : {metrics.best_trade_pct:+.2f}%")
    print(f"  Worst trade       : {metrics.worst_trade_pct:+.2f}%")
    print()

    print("TIMING")
    print("-" * 76)
    print(f"  Avg bars held     : {metrics.avg_bars_held:.1f}")
    if interval == "1h":
        print(f"  Avg hours held    : {metrics.avg_bars_held:.1f}")
    print()

    if metrics.by_setup:
        print("BY SETUP TYPE")
        print("-" * 76)
        print(f"  {'Setup':<12} {'Trades':>8} {'Win%':>8} "
              f"{'Avg PnL':>10} {'Total PnL':>12}")
        print("  " + "-" * 70)
        for setup, data in metrics.by_setup.items():
            print(
                f"  {setup:<12} {data['trades']:>8} "
                f"{data['win_rate']:>7.1%} "
                f"{data['avg_pnl']:>+9.2f}% "
                f"{data['total_pnl']:>+11.2f}%"
            )
        print()

    if metrics.by_exit_reason:
        print("BY EXIT REASON")
        print("-" * 76)
        for reason, count in sorted(
            metrics.by_exit_reason.items(),
            key=lambda x: -x[1],
        ):
            pct = count / metrics.total_trades * 100
            print(f"  {reason:<20} {count:>5} ({pct:>5.1f}%)")
        print()

    # Recent trades
    recent = sorted(trades, key=lambda t: t.entry_ts)[-10:]
    if recent:
        print("LAST 10 TRADES")
        print("-" * 76)
        print(
            f"  {'Entry TS':<12} {'Dir':<6} {'Setup':<10} "
            f"{'Conf':>6} {'TPs':>4} {'PnL':>8} {'Exit':<12}"
        )
        print("  " + "-" * 70)
        for t in recent:
            ts = datetime.utcfromtimestamp(
                t.entry_ts / 1000
            ).strftime("%m-%d %H:%M")
            print(
                f"  {ts:<12} {t.direction:<6} {t.setup_type:<10} "
                f"{t.confidence:>5.1%} {t.tp_hits:>3}/3 "
                f"{t.net_pnl_pct:>+7.2f}% {t.exit_reason:<12}"
            )
        print()

    # Equity curve text sparkline
    if metrics.equity_curve:
        print("EQUITY CURVE (cumulative %)")
        print("-" * 76)
        eq = metrics.equity_curve
        n_points = min(60, len(eq))
        step = max(1, len(eq) // n_points)
        sampled = eq[::step]
        mn, mx = min(sampled), max(sampled)
        rows = []
        for row in range(10, 0, -1):
            threshold = mn + (mx - mn) * row / 10
            line = "".join(
                "█" if v >= threshold else " "
                for v in sampled
            )
            rows.append(f"  {threshold:>+6.2f}% │{line}")
        print("\n".join(rows))
        print("          └" + "─" * len(sampled))
        print()

    print("=" * 76)


def export_json(
    metrics: BacktestMetrics,
    trades: List,
    path: str,
):
    """Dump trades + metrics to JSON for further analysis."""
    data = {
        "metrics": {
            "total_trades": metrics.total_trades,
            "wins": metrics.wins,
            "losses": metrics.losses,
            "win_rate": metrics.win_rate,
            "total_pnl_pct": metrics.total_pnl_pct,
            "avg_pnl_pct": metrics.avg_pnl_pct,
            "profit_factor": (
                None if metrics.profit_factor == float("inf")
                else metrics.profit_factor
            ),
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "sharpe": metrics.sharpe,
            "sortino": metrics.sortino,
            "by_setup": metrics.by_setup,
            "by_exit_reason": metrics.by_exit_reason,
        },
        "trades": [
            {
                "signal_ts": t.signal_ts,
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "direction": t.direction,
                "setup_type": t.setup_type,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "sl": t.sl,
                "tps": t.tps,
                "tp_hits": t.tp_hits,
                "confidence": t.confidence,
                "gross_pnl_pct": t.gross_pnl_pct,
                "net_pnl_pct": t.net_pnl_pct,
                "fees_pct": t.fees_pct,
                "bars_held": t.bars_held,
                "exit_reason": t.exit_reason,
            }
            for t in trades
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"JSON written to {path}")


def export_csv(trades: List, path: str):
    """Export trade log to CSV."""
    import csv
    if not trades:
        print("No trades to export")
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "signal_ts", "entry_ts", "exit_ts", "direction",
            "setup_type", "entry_price", "exit_price",
            "sl", "tp1", "tp2", "tp3", "tp_hits",
            "confidence", "gross_pnl_pct", "net_pnl_pct",
            "fees_pct", "bars_held", "exit_reason",
        ])
        for t in trades:
            writer.writerow([
                t.signal_ts, t.entry_ts, t.exit_ts, t.direction,
                t.setup_type, t.entry_price, t.exit_price,
                t.sl, t.tps[0], t.tps[1], t.tps[2], t.tp_hits,
                t.confidence, t.gross_pnl_pct, t.net_pnl_pct,
                t.fees_pct, t.bars_held, t.exit_reason,
            ])
    print(f"CSV written to {path}")


async def send_discord_report(
    webhook_url: str,
    metrics: BacktestMetrics,
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
):
    """Send a summary embed to Discord."""
    import aiohttp

    if metrics.total_trades == 0:
        desc = "No trades executed in this backtest period."
        color = 0x808080
    else:
        pnl = metrics.total_pnl_pct
        color = 0x3BA55D if pnl >= 0 else 0xED4245
        desc = (
            f"**{symbol} {interval}** — {start_date} → {end_date}\n"
            f"Trades: {metrics.total_trades} · "
            f"Win rate: {metrics.win_rate:.1%} · "
            f"PnL: **{pnl:+.2f}%**"
        )

    fields = []
    if metrics.total_trades > 0:
        fields = [
            {"name": "Wins/Losses",
             "value": f"{metrics.wins} / {metrics.losses}",
             "inline": True},
            {"name": "Avg PnL",
             "value": f"{metrics.avg_pnl_pct:+.2f}%",
             "inline": True},
            {"name": "Profit Factor",
             "value": (
                 "∞" if metrics.profit_factor == float("inf")
                 else f"{metrics.profit_factor:.2f}"
             ),
             "inline": True},
            {"name": "Max DD",
             "value": f"{metrics.max_drawdown_pct:+.2f}%",
             "inline": True},
            {"name": "Sharpe (ann.)",
             "value": f"{metrics.sharpe:.2f}",
             "inline": True},
            {"name": "Avg bars held",
             "value": f"{metrics.avg_bars_held:.1f}",
             "inline": True},
        ]
        for setup, data in list(metrics.by_setup.items())[:4]:
            fields.append({
                "name": f"Setup: {setup}",
                "value": (
                    f"{data['trades']} trades · "
                    f"{data['win_rate']:.0%} WR · "
                    f"{data['total_pnl']:+.2f}%"
                ),
                "inline": True,
            })

    embed = {
        "title": f"📊 Backtest · {symbol} {interval}",
        "color": color,
        "description": desc,
        "fields": fields,
        "footer": {
            "text": f"Bybit Quant v9.2 backtest · "
                    f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
        },
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            webhook_url, json={"embeds": [embed]},
        ) as resp:
            if resp.status >= 400:
                print(
                    f"Discord error {resp.status}: "
                    f"{await resp.text()}"
                )
            else:
                print("Discord report sent")

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
