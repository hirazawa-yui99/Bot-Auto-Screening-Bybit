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