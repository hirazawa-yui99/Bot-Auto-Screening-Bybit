"""
backtest/run_backtest.py
CLI entry point for the backtester.

Usage:
  python -m backtest.run_backtest --symbol BTCUSDT --days 30
  python -m backtest.run_backtest --symbol ETHUSDT --days 14 --interval 1h
  python -m backtest.run_backtest --symbol SOLUSDT --config config.json
  python -m backtest.run_backtest --symbol BTCUSDT --csv path/to/data.csv

Modes:
  --symbol       Fetch live data from Bybit (network required)
  --csv          Use a local CSV file instead
  --days         Lookback period (default 30)
  --interval     1h default, also 15m, 4h, etc.
  --config       Path to config.json (default: ./config.json)
  --min-conf     Override min confidence (default: from config)
  --json-out     Export trades to JSON
  --csv-out      Export trades to CSV
  --discord      Send summary to Discord webhook
  --verbose      Debug logging
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
from tqdm import tqdm

from backtest.data_fetcher import BybitDataFetcher
from backtest.strategy import BacktestStrategy
from backtest.simulator import TradeSimulator, SimulationConfig
from backtest.metrics import compute_metrics
from backtest.report import (
    print_report, export_json, export_csv, send_discord_report,
)


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def load_data(args) -> dict:
    """Load klines + funding + OI from Bybit or CSV."""
    if args.csv:
        print(f"Loading CSV: {args.csv}")
        df = pd.read_csv(args.csv)
        # Normalize columns
        renames = {
            "start": "ts", "startTime": "ts", "start_time": "ts",
            "timestamp": "ts", "o": "open", "h": "high",
            "l": "low", "c": "close", "v": "volume",
        }
        df = df.rename(columns={k: v for k, v in renames.items()
                                if k in df.columns})
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("ts").reset_index(drop=True)
        return {"klines": df, "funding": pd.DataFrame(),
                "oi": pd.DataFrame()}

    fetcher = BybitDataFetcher()
    try:
        bundle = await fetcher.fetch_bundle(
            args.symbol, args.interval, args.days,
        )
    finally:
        await fetcher.close()
    return bundle


def run_backtest(
    df: pd.DataFrame,
    funding_df: pd.DataFrame,
    oi_df: pd.DataFrame,
    symbol: str,
    interval: str,
    config: dict,
) -> list:
    """
    Walk forward through df, evaluating the strategy at each bar.
    Collects all Trade objects.
    """
    strategy = BacktestStrategy(config)
    sim_config = SimulationConfig(
        market_tolerance_pct=config.get("execution", {}).get(
            "market_tolerance_pct", 0.15,
        ),
        max_deviation_pct=config.get("execution", {}).get(
            "max_deviation_pct", 4.0,
        ),
        cancel_on_rally_pct=config.get("execution", {}).get(
            "cancel_on_rally_pct", 2.5,
        ),
        limit_expiry_bars=config.get("execution", {}).get(
            "stall_timeout_candles", 8,
        ),
        max_hold_bars=config.get("execution", {}).get(
            "max_hold_hours", 72,
        ),
    )
    simulator = TradeSimulator(sim_config)

    trades = []
    # Track open trade end bar to prevent overlapping entries on same symbol
    next_available_bar = 100

    bars = range(100, len(df) - 1)
    pbar = tqdm(bars, desc=f"Backtest {symbol}", unit="bar")

    for i in pbar:
        if i < next_available_bar:
            continue

        signal = strategy.evaluate(
            df_1h=df,
            funding_df=funding_df,
            oi_df=oi_df,
            bar_idx=i,
            symbol=symbol,
        )
        if signal is None:
            continue

        trade = simulator.simulate(signal, df, i)
        if trade is None:
            continue

        trades.append(trade)
        # Block re-entry until this trade closes
        next_available_bar = trade.entry_ts and (
            df.index[df["ts"] == trade.exit_ts][0] + 1
            if trade.exit_ts in df["ts"].values else i + 1
        )
        pbar.set_postfix(
            trades=len(trades),
            pnl=f"{sum(t.net_pnl_pct for t in trades):+.1f}%",
        )

    return trades


async def main():
    parser = argparse.ArgumentParser(
        description="Backtest Bybit Quant Bot strategy",
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h",
                       choices=["1m", "5m", "15m", "30m",
                                "1h", "4h", "1d"])
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--discord", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Load config
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        sys.exit(1)
    with open(args.config) as f:
        config = json.load(f)

    # Fetch data
    print(f"\n=== FETCHING DATA ===")
    bundle = await load_data(args)
    df = bundle["klines"]
    funding_df = bundle["funding"]
    oi_df = bundle["oi"]

    if df is None or len(df) < 150:
        print(
            f"Not enough data: {len(df) if df is not None else 0} bars "
            f"(need >= 150)"
        )
        sys.exit(1)

    start_date = datetime.utcfromtimestamp(
        int(df["ts"].iloc[0]) / 1000
    ).strftime("%Y-%m-%d")
    end_date = datetime.utcfromtimestamp(
        int(df["ts"].iloc[-1]) / 1000
    ).strftime("%Y-%m-%d")

    print(
        f"\nData: {len(df)} bars · "
        f"{start_date} → {end_date}"
    )
    print(f"Funding rows: {len(funding_df)}")
    print(f"OI rows: {len(oi_df)}")
    print()

    # Run backtest
    print("=== RUNNING BACKTEST ===\n")
    trades = run_backtest(
        df, funding_df, oi_df,
        args.symbol, args.interval, config,
    )

    # Compute metrics
    metrics = compute_metrics(trades)

    # Print report
    print("\n")
    print_report(
        metrics, trades,
        args.symbol, args.interval,
        start_date, end_date,
    )

    # Exports
    if args.json_out:
        export_json(metrics, trades, args.json_out)
    if args.csv_out:
        export_csv(trades, args.csv_out)

    # Discord
    if args.discord and metrics.total_trades > 0:
        await send_discord_report(
            args.discord, metrics,
            args.symbol, args.interval,
            start_date, end_date,
        )


if __name__ == "__main__":
    asyncio.run(main())