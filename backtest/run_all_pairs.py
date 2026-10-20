"""
backtest/run_all_pairs.py
Fetch every Bybit USDT perpetual, backtest each over N days,
simulate portfolio, and optionally send results to DeepSeek for
config recommendations.

Usage:
  # Backtest only
  python -m backtest.run_all_pairs --days 30

  # Backtest + DeepSeek recommendation (saves to config.json, does not apply)
  python -m backtest.run_all_pairs --days 30 --recommend

  # Backtest + recommendation + apply changes to config.json (with backup)
  python -m backtest.run_all_pairs --days 30 --recommend --apply

  # Analyze an existing JSON result without re-running backtest
  python -m backtest.run_all_pairs --advise-only portfolio_result.json
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from backtest.data_fetcher import BybitDataFetcher
from backtest.strategy import BacktestStrategy
from backtest.simulator import TradeSimulator, SimulationConfig
from backtest.portfolio_simulator import (
    PortfolioSimulator, PortfolioConfig, PortfolioPosition,
)
from backtest.metrics import compute_metrics, compute_portfolio_metrics
from backtest.report import (
    print_report, print_portfolio_report,
    export_json, export_csv,
)
from backtest.deepseek_advisor import (
    DeepSeekAdvisor, load_config, save_config,
    format_recommendation_report,
)


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ====================================================================== #
#  Per-symbol backtest
# ====================================================================== #
async def backtest_symbol(
    fetcher: BybitDataFetcher,
    symbol: str,
    interval: str,
    days: int,
    strategy: BacktestStrategy,
    simulator: TradeSimulator,
    sem: asyncio.Semaphore,
    fetch_only: bool = False,
) -> Dict:
    async with sem:
        try:
            bundle = await fetcher.fetch_bundle(
                symbol, interval, days,
            )
        except Exception as e:
            return {"symbol": symbol, "trades": [], "error": str(e)}

    df = bundle["klines"]
    funding_df = bundle["funding"]
    oi_df = bundle["oi"]

    if df is None or len(df) < 150:
        return {
            "symbol": symbol, "trades": [],
            "error": f"insufficient_data:"
                     f"{len(df) if df is not None else 0}",
        }

    if fetch_only:
        return {"symbol": symbol, "trades": [], "error": None}

    trades = []
    next_available_bar = 100

    for i in range(100, len(df) - 1):
        if i < next_available_bar:
            continue
        try:
            signal = strategy.evaluate(
                df_1h=df,
                funding_df=funding_df,
                oi_df=oi_df,
                bar_idx=i,
                symbol=symbol,
            )
        except Exception as e:
            logging.debug(f"{symbol} eval error at {i}: {e}")
            continue

        if signal is None:
            continue

        trade = simulator.simulate(signal, df, i)
        if trade is None:
            continue

        if not hasattr(trade, "symbol") or not trade.symbol:
            trade.symbol = symbol
        trades.append(trade)

        try:
            exit_idx = df.index[df["ts"] == trade.exit_ts]
            if len(exit_idx) > 0:
                next_available_bar = int(exit_idx[0]) + 1
            else:
                next_available_bar = i + 1
        except Exception:
            next_available_bar = i + 1

    return {"symbol": symbol, "trades": trades, "error": None}


# ====================================================================== #
#  Advisor integration
# ====================================================================== #
async def run_advisor(
    config: Dict,
    pmetrics: Dict,
    result,
    config_path: str,
    apply_changes: bool,
) -> int:
    """
    Call DeepSeek with the backtest result.
    Save recommendation to config.json.backtest_recommendation.
    If apply_changes: also write recommended values to live config.
    Returns exit code (0 = success).
    """
    ds_cfg = config.get("deepseek", {})
    api_key = ds_cfg.get("api_key", "")
    if not api_key:
        print()
        print("⚠ DeepSeek advisor skipped: no api_key in config.json")
        return 0

    advisor = DeepSeekAdvisor(
        api_key=api_key,
        base_url=ds_cfg.get("base_url", "https://api.deepseek.com/v1"),
        model=ds_cfg.get("model", "deepseek-v4-pro"),
        timeout=ds_cfg.get("timeout", 60),
    )

    print()
    print("=== CALLING DEEPSEEK ADVISOR ===")
    print(f"  Model: {advisor.model}")
    print(f"  Trades to analyze: {pmetrics.get('total_trades', 0)}")
    print("  Waiting for response...")

    recommendation = await advisor.analyze(
        metrics=pmetrics,
        positions=result.positions,
        current_config=config,
    )

    if recommendation is None:
        print("  ✗ DeepSeek call failed — see log for details")
        return 1

    # Validate
    valid, rejected = advisor.validate(recommendation, config)

    # Print
    print()
    print(format_recommendation_report(recommendation, valid, rejected))

    # Build recommendation block
    rec_block = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": advisor.model,
        "backtest_summary": {
            "total_trades": pmetrics.get("total_trades", 0),
            "win_rate": pmetrics.get("win_rate", 0),
            "total_return_pct": pmetrics.get("total_return_pct", 0),
            "max_drawdown_pct": pmetrics.get("max_drawdown_pct", 0),
            "ruined": pmetrics.get("ruined", False),
        },
        "verdict": recommendation.get("verdict", "?"),
        "summary": recommendation.get("summary", ""),
        "parameter_changes": valid,
        "rejected_changes": rejected,
        "risk_warnings": recommendation.get("risk_warnings", []),
        "keep_unchanged": recommendation.get("keep_unchanged", []),
        "applied": False,
        "applied_at": None,
    }

    # Save to config.json
    config["backtest_recommendation"] = rec_block

    if apply_changes and valid:
        print(f"  Applying {len(valid)} validated changes to config.json")
        config = advisor.apply_changes(config, valid)
        rec_block["applied"] = True
        rec_block["applied_at"] = datetime.now(timezone.utc).isoformat()
        config["backtest_recommendation"] = rec_block

    save_config(config_path, config, backup=True)
    print(f"  → config.json updated (backup saved)")

    if not valid:
        print()
        print("  ℹ No parameter changes were recommended/applied")
    elif not apply_changes:
        print()
        print("  ℹ Recommendation saved to config.json.backtest_recommendation")
        print("    Run with --apply to actually update the live parameters")

    return 0


async def run_advisor_from_json(
    json_path: str,
    config_path: str,
    apply_changes: bool,
) -> int:
    """Load an existing portfolio_btc_*.json and run the advisor."""
    with open(json_path) as f:
        data = json.load(f)

    pmetrics = data.get("portfolio_metrics", {})
    positions_raw = data.get("portfolio_positions", [])

    # Reconstruct minimal PortfolioPosition objects
    class _StubTrade:
        def __init__(self, d):
            self.setup_type = d.get("setup_type", "unknown")
            self.symbol = d.get("symbol", "")

    class _StubPos:
        def __init__(self, d):
            self.symbol = d["symbol"]
            self.direction = d["direction"]
            self.entry_ts = d["entry_ts"]
            self.exit_ts = d["exit_ts"]
            self.equity_delta_usd = d["equity_delta_usd"]
            self.equity_delta_pct = d["equity_delta_pct"]
            self.trade = _StubTrade(d)

    positions = [_StubPos(d) for d in positions_raw]

    class _StubResult:
        pass

    result = _StubResult()
    result.positions = positions

    config = load_config(config_path)
    return await run_advisor(
        config, pmetrics, result, config_path, apply_changes,
    )


# ====================================================================== #
#  Main
# ====================================================================== #
async def main():
    parser = argparse.ArgumentParser(
        description="Multi-pair portfolio backtest + DeepSeek advisor",
    )
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--interval", default="1h",
                       choices=["15m", "30m", "1h", "4h"])
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--min-turnover", type=float, default=1_000_000)
    parser.add_argument("--concurrent", type=int, default=3)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    # New: advisor flags
    parser.add_argument(
        "--recommend", action="store_true",
        help="Call DeepSeek after backtest and save recommendation "
             "to config.json.backtest_recommendation",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Apply validated recommendation to live config "
             "(requires --recommend). Backup is created.",
    )
    parser.add_argument(
        "--advise-only", default=None, metavar="JSON_FILE",
        help="Skip backtest; analyze an existing JSON result file",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # ---- Path: advise-only ----
    if args.advise_only:
        if not os.path.exists(args.advise_only):
            print(f"File not found: {args.advise_only}")
            sys.exit(1)
        exit_code = await run_advisor_from_json(
            args.advise_only, args.config, args.apply,
        )
        sys.exit(exit_code)

    # ---- Path: full backtest ----
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        sys.exit(1)
    with open(args.config) as f:
        config = json.load(f)

    risk_cfg = config.get("risk", {})
    exec_cfg = config.get("execution", {})

    portfolio_cfg = PortfolioConfig(
        starting_equity_usd=100.0,
        max_position_pct=float(risk_cfg.get("max_position_pct", 2.0)),
        leverage=int(risk_cfg.get("default_leverage", 10)),
        max_open_positions=int(risk_cfg.get("max_open_positions", 5)),
        max_daily_loss_pct=float(risk_cfg.get("max_daily_loss_pct", 100.0)),
        max_drawdown_pct=float(risk_cfg.get("max_drawdown_pct", 100.0)),
    )

    strategy = BacktestStrategy(config)
    sim_config = SimulationConfig(
        market_tolerance_pct=exec_cfg.get("market_tolerance_pct", 0.15),
        max_deviation_pct=exec_cfg.get("max_deviation_pct", 4.0),
        cancel_on_rally_pct=exec_cfg.get("cancel_on_rally_pct", 2.5),
        limit_expiry_bars=exec_cfg.get("stall_timeout_candles", 8),
        max_hold_bars=exec_cfg.get("max_hold_hours", 72),
    )
    trade_sim = TradeSimulator(sim_config)

    fetcher = BybitDataFetcher()

    # Resolve symbols
    if args.symbols:
        symbols = [
            s.strip() for s in args.symbols.split(",") if s.strip()
        ]
        print(f"Using {len(symbols)} provided symbols")
    else:
        print("Fetching symbol list from Bybit...")
        symbols = await fetcher.list_all_symbols(
            min_turnover_24h=args.min_turnover,
        )
        if args.max_symbols > 0:
            symbols = symbols[:args.max_symbols]
        print(f"Resolved {len(symbols)} symbols")

    if not symbols:
        print("No symbols to backtest")
        await fetcher.close()
        return

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    start_date = start.strftime("%Y-%m-%d")
    end_date = end.strftime("%Y-%m-%d")

    print(f"Range: {start_date} → {end_date} ({args.days} days)")
    print(f"Interval: {args.interval}")
    print(
        f"Portfolio: ${portfolio_cfg.starting_equity_usd:.0f} · "
        f"{portfolio_cfg.max_position_pct}% × "
        f"{portfolio_cfg.leverage}x · "
        f"max {portfolio_cfg.max_open_positions} positions"
    )
    print()

    # Run backtests
    print(f"=== BACKTESTING {len(symbols)} SYMBOLS ===\n")
    t_start = time.time()
    sem = asyncio.Semaphore(args.concurrent)

    tasks = [
        backtest_symbol(
            fetcher, sym, args.interval, args.days,
            strategy, trade_sim, sem, fetch_only=args.fetch_only,
        )
        for sym in symbols
    ]

    all_trades = []
    per_symbol_stats = {}
    errors = []
    completed = 0

    pbar = tqdm(total=len(symbols), desc="Symbols", unit="sym")
    for coro in asyncio.as_completed(tasks):
        res = await coro
        sym = res["symbol"]
        trades = res["trades"]
        err = res["error"]
        completed += 1

        if err:
            errors.append(f"{sym}: {err}")
            pbar.set_postfix(failed=len(errors))
        else:
            if trades:
                all_trades.extend(trades)
                per_symbol_stats[sym] = {
                    "trades": len(trades),
                    "wins": sum(
                        1 for t in trades if t.net_pnl_pct > 0
                    ),
                    "total_pnl_pct": sum(
                        t.net_pnl_pct for t in trades
                    ),
                }
            pbar.set_postfix(trades=len(all_trades))

        pbar.update(1)

    pbar.close()
    elapsed = time.time() - t_start
    print(f"\nBacktest complete in {elapsed:.1f}s")
    print(f"  Symbols processed : {completed}")
    print(f"  Symbols with trades: {len(per_symbol_stats)}")
    print(f"  Total raw signals  : {len(all_trades)}")
    if errors:
        print(f"  Errors / skips     : {len(errors)}")
        for e in errors[:5]:
            print(f"    {e}")
        if len(errors) > 5:
            print(f"    ... and {len(errors)-5} more")
    print()

    await fetcher.close()

    if args.fetch_only:
        print("Fetch-only mode — no backtest performed")
        return

    if not all_trades:
        print("No trades generated across all symbols")
        return

    # Portfolio simulation
    print("=== PORTFOLIO SIMULATION ===\n")
    portfolio_sim = PortfolioSimulator(portfolio_cfg)
    result = portfolio_sim.simulate(all_trades)

    pmetrics = compute_portfolio_metrics(result)
    print_portfolio_report(
        result, pmetrics, start_date, end_date, len(symbols),
    )

    # Exports
    if args.json_out:
        export_data = {
            "config_summary": {
                "symbols": len(symbols),
                "days": args.days,
                "interval": args.interval,
                "starting_equity": portfolio_cfg.starting_equity_usd,
                "max_position_pct": portfolio_cfg.max_position_pct,
                "leverage": portfolio_cfg.leverage,
                "max_open_positions": portfolio_cfg.max_open_positions,
            },
            "portfolio_metrics": pmetrics,
            "per_symbol_raw": per_symbol_stats,
            "portfolio_positions": [
                {
                    "symbol": p.symbol,
                    "direction": p.direction,
                    "entry_ts": p.entry_ts,
                    "exit_ts": p.exit_ts,
                    "margin_usd": p.margin_usd,
                    "notional_usd": p.notional_usd,
                    "net_pnl_pct": p.net_pnl_pct,
                    "equity_delta_pct": p.equity_delta_pct,
                    "equity_delta_usd": p.equity_delta_usd,
                    "setup_type": getattr(
                        p.trade, "setup_type", "unknown",
                    ),
                }
                for p in result.positions
            ],
            "equity_curve": result.equity_curve,
        }
        with open(args.json_out, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"JSON written to {args.json_out}")

    if args.csv_out:
        export_csv([p.trade for p in result.positions], args.csv_out)

    # ---- DeepSeek advisor ----
    if args.recommend:
        # Reload config in case anything changed since we read it
        config = load_config(args.config)
        exit_code = await run_advisor(
            config=config,
            pmetrics=pmetrics,
            result=result,
            config_path=args.config,
            apply_changes=args.apply,
        )
        if exit_code != 0:
            sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())