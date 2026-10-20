# Backtest Engine

Replays the Bybit Quant Bot strategy over historical data.

## Quick start

```bash
cd /home/nazhier/Bot-Auto-Screening-Bybit
source venv/bin/activate
pip install tqdm

# Backtest last 30 days of BTCUSDT
python -m backtest.run_backtest --symbol BTCUSDT --days 30

# Backtest with JSON export
python -m backtest.run_backtest \
    --symbol ETHUSDT --days 14 \
    --json-out eth_result.json \
    --csv-out eth_trades.csv

# Send result to Discord
python -m backtest.run_backtest \
    --symbol SOLUSDT --days 30 \
    --discord "https://discord.com/api/webhooks/..."

# Use a local CSV
python -m backtest.run_backtest \
    --symbol TEST --csv tests/data/BTCUSDT_60.csv
```

## What's simulated

| Component | Realistic | Notes |
|---|---|---|
| OHLCV | ✅ | Real Bybit kline data |
| Funding rate | ✅ | Historical funding history |
| Open interest | ✅ | Historical OI snapshots |
| CVD | ⚠️ | Approximated from kline direction × volume |
| Orderbook heatmap | ❌ | No historical data — neutral contribution |
| Live L/S ratio | ❌ | Not available — skipped in agreement gate |
| SMC (BOS/CHoCH) | ⚠️ | Simplified swing-break proxy |
| Fibonacci zones | ✅ | Real computation from OHLCV |
| Momentum guard | ✅ | Full direction-aware + sigma check |
| ATR filter | ✅ | Full band check |
| Setup classifier | ⚠️ | All timeframes use 1h data (proxy) |
| Agreement gate | ✅ | Requires 2-3 of {smc, deriv, quant, fib} |
| Confidence scorer | ✅ | Cold-start calibration |
| Fill simulation | ✅ | Taker + maker fees, slippage, rally cancel |
| TP/SL walk | ✅ | Intrabar high/low, conservative SL-first |
| Stall detection | ✅ | 8-bar limit expiry |

## Fees and slippage

Default assumptions:
- Taker fee: 0.055% (market entry, SL exit)
- Maker fee: 0.020% (limit entry, TP fills)
- Slippage: 0.02% on market orders

To change, edit `SimulationConfig` in `backtest/simulator.py` or extend
`run_backtest.py` to accept CLI flags.

## Interpreting results

**Key metrics to watch:**

- **Win rate**: 40–60% is normal for this strategy
- **Profit factor**: >1.5 is healthy, >2.0 is excellent
- **Max drawdown**: should be < 15% for a $100 account
- **Sharpe**: >1.0 annualized is tradeable, >2.0 is very good
- **Avg bars held**: if >24, your TPs may be too far

**Red flags:**

- Win rate > 80% with avg win < 0.5% → tiny gains, hidden risk
- Profit factor < 1.0 → negative expectancy
- Max drawdown > 20% → position sizing too aggressive
- 0 trades in 30 days → thresholds too strict

## What the backtest does NOT capture

1. **Slippage on illiquid symbols** — uses fixed 0.02% regardless of volume
2. **Exchange downtime** — assumes Bybit is always up
3. **API failures** — assumes order placement always succeeds
4. **Funding paid during hold** — funding is applied at signal time only
5. **Partial fills** — simulated as all-or-nothing per TP level
6. **Concurrent positions** — backtests one symbol at a time

For a multi-symbol portfolio backtest, run each symbol separately and
aggregate the `total_pnl_pct` values weighted by allocation %.

## Files

- `data_fetcher.py` — REST pulls with pagination + parquet cache
- `strategy.py` — production-equivalent signal logic
- `simulator.py` — bar-by-bar trade execution
- `metrics.py` — Sharpe, drawdown, per-setup breakdown
- `report.py` — text report + JSON/CSV export + Discord
- `run_backtest.py` — CLI entry point