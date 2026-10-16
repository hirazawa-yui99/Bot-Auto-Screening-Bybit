# Bybit Quant Bot v9.1

<div align="center">

**Agentic quantitative futures trading bot for Bybit V5**

Scans every USDT perpetual · Zeta mean-reversion · Basis spread · Volume anomaly · CVD/OI/L-S · DeepSeek agentic confirmation

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Bybit V5](https://img.shields.io/badge/Bybit-V5%20Unified-yellow.svg)](https://bybit-exchange.github.io/docs/v5/intro)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-purple.svg)](https://api.deepseek.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## Why v9.1 Exists

The previous version (`v8.1`) stopped producing signals for **4 months** due to compounding design faults:

| # | Fault | Impact |
|---|---|---|
| 1 | Silent crash in threaded scan loop | Scheduler died without recovery |
| 2 | Over-filtering (7+ hard filters stacked multiplicatively) | Pass rate < 0.1% of scanned pairs |
| 3 | Deprecated V2 API endpoints (`/v2/public/account-ratio`) | Silent failures |
| 4 | Binary pass/fail scoring with no calibration | No confidence gradient |
| 5 | No auto-recovery for Discord/DB failures | Bot halts on transient errors |
| 6 | No feedback loop from outcomes | Parameters never adapted |

**v9.1 addresses every one of these** while adding Zeta mean-reversion, basis spread analysis, volume anomaly detection, agentic DeepSeek reasoning, and a statistically honest auto-improvement loop.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Bot](#running-the-bot)
- [How the Scanner Works](#how-the-scanner-works)
- [Quantitative Methods](#quantitative-methods)
- [Signal Confidence Gate](#signal-confidence-gate)
- [Auto-Improvement Loop](#auto-improvement-loop)
- [Risk Management](#risk-management)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)

---

## Features

### Core Upgrades over v8.1

- ✅ **Bybit V5 Unified API** — REST + WebSocket Trade API for order placement
- ✅ **Full-pair scanner** — paginates `/v5/market/instruments-info` and scans **every active USDT perpetual** on Bybit
- ✅ **Async architecture** — no more thread crashes; graceful degradation with heartbeat monitoring
- ✅ **Calibrated confidence scoring** — Bayesian shrinkage toward 0.5 until enough live outcomes accumulate
- ✅ **Agentic DeepSeek** — tool-calling LLM that requests additional data on demand

### New Quantitative Modules

- 📊 **Zeta mean-reversion** — Ornstein-Uhlenbeck fitting + half-life + Hurst exponent + ADF stationarity test
- 📊 **Basis spread rate** — spot vs perp + annualized funding carry
- 📊 **Volume anomaly detection** — robust Z-score (median/MAD) + percentile rank + volume-weighted price divergence
- 📊 **CVD, OI, L/S ratios** — V5 endpoints with top-trader vs retail divergence
- 📊 **Composite scoring** — fuses 5 signals into `[−1, +1]` with agreement ratio

### Execution & Risk

- 🔒 **WebSocket order placement** with native TP/SL attached at entry
- 🔒 **Real-time position monitor** — trailing stops, emergency exits, max hold time
- 🔒 **Risk manager** — position sizing, daily loss limit, max drawdown kill switch
- 🔒 **Emergency close** on adverse 2.5% move

### Learning & Adaptation

- 🧠 **Daily 07:00 auto-improvement** — compares signals to outcomes, tunes parameters
- 🧠 **Anti-overfitting guards** — walk-forward split, Bonferroni correction, monotone safety, deflated Sharpe
- 🧠 **Full PostgreSQL persistence** — signals, outcomes, tuning history, positions
- 🧠 **Discord notifications** — rich embeds with full breakdowns

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         main.py (orchestrator)                      │
└─────────────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Scan Loop    │   │  Daily 07:00  │   │  Position     │
│  (hourly)     │   │  Tuner        │   │  Monitor (3s) │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BybitV5Client  ←→  REST + WebSocket Trade/Private                  │
│  get_all_linear_symbols() · get_klines() · place_order_ws()         │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Analysis Layer                                                      │
│  ┌────────────┐ ┌────────────┐ ┌──────────────┐ ┌────────────────┐  │
│  │Technicals  │ │Derivatives │ │MeanReversion │ │Basis / Volume  │  │
│  │EMA RSI ATR │ │CVD OI L/S  │ │Zeta OU Hurst │ │Basis Anomaly   │  │
│  └────────────┘ └────────────┘ └──────────────┘ └────────────────┘  │
│                          │                                           │
│                          ▼                                           │
│              ┌───────────────────────┐                               │
│              │ CompositeQuantEngine  │  → [−1, +1] score             │
│              └───────────────────────┘                               │
│                          │                                           │
│                          ▼                                           │
│              ┌───────────────────────┐                               │
│              │ ConfidenceScorer      │  → high-confidence gate       │
│              └───────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Agent Layer                                                         │
│  DeepSeekAgent → structured JSON (direction, invalidation, risks)   │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Execution Layer                                                     │
│  RiskManager → TradeExecutor → WS order + native TP/SL              │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Learning Layer                                                      │
│  SignalJournal → OutcomeTracker → AutoTuner → config.json           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- **Python 3.10+**
- **PostgreSQL 13+**
- **Bybit V5 API key** (Unified Trading account, Read + Trade permissions)
- **DeepSeek API key** (from https://platform.deepseek.com/)
- **Discord webhook URL** (optional, for alerts)

### Step 1 — Clone & Environment

```bash
git clone https://github.com/youruser/bybit_quant_v9.git
cd bybit_quant_v9

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Database

```bash
createdb bybit_bot_v9
```

### Step 4 — Package Markers

```bash
touch core/__init__.py analysis/__init__.py agent/__init__.py \
      execution/__init__.py learning/__init__.py \
      persistence/__init__.py notification/__init__.py
```

### Step 5 — Configuration

```bash
cp config.json.example config.json
cp .env.example .env
```

Fill in your API keys in both files. See [Configuration](#configuration).

### Step 6 — First Run (Testnet)

Set `"testnet": true` in `config.json`, then:

```bash
python main.py
```

Watch the log:

```bash
tail -f bot_v9.log
```

You should see:

```
[INFO] main: Bybit Quant Bot v9.1 starting — FULL-PAIR SCANNER
[INFO] core.bybit_client: Instruments page 1: 500 symbols (total so far: 500)
[INFO] core.bybit_client: Instruments page 2: 500 symbols (total so far: 1000)
[INFO] core.bybit_client: Filtered to 412 USDT perpetuals with >5,000,000 24h turnover
[INFO] main: Scanning 412 symbols
```

### Step 7 — Production

After 24–48h on testnet:

1. Set `"testnet": false` in `config.json`
2. Install systemd service:

```bash
sudo cp deploy/bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now bot
sudo systemctl status bot
```

---

## Configuration

### `config.json`

```json
{
  "api": {
    "bybit_key": "YOUR_V5_API_KEY",
    "bybit_secret": "YOUR_V5_API_SECRET",
    "testnet": false,
    "discord_webhook": "https://discord.com/api/webhooks/..."
  },
  "deepseek": {
    "api_key": "sk-...",
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
    "temperature": 0.3,
    "max_tokens": 1500
  },
  "database": {
    "host": "localhost",
    "database": "bybit_bot_v9",
    "user": "postgres",
    "password": "",
    "port": "5432"
  },
  "system": {
    "check_interval_hours": 1,
    "max_concurrent": 10,
    "scan_top_n": 0,
    "min_turnover_24h": 5000000
  },
  "strategy": {
    "min_raw_threshold": 65.0,
    "min_calibrated": 0.72,
    "composite_weights": {
      "mean_reversion": 0.25,
      "basis": 0.15,
      "volume": 0.15,
      "derivatives": 0.30,
      "momentum": 0.15
    }
  },
  "risk": {
    "default_leverage": 5,
    "max_position_pct": 2.0,
    "max_open_positions": 5,
    "max_daily_loss_pct": 5.0,
    "max_drawdown_pct": 10.0
  },
  "execution": {
    "trailing_activation_pct": 1.0,
    "trailing_distance_pct": 0.5,
    "emergency_sl_pct": 2.5,
    "max_hold_hours": 72
  },
  "auto_tune": {
    "run_hour": 7,
    "auto_tune_min_samples": 30,
    "proven_edge_samples": 100
  }
}
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `check_interval_hours` | `1` | Hours between full market scans |
| `max_concurrent` | `10` | Concurrent symbol analyses (rate-limit safety) |
| `scan_top_n` | `0` | `0` = scan all pairs, `N>0` = top N by turnover |
| `min_turnover_24h` | `5000000` | Minimum 24h turnover in USDT |
| `min_raw_threshold` | `65.0` | Raw score must exceed this (0–100 scale) |
| `min_calibrated` | `0.72` | Calibrated confidence must exceed this |
| `max_position_pct` | `2.0` | Max % of equity per trade |
| `max_open_positions` | `5` | Hard cap on concurrent positions |
| `run_hour` | `7` | UTC hour for daily auto-improvement |

### `.env`

```bash
BYBIT_API_KEY=your_v5_key
BYBIT_API_SECRET=your_v5_secret
DEEPSEEK_API_KEY=sk-...
DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
DB_HOST=localhost
DB_USER=postgres
DB_PASSWORD=
DB_NAME=bybit_bot_v9
```

### Bybit API Key Permissions

Navigate to **Bybit → API Management → Create New Key**:

- ✅ **Read** (required for market data + account)
- ✅ **Trade** (required for order placement)
- ❌ **Withdraw** (never enable)
- ✅ **IP whitelist** (strongly recommended in production)

---

## Running the Bot

### Development

```bash
python main.py
```

### Production (systemd)

```bash
sudo systemctl start bot
sudo systemctl status bot
sudo journalctl -u bot -f
```

### Quick Restart Script

```bash
./deploy/restart_bot.sh
```

### Graceful Shutdown

`Ctrl+C` or `SIGTERM` triggers:

1. Cancel scan loop
2. Stop position monitor
3. Close all open positions (with confirmation)
4. Flush database connections
5. Exit cleanly

---

## How the Scanner Works

Every `check_interval_hours` (default 1h):

```
1. Fetch all instruments (paginated, 500/page)
   GET /v5/market/instruments-info?category=linear&limit=500
   → follows nextPageCursor until exhausted

2. Filter by quoteCoin="USDT", status="Trading",
   contractType="LinearPerpetual"

3. Fetch all tickers in one call (no symbol parameter)
   GET /v5/market/tickers?category=linear

4. Filter by turnover24h >= min_turnover_24h (default 5M USDT)

5. Sort by turnover descending

6. For each symbol (semaphore=10 concurrent):
   ├─ get_klines(60, 200)          → OHLCV
   ├─ compute_indicators()          → EMA, RSI, ATR, RVOL
   ├─ get_full_snapshot()           → CVD, OI, L/S ratio
   ├─ composite_engine.compute()    → Zeta, basis, volume
   ├─ scorer.compute(prelim)        → gate check (raw < 40 → skip)
   ├─ deepseek_agent.analyse()      → LLM confirmation
   ├─ scorer.compute(final)         → calibrated confidence
   ├─ journal.log_signal()          → ALWAYS log (for auto-tuner)
   └─ if high_confidence:
        executor.execute_signal()   → WS order + native TP/SL
```

### Rate-Limit Safety

- `asyncio.Semaphore(8)` caps concurrent REST requests
- Ticker fetch is batched (single request for all symbols)
- Exponential back-off on `retCode: 10006` (rate limited)
- Bybit's public limit (600 req/5s/IP) is respected with ~40× margin

### Expected Throughput

On typical hardware with a good connection:

| Symbol count | Scan duration | API calls |
|---|---|---|
| 100 pairs | ~2 min | ~700 |
| 400 pairs | ~7 min | ~2,800 |
| 800 pairs | ~15 min | ~5,600 |

---

## Quantitative Methods

All formulas implemented in `analysis/`. Full details in code docstrings.

### 1. Zeta (ζ) — Mean-Reversion Score

Unlike a raw Z-score, Zeta **weights deviation by the actual reversion speed** of the series.

**Ornstein-Uhlenbeck fit:**

```
X_t = α + φ · X_{t-1} + ε
θ = −ln(φ)                # reversion speed
HL = ln(2) / θ             # half-life in bars
μ = α / (1 − φ)            # long-run mean
σ = std(residuals)
```

**Zeta:**

```
z  = (X_t − μ) / σ
ζ  = z × (1 − φ) / (1 + φ)
```

**Interpretation:**

- `|ζ| > 1.5` → actionable reversion signal
- `φ → 1` (random walk) → factor → 0, signal suppressed
- `φ → 0` (fast reversion) → factor → 1, full strength

### 2. Hurst Exponent (R/S Analysis)

```
H < 0.45  → anti-persistent (mean-reverting regime)
H ≈ 0.50  → random walk
H > 0.55  → persistent (trending regime)
```

Used to **gate** the Zeta signal: Zeta is ignored when `H > 0.55` (don't fight trends).

### 3. Augmented Dickey-Fuller (ADF) Test

Tests whether the price series is stationary. Our implementation uses a lightweight OLS-based version without statsmodels.

```
p < 0.10  → stationary → mean reversion valid
p ≥ 0.10  → non-stationary → Zeta signal rejected
```

### 4. Basis Spread Rate

```
basis_abs  = Perp_mark − Spot_last
basis_pct  = basis_abs / Spot_last × 100

funding_annualized = funding_rate × 3 × 365 × 100   (Bybit 8h funding)
```

**Regime classification:**

| Basis % | Regime | Signal Bias |
|---|---|---|
| > +0.15% & funding > 0.03% | Contango overheated | `short_perp` (contrarian) |
| < −0.15% & funding < −0.03% | Backwardation capitulated | `long_perp` (contrarian) |
| Else | Neutral | `neutral` |

### 5. Volume Anomaly Detection

**Robust Z-score (resistant to outlier spikes):**

```
Z = (V_t − median(V, N)) / (1.4826 · MAD(V, N))
```

**Percentile rank:**

```
P = P(V_t > V_{t-N .. t-1})
```

**Anomaly classification:**

| Z | Percentile | Level |
|---|---|---|
| ≥ 3.0 | ≥ 0.98 | `extreme_high` |
| ≥ 2.0 | ≥ 0.90 | `high` |
| ≤ −2.0 | ≤ 0.05 | `extreme_low` |
| ≤ −1.5 | ≤ 0.15 | `low` |
| Else | — | `normal` |

**Volume-Weighted Price Divergence:**

```
VWPD = corr(sign(Δprice), Δvol_z)  over 10 bars
```

- `VWPD < −0.3` → price rising on shrinking volume → bearish divergence
- `VWPD > +0.3` with positive Z → confirmed breakout

### 6. CVD (Cumulative Volume Delta)

Computed from kline data using the close-vs-open directional proxy:

```
delta_t = +volume  if close > open
        = −volume  if close < open
        = 0        otherwise
CVD     = cumulative sum of delta
```

Divergence detection (20-bar lookback):

- Price ↑ + CVD ↓ → **bearish** (distribution)
- Price ↓ + CVD ↑ → **bullish** (accumulation)

### 7. Open Interest Divergence Matrix

| Price | OI | Interpretation |
|---|---|---|
| ↑ | ↑ | `bullish_continuation` (new money) |
| ↑ | ↓ | `short_covering` (weak rally) |
| ↓ | ↑ | `bearish_continuation` (short buildup) |
| ↓ | ↓ | `long_liquidation` (potential bottom) |

### 8. Composite Score

All sub-signals are normalized to `[−1, +1]` and weighted:

```
composite = Σ w_i × (contribution_i / normalizer_i)
```

Default weights:

| Component | Weight | Range |
|---|---|---|
| Mean reversion | 25% | [−15, +15] |
| Basis | 15% | [−10, +10] |
| Volume | 15% | [−10, +10] |
| Derivatives | 30% | [−25, +25] |
| Momentum | 15% | [−10, +10] |

Direction thresholds:

- `composite > +0.20` → long bias
- `composite < −0.20` → short bias
- else → neutral

Conviction:

- `|composite| ≥ 0.55` → high
- `|composite| ≥ 0.30` → medium
- else → low

---

## Signal Confidence Gate

A signal is only executed when **all three conditions** hold:

```
1. raw_total ≥ 65 / 100
2. calibrated_confidence ≥ 0.72
3. no hard gates triggered
```

### Score Components (max 100)

| Component | Max | Source |
|---|---|---|
| Technical | 20 | EMA alignment, RSI, RVOL |
| Composite quant | 20 | Zeta + basis + volume + derivatives + momentum |
| Derivatives | 20 | CVD + OI + L/S ratio divergence |
| Sentiment | 15 | Top-trader L/S + retail divergence |
| Pattern | 15 | Chart patterns |
| LLM | 10 | DeepSeek agreement + invalidation clarity |

### Hard Gates (immediate rejection)

- Funding rate > |0.05%|
- 24h avg volume < 1M USDT
- RSI > 85 (overbought extreme)
- OI change > 15% in 1h (anomaly)

### Bayesian Calibration

Until 30 live outcomes per score bucket accumulate, calibrated confidence is **shrunk toward 0.5**:

```
α = n / (n + 30)
calibrated = α × historical_wr + (1 − α) × 0.5
```

This prevents false precision on small samples — a critical fallacy guard.

---

## Auto-Improvement Loop

Runs daily at **07:00 UTC**. Compares last 24h signals against actual market outcomes.

### Step 1 — Outcome Resolution

For each unresolved signal:

1. Fetch klines from signal time to now
2. Walk forward through candles
3. Classify: `tp_hit`, `sl_hit`, `expired_profit`, `expired_loss`
4. Persist to `outcomes` table

### Step 2 — Parameter Tuning with Fallacy Guards

**Walk-forward split:** 70% train, 30% validation.

**Bonferroni correction:** Testing 16 parameter variations (8 params × 2 directions) sets `α_adjusted = 0.10 / 16 = 0.00625`.

**Minimum sample size:** 30 outcomes required before any tuning.

**Monotone safety:** While `n < 100` outcomes, parameters can only move in the **risk-reducing** direction:

| Parameter | Safe direction |
|---|---|
| `min_raw_threshold` | up (fewer signals) |
| `min_calibrated` | up |
| `min_rvol` | up |
| `risk_reward_min` | up |
| `trailing_activation_pct` | down (tighter trailing) |
| `trailing_distance_pct` | down |
| `zeta_entry_threshold` | up |
| `volume_z_high_threshold` | up |

**Deflated Sharpe Ratio:** Available for the strategy-level assessment.

### Tuned Parameters

- `min_raw_threshold` ∈ [50, 85]
- `min_calibrated` ∈ [0.55, 0.90]
- `min_rvol` ∈ [1.0, 4.0]
- `risk_reward_min` ∈ [1.5, 5.0]
- `trailing_activation_pct` ∈ [0.5, 3.0]
- `trailing_distance_pct` ∈ [0.2, 2.0]
- `zeta_entry_threshold` ∈ [1.0, 3.0]
- `volume_z_high_threshold` ∈ [1.5, 3.0]

### What the Fallacy Guards Actually Prevent

| Fallacy | Guard |
|---|---|
| Multiple comparisons inflate false positives | Bonferroni correction |
| In-sample overfitting | Walk-forward split |
| Storytelling bias (cherry-picked signals) | Minimum sample size + validation-set-only evaluation |
| LLM confidence treated as probability | LLM contributes at most 10/100 and only structural signals |
| False precision from small samples | Bayesian shrinkage toward 0.5 |
| Best-of-N random noise | Deflated Sharpe Ratio |
| Dangerous drift | Monotone safety + hard parameter bounds |

---

## Risk Management

### Position Sizing

```
size_usd = equity × max_position_pct / 100
size_usd *= min(1.5, max(0.5, confidence / 0.7))   # confidence scaling
size_usd  = min(size_usd, equity × leverage × max_position_pct / 100)
```

### Hard Risk Limits

| Limit | Default | Behavior |
|---|---|---|
| Max daily loss | 5% | Kill switch activates |
| Max drawdown | 10% | Kill switch activates |
| Max open positions | 5 | New signals rejected |
| Min risk-reward | 1.5 | Signal rejected |
| Emergency SL per position | 2.5% adverse | Immediate market close |
| Max hold time | 72h | Position closed at market |

### Trade Execution

Every entry is placed via **WebSocket Trade API** with:

1. Market order (IOC)
2. **Native TP/SL attached at entry** — the exchange manages protection even if the bot disconnects
3. Reduce-only flag on emergency closes
4. Trailing stop activated once profit ≥ 1%
5. Trailing distance 0.5% behind highest favorable price

### Kill Switch

When triggered (daily loss or max drawdown):

- All new signals rejected
- Existing positions still managed (trailing stops continue)
- Manual reset required: `risk_manager.reset_kill_switch()`

---

## File Structure

```
bybit_quant_v9/
├── main.py                          # Async orchestrator + full-pair scanner
├── config.json                      # Configuration
├── requirements.txt
├── .env                             # Secrets
├── README.md                        # This file
│
├── core/
│   ├── bybit_client.py              # V5 REST + WebSocket client
│   ├── health_monitor.py            # Heartbeat + watchdog
│   └── event_bus.py                 # In-process pub/sub
│
├── analysis/
│   ├── derivatives.py               # CVD, OI, L/S ratio
│   ├── confidence.py                # Calibrated scorer
│   ├── mean_reversion.py            # Zeta, OU, Hurst, ADF
│   ├── basis.py                     # Spot-perp basis spread
│   ├── volume_anomaly.py            # Robust Z + VWPD
│   ├── quant_composite.py           # Composite engine
│   ├── technicals.py                # EMA, RSI, ATR, RVOL, ADX
│   ├── smc.py                       # Order blocks, BOS, CHoCH
│   └── patterns.py                  # Chart patterns
│
├── agent/
│   ├── deepseek_agent.py            # LLM with tool-calling
│   └── prompts.py                   # System prompts
│
├── execution/
│   ├── trade_executor.py            # WS orders + position monitor
│   ├── risk_manager.py              # Sizing, drawdown, kill switch
│   └── order_state.py               # Lifecycle FSM
│
├── learning/
│   ├── signal_journal.py            # Persistence
│   ├── outcome_tracker.py           # 07:00 resolution
│   └── auto_tuner.py                # Walk-forward tuner
│
├── persistence/
│   ├── database.py                  # asyncpg pool + schema
│   └── models.py                    # Dataclasses
│
├── notification/
│   ├── discord_bot.py               # Rich embeds
│   └── chart_generator.py           # Matplotlib charts
│
└── deploy/
    ├── bot.service                  # systemd unit
    └── restart_bot.sh
```

---

## Troubleshooting

### Bot starts but no signals for hours

**Normal** if markets are choppy — the composite engine is designed to be quiet during random-walk regimes. Check:

```bash
grep "not high confidence" bot_v9.log | tail -20
```

If the vast majority show `raw=X (X < 40)`, the market is genuinely quiet. If they show `raw=50-64`, lower `min_raw_threshold` slightly.

### `Bybit API error 10006` (rate limited)

Reduce `max_concurrent` in config to 5. The semaphore-based rate limiter already paces requests but extreme conditions can still trigger back-off.

### `Bybit API error 10003` (invalid API key)

- Confirm the key has **Read + Trade** permissions on a **Unified Trading** account.
- Testnet keys only work with `"testnet": true`.
- IP whitelist must include your server's public IP.

### WebSocket disconnects repeatedly

The client auto-reconnects. If persistent:

- Check firewall allows outbound WSS on 443
- Reduce `check_interval_hours` to lighten request load
- Confirm system time is correct (`timedatectl`)

### DeepSeek returns non-JSON

The agent handles markdown fences and falls back to `{"direction": "neutral"}`. Check `DEEPSEEK_API_KEY` and account balance. Non-JSON responses do **not** block the quant pipeline.

### Database connection refused

```bash
sudo systemctl status postgresql
psql -d bybit_bot_v9 -c "SELECT COUNT(*) FROM signals;"
```

If schema is missing, run `python main.py` once — the schema is applied on connect.

### 07:00 tuner never fires

Check the log for `"Next auto-improvement in X.Xh"`. If it shows a value > 24h, your server timezone differs from UTC. Set server time to UTC:

```bash
sudo timedatectl set-timezone UTC
```

---

## Development

### Adding a New Quant Signal

1. Create `analysis/your_signal.py` with a dataclass output
2. Implement `analyze(df)` returning the dataclass
3. Include `confidence_contribution` field (clipped to a sane range)
4. Add weight + normalizer to `CompositeQuantEngine`
5. Wire into `quant_composite.compute()`
6. Add logging fields to `signal_journal.log_signal()`

### Running Tests

```bash
pytest tests/ -v
```

### Linting

```bash
ruff check .
black .
```

---

## Roadmap

- [ ] Binance USDT-M mirror support
- [ ] Portfolio-level correlation hedge
- [ ] Cross-exchange basis arbitrage
- [ ] Sentiment analysis from news + Twitter
- [ ] On-chain metrics (funding, staking flows)
- [ ] Live dashboard (FastAPI + WebSocket)
- [ ] Multi-account support

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/x`)
3. Commit changes (`git commit -am 'Add x'`)
4. Push (`git push origin feature/x`)
5. Open a Pull Request

Please include:

- Unit tests for new quant formulas
- Docstring with the mathematical derivation
- Justification for weight changes in the composite engine

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Disclaimer

**This software is for research and educational purposes only.**

Trading cryptocurrency futures involves substantial risk of loss and is not suitable for every investor. The past performance of any trading system or methodology is not necessarily indicative of future results.

- The auto-improvement loop adjusts parameters on historical data and cannot guarantee future profitability.
- Mean-reversion strategies fail during sustained trends.
- LLM-based analysis (DeepSeek) can be wrong, hallucinate, or produce inconsistent outputs — it contributes only 10/100 to the confidence score for this reason.
- Always run on **testnet first** for at least 2 weeks.
- Start with **small position sizes** on mainnet (1% max per trade).
- Never risk more than you can afford to lose.

The authors assume **no responsibility** for financial losses incurred through use of this software.

---

<div align="center">

**Built with care for the quant trading community**

[Report Bug](../../issues) · [Request Feature](../../issues) · [Documentation](docs/)

</div>